"""User-facing simulated experiments built on the PulseEngine protocol (PLAN.md Sections 7.5, 7.9; M2 for the single-ion
Rabi, Ramsey and sideband scans, M4 for the entangling-gate amplitude scan and the parity scan, M8 for the rest).

M4 (Section 7.5 item 4, Section 7.9): ``ms_scan`` plays the pair's waveform at scaled amplitudes and detuning offsets and
records P_00, P_01 + P_10, P_11 from |00>, fitting the closure amplitude at which P_00 = P_11 (chi = pi/4 has P_11 = sin^2 chi
= 1/2 with maximal slope); ``parity_scan`` follows the gate with a pi/2 analysis pulse of scanned phase on both ions and fits
the parity oscillation Pi(phi) = C cos(2 phi + phi_0) + B, whose contrast C bounds the Bell fidelity F = (P_00 + P_11 + C)/2
(Wright 2019).

Each returns data plus the fitted parameters with uncertainties; ``calibration/`` uses them. The M2 experiments drive one
ion of the device through the JOINT_EXACT engine with the derived drive of ``light/`` (Raman or single-photon optical
from the device's beams, microwave from a supplied Rabi frequency), an initial internal |0> and per-mode thermal
occupations; fits use the closed forms of Sections 4.2.7 and 13: the carrier Rabi curve with the thermal Debye-Waller
envelope sum_n P_n sin^2(Omega_n t/2), Omega_n = Omega e^{-eta^2/2} L_n(eta^2), and the Ramsey fringe
P = A cos(2 pi delta t + phi_0) + B.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import least_squares
from scipy.special import eval_genlaguerre

if TYPE_CHECKING:
    from qutip_trap.control.pulses import Drive
    from qutip_trap.control.schedule import GateDrive
    from qutip_trap.device.model import Device
    from qutip_trap.dynamics.engine import SolverOptions
    from qutip_trap.dynamics.hamiltonian import BuilderOptions
    from qutip_trap.hilbert.space import HilbertSpace

M8 = "milestone M8 (experiments/, PLAN.md Section 7.5)"


@dataclass(frozen=True)
class ExperimentResult:
    data: np.ndarray
    fitted: dict[str, tuple[float, float]]
    """Parameter -> (value, uncertainty)."""
    model: str
    provenance_id: str


# ---- shared machinery of the M2 experiments ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Setup:
    drive: Drive
    space: HilbertSpace
    eta_driven: float
    driven_mode: int | None
    rabi_hz: float
    nbar: dict[int, float]


def _setup(device: Device, ion: int, kw: dict[str, Any]) -> _Setup:
    from qutip_trap.control.schedule import default_gate_drives
    from qutip_trap.hilbert.operators import required_margin
    from qutip_trap.hilbert.space import HilbertSpace, ModeTruncation
    from qutip_trap.light.microwave import square_microwave_drive
    from qutip_trap.light.raman import derive_optical_drive, derive_raman_drive, square_drive

    nbar: dict[int, float] = {int(k): float(v) for k, v in dict(kw.get("nbar", {})).items()}
    gate_drive: GateDrive = kw.get("gate_drive") or default_gate_drives(device)[ion]
    detuning = float(kw.get("detuning_hz", 0.0))
    phase = float(kw.get("phase_rad", 0.0))
    n_modes = len(device.crystal.modes)
    if gate_drive.kind == "microwave":
        rabi = kw.get("rabi_hz")
        if rabi is None:
            raise ValueError("a microwave experiment needs rabi_hz (no beams to derive it from)")
        drive = square_microwave_drive(
            ion,
            float(rabi),
            detuning_hz=detuning,
            phase_rad=phase,
            stark_shift_hz=float(kw.get("stark_shift_hz", 0.0)),
        )
        etas = {m: 0.0 for m in range(n_modes)}
        rabi_hz = float(rabi)
    else:
        if gate_drive.kind == "raman":
            derived = derive_raman_drive(
                device, ion, (gate_drive.beams[0], gate_drive.beams[1]), scattering=False
            )
        else:
            derived = derive_optical_drive(device, ion, gate_drive.beams[0], scattering=False)
        drive = square_drive(
            derived, detuning_hz=detuning, phase_rad=phase, include_stark=bool(kw.get("include_stark", True))
        )
        etas = derived.etas
        rabi_hz = derived.carrier_rabi_hz
    space = kw.get("space")
    driven = max(range(n_modes), key=lambda m: abs(etas[m])) if n_modes else None
    eta_driven = abs(etas[driven]) if driven is not None else 0.0
    if space is None:
        ion_dims = tuple(2 for _ in range(device.crystal.n_ions))
        if driven is None or eta_driven == 0.0:
            space = HilbertSpace(ion_dims, (), None, tuple(range(n_modes)))
            driven = None
        else:
            nb = nbar.get(driven, 0.0)
            n_hi = int(math.ceil(nb + 5.0 * math.sqrt(nb * (nb + 1.0)) + 2.0))
            d = int(kw.get("d", n_hi + 1 + required_margin(eta_driven)))
            space = HilbertSpace(
                ion_dims,
                (ModeTruncation(driven, d, (0, min(n_hi, d - 1)), eta_driven * 1.5),),
                None,
                tuple(m for m in range(n_modes) if m != driven),
            )
    return _Setup(
        drive=drive, space=space, eta_driven=eta_driven, driven_mode=driven, rabi_hz=rabi_hz, nbar=nbar
    )


def _run(
    device: Device,
    ion: int,
    pulses: Sequence[Any],
    setup: _Setup,
    kw: dict[str, Any],
    store_times: Sequence[float] = (),
) -> Any:
    from qutip_trap.control.schedule import Schedule
    from qutip_trap.dynamics.engine import JointExactEngine, SeedSpec, SolverOptions
    from qutip_trap.noise.sampling import quiet_sample

    idle = tuple(kw.get("idle", ()))
    schedule = Schedule(tuple(pulses), idle, (), {q: 0.0 for q in range(device.crystal.n_ions)})
    options: SolverOptions = kw.get("options") or SolverOptions()
    bopts: BuilderOptions | None = kw.get("builder_options")
    engine = JointExactEngine(
        builder_options=bopts,
        store_per_segment=2,
        store_times_s=tuple(store_times),
        qubit_shifts_hz={int(k): float(v) for k, v in dict(kw.get("qubit_shifts_hz", {})).items()},
    )
    internal = [0] * device.crystal.n_ions
    state = setup.space.initial_state(internal, thermal=setup.nbar)
    return engine.run_pulses(
        device, schedule, state, setup.space, quiet_sample(), SeedSpec(int(kw.get("seed", 0))), options
    )


def _fit(
    model: Any,
    x0: Sequence[float],
    t: np.ndarray,
    y: np.ndarray,
    bounds: tuple[Sequence[float], Sequence[float]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    def resid(p: np.ndarray) -> np.ndarray:
        return np.asarray(model(p, t) - y)

    kwargs: dict[str, Any] = {"x_scale": "jac"}
    if bounds is not None:
        kwargs["bounds"] = bounds
    res = least_squares(resid, np.asarray(x0, dtype=float), **kwargs)
    dof = max(len(t) - len(x0), 1)
    s2 = float(np.sum(res.fun**2)) / dof
    try:
        cov = np.linalg.inv(res.jac.T @ res.jac) * s2
        err = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        err = np.full(len(x0), np.nan)
    return np.asarray(res.x), err


def thermal_rabi_model(p: np.ndarray, t: np.ndarray, eta: float, n_max: int = 200) -> np.ndarray:
    """P_1(t) = A sum_n P_n(nbar) sin^2(Omega_n t/2), Omega_n = 2 pi f e^{-eta^2/2} L_n(eta^2): p = (f_hz, nbar, A) (Section 4.2.7)."""
    f, nbar, amp = float(p[0]), max(float(p[1]), 0.0), float(p[2])
    n = np.arange(n_max)
    if nbar == 0.0:
        weights = np.zeros(n_max)
        weights[0] = 1.0
    else:
        weights = np.exp(n * math.log(nbar) - (n + 1) * math.log1p(nbar))
    omegas = 2.0 * math.pi * f * math.exp(-(eta**2) / 2.0) * eval_genlaguerre(n, 0, eta**2)
    out = amp * np.sum(weights[None, :] * np.sin(0.5 * omegas[None, :] * np.asarray(t)[:, None]) ** 2, axis=1)
    return np.asarray(out)


# ---- the M2 experiments ---------------------------------------------------------------------------------------------------------


def rabi_scan(device: Device, ion: int, durations_s: Sequence[float], **kw: Any) -> ExperimentResult:
    """Carrier (or sideband, with ``detuning_hz``) Rabi flopping of ``ion`` against pulse duration, fitted with the thermal
    Debye-Waller envelope (Section 4.2.7): fitted f_rabi_hz, nbar and contrast; data columns (t, P1)."""
    from qutip_trap.control.pulses import Pulse

    setup = _setup(device, ion, kw)
    ts = np.array(sorted(float(t) for t in durations_s))
    if ts.size < 2 or ts[0] < 0.0:
        raise ValueError("durations_s: at least two non-negative durations")
    t_max = float(ts[-1]) if ts[-1] > 0.0 else 1e-9
    pulse = Pulse(setup.drive, 0.0, t_max, "rabi_scan", ())
    traces = _run(device, ion, [pulse], setup, kw, store_times=[t for t in ts if 0.0 < t < t_max])
    p1 = np.interp(ts, traces.times_s, np.real(traces.expectations[f"P1[{ion}]"]))
    data = np.column_stack([ts, p1])
    eta = setup.eta_driven if float(kw.get("detuning_hz", 0.0)) == 0.0 else 0.0
    nb0 = setup.nbar.get(setup.driven_mode, 0.0) if setup.driven_mode is not None else 0.0
    guess = [setup.rabi_hz, nb0, 1.0]
    fitted: dict[str, tuple[float, float]] = {}
    if float(kw.get("detuning_hz", 0.0)) == 0.0 and ts.size >= 4:
        x, err = _fit(
            lambda p, t: thermal_rabi_model(p, t, eta),
            guess,
            ts,
            p1,
            bounds=([0.0, 0.0, 0.0], [np.inf, np.inf, 1.0]),
        )
        fitted = {
            "f_rabi_hz": (float(x[0]), float(err[0])),
            "nbar": (float(x[1]), float(err[1])),
            "contrast": (float(x[2]), float(err[2])),
        }
    return ExperimentResult(
        data=data, fitted=fitted, model="thermal_debye_waller_rabi", provenance_id="conv.rabi_frequency"
    )


def ramsey(device: Device, ion: int, delays_s: Sequence[float], **kw: Any) -> ExperimentResult:
    """pi/2 - delay - pi/2(phase ``analysis_phase_rad``) on ``ion`` at drive detuning ``detuning_hz``; data columns (delay, P1);
    fitted fringe P = A cos(2 pi delta t + phi_0) + B: delta_hz, contrast (Section 7.5, the Ramsey-frequency experiment's core)."""
    from qutip_trap.control.pulses import Pulse

    setup = _setup(device, ion, kw)
    detuning = float(kw.get("detuning_hz", 0.0))
    dead = float(device.hardware.dead_time_s)
    t_half = 0.25 / setup.rabi_hz
    analysis = float(kw.get("analysis_phase_rad", 0.0))
    rows = []
    for delay in sorted(float(x) for x in delays_s):
        if delay < 0.0:
            raise ValueError("delays are non-negative")
        gap = max(delay, 0.0)
        p1_pulse = Pulse(setup.drive, 0.0, t_half, "ramsey_1", ())
        start2 = t_half + gap
        base_phase = float(kw.get("phase_rad", 0.0))
        drive2 = replace(setup.drive, tones=(replace(setup.drive.tones[0], phase_rad=base_phase + analysis),))
        p2_pulse = Pulse(drive2, start2, start2 + t_half, "ramsey_2", ())
        idle = ((t_half, start2),) if gap > 0.0 else ()
        traces = _run(device, ion, [p1_pulse, p2_pulse], setup, {**kw, "idle": idle})
        rows.append((delay, float(np.real(traces.expectations[f"P1[{ion}]"][-1]))))
        _ = dead
    data = np.array(rows)
    fitted: dict[str, tuple[float, float]] = {}
    if data.shape[0] >= 4:
        t, y = data[:, 0], data[:, 1]

        def model(p: np.ndarray, tt: np.ndarray) -> np.ndarray:
            return np.asarray(p[0] * np.cos(2.0 * math.pi * p[1] * tt + p[2]) + p[3])

        guess = [0.5, detuning if detuning != 0.0 else 1.0 / max(float(t[-1]), 1e-9), analysis, 0.5]
        x, err = _fit(model, guess, t, y)
        fitted = {
            "contrast": (abs(float(x[0])), float(err[0])),
            "delta_hz": (float(x[1]), float(err[1])),
            "phi0_rad": (float(x[2]), float(err[2])),
            "offset": (float(x[3]), float(err[3])),
        }
    return ExperimentResult(
        data=data, fitted=fitted, model="ramsey_fringe", provenance_id="conv.detuning_symbols"
    )


def ramsey_frequency(device: Device, ion: int, delays_s: Sequence[float], **kw: Any) -> ExperimentResult:
    """Qubit frequency for the table: the frame (drive reference) frequency plus the fitted Ramsey fringe frequency, with the
    sign resolved by two scans at drive detunings +-``probe_hz`` (default 1 kHz); the scheduler never reads the true value."""
    probe = abs(float(kw.get("probe_hz", 1e3)))
    frame_hz = float(kw.get("frame_hz", 0.0))
    plus = ramsey(device, ion, delays_s, **{**kw, "detuning_hz": +probe})
    minus = ramsey(device, ion, delays_s, **{**kw, "detuning_hz": -probe})
    fp = abs(plus.fitted["delta_hz"][0]) if plus.fitted else math.nan
    fm = abs(minus.fitted["delta_hz"][0]) if minus.fitted else math.nan
    # a fringe at |probe - x| for the + scan and |probe + x| for the - scan pins the true offset x of the transition from the frame
    x = 0.5 * (fm - fp)
    unc = 0.5 * math.hypot(
        plus.fitted["delta_hz"][1] if plus.fitted else 0.0,
        minus.fitted["delta_hz"][1] if minus.fitted else 0.0,
    )
    data = np.vstack(
        [
            np.column_stack([plus.data, np.full(len(plus.data), +probe)]),
            np.column_stack([minus.data, np.full(len(minus.data), -probe)]),
        ]
    )
    return ExperimentResult(
        data=data,
        fitted={
            "qubit_offset_hz": (x, unc),
            "qubit_freq_hz": (frame_hz + x, unc),
            "fringe_plus_hz": (fp, 0.0),
            "fringe_minus_hz": (fm, 0.0),
        },
        model="ramsey_two_probe",
        provenance_id="conv.detuning_symbols",
    )


def sideband_spectroscopy(
    device: Device, ion: int, detunings_hz: Sequence[float], **kw: Any
) -> ExperimentResult:
    """P1 after a pulse of ``duration_s`` (default the carrier pi time) against the drive detuning; the fitted entries are the
    detunings of the local maxima nearest the carrier and the driven mode's first sidebands (Section 7.9)."""
    from qutip_trap.control.pulses import Pulse

    base = _setup(device, ion, {**kw, "detuning_hz": 0.0})
    duration = float(kw.get("duration_s", 0.5 / base.rabi_hz))
    rows = []
    for mu in sorted(float(x) for x in detunings_hz):
        setup = _setup(device, ion, {**kw, "detuning_hz": mu, "space": base.space})
        pulse = Pulse(setup.drive, 0.0, duration, "sideband_spectroscopy", ())
        traces = _run(device, ion, [pulse], setup, kw)
        rows.append((mu, float(np.real(traces.expectations[f"P1[{ion}]"][-1]))))
    data = np.array(rows)
    fitted: dict[str, tuple[float, float]] = {}
    mus, p1 = data[:, 0], data[:, 1]
    step = float(np.median(np.diff(mus))) if len(mus) > 1 else 0.0
    if base.driven_mode is not None:
        f_mode = device.crystal.modes[base.driven_mode].omega_hz
        for name, centre in (("carrier_hz", 0.0), ("blue_sideband_hz", f_mode), ("red_sideband_hz", -f_mode)):
            window = np.abs(mus - centre) <= max(0.25 * f_mode, 2 * step)
            if np.any(window):
                idx = int(np.argmax(np.where(window, p1, -np.inf)))
                fitted[name] = (float(mus[idx]), step)
        if "blue_sideband_hz" in fitted and "red_sideband_hz" in fitted:
            fitted["mode_hz"] = (0.5 * (fitted["blue_sideband_hz"][0] - fitted["red_sideband_hz"][0]), step)
    else:
        idx = int(np.argmax(p1))
        fitted["carrier_hz"] = (float(mus[idx]), step)
    return ExperimentResult(
        data=data, fitted=fitted, model="sideband_spectrum_peaks", provenance_id="conv.detuning_symbols"
    )


# ---- milestone M8 ----------------------------------------------------------------------------------------------------------------


def micromotion_scan(
    device: Device,
    ion: int,
    beam: int,
    shim_ranges_v: Mapping[str, tuple[float, float]],
    method: str = "rf_photon_correlation",
    **kw: Any,
) -> ExperimentResult:
    raise NotImplementedError(f"micromotion_scan is {M8}")


def _entangling_setup(
    device: Device, pair: tuple[int, int], kw: dict[str, Any]
) -> tuple[Any, dict[int, Any], Any, Any, Any]:
    """(waveform, gate drives, table, modes, space) for the pair from the keyword arguments or the device's defaults."""
    from qutip_trap.calibration.entangling import gate_space
    from qutip_trap.control.schedule import default_gate_drives
    from qutip_trap.control.shaping import gate_modes

    table = kw.get("table")
    waveform = kw.get("waveform") or (table.waveform_for(pair) if table is not None else None)
    if waveform is None:
        raise ValueError(
            "ms_scan/parity_scan need the pair's waveform (kw waveform=, or a table with an ms entry)"
        )
    drives: dict[int, Any] = kw.get("gate_drives") or default_gate_drives(device)
    beams = drives[pair[0]].beams
    if len(beams) != 2:
        raise ValueError("the entangling experiments take a Raman (two-beam) gate drive")
    nbar: dict[int, float] = {int(k): float(v) for k, v in dict(kw.get("nbar", {})).items()}
    modes = kw.get("modes") or gate_modes(device, pair, (beams[0], beams[1]), nbar=nbar)
    space = kw.get("space") or gate_space(modes, device.crystal.n_ions, nbar=nbar, waveform=waveform)
    if table is None:
        raise ValueError(
            "the entangling experiments read the CalibrationTable (Stark and Rabi entries): pass table="
        )
    return waveform, drives, table, modes, space


def ms_scan(
    device: Device,
    pair: tuple[int, int],
    amplitudes: Sequence[float],
    detunings_hz: Sequence[float],
    **kw: Any,
) -> ExperimentResult:
    """Populations after the pair's waveform at amplitude scale factors ``amplitudes`` and detuning offsets ``detunings_hz`` (added to
    every leg's beat note, red legs moving opposite so the tones stay symmetric); data columns (scale, offset_hz, P00, P01 + P10, P11);
    fitted: the closure scale where P_00 = P_11 at the smallest |offset| (the chi = pi/4 amplitude, Section 7.5 item 4)."""
    from qutip_trap.calibration.entangling import exact_gate_check
    from qutip_trap.control.shaping import scaled

    waveform, drives, table, _modes, space = _entangling_setup(device, pair, kw)
    rows = []
    for off in sorted(float(x) for x in detunings_hz):
        wf_off = _shift_detuning(waveform, off)
        for s in sorted(float(x) for x in amplitudes):
            if s <= 0.0:
                raise ValueError("amplitude scale factors are positive")
            check, _ = exact_gate_check(
                device,
                scaled(wf_off, s),
                pair,
                drives,
                table,
                space=space,
                nbar={int(k): float(v) for k, v in dict(kw.get("nbar", {})).items()},
                options=kw.get("options"),
                builder_options=kw.get("builder_options"),
            )
            rows.append((s, off, check.populations["P00"], check.leakage, check.populations["P11"]))
    data = np.array(rows)
    fitted: dict[str, tuple[float, float]] = {}
    offs = np.unique(data[:, 1])
    off0 = float(offs[np.argmin(np.abs(offs))])
    sel = data[:, 1] == off0
    sc, p00, p11 = data[sel, 0], data[sel, 2], data[sel, 4]
    diff = p00 - p11
    crossings = np.flatnonzero(np.diff(np.sign(diff)) != 0)
    if crossings.size:
        i = int(crossings[0])
        x0, x1, y0, y1 = sc[i], sc[i + 1], diff[i], diff[i + 1]
        root = float(x0 - y0 * (x1 - x0) / (y1 - y0)) if y1 != y0 else float(x0)
        fitted["closure_scale"] = (root, float(abs(x1 - x0)))
    return ExperimentResult(
        data=data, fitted=fitted, model="ms_population_scan", provenance_id="conv.ms_closure"
    )


def _shift_detuning(waveform: Any, offset_hz: float) -> Any:
    """Every blue leg + offset, every red leg - offset (the symmetric detuning scan of Section 7.5)."""
    from dataclasses import replace

    from qutip_trap.control.table import Segment

    if offset_hz == 0.0 or waveform.segments is None:
        return waveform

    def shift(v: Any, sign: float) -> Any:
        if callable(v):
            fn = v
            return lambda tau: float(fn(tau)) + sign * offset_hz
        return float(v) + sign * offset_hz

    segs = tuple(
        Segment(
            s.duration_s,
            dict(s.amplitude_hz),
            dict(s.phase_rad),
            {leg: shift(v, 1.0 if leg == "blue" else -1.0) for leg, v in s.detuning_hz.items()},
        )
        for s in waveform.segments
    )
    return replace(waveform, segments=segs)


def parity_scan(
    device: Device, pair: tuple[int, int], analysis_phases_rad: Sequence[float], **kw: Any
) -> ExperimentResult:
    """Parity after the gate and a pi/2 analysis pulse of phase phi on both ions (Section 7.9); data columns (phi, parity, P00, P11);
    fitted: contrast C, phase phi_0 and offset of Pi(phi) = C cos(2 phi + phi_0) + B, and the Bell-fidelity bound (P_00 + P_11 + C)/2
    with the populations read without the analysis pulse."""
    from qutip_trap.calibration.entangling import exact_gate_check, parity_after_analysis_pulse
    from qutip_trap.control.schedule import carrier_rabi_hz

    waveform, drives, table, _modes, space = _entangling_setup(device, pair, kw)
    rabi = kw.get("analysis_rabi_hz")
    if rabi is None:
        rabi = {q: carrier_rabi_hz(table, q, drives[q]) for q in pair}
    nbar = {int(k): float(v) for k, v in dict(kw.get("nbar", {})).items()}
    rows = []
    for phi in sorted(float(x) for x in analysis_phases_rad):
        par, pops = parity_after_analysis_pulse(
            device,
            waveform,
            pair,
            drives,
            table,
            space=space,
            analysis_phase_rad=phi,
            analysis_rabi_hz=rabi,
            nbar=nbar,
            options=kw.get("options"),
            builder_options=kw.get("builder_options"),
        )
        rows.append((phi, par, pops["P00"], pops["P11"]))
    data = np.array(rows)
    fitted: dict[str, tuple[float, float]] = {}
    if data.shape[0] >= 4:
        phis, parity = data[:, 0], data[:, 1]

        def model(p: np.ndarray, x: np.ndarray) -> np.ndarray:
            return np.asarray(p[0] * np.cos(2.0 * x + p[1]) + p[2])

        x, err = _fit(
            model, [float(np.max(parity) - np.min(parity)) / 2.0, 0.0, float(np.mean(parity))], phis, parity
        )
        contrast = abs(float(x[0]))
        fitted = {
            "contrast": (contrast, float(err[0])),
            "phi0_rad": (float(x[1]) + (math.pi if x[0] < 0 else 0.0), float(err[1])),
            "offset": (float(x[2]), float(err[2])),
        }
        base, _ = exact_gate_check(
            device,
            waveform,
            pair,
            drives,
            table,
            space=space,
            nbar=nbar,
            options=kw.get("options"),
            builder_options=kw.get("builder_options"),
        )
        fitted["bell_fidelity_bound"] = (
            0.5 * (base.populations["P00"] + base.populations["P11"] + contrast),
            float(err[0]) / 2.0,
        )
    return ExperimentResult(
        data=data, fitted=fitted, model="parity_oscillation", provenance_id="conv.entangling_angle"
    )


def heating_rate(device: Device, mode: int, delays_s: Sequence[float], **kw: Any) -> ExperimentResult:
    raise NotImplementedError(f"heating_rate is {M8}")


def detection_histogram(device: Device, ion: int, n_records: int, **kw: Any) -> ExperimentResult:
    """Section 7.5 item 5 (M5): histogram bright and dark photon counts on the simulated readout model of ion ``ion`` and
    choose the threshold and window minimizing the average error.

    The rates come from the M3a Bloch model of the device's detection beams at the ion (``detection_beams`` overrides the
    beams near the species' cycling wavelength; ``scheme`` a :class:`~qutip_trap.readout.fluorescence.ReadoutScheme`;
    ``levels`` the included fine-structure levels; ``windows_s`` the candidate bin times, default 0.25 to 2.5 times the
    detector's window; ``seed`` the record generator). ``data`` holds the bright and dark histograms at the chosen window
    (rows) and ``fitted`` the threshold, window, eps_B, eps_D and the fitted rates with their uncertainties.
    """
    from qutip_trap.calibration.readout import calibrate_detection
    from qutip_trap.light.roles import detection_beams
    from qutip_trap.readout.detection import RecordModel
    from qutip_trap.readout.fluorescence import detection_rates_for_ion

    species = device.crystal.species[ion]
    beams = kw.get("detection_beams")
    if beams is None:
        beams = [device.beams[k] for k in detection_beams(device, ion)]
    rates, scheme, _model = detection_rates_for_ion(
        species,
        device.field.B_gauss,
        device.field.direction,
        beams,
        position_m=tuple(float(x) for x in device.crystal.positions_m[ion]),
        levels=kw.get("levels"),
        scheme=kw.get("scheme"),
    )
    record_model = RecordModel.from_rates(rates, device.detector)
    windows = kw.get("windows_s")
    if windows is None:
        windows = tuple(float(x) for x in np.geomspace(0.25, 2.5, 12) * device.detector.window_s)
    cal = calibrate_detection(
        record_model,
        scheme,
        windows_s=windows,
        n_records=n_records,
        rng=np.random.default_rng(int(kw.get("seed", 0))),
        fitted_at_s=float(kw.get("t0_s", 0.0)),
    )
    n = max(cal.bright_histogram.size, cal.dark_histogram.size)
    data = np.zeros((2, n))
    data[0, : cal.bright_histogram.size] = cal.bright_histogram
    data[1, : cal.dark_histogram.size] = cal.dark_histogram
    fitted = {name: (e.value, e.uncertainty) for name, e in cal.entries.items()}
    fitted["R_bright_scattered_per_s"] = (rates.R_bright_per_s, 0.0)
    return ExperimentResult(
        data=data, fitted=fitted, model="detection_histogram", provenance_id="conv.readout_figure_of_merit"
    )


__all__ = [
    "ExperimentResult",
    "detection_histogram",
    "heating_rate",
    "micromotion_scan",
    "ms_scan",
    "parity_scan",
    "rabi_scan",
    "ramsey",
    "ramsey_frequency",
    "sideband_spectroscopy",
    "thermal_rabi_model",
]
