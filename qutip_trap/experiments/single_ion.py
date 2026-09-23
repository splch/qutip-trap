"""Single-ion experiments on the JOINT_EXACT engine: Rabi flopping, Ramsey, the Ramsey frequency, sideband spectroscopy.

The most strongly coupled mode (or ``mode``) is resolved; the other coupled modes are frozen and their thermal
Debye-Waller statistics averaged exactly. Shared ``**kw``: ``nbar``, ``gate_drive``, ``rabi_hz`` (microwave),
``detuning_hz``, ``phase_rad``, ``include_stark``, ``space``, ``d``, ``crosstalk``, ``rf_locked``, ``rf_phase_rad``,
``sample``, ``table``, ``qubit_shifts_hz``, ``frame``, ``builder_options``, ``options``, ``shots``, ``readout``, ``seed``,
``branch_weight_min``, ``fock_branches``, ``device_channels``, ``internal``.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from qutip_trap.experiments.fitting import (
    Observation,
    ReadoutErrors,
    fit_lineshape,
    multimode_rabi_model,
    ramsey_model,
    readout_errors_for,
    sigmas_or_none,
    thermal_rabi_model,
    weighted_fit,
)
from qutip_trap.experiments.result import (
    ExperimentResult,
    RabiScan,
    RamseyFringe,
    ScanParameters,
    SidebandSpectrum,
    realized_drive,
    requested_drive,
)
from qutip_trap.machine import Machine, laboratory_kwargs

if TYPE_CHECKING:
    from qutip_trap.control.pulses import Drive
    from qutip_trap.control.schedule import GateDrive
    from qutip_trap.device.model import Device
    from qutip_trap.dynamics.hamiltonian import BuilderOptions
    from qutip_trap.hilbert.space import HilbertSpace
    from qutip_trap.machine import Machine
    from qutip_trap.noise.sampling import NoiseSample


# ---- shared machinery -------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Setup:
    drive: Drive
    space: HilbertSpace
    eta_driven: float
    driven_mode: int | None
    rabi_hz: float
    """The physical carrier Rabi frequency of the drive, not the table's belief."""
    nbar: dict[int, float]
    etas: dict[int, float]
    stark_shift_hz: float
    gate_drive: GateDrive


def thermal_n_max(nbar: float, tail: float = 1e-7) -> int:
    """The Fock level above which a thermal state of ``nbar`` holds less than ``tail`` of its population, plus two, from
    the geometric tail (nbar/(nbar + 1))^(n + 1) < tail (a Gaussian rule under-sizes a hot state's exponential tail)."""
    if nbar <= 0.0:
        return 2
    return int(math.ceil(math.log(tail) / math.log(nbar / (nbar + 1.0)))) + 2


def _builder_options(kw: dict[str, Any]) -> BuilderOptions | None:
    bopts = kw.get("builder_options")
    if bopts is not None:
        return bopts  # type: ignore[no-any-return]
    frame = kw.get("frame")
    if frame is None:
        return None
    from qutip_trap.dynamics.hamiltonian import BuilderOptions

    return BuilderOptions(frame=frame)


def _observation(device: Device, kw: dict[str, Any]) -> Observation:
    """The observation model from the keyword arguments: ``shots`` and ``readout`` (True builds the device's readout errors)."""
    readout = kw.get("readout")
    errors: ReadoutErrors | None
    if readout is True:
        errors = readout_errors_for(device, kw.get("table"))
    elif readout is False or readout is None:
        errors = None
    else:
        errors = readout
    sample: NoiseSample | None = kw.get("sample")
    shots = kw.get("shots")
    return Observation(
        shots=None if shots is None else int(shots),
        readout=errors,
        seed=int(kw.get("seed", 0)),
        sample_id=0 if sample is None else int(sample.sample_id),
        stream=str(kw.get("stream", "")),
    )


def sub_stream(kw: Mapping[str, Any], label: str) -> dict[str, Any]:
    """``kw`` for a sub-experiment with its own shot-noise stream: ``label`` appended to the parent's stream."""
    parent = str(kw.get("stream", ""))
    return {**kw, "stream": f"{parent}/{label}" if parent else label}


def _setup(device: Device, ion: int, kw: dict[str, Any]) -> _Setup:
    from qutip_trap.control.schedule import default_gate_drives
    from qutip_trap.hilbert.operators import required_margin
    from qutip_trap.hilbert.space import HilbertSpace, ModeTruncation
    from qutip_trap.light.microwave import square_microwave_drive
    from qutip_trap.light.raman import (
        crosstalk_ratios,
        derive_optical_drive,
        derive_raman_drive,
        square_drive,
    )

    nbar: dict[int, float] = {int(k): float(v) for k, v in dict(kw.get("nbar", {})).items()}
    gate_drive: GateDrive = default_gate_drives(device)[ion]
    detuning = float(kw.get("detuning_hz", 0.0))
    phase = float(kw.get("phase_rad", 0.0))
    n_modes = len(device.crystal.modes)
    stark = 0.0
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
        xt = (
            dict(crosstalk_ratios(device, ion, gate_drive.beams, kind=gate_drive.kind))
            if kw.get("crosstalk", False)
            else None
        )
        scale = float(kw.get("amplitude_scale", 1.0))  # the rf amplitude relative to the beams' nominal power
        drive = square_drive(
            derived,
            detuning_hz=detuning,
            phase_rad=phase,
            rabi_scale=scale,
            include_stark=bool(kw.get("include_stark", True)),
            crosstalk=xt,
            rf_locked=bool(kw.get("rf_locked", False)),
            rf_phase_rad=kw.get("rf_phase_rad") if kw.get("rf_locked", False) else None,
        )
        if scale != 1.0 and kw.get("include_stark", True):
            # a two-photon shift scales with the intensity, a single-photon one with its square
            power = 1 if derived.kind == "raman" else 2
            drive = replace(drive, stark_shift_hz=float(derived.stark_shift_hz) * scale**power)
        etas = derived.etas
        rabi_hz = derived.carrier_rabi_hz * scale
        stark = float(drive.stark_shift_hz) if kw.get("include_stark", True) else 0.0  # type: ignore[arg-type]
    space = kw.get("space")
    if kw.get("mode") is not None:
        driven: int | None = int(kw["mode"])
        assert driven is not None
        if abs(etas[driven]) == 0.0:
            raise ValueError(f"the drive does not couple ion {ion} to mode {driven} (eta = 0)")
    else:
        driven = max(range(n_modes), key=lambda m: abs(etas[m])) if n_modes else None
    eta_driven = abs(etas[driven]) if driven is not None else 0.0
    if space is None:
        ion_dims = tuple(2 for _ in range(device.crystal.n_ions))
        if driven is None or eta_driven == 0.0:
            space = HilbertSpace(ion_dims, (), None, tuple(range(n_modes)))
            driven = None
        else:
            # a Gaussian cap for the cold Fock-branch experiments (a hot mixture under mesolve takes ``thermal_n_max``)
            nb = nbar.get(driven, 0.0)
            n_hi = int(math.ceil(nb + 5.0 * math.sqrt(nb * (nb + 1.0)) + 2.0))
            if kw.get("n_max") is not None:
                n_hi = max(n_hi, int(kw["n_max"]))
            d = int(kw.get("d", n_hi + 1 + required_margin(eta_driven)))
            space = HilbertSpace(
                ion_dims,
                (ModeTruncation(driven, d, (0, min(n_hi, d - 1)), eta_driven * 1.5),),
                None,
                tuple(m for m in range(n_modes) if m != driven),
            )
    return _Setup(
        drive=drive,
        space=space,
        eta_driven=eta_driven,
        driven_mode=driven,
        rabi_hz=rabi_hz,
        nbar=nbar,
        etas=etas,
        stark_shift_hz=stark,
        gate_drive=gate_drive,
    )


@dataclass(frozen=True)
class _Averaged:
    """Traces averaged over the initial-mixture branches: the expectation values against time and the final occupations."""

    times_s: np.ndarray
    expectations: dict[str, np.ndarray]
    nbar: dict[int, float]
    dropped_weight: float
    branches: int

    def p1(self, ion: int) -> np.ndarray:
        return np.asarray(np.real(self.expectations[f"P1[{ion}]"]), dtype=float)

    def final_p1(self, ion: int) -> float:
        return float(self.p1(ion)[-1])


def _branches(
    space: HilbertSpace,
    nbar: dict[int, float],
    etas: dict[int, float],
    weight_min: float,
    fock_resolved: bool,
) -> tuple[list[tuple[float, dict[int, int], dict[int, int]]], float]:
    """(weight, resolved Fock states, frozen Fock states) of the thermal initial mixture over the coupled modes, and the
    dropped weight: the frozen coupled modes always, the resolved modes when ``fock_resolved``."""
    from qutip_trap.hilbert.operators import thermal_populations

    options: list[tuple[int, bool, list[tuple[int, float]]]] = []
    for m, nb in nbar.items():
        if nb <= 0.0 or abs(etas.get(m, 0.0)) <= 1e-12:
            continue
        cls = space.mode_class(m)
        if cls == "frozen":
            resolved_flag = False
        elif cls == "resolved" and fock_resolved:
            resolved_flag = True
        else:
            continue
        d = int(math.ceil(math.log(weight_min) / math.log(nb / (1.0 + nb)))) + 2
        probs = thermal_populations(nb, max(d, 2))
        opts = [(n, float(p)) for n, p in enumerate(probs) if p >= weight_min]
        options.append((m, resolved_flag, opts or [(0, 1.0)]))
    if not options:
        return [(1.0, {}, {})], 0.0
    out: list[tuple[float, dict[int, int], dict[int, int]]] = []
    for choice in itertools.product(*[o[2] for o in options]):
        w = float(np.prod([p for _n, p in choice]))
        if w < weight_min:
            continue
        res = {options[k][0]: n for k, (n, _p) in enumerate(choice) if options[k][1]}
        fro = {options[k][0]: n for k, (n, _p) in enumerate(choice) if not options[k][1]}
        out.append((w, res, fro))
    total = sum(w for w, _r, _f in out)
    return [(w / total, r, f) for w, r, f in out], float(max(1.0 - total, 0.0))


def _run(
    device: Device,
    ion: int,
    pulses: Sequence[Any],
    setup: _Setup,
    kw: dict[str, Any],
    store_times: Sequence[float] = (),
) -> _Averaged:
    """Play ``pulses`` from |internal> x thermal(nbar) on the setup's space, averaged over the initial-mixture branches."""
    from qutip_trap.control.schedule import Schedule
    from qutip_trap.dynamics.engine import JointExactEngine, SeedSpec, SolverOptions
    from qutip_trap.noise.sampling import NoiseSample, key_frozen_n, quiet_sample

    idle = tuple(kw.get("idle", ()))
    schedule = Schedule(tuple(pulses), idle, (), {q: 0.0 for q in range(device.crystal.n_ions)})
    options: SolverOptions = kw.get("options") or SolverOptions()
    bopts = _builder_options(kw)
    base_sample: NoiseSample = kw.get("sample") or quiet_sample()
    engine = JointExactEngine(
        builder_options=bopts,
        store_per_segment=int(kw.get("store_per_segment", 2)),
        store_times_s=tuple(store_times),
        qubit_shifts_hz={int(k): float(v) for k, v in dict(kw.get("qubit_shifts_hz", {})).items()},
        table=kw.get("table"),
        device_channels=bool(kw.get("device_channels", False)),
    )
    internal = list(kw.get("internal") or [0] * device.crystal.n_ions)
    seeds = SeedSpec(int(kw.get("seed", 0)))
    weight_min = float(kw.get("branch_weight_min", 1e-3))
    branches, dropped = _branches(
        setup.space, setup.nbar, setup.etas, weight_min, bool(kw.get("fock_branches", True))
    )
    times: np.ndarray | None = None
    acc: dict[str, np.ndarray] = {}
    nbar_final: dict[int, float] = {}
    for weight, res_fock, frozen_fock in branches:
        thermal = {m: v for m, v in setup.nbar.items() if m not in res_fock}
        state = setup.space.initial_state(internal, fock=res_fock, thermal=thermal)
        values = dict(base_sample.values)
        values.update({key_frozen_n(m): float(n) for m, n in frozen_fock.items()})
        sample = NoiseSample(
            sample_id=base_sample.sample_id,
            values=values,
            ou_grids=dict(base_sample.ou_grids),
            t_s=base_sample.t_s,
        )
        tr = engine.run_pulses(device, schedule, state, setup.space, sample, seeds, options)
        if times is None:
            times = np.asarray(tr.times_s, dtype=float)
        for key, arr in tr.expectations.items():
            vals = np.asarray(arr)
            if key in acc and acc[key].shape == vals.shape:
                acc[key] = acc[key] + weight * vals
            elif key not in acc:
                acc[key] = weight * vals
            else:
                # differing time grids (a boundary retry): interpolate onto the first grid
                acc[key] = acc[key] + weight * np.interp(times, np.asarray(tr.times_s), np.real(vals))
        for m, v in tr.final.motional.nbar.items():
            nbar_final[m] = nbar_final.get(m, 0.0) + weight * float(v)
    assert times is not None
    return _Averaged(times, acc, nbar_final, dropped, len(branches))


# ---- the experiments -----------------------------------------------------------------------------------------------------------


def rabi_scan(machine: Machine, ion: int, durations_s: Sequence[float], **kw: Any) -> ExperimentResult:
    """Scan the pulse duration of Rabi flopping (carrier, or sideband with ``detuning_hz``) on ``ion``; returns a
    ``RabiScan`` fitted with the thermal Debye-Waller envelope (f_rabi_hz, nbar, contrast). ``nbar_fixed`` (from the
    thermometry) fits (f_rabi, contrast, offset) instead: a free-nbar fit over too short a scan is degenerate."""
    device, kw = laboratory_kwargs(machine, kw)
    from qutip_trap.control.pulses import Pulse

    setup = _setup(device, ion, kw)
    obs = _observation(device, kw)
    ts = np.array(sorted(float(t) for t in durations_s))
    if ts.size < 2 or ts[0] < 0.0:
        raise ValueError("durations_s: at least two non-negative durations")
    t_max = float(ts[-1]) if ts[-1] > 0.0 else 1e-9
    pulse = Pulse(setup.drive, 0.0, t_max, "rabi_scan", ())
    avg = _run(device, ion, [pulse], setup, kw, store_times=[t for t in ts if 0.0 < t < t_max])
    exact = np.interp(ts, avg.times_s, avg.p1(ion))
    measured = [obs.p1(float(p), ion, "rabi_scan", k) for k, p in enumerate(exact)]
    p1 = np.array([m[0] for m in measured])
    sigma = sigmas_or_none([m[1] for m in measured])
    data = np.column_stack([ts, p1])
    detuning = float(kw.get("detuning_hz", 0.0))
    eta = setup.eta_driven if detuning == 0.0 else 0.0
    nb0 = setup.nbar.get(setup.driven_mode, 0.0) if setup.driven_mode is not None else 0.0
    fitted: dict[str, tuple[float, float]] = {}
    converged = True
    notes: list[str] = []
    if avg.dropped_weight > 0.0:
        notes.append(
            f"initial-mixture branches below {kw.get('branch_weight_min', 1e-3):g} dropped: weight {avg.dropped_weight:.2e}"
        )
    if ts.size >= 4:
        nbar_fixed = kw.get("nbar_fixed")
        if eta == 0.0 and nbar_fixed is None:
            nbar_fixed = (
                0.0  # no Debye-Waller envelope to fit nbar from (a sideband scan, a microwave carrier)
            )
        if nbar_fixed is None:
            fit = weighted_fit(
                lambda p, t: thermal_rabi_model(p, t, eta),
                [setup.rabi_hz, nb0, 1.0],
                ts,
                p1,
                sigma=sigma,
                bounds=([0.0, 0.0, 0.0], [np.inf, np.inf, 1.0]),
            )
            fitted = {
                "f_rabi_hz": fit.value(0),
                "nbar": fit.value(1),
                "contrast": fit.value(2),
            }
        else:
            # every coupled mode's Debye-Waller factor: the driven mode at nbar_fixed, the spectators at ``nbar``
            nb_fixed = float(nbar_fixed)
            etas_all = [abs(e) for m, e in sorted(setup.etas.items()) if e != 0.0]
            nbars_all = [
                nb_fixed if m == setup.driven_mode else float(setup.nbar.get(m, 0.0))
                for m, e in sorted(setup.etas.items())
                if e != 0.0
            ]
            if eta == 0.0:
                etas_all, nbars_all = [], []
            fit = weighted_fit(
                lambda p, t: multimode_rabi_model(p, t, etas_all, nbars_all),
                [setup.rabi_hz, 1.0, 0.0],
                ts,
                p1,
                sigma=sigma,
                bounds=([0.0, 0.0, -0.5], [np.inf, 1.5, 0.5]),
            )
            fitted = {
                "f_rabi_hz": fit.value(0),
                "contrast": fit.value(1),
                "offset": fit.value(2),
                "nbar": (nb_fixed, 0.0),
            }
        fitted["chi2_per_dof"] = (fit.chi2_per_dof, 0.0)
        converged = fit.converged and fitted["f_rabi_hz"][0] > 0.0
        if not fit.converged:
            notes.append(f"Rabi fit did not converge: {fit.message}")
    return RabiScan(
        data=data,
        fitted=fitted,
        model="thermal_debye_waller_rabi",
        provenance_id="conv.rabi_frequency",
        converged=converged,
        notes=tuple(notes),
        sigma=sigma,
        requested=ScanParameters({"durations_s": ts, **requested_drive(setup.drive, t_max)}),
        realized=ScanParameters({"durations_s": ts, **realized_drive(device, setup.drive, t_max)}),
        chi2=fitted["chi2_per_dof"][0] if "chi2_per_dof" in fitted else None,
        subject={"ion": int(ion), "beam": setup.gate_drive.table_key_beam},
    )


def ramsey(machine: Machine, ion: int, delays_s: Sequence[float], **kw: Any) -> ExperimentResult:
    """Scan the delay of pi/2 - delay - pi/2(``analysis_phase_rad``) on ``ion``; returns a ``RamseyFringe`` fitted with
    P = A cos(2 pi delta t + phi_0) + B. ``pi_half_s`` defaults to 1/(4 f) at ``rabi_hz_belief`` (else the physical Rabi
    frequency); ``delay_pulses`` is a callable (start, end) -> pulses filling the delay."""
    device, kw = laboratory_kwargs(machine, kw)
    from qutip_trap.control.pulses import Pulse

    setup = _setup(device, ion, kw)
    obs = _observation(device, kw)
    detuning = float(kw.get("detuning_hz", 0.0))
    belief = kw.get("rabi_hz_belief")
    t_half = float(kw.get("pi_half_s", 0.25 / (float(belief) if belief else setup.rabi_hz)))
    analysis = float(kw.get("analysis_phase_rad", 0.0))
    delay_pulses = kw.get("delay_pulses")
    rows: list[tuple[float, float]] = []
    sig: list[float | None] = []
    for k, delay in enumerate(sorted(float(x) for x in delays_s)):
        if delay < 0.0:
            raise ValueError("delays are non-negative")
        gap = max(delay, 0.0)
        p1_pulse = Pulse(setup.drive, 0.0, t_half, "ramsey_1", ())
        start2 = t_half + gap
        base_phase = float(kw.get("phase_rad", 0.0))
        drive2 = replace(setup.drive, tones=(replace(setup.drive.tones[0], phase_rad=base_phase + analysis),))
        p2_pulse = Pulse(drive2, start2, start2 + t_half, "ramsey_2", ())
        pulses = [p1_pulse, p2_pulse]
        idle: tuple[tuple[float, float], ...] = ()
        if gap > 0.0:
            if delay_pulses is not None:
                pulses.extend(delay_pulses(t_half, start2))
            else:
                idle = ((t_half, start2),)
        avg = _run(device, ion, pulses, setup, {**kw, "idle": idle})
        p, s = obs.p1(avg.final_p1(ion), ion, "ramsey", k)
        rows.append((delay, p))
        sig.append(s)
    data = np.array(rows)
    sigma = sigmas_or_none(sig)
    fitted: dict[str, tuple[float, float]] = {}
    converged = True
    if data.shape[0] >= 4:
        t, y = data[:, 0], data[:, 1]
        guess = [0.5, detuning if detuning != 0.0 else 1.0 / max(float(t[-1]), 1e-9), analysis, 0.5]
        fit = weighted_fit(ramsey_model, guess, t, y, sigma=sigma)
        fitted = {
            "contrast": (abs(float(fit.params[0])), float(fit.errors[0])),
            "delta_hz": fit.value(1),
            "phi0_rad": fit.value(2),
            "offset": fit.value(3),
            "chi2_per_dof": (fit.chi2_per_dof, 0.0),
        }
        converged = fit.converged
    return RamseyFringe(
        data=data,
        fitted=fitted,
        model="ramsey_fringe",
        provenance_id="conv.detuning_symbols",
        converged=converged,
        sigma=sigma,
        requested=ScanParameters(
            {"delays_s": data[:, 0], "pi_half_s": (t_half,), **requested_drive(setup.drive, t_half)}
        ),
        realized=ScanParameters(
            {"delays_s": data[:, 0], "pi_half_s": (t_half,), **realized_drive(device, setup.drive, t_half)}
        ),
        chi2=fitted["chi2_per_dof"][0] if "chi2_per_dof" in fitted else None,
        subject={"ion": int(ion), "beam": setup.gate_drive.table_key_beam},
    )


def ramsey_frequency(machine: Machine, ion: int, delays_s: Sequence[float], **kw: Any) -> ExperimentResult:
    """Two Ramsey delay scans at drive detunings +-``probe_hz`` (default 1 kHz); returns a ``RamseyFringe`` with
    qubit_freq_hz = ``frame_hz`` plus the signed fringe offset (valid while |offset| < probe)."""
    device, kw = laboratory_kwargs(machine, kw)
    probe = abs(float(kw.get("probe_hz", 1e3)))
    frame_hz = float(kw.get("frame_hz", 0.0))
    plus = ramsey(Machine(device), ion, delays_s, **{**kw, "detuning_hz": +probe})
    minus = ramsey(Machine(device), ion, delays_s, **{**kw, "detuning_hz": -probe})
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
    converged = plus.converged and minus.converged and math.isfinite(x) and abs(x) < probe
    notes: list[str] = []
    if math.isfinite(x) and abs(x) >= probe:
        notes.append(
            f"the offset {x:.3g} Hz exceeds the probe {probe:.3g} Hz: the sign resolution is ambiguous; raise probe_hz"
        )
    return RamseyFringe(
        data=data,
        fitted={
            "qubit_offset_hz": (x, unc),
            "qubit_freq_hz": (frame_hz + x, unc),
            "fringe_plus_hz": (fp, plus.fitted["delta_hz"][1] if plus.fitted else 0.0),
            "fringe_minus_hz": (fm, minus.fitted["delta_hz"][1] if minus.fitted else 0.0),
        },
        model="ramsey_two_probe",
        provenance_id="conv.detuning_symbols",
        converged=converged,
        notes=tuple(notes),
        requested=plus.requested,
        realized=plus.realized,
        chi2=max(c for c in (plus.chi2, minus.chi2) if c is not None) if (plus.chi2 or minus.chi2) else None,
        subject=dict(plus.subject),
    )


def sideband_spectroscopy(
    machine: Machine, ion: int, detunings_hz: Sequence[float], **kw: Any
) -> ExperimentResult:
    """Scan the drive detuning of a pulse of ``duration_s`` (default the carrier pi time); returns a ``SidebandSpectrum``
    with the peaks nearest the carrier and the driven mode's sidebands; ``fit=True`` also fits the blue-sideband and
    carrier lineshapes where five points lie near them (blue_sideband_fit_hz, omega_bsb_hz, carrier_fit_hz)."""
    device, kw = laboratory_kwargs(machine, kw)
    from qutip_trap.control.pulses import Pulse

    base = _setup(device, ion, {**kw, "detuning_hz": 0.0})
    obs = _observation(device, kw)
    duration = float(kw.get("duration_s", 0.5 / base.rabi_hz))
    rows: list[tuple[float, float]] = []
    sig: list[float | None] = []
    realized_mus: list[float] = []
    for k, mu in enumerate(sorted(float(x) for x in detunings_hz)):
        setup = _setup(device, ion, {**kw, "detuning_hz": mu, "space": base.space})
        pulse = Pulse(setup.drive, 0.0, duration, "sideband_spectroscopy", ())
        realized_mus.append(realized_drive(device, setup.drive, duration)["detuning_hz"])
        avg = _run(device, ion, [pulse], setup, kw)
        p, s = obs.p1(avg.final_p1(ion), ion, "sideband_spectroscopy", k)
        rows.append((mu, p))
        sig.append(s)
    data = np.array(rows)
    sigma = sigmas_or_none(sig)
    fitted: dict[str, tuple[float, float]] = {}
    mus, p1 = data[:, 0], data[:, 1]
    step = float(np.median(np.diff(mus))) if len(mus) > 1 else 0.0
    if base.driven_mode is not None:
        f_mode = float(kw.get("mode_hz", device.crystal.modes[base.driven_mode].omega_hz))
        for name, centre in (("carrier_hz", 0.0), ("blue_sideband_hz", f_mode), ("red_sideband_hz", -f_mode)):
            window = np.abs(mus - centre) <= max(0.25 * f_mode, 2 * step)
            if np.any(window):
                idx = int(np.argmax(np.where(window, p1, -np.inf)))
                fitted[name] = (float(mus[idx]), step)
                if kw.get("fit", False) and int(np.sum(window)) >= 5 and name != "red_sideband_hz":
                    sel = np.flatnonzero(window)
                    fit = fit_lineshape(
                        mus[sel],
                        p1[sel],
                        duration,
                        sigma=None if sigma is None else sigma[sel],
                        guess=(float(mus[idx]), 0.5 / duration),
                    )
                    tag = "carrier_fit_hz" if name == "carrier_hz" else "blue_sideband_fit_hz"
                    fitted[tag] = fit.value(0)
                    fitted["omega_carrier_hz" if name == "carrier_hz" else "omega_bsb_hz"] = fit.value(1)
        if "blue_sideband_hz" in fitted and "red_sideband_hz" in fitted:
            fitted["mode_hz"] = (0.5 * (fitted["blue_sideband_hz"][0] - fitted["red_sideband_hz"][0]), step)
    else:
        idx = int(np.argmax(p1))
        fitted["carrier_hz"] = (float(mus[idx]), step)
    return SidebandSpectrum(
        data=data,
        fitted=fitted,
        model="sideband_spectrum_peaks",
        provenance_id="conv.detuning_symbols",
        sigma=sigma,
        requested=ScanParameters({"detunings_hz": mus, "duration_s": (duration,)}),
        realized=ScanParameters({"detunings_hz": realized_mus, "duration_s": (duration,)}),
        subject={"ion": int(ion), "beam": base.gate_drive.table_key_beam}
        | ({"mode": int(base.driven_mode)} if base.driven_mode is not None else {}),
    )
