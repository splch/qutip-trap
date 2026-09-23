"""Light and field experiments: the Stark scan, the crosstalk scan and the field scan."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import brentq

from qutip_trap.experiments.fitting import (
    at_scan_edge,
    sigmas_or_none,
    thermal_rabi_model_fixed_nbar,
    weighted_fit,
)
from qutip_trap.experiments.result import (
    CrosstalkScan,
    ExperimentResult,
    FieldScan,
    ScanParameters,
    StarkScan,
    realized_drive,
    requested_drive,
)
from qutip_trap.experiments.single_ion import _observation, _run, _setup, ramsey, ramsey_frequency, sub_stream
from qutip_trap.machine import Machine, laboratory_kwargs

if TYPE_CHECKING:
    from qutip_trap.machine import Machine


def _wrap(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def _nyquist_hz(delays_s: Sequence[float]) -> float | None:
    """The highest fringe frequency the delay grid resolves, 1/(2 dt) at the median step; None for fewer than two delays."""
    ts = np.array(sorted(float(x) for x in delays_s))
    if ts.size < 2:
        return None
    step = float(np.median(np.diff(ts)))
    return 0.5 / step if step > 0.0 else None


def stark_scan(machine: Machine, ion: int, delays_s: Sequence[float], **kw: Any) -> ExperimentResult:
    """Scan the Ramsey delay with ``ion``'s gate-beam light on; returns a ``StarkScan`` of the differential light shift.

    ``mode="per_beam"`` (default): one beam on at a time, at both probe signs so shift = (f_minus - f_plus)/2 keeps its
    sign, valid while |shift| < ``probe_hz`` (1 kHz) and the fringe is below the delay grid's Nyquist frequency.
    ``mode="beat_note"``: both beams on, beat note at +-``stark_detuning_hz``; the even part of the two shifts is the
    light shift, the odd part the coupling shift Omega^2/(2 delta).
    """
    device, kw = laboratory_kwargs(machine, kw)
    from qutip_trap.control.pulses import Drive, Pulse, Tone
    from qutip_trap.control.schedule import default_gate_drives
    from qutip_trap.light.raman import (
        derive_optical_drive,
        derive_raman_drive,
        differential_stark_shift_hz,
    )

    gate_drive = default_gate_drives(device)[ion]
    if gate_drive.kind == "raman":
        derived = derive_raman_drive(
            device, ion, (gate_drive.beams[0], gate_drive.beams[1]), scattering=False
        )
    elif gate_drive.kind in ("optical_E1", "optical_E2"):
        derived = derive_optical_drive(device, ion, gate_drive.beams[0], scattering=False)
    else:
        raise ValueError(
            "the Stark scan measures the light shift of laser beams; a microwave drive has none to scan"
        )
    omega = derived.carrier_rabi_hz
    belief = float(kw.get("rabi_hz_belief") or omega)
    probe = abs(float(kw.get("probe_hz", 1e3)))
    mode = kw.get("mode", "per_beam")
    rows: list[np.ndarray] = []
    sigmas: list[np.ndarray | None] = []
    notes: list[str] = []
    fitted: dict[str, tuple[float, float]] = {}
    converged = True

    # ``mode`` is this experiment's switch; ``_setup`` would read it as a driven-mode index
    base_kw = {k: v for k, v in kw.items() if k != "mode"}

    def run_with(label: float, delay_pulses: Any, tag: str, detuning_hz: float) -> tuple[float, float, bool]:
        # every sub-run draws its own shot noise
        res = ramsey(
            Machine(device),
            ion,
            delays_s,
            **{
                **sub_stream(base_kw, tag),
                "detuning_hz": detuning_hz,
                "delay_pulses": delay_pulses,
                "rabi_hz_belief": belief,
            },
        )
        f, s_f = res.fitted.get("delta_hz", (math.nan, 0.0))
        rows.append(np.column_stack([np.full(len(res.data), label), res.data]))
        sigmas.append(res.sigma)
        return abs(f), s_f, res.converged

    if mode == "per_beam":
        total, var = 0.0, 0.0
        nyquist = _nyquist_hz(delays_s)
        for b in derived.beams:
            shift_b = float(differential_stark_shift_hz(device, ion, (b,)))

            def delay_pulses(t0: float, t1: float, _b: int = b, _shift: float = shift_b) -> list[Pulse]:
                # one beam on: no two-photon coupling (a zero envelope), the beam's own light shift on the qubit
                drive = Drive(
                    kind=derived.kind,
                    ions=(ion,),
                    tones=(Tone(detuning_hz=0.0, phase_rad=0.0, envelope_hz=0.0),),
                    beams=derived.beams,
                    stark_shift_hz=_shift,
                    crosstalk={},
                )
                return [Pulse(drive, t0, t1, f"stark_probe[beam {_b}]", ())]

            # fringes at |probe - delta| and |probe + delta|: half their difference is delta while |delta| < probe
            f_p, s_p, ok_p = run_with(float(b), delay_pulses, f"beam{b}/plus", +probe)
            f_m, s_m, ok_m = run_with(float(b), delay_pulses, f"beam{b}/minus", -probe)
            shift = 0.5 * (f_m - f_p)
            s_shift = 0.5 * math.hypot(s_p, s_m)
            fitted[f"stark_shift_hz[{b}]"] = (shift, s_shift)
            fitted[f"fringe_plus_hz[{b}]"] = (f_p, s_p)
            fitted[f"fringe_minus_hz[{b}]"] = (f_m, s_m)
            total += shift
            var += s_shift**2
            ok = ok_p and ok_m and abs(shift) < probe
            residual = abs(f_p + f_m - 2.0 * probe)
            tol = max(4.0 * math.hypot(s_p, s_m), 1e-3 * probe)
            if residual > tol:
                ok = False
                notes.append(
                    f"beam {b}: the two fringes sum to {f_p + f_m:.6g} Hz, not 2 x probe = {2.0 * probe:.6g} Hz "
                    f"(residual {residual:.3g} above {tol:.3g}): the shift lies outside the probe; raise probe_hz"
                )
            if nyquist is not None and abs(shift) + probe > nyquist:
                ok = False
                notes.append(
                    f"beam {b}: the fringe at {abs(shift) + probe:.6g} Hz exceeds the delay grid's Nyquist frequency "
                    f"{nyquist:.6g} Hz: shorten the delay step or lower probe_hz"
                )
            converged = converged and ok
        fitted["stark_shift_hz"] = (total, math.sqrt(var))
    elif mode == "beat_note":
        far = kw.get("stark_detuning_hz")
        if far is None:
            far = max(m.omega_hz for m in device.crystal.modes) + 10.0 * belief
        far = abs(float(far))
        fringes: dict[int, tuple[float, float, bool]] = {}
        for sign in (+1, -1):

            def delay_pulses_beat(t0: float, t1: float, _sign: int = sign) -> list[Pulse]:
                drive = Drive(
                    kind=derived.kind,
                    ions=(ion,),
                    tones=(Tone(detuning_hz=_sign * far, phase_rad=0.0, envelope_hz=float(omega)),),
                    beams=derived.beams,
                    stark_shift_hz=float(derived.stark_shift_hz),
                    crosstalk={},
                )
                return [Pulse(drive, t0, t1, f"stark_probe[{_sign:+d}]", ())]

            fringes[sign] = run_with(float(sign), delay_pulses_beat, f"sign{sign:+d}", +probe)
        shift_p = probe - fringes[+1][0]
        shift_m = probe - fringes[-1][0]
        fitted["stark_shift_hz"] = (
            0.5 * (shift_p + shift_m),
            0.5 * math.hypot(fringes[+1][1], fringes[-1][1]),
        )
        fitted["coupling_shift_hz"] = (0.5 * (shift_p - shift_m), fitted["stark_shift_hz"][1])
        fitted["coupling_shift_expected_hz"] = (belief**2 / (2.0 * far), 0.0)
        fitted["stark_detuning_hz"] = (far, 0.0)
        converged = fringes[+1][2] and fringes[-1][2] and max(abs(shift_p), abs(shift_m)) < probe
        notes.append("beat-note mode: the delays must resolve a fringe at probe minus Omega^2/(2 delta)")
    else:
        raise ValueError("mode is 'per_beam' or 'beat_note'")
    if not converged:
        notes.append("a fringe fit failed or a shift exceeded the probe: raise probe_hz")
    return StarkScan(
        data=np.vstack(rows),
        fitted=fitted,
        model=f"ramsey_beams_on_{mode}",
        provenance_id="conv.stark_scaling_with_amplitude",
        converged=converged,
        notes=tuple(notes),
        sigma=None
        if any(s is None for s in sigmas)
        else np.concatenate([s for s in sigmas if s is not None]),
        requested=ScanParameters({"delays_s": [float(x) for x in delays_s]}),
        subject={"ion": int(ion), "beam": gate_drive.table_key_beam},
    )


def _rabi_rate_fit(
    ts: np.ndarray, p1: np.ndarray, sigma: np.ndarray | None, eta: float, nbar: float, guess_hz: float
) -> tuple[float, float, bool]:
    fit = weighted_fit(
        lambda p, t: thermal_rabi_model_fixed_nbar(p, t, eta, nbar),
        [guess_hz, 1.0, 0.0],
        ts,
        p1,
        sigma=sigma,
        bounds=([0.0, 0.0, -0.5], [np.inf, 1.5, 0.5]),
    )
    f, s = fit.value(0)
    return f, s, fit.converged and f > 0.0


def _ideal_phase_response(alpha_rad: float, axis_rad: float, phases: np.ndarray) -> np.ndarray:
    """Ideal P1 of |0> after GPi2(0), a rotation by ``alpha`` about the equatorial axis ``axis``, and GPi2(phi)."""
    from qutip_trap.control.native import gpi2

    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    gen = math.cos(axis_rad) * x + math.sin(axis_rad) * y
    rot = math.cos(alpha_rad / 2.0) * np.eye(2) - 1j * math.sin(alpha_rad / 2.0) * gen
    ket0 = np.array([1.0, 0.0], dtype=complex)
    mid = rot @ (gpi2(0.0) @ ket0)
    return np.array([abs((gpi2(float(phi)) @ mid)[1]) ** 2 for phi in phases])


def _fit_phase(phases: np.ndarray, signal: np.ndarray, sigma: np.ndarray | None) -> tuple[float, float, bool]:
    def model(p: np.ndarray, x: np.ndarray) -> np.ndarray:
        return np.asarray(float(p[0]) * np.cos(np.asarray(x) + float(p[1])) + float(p[2]))

    c0 = 0.5 * float(signal.max() - signal.min())
    best: tuple[float, float, bool, float] | None = None
    for phi_guess in np.linspace(-math.pi, math.pi, 8, endpoint=False):
        fit = weighted_fit(model, [c0, phi_guess, float(signal.mean())], phases, signal, sigma=sigma)
        phi0 = float(fit.params[1]) + (math.pi if fit.params[0] < 0.0 else 0.0)
        cand = (_wrap(phi0), float(fit.errors[1]), fit.converged, fit.chi2_per_dof)
        if best is None or cand[3] < best[3]:
            best = cand
    assert best is not None
    return best[0], best[1], best[2]


def crosstalk_scan(machine: Machine, ion: int, durations_s: Sequence[float], **kw: Any) -> ExperimentResult:
    """Scan the duration of a carrier pulse on ``ion`` and fit every reached neighbour's Rabi rate; returns a
    ``CrosstalkScan`` with eps[j] = f_j/f_i and, with ``phase=True`` (default), each neighbour's crosstalk axis from a
    pi/2 - crosstalk pi pulse - pi/2(phi) scan (phase_rad[j], the axis minus the geometric phase Delta k . (x_j - x_i))."""
    device, kw = laboratory_kwargs(machine, kw)
    from qutip_trap.control.pulses import Pulse
    from qutip_trap.light.raman import lamb_dicke_parameters

    setup = _setup(device, ion, {**kw, "crosstalk": True})
    obs = _observation(device, kw)
    neighbours = sorted(setup.drive.crosstalk)
    ts = np.array(sorted(float(t) for t in durations_s))
    if ts.size < 4 or ts[0] < 0.0:
        raise ValueError("durations_s: at least four non-negative durations")
    notes: list[str] = []
    if not neighbours:
        return CrosstalkScan(
            data=np.zeros((0, 2)),
            fitted={"rate_hz": (setup.rabi_hz, 0.0)},
            model="crosstalk_rabi_rates",
            provenance_id="conv.crosstalk_ratio",
            notes=("the device derives no light on any neighbour under this drive",),
            subject={"ion": int(ion), "beam": setup.gate_drive.table_key_beam},
        )
    t_max = float(ts[-1])
    pulse = Pulse(setup.drive, 0.0, t_max, "crosstalk_scan", ())
    avg = _run(device, ion, [pulse], setup, kw, store_times=[t for t in ts if 0.0 < t < t_max])
    n = device.crystal.n_ions
    curves: dict[int, tuple[np.ndarray, np.ndarray | None]] = {}
    for q in [ion] + neighbours:
        exact = np.interp(ts, avg.times_s, avg.p1(q))
        meas = [obs.p1(float(p), q, "crosstalk_scan", k) for k, p in enumerate(exact)]
        curves[q] = (np.array([m[0] for m in meas]), sigmas_or_none([m[1] for m in meas]))
    driven = setup.driven_mode
    nbar_m = setup.nbar.get(driven, 0.0) if driven is not None else 0.0
    f_i, s_i, ok_i = _rabi_rate_fit(
        ts, curves[ion][0], curves[ion][1], setup.eta_driven, nbar_m, setup.rabi_hz
    )
    fitted: dict[str, tuple[float, float]] = {"rate_hz": (f_i, s_i)}
    converged = ok_i
    delta_k = setup.drive.delta_k(device.beams)
    for j in neighbours:
        eta_j = (
            abs(lamb_dicke_parameters(device, j, delta_k)[0].get(driven, 0.0)) if driven is not None else 0.0
        )
        guess = abs(setup.drive.crosstalk[j]) * setup.rabi_hz
        f_j, s_j, ok_j = _rabi_rate_fit(ts, curves[j][0], curves[j][1], eta_j, nbar_m, guess)
        eps = f_j / f_i if f_i > 0.0 else math.nan
        s_eps = eps * math.hypot(s_j / max(f_j, 1e-300), s_i / max(f_i, 1e-300)) if f_j > 0.0 else math.nan
        fitted[f"rate_hz[{j}]"] = (f_j, s_j)
        fitted[f"eps[{j}]"] = (eps, s_eps)
        converged = converged and ok_j and math.isfinite(eps)
        if kw.get("phase", True) and ok_j and f_j > 0.0:
            # a pi (not pi/2) rotation about the crosstalk axis, which leaves a fringe for every axis
            t_x = float(kw.get("phase_duration_s") or 0.5 / f_j)
            phases = np.asarray(
                kw.get("analysis_phases_rad", np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False))
            )
            own = _setup(device, j, {**kw, "crosstalk": False, "space": setup.space})
            belief_j = float(kw.get("rabi_hz_belief_neighbour", own.rabi_hz))
            t_h = 0.25 / belief_j
            dead = float(device.hardware.dead_time_s)
            signal: list[float] = []
            sig: list[float | None] = []
            for k, phi in enumerate(phases):
                from dataclasses import replace as dc_replace

                p1 = Pulse(own.drive, 0.0, t_h, "xt_phase_1", ())
                px = Pulse(setup.drive, t_h + dead, t_h + dead + t_x, "xt_phase_x", ())
                drive3 = dc_replace(own.drive, tones=(dc_replace(own.drive.tones[0], phase_rad=float(phi)),))
                p3 = Pulse(drive3, t_h + 2.0 * dead + t_x, 2.0 * t_h + 2.0 * dead + t_x, "xt_phase_2", ())
                idle = ((t_h, t_h + dead), (t_h + dead + t_x, t_h + 2.0 * dead + t_x)) if dead > 0.0 else ()
                res = _run(device, ion, [p1, px, p3], setup, {**kw, "idle": idle})
                p, s = obs.p1(res.final_p1(j), j, f"crosstalk_phase[{j}]", k)
                signal.append(p)
                sig.append(s)
            phi_meas, s_phi, ok_phi = _fit_phase(phases, np.array(signal), sigmas_or_none(sig))
            alpha = 2.0 * math.pi * f_j * t_x
            phi_ideal0, _s0, _ok0 = _fit_phase(phases, _ideal_phase_response(alpha, 0.0, phases), None)
            phi_ideal1, _s1, _ok1 = _fit_phase(phases, _ideal_phase_response(alpha, 0.1, phases), None)
            response = _wrap(phi_ideal1 - phi_ideal0) / 0.1  # d phi_0 / d axis: +-2 for a pi rotation
            # the fringe phase carries twice the axis angle, so the axis is defined modulo pi: wrap the phase difference first
            axis = _wrap(_wrap(phi_meas - phi_ideal0) / response) if abs(response) > 0.1 else math.nan
            geometric = float(
                np.dot(
                    delta_k,
                    np.asarray(device.crystal.positions_m[j]) - np.asarray(device.crystal.positions_m[ion]),
                )
            )
            fitted[f"phase_total_rad[{j}]"] = (axis, s_phi)
            fitted[f"phase_rad[{j}]"] = (_wrap(axis - geometric), s_phi)
            converged = converged and ok_phi and math.isfinite(axis)
    data = np.column_stack([ts] + [curves[q][0] for q in [ion] + neighbours])
    sig_all = [curves[q][1] for q in [ion] + neighbours]
    return CrosstalkScan(
        data=data,
        fitted=fitted,
        model="crosstalk_rabi_rates",
        provenance_id="conv.crosstalk_ratio",
        converged=converged,
        notes=tuple(notes) + (f"ions in data columns: {[ion] + neighbours} of {n}",),
        sigma=None
        if any(s is None for s in sig_all)
        else np.column_stack([s for s in sig_all if s is not None]),
        requested=ScanParameters({"durations_s": ts, **requested_drive(setup.drive, t_max)}),
        realized=ScanParameters({"durations_s": ts, **realized_drive(device, setup.drive, t_max)}),
        subject={"ion": int(ion), "beam": setup.gate_drive.table_key_beam},
    )


def field_scan(machine: Machine, ion: int, delays_s: Sequence[float], **kw: Any) -> ExperimentResult:
    """Two-probe Ramsey delay scan with the frame at nu(B_seed), inverted through nu(B); returns a ``FieldScan`` (B_gauss).

    ``b_seed_gauss`` (default the device's field), ``sigma_max_gauss`` above which the entry is uncalibrated (0.05 G),
    ``sensitivity_min_hz_per_g`` below which the transition (a clock point) cannot fix the field (10 Hz/G).
    """
    device, kw = laboratory_kwargs(machine, kw)
    sp = device.crystal.species[ion]
    lower, upper = sp.qubit
    b_seed = float(kw.get("b_seed_gauss", device.field.B_gauss))
    f_seed, d1_seed, _d2 = sp.transition_frequency_hz(lower, upper, b_seed)
    f_true, _d1t, _d2t = sp.transition_frequency_hz(lower, upper, device.field.B_gauss)
    shifts = {int(k): float(v) for k, v in dict(kw.get("qubit_shifts_hz", {})).items()}
    shifts[ion] = shifts.get(ion, 0.0) + (f_true - f_seed)
    res = ramsey_frequency(
        Machine(device), ion, delays_s, **{**kw, "qubit_shifts_hz": shifts, "frame_hz": f_seed}
    )
    x, s_x = res.fitted["qubit_offset_hz"]
    f_meas = f_seed + x
    notes = list(res.notes)
    sens_min = float(kw.get("sensitivity_min_hz_per_g", 10.0))
    converged = res.converged and math.isfinite(x)
    b_fit, s_b, d1 = b_seed, math.inf, d1_seed
    if abs(d1_seed) < sens_min:
        notes.append(
            f"|d nu/dB| = {abs(d1_seed):.3g} Hz/G at the seed field: the qubit transition cannot fix the field (a clock point); "
            "use a field-sensitive transition"
        )
        converged = False
    elif converged:

        def g(b: float) -> float:
            return float(sp.transition_frequency_hz(lower, upper, b)[0]) - f_meas

        lo, hi = max(b_seed * 0.5, 1e-6), b_seed * 1.5 + 1e-3
        try:
            if g(lo) * g(hi) > 0.0:
                raise ValueError("no root in the bracket")
            b_fit = float(brentq(g, lo, hi, xtol=1e-12))
            d1 = float(sp.transition_frequency_hz(lower, upper, b_fit)[1])
            s_b = s_x / max(abs(d1), 1e-300)
        except ValueError as exc:
            notes.append(f"field inversion failed: {exc}")
            converged = False
    s_max = float(kw.get("sigma_max_gauss", 0.05))
    if converged and s_b > s_max:
        notes.append(f"field uncertainty {s_b:.3g} G exceeds {s_max:.3g} G")
        converged = False
    fitted = {
        "B_gauss": (b_fit, s_b if math.isfinite(s_b) else 0.0),
        "qubit_freq_hz": (f_meas, s_x),
        "qubit_offset_hz": (x, s_x),
        "dnu_dB_hz_per_g": (d1, 0.0),
        "b_seed_gauss": (b_seed, 0.0),
    }
    return FieldScan(
        data=res.data,
        fitted=fitted,
        model="ramsey_frequency_zeeman_inversion",
        provenance_id="conv.curvature_naming",
        converged=converged,
        notes=tuple(notes),
        requested=res.requested,
        realized=res.realized,
        chi2=res.chi2,
        subject={"ion": int(ion)},
    )


def wrap_angle(angle: float) -> float:
    return _wrap(angle)


__all__ = ["at_scan_edge", "crosstalk_scan", "field_scan", "stark_scan", "wrap_angle"]
