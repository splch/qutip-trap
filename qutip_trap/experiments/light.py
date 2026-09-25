"""Light-and-field experiments: the Stark scan, the crosstalk scan and the field scan (PLAN.md Section 7.5 items 7, 8, 9).

- ``stark_scan``: a Ramsey experiment with ONE beam of the gate drive on during the delay (the other blocked, so no
  two-photon coupling exists): the fringe shift is that beam's differential light shift, the drive's the sum over its
  beams. The fringe fit returns |delta - probe|, so the scan runs at both probe signs and the shift is the half-difference
  (``ramsey_frequency``'s estimator). ``mode="beat_note"`` instead keeps both beams on with the beat note detuned far from
  the carrier at both signs: the light shift is even in the detuning, the coupling shift Omega^2/(2 delta) odd.
- ``crosstalk_scan``: drive ion i on its carrier with the light its beams put on the neighbours and fit every neighbour's
  Rabi flopping: the rate ratio is epsilon_ij. The crosstalk axis on each neighbour comes from a pi/2 - crosstalk pi -
  pi/2(phi) sequence compared with the same sequence on ideal matrices.
- ``field_scan``: the Ramsey-frequency experiment with the frame at the transition frequency of a seed field, inverted
  through the atomic layer's exact nu(B) for B with uncertainty sigma_nu/|d nu/dB| (2 x 310.87 B = 3.1 kHz/G for the
  171Yb+ clock qubit at 5 G); a transition at a clock point cannot fix the field and is left uncalibrated.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Literal, Unpack

import numpy as np
from scipy.optimize import brentq

from qutip_trap.experiments.fitting import (
    fit_fringe,
    sigmas_or_none,
    thermal_rabi_model,
    weighted_fit,
    wrap_angle,
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
from qutip_trap.experiments.single_ion import (
    _check_lab,
    _Lab,
    _LabOptions,
    _Probe,
    _run,
    _setup,
    ramsey,
    ramsey_frequency,
    sub_stream,
)

if TYPE_CHECKING:
    from qutip_trap.control.pulses import Pulse
    from qutip_trap.machine import Machine


def _nyquist_hz(delays_s: Sequence[float]) -> float | None:
    """The highest fringe frequency the delay grid resolves, 1/(2 dt) at the median step; None for fewer than two delays."""
    ts = np.array(sorted(float(x) for x in delays_s))
    if ts.size < 2:
        return None
    step = float(np.median(np.diff(ts)))
    return 0.5 / step if step > 0.0 else None


def stark_scan(
    machine: Machine,
    ion: int,
    delays_s: Sequence[float],
    *,
    mode: Literal["per_beam", "beat_note"] = "per_beam",
    probe_hz: float = 1e3,
    rabi_hz_belief: float | None = None,
    **kw: Unpack[_LabOptions],
) -> ExperimentResult:
    """The differential light shift of ``ion``'s gate beams per (ion, beam) (Section 7.5 item 7).

    ``mode="per_beam"``: a Ramsey scan per beam with that beam alone on during the delay, at both probe signs: delta =
    (f_minus - f_plus)/2, exact while |delta| < probe, with f_plus + f_minus = 2 probe and the delay grid's Nyquist
    frequency as the range guards; the drive's shift is the sum over its beams. ``mode="beat_note"``: both beams on, the
    beat note detuned by the highest mode frequency plus ten Rabi frequencies at both signs; the light shift is the even
    part of the two fringe shifts and the odd part the coupling shift Omega^2/(2 delta). ``probe_hz`` is the Ramsey probe,
    ``rabi_hz_belief`` the table's Rabi frequency for the pi/2 pulses. Data rows (beam or sign, delay_s, P1); fitted
    stark_shift_hz (the drive's total) and, per beam, stark_shift_hz[b], fringe_plus_hz[b], fringe_minus_hz[b]
    (per_beam), or coupling_shift_hz, coupling_shift_expected_hz and stark_detuning_hz (beat_note)."""
    from qutip_trap.control.pulses import Drive, Pulse, Tone
    from qutip_trap.control.schedule import default_gate_drives
    from qutip_trap.light.raman import derive_optical_drive, derive_raman_drive, differential_stark_shift_hz

    _check_lab(kw)
    device = machine.device
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
    belief = float(rabi_hz_belief or omega)
    probe = abs(float(probe_hz))
    rows: list[np.ndarray] = []
    sigmas: list[np.ndarray | None] = []
    notes: list[str] = []
    fitted: dict[str, tuple[float, float]] = {}
    converged = True

    def probe_drive(detuning_hz: float, envelope_hz: float, stark_hz: float) -> Drive:
        tone = Tone(detuning_hz=detuning_hz, phase_rad=0.0, envelope_hz=envelope_hz)
        return Drive(
            kind=derived.kind,
            ions=(ion,),
            tones=(tone,),
            beams=derived.beams,
            stark_shift_hz=stark_hz,
            crosstalk={},
        )

    def fringe(
        label: float, drive: Drive, gate_id: str, tag: str, detuning_hz: float
    ) -> tuple[float, float, bool]:
        """|fringe frequency|, its sigma and convergence of a Ramsey scan with ``drive`` on during the delay; each sub-run
        draws its own shot noise."""

        def delay_pulses(t0: float, t1: float) -> list[Pulse]:
            return [Pulse(drive, t0, t1, gate_id, ())]

        res = ramsey(
            machine,
            ion,
            delays_s,
            detuning_hz=detuning_hz,
            delay_pulses=delay_pulses,
            rabi_hz_belief=belief,
            **sub_stream(kw, tag),
        )
        f, s_f = res.fitted.get("delta_hz", (math.nan, 0.0))
        rows.append(np.column_stack([np.full(len(res.data), label), res.data]))
        sigmas.append(res.sigma)
        return abs(f), s_f, res.converged

    if mode == "per_beam":
        total, var = 0.0, 0.0
        nyquist = _nyquist_hz(delays_s)
        for b in derived.beams:
            # one beam on: no two-photon coupling (a zero envelope), the beam's own light shift on the qubit
            drive = probe_drive(0.0, 0.0, float(differential_stark_shift_hz(device, ion, (b,))))
            gate_id = f"stark_probe[beam {b}]"
            # the fringe runs at |probe - delta| for the + probe and |probe + delta| for the -, so the half-difference is
            # delta with its sign and the sum is 2 probe only while |delta| < probe
            f_p, s_p, ok_p = fringe(float(b), drive, gate_id, f"beam{b}/plus", +probe)
            f_m, s_m, ok_m = fringe(float(b), drive, gate_id, f"beam{b}/minus", -probe)
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
        far = abs(float(max(m.omega_hz for m in device.crystal.modes) + 10.0 * belief))
        fringes = {
            sign: fringe(
                float(sign),
                probe_drive(sign * far, float(omega), float(derived.stark_shift_hz)),
                f"stark_probe[{sign:+d}]",
                f"sign{sign:+d}",
                +probe,
            )
            for sign in (+1, -1)
        }
        shift_p = probe - fringes[+1][0]
        shift_m = probe - fringes[-1][0]
        s_total = 0.5 * math.hypot(fringes[+1][1], fringes[-1][1])
        fitted["stark_shift_hz"] = (0.5 * (shift_p + shift_m), s_total)
        fitted["coupling_shift_hz"] = (0.5 * (shift_p - shift_m), s_total)
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
    """(f, sigma, converged) of the thermal Debye-Waller Rabi curve with nbar fixed: p = (f_hz, A, B)."""

    def model(p: np.ndarray, t: np.ndarray) -> np.ndarray:
        return np.asarray(thermal_rabi_model(np.array([p[0], nbar, p[1]]), t, eta) + float(p[2]))

    fit = weighted_fit(
        model, [guess_hz, 1.0, 0.0], ts, p1, sigma=sigma, bounds=([0.0, 0.0, -0.5], [np.inf, 1.5, 0.5])
    )
    f, s = fit.value(0)
    return f, s, fit.converged and f > 0.0


def _ideal_phase_response(alpha_rad: float, axis_rad: float, phases: np.ndarray) -> np.ndarray:
    """P1 of |0> after GPi2(0), a rotation by ``alpha`` about the equatorial axis at ``axis`` and GPi2(phi): the
    neighbour's Ramsey signal under a crosstalk rotation, on ideal matrices."""
    from qutip_trap.control.native import PAULI_X, PAULI_Y, gpi2

    gen = math.cos(axis_rad) * PAULI_X + math.sin(axis_rad) * PAULI_Y
    rot = math.cos(alpha_rad / 2.0) * np.eye(2) - 1j * math.sin(alpha_rad / 2.0) * gen
    mid = rot @ (gpi2(0.0) @ np.array([1.0, 0.0], dtype=complex))
    return np.array([abs((gpi2(float(phi)) @ mid)[1]) ** 2 for phi in phases])


def crosstalk_scan(
    machine: Machine,
    ion: int,
    durations_s: Sequence[float],
    *,
    analysis_phases_rad: Sequence[float] | np.ndarray | None = None,
    rabi_hz_belief: float | None = None,
    **kw: Unpack[_LabOptions],
) -> ExperimentResult:
    """Drive ``ion`` on its carrier and fit the Rabi rate on every neighbour its light reaches (Section 7.5 item 8):
    epsilon_ij = f_j/f_i, the rates fitted with nbar fixed at ``nbar``. The crosstalk axis on each neighbour comes from
    pi/2 - crosstalk pi - pi/2(phi) over ``analysis_phases_rad`` (default eight), the pi/2 pulses at the neighbour's own
    Rabi frequency (``rabi_hz_belief`` is not read). Data rows (t, P1 of the driven ion and each neighbour); fitted rate_hz
    and per neighbour j rate_hz[j], eps[j], phase_total_rad[j] (the axis in j's frame) and phase_rad[j] (minus the
    geometric phase Delta k . (x_j - x_i), i.e. arg epsilon_ij)."""
    from qutip_trap.control.pulses import Pulse
    from qutip_trap.light.raman import lamb_dicke_parameters

    del rabi_hz_belief
    lab = _Lab.of(machine, kw)
    device = lab.device
    setup = _setup(lab, ion, _Probe(crosstalk=True))
    neighbours = sorted(setup.drive.crosstalk)
    ts = np.array(sorted(float(t) for t in durations_s))
    if ts.size < 4 or ts[0] < 0.0:
        raise ValueError("durations_s: at least four non-negative durations")
    subject = {"ion": int(ion), "beam": setup.gate_drive.table_key_beam}
    if not neighbours:
        return CrosstalkScan(
            data=np.zeros((0, 2)),
            fitted={"rate_hz": (setup.rabi_hz, 0.0)},
            model="crosstalk_rabi_rates",
            provenance_id="conv.crosstalk_ratio",
            notes=("the device derives no light on any neighbour under this drive",),
            subject=subject,
        )
    t_max = float(ts[-1])
    pulse = Pulse(setup.drive, 0.0, t_max, "crosstalk_scan", ())
    avg = _run(lab, ion, [pulse], setup, store_times=[t for t in ts if 0.0 < t < t_max])
    curves: dict[int, tuple[np.ndarray, np.ndarray | None]] = {}
    for q in [ion, *neighbours]:
        exact = np.interp(ts, avg.times_s, avg.p1(q))
        meas = [lab.obs.p1(float(p), q, "crosstalk_scan", k) for k, p in enumerate(exact)]
        curves[q] = (np.array([m[0] for m in meas]), sigmas_or_none([m[1] for m in meas]))
    driven = setup.driven_mode
    nbar_m = lab.nbar.get(driven, 0.0) if driven is not None else 0.0
    f_i, s_i, ok_i = _rabi_rate_fit(ts, *curves[ion], setup.eta_driven, nbar_m, setup.rabi_hz)
    fitted: dict[str, tuple[float, float]] = {"rate_hz": (f_i, s_i)}
    converged = ok_i
    delta_k = setup.drive.delta_k(device.beams)
    phases = np.asarray(
        np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False)
        if analysis_phases_rad is None
        else analysis_phases_rad
    )
    positions = np.asarray(device.crystal.positions_m)
    dead = float(device.hardware.dead_time_s)
    for j in neighbours:
        eta_j = (
            abs(lamb_dicke_parameters(device, j, delta_k)[0].get(driven, 0.0)) if driven is not None else 0.0
        )
        guess = abs(setup.drive.crosstalk[j]) * setup.rabi_hz
        f_j, s_j, ok_j = _rabi_rate_fit(ts, *curves[j], eta_j, nbar_m, guess)
        eps = f_j / f_i if f_i > 0.0 else math.nan
        s_eps = eps * math.hypot(s_j / max(f_j, 1e-300), s_i / max(f_i, 1e-300)) if f_j > 0.0 else math.nan
        fitted[f"rate_hz[{j}]"] = (f_j, s_j)
        fitted[f"eps[{j}]"] = (eps, s_eps)
        converged = converged and ok_j and math.isfinite(eps)
        if not (ok_j and f_j > 0.0):
            continue
        # a PI rotation about the crosstalk axis reflects the neighbour's Bloch vector across it (a pi/2 one maps the
        # equatorial state onto the pole for an axis parallel to the preparation pulse's and gives no fringe)
        t_x = 0.5 / f_j
        own = _setup(lab, j, _Probe(), setup.space)
        t_h = 0.25 / own.rabi_hz
        signal: list[float] = []
        sig: list[float | None] = []
        for k, phi in enumerate(phases):
            drive3 = replace(own.drive, tones=(replace(own.drive.tones[0], phase_rad=float(phi)),))
            pulses = [
                Pulse(own.drive, 0.0, t_h, "xt_phase_1", ()),
                Pulse(setup.drive, t_h + dead, t_h + dead + t_x, "xt_phase_x", ()),
                Pulse(drive3, t_h + 2.0 * dead + t_x, 2.0 * t_h + 2.0 * dead + t_x, "xt_phase_2", ()),
            ]
            idle = ((t_h, t_h + dead), (t_h + dead + t_x, t_h + 2.0 * dead + t_x)) if dead > 0.0 else ()
            p, s = lab.obs.p1(
                _run(lab, ion, pulses, setup, idle=idle).final_p1(j), j, f"crosstalk_phase[{j}]", k
            )
            signal.append(p)
            sig.append(s)
        measured = fit_fringe(phases, np.array(signal), sigmas_or_none(sig), 1)
        alpha = 2.0 * math.pi * f_j * t_x
        ideal0 = fit_fringe(phases, _ideal_phase_response(alpha, 0.0, phases), None, 1).phase_rad[0]
        ideal1 = fit_fringe(phases, _ideal_phase_response(alpha, 0.1, phases), None, 1).phase_rad[0]
        response = (
            wrap_angle(ideal1 - ideal0) / 0.1
        )  # d phi_0/d axis: +-2 for a pi rotation (the reflection doubles it)
        # the fringe phase carries twice the axis angle, so the axis is defined modulo pi: wrap the phase difference first
        axis = (
            wrap_angle(wrap_angle(measured.phase_rad[0] - ideal0) / response)
            if abs(response) > 0.1
            else math.nan
        )
        geometric = float(np.dot(delta_k, positions[j] - positions[ion]))
        fitted[f"phase_total_rad[{j}]"] = (axis, measured.phase_rad[1])
        fitted[f"phase_rad[{j}]"] = (wrap_angle(axis - geometric), measured.phase_rad[1])
        converged = converged and measured.converged and math.isfinite(axis)
    sig_all = [curves[q][1] for q in [ion, *neighbours]]
    return CrosstalkScan(
        data=np.column_stack([ts] + [curves[q][0] for q in [ion, *neighbours]]),
        fitted=fitted,
        model="crosstalk_rabi_rates",
        provenance_id="conv.crosstalk_ratio",
        converged=converged,
        notes=(f"ions in data columns: {[ion, *neighbours]} of {device.crystal.n_ions}",),
        sigma=None
        if any(s is None for s in sig_all)
        else np.column_stack([s for s in sig_all if s is not None]),
        requested=ScanParameters({"durations_s": ts, **requested_drive(setup.drive, t_max)}),
        realized=ScanParameters({"durations_s": ts, **realized_drive(device, setup.drive, t_max)}),
        subject=subject,
    )


def field_scan(
    machine: Machine,
    ion: int,
    delays_s: Sequence[float],
    *,
    b_seed_gauss: float | None = None,
    probe_hz: float = 1e3,
    rabi_hz_belief: float | None = None,
    include_stark: bool = True,
    **kw: Unpack[_LabOptions],
) -> ExperimentResult:
    """B from the qubit transition frequency (Section 7.5 item 9): ``ramsey_frequency`` with the frame at nu(B_seed)
    (``b_seed_gauss``, default the device's field), inverted through the atomic layer's nu(B); uncalibrated when |d nu/dB|
    is below 10 Hz/G (a clock point) or sigma_B above 0.05 G. Fitted B_gauss, qubit_freq_hz (at the measured field),
    qubit_offset_hz (from the seed frame), dnu_dB_hz_per_g and b_seed_gauss."""
    _check_lab(kw)
    device = machine.device
    sp = device.crystal.species[ion]
    lower, upper = sp.qubit
    b_seed = float(device.field.B_gauss if b_seed_gauss is None else b_seed_gauss)
    f_seed, d1_seed, _d2 = sp.transition_frequency_hz(lower, upper, b_seed)
    f_true, _d1t, _d2t = sp.transition_frequency_hz(lower, upper, device.field.B_gauss)
    shifts = {int(k): float(v) for k, v in (kw.get("qubit_shifts_hz") or {}).items()}
    shifts[ion] = shifts.get(ion, 0.0) + (f_true - f_seed)
    seeded: _LabOptions = {**kw, "qubit_shifts_hz": shifts}
    res = ramsey_frequency(
        machine,
        ion,
        delays_s,
        probe_hz=probe_hz,
        frame_hz=f_seed,
        rabi_hz_belief=rabi_hz_belief,
        include_stark=include_stark,
        **seeded,
    )
    x, s_x = res.fitted["qubit_offset_hz"]
    f_meas = f_seed + x
    notes = list(res.notes)
    converged = res.converged and math.isfinite(x)
    b_fit, s_b, d1 = b_seed, math.inf, d1_seed
    if abs(d1_seed) < 10.0:
        notes.append(
            f"|d nu/dB| = {abs(d1_seed):.3g} Hz/G at the seed field: the qubit transition cannot fix the field (a clock "
            "point); use a field-sensitive transition"
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
    if converged and s_b > 0.05:
        notes.append(f"field uncertainty {s_b:.3g} G exceeds 0.05 G")
        converged = False
    return FieldScan(
        data=res.data,
        fitted={
            "B_gauss": (b_fit, s_b if math.isfinite(s_b) else 0.0),
            "qubit_freq_hz": (f_meas, s_x),
            "qubit_offset_hz": (x, s_x),
            "dnu_dB_hz_per_g": (d1, 0.0),
            "b_seed_gauss": (b_seed, 0.0),
        },
        model="ramsey_frequency_zeeman_inversion",
        provenance_id="conv.curvature_naming",
        converged=converged,
        notes=tuple(notes),
        requested=res.requested,
        realized=res.realized,
        chi2=res.chi2,
        subject={"ion": int(ion)},
    )
