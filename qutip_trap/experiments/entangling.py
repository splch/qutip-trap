"""Entangling-gate experiments: the MS amplitude and detuning scan, the parity scan and the MS phase scan, all in the
frame the machine programs (``qubit_shifts_hz``) at the mode frequencies the table believes (``mode_frequencies_hz``)."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from qutip_trap.experiments.fitting import at_scan_edge, parity_model, weighted_fit
from qutip_trap.experiments.result import ExperimentResult, MSScan, ParityScan, ScanParameters
from qutip_trap.experiments.single_ion import _observation
from qutip_trap.machine import laboratory_kwargs

if TYPE_CHECKING:
    from qutip_trap.device.model import Device
    from qutip_trap.machine import Machine


def _entangling_setup(
    device: Device, pair: tuple[int, int], kw: dict[str, Any]
) -> tuple[Any, dict[int, Any], dict[int, Any], Any, Any, Any]:
    """(waveform, entangling drives, single-qubit drives, table, modes, space) from the keyword arguments or the device's defaults."""
    from qutip_trap.calibration.entangling import gate_space
    from qutip_trap.control.schedule import resolve_drives
    from qutip_trap.control.shaping import gate_modes

    table = kw.get("table")
    waveform = kw.get("waveform") or (table.waveform_for(pair) if table is not None else None)
    if waveform is None:
        raise ValueError(
            "ms_scan/parity_scan need the pair's waveform (kw waveform=, or a table with an ms entry)"
        )
    sq_resolved, ent_resolved = resolve_drives(device)
    sq: dict[int, Any] = dict(sq_resolved)
    ent: dict[int, Any] = dict(ent_resolved)
    beams = ent[pair[0]].beams
    if len(beams) != 2:
        raise ValueError("the entangling experiments take a Raman (two-beam) entangling drive")
    nbar: dict[int, float] = {int(k): float(v) for k, v in dict(kw.get("nbar", {})).items()}
    # built at the mode frequencies the machine believes; a supplied GateModes carries its own, so both at once is refused
    mode_hz = {int(k): float(v) for k, v in dict(kw.get("mode_frequencies_hz") or {}).items()}
    supplied = kw.get("modes")
    if supplied is not None and mode_hz:
        raise ValueError(
            "give the gate modes or the mode_frequencies_hz they are built from, not both: the supplied GateModes carries "
            "its own frequencies and mode_frequencies_hz would be discarded"
        )
    modes = supplied or gate_modes(
        device, pair, (beams[0], beams[1]), nbar=nbar, mode_frequencies_hz=mode_hz or None
    )
    space = kw.get("space") or gate_space(modes, device.crystal.n_ions, nbar=nbar, waveform=waveform)
    if table is None:
        raise ValueError(
            "the entangling experiments read the CalibrationTable (Stark and Rabi entries): pass table="
        )
    return waveform, ent, sq, table, modes, space


def _solver_options(kw: Mapping[str, Any]) -> Any:
    """The caller's SolverOptions, else the defaults with ``branch_weight_min`` (default 1e-3) for the thermal branches."""
    from qutip_trap.dynamics.engine import SolverOptions

    options = kw.get("options")
    if options is not None:
        return options
    return SolverOptions(branch_weight_min=float(kw.get("branch_weight_min", 1e-3)))


def _pops4(pops: Mapping[str, float]) -> np.ndarray:
    return np.array([pops["P00"], pops["P01"], pops["P10"], pops["P11"]], dtype=float)


def _frame_shifts(kw: Mapping[str, Any]) -> dict[int, float]:
    """``qubit_shifts_hz``: the true transition minus the table's frame per ion."""
    return {int(k): float(v) for k, v in dict(kw.get("qubit_shifts_hz") or {}).items()}


def _observe_pops(
    obs: Any, pops: Mapping[str, float], pair: tuple[int, int], key: str, index: int
) -> tuple[np.ndarray, np.ndarray | None]:
    measured, sigma = obs.joint(_pops4(pops), pair, key, index)
    return measured, sigma


@dataclass(frozen=True, eq=False)
class OffsetFn:
    """tau -> fn(tau) + offset: an FM leg's detuning shifted by a scan offset, picklable when ``fn`` is (unlike a lambda)
    for the default parallel solver map."""

    fn: Callable[[float], float]
    offset: float

    def __call__(self, tau: float) -> float:
        return float(self.fn(tau)) + self.offset


def _shift_detuning(waveform: Any, offset_hz: float) -> Any:
    """Every blue leg + offset, every red leg - offset."""
    from qutip_trap.control.table import Segment

    if offset_hz == 0.0 or waveform.segments is None:
        return waveform

    def shift(v: Any, sign: float) -> Any:
        if callable(v):
            return OffsetFn(v, sign * offset_hz)
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


def ms_scan(
    machine: Machine,
    pair: tuple[int, int],
    amplitudes: Sequence[float],
    detunings_hz: Sequence[float],
    **kw: Any,
) -> ExperimentResult:
    """Scan the pair's waveform over amplitude scales ``amplitudes`` and detuning offsets ``detunings_hz`` (blue legs +,
    red legs -) from |00>; returns an ``MSScan``. Data columns (scale, offset_hz, P00, P01 + P10, P11).

    Fitted ``closure_offset_hz`` (the leakage minimum, a parabola over three or more offsets), ``chi_unit_rad`` from
    (P00 - P11)/(P00 + P11) = cos(2 chi_1 s^2) and ``closure_scale`` = sqrt((pi/4)/chi_1), measured at
    ``closure_offset_used_hz`` (the fitted offset when the parabola converged, so the table may apply the two together).
    """
    device, kw = laboratory_kwargs(machine, kw)
    from qutip_trap.calibration.entangling import exact_gate_check
    from qutip_trap.control.shaping import scaled

    waveform, ent, _sq, table, _modes, space = _entangling_setup(device, pair, kw)
    obs = _observation(device, kw)
    scales = sorted(float(x) for x in amplitudes)
    offsets = sorted(float(x) for x in detunings_hz)
    if any(s <= 0.0 for s in scales):
        raise ValueError("amplitude scale factors are positive")
    rows: list[tuple[float, float, float, float, float]] = []
    sig: list[np.ndarray | None] = []
    idx = 0

    def amplitude_row(off: float) -> None:
        """One amplitude scan at detuning offset ``off``, appended to the rows."""
        nonlocal idx
        wf_off = _shift_detuning(waveform, off)
        for s in scales:
            check, _ = exact_gate_check(
                device,
                scaled(wf_off, s),
                pair,
                ent,
                table,
                space=space,
                nbar={int(k): float(v) for k, v in dict(kw.get("nbar", {})).items()},
                options=_solver_options(kw),
                builder_options=kw.get("builder_options"),
                sample=kw.get("sample"),
                qubit_shifts_hz=_frame_shifts(kw),
            )
            p, sg = _observe_pops(obs, check.populations, pair, "ms_scan", idx)
            idx += 1
            rows.append((s, off, float(p[0]), float(p[1] + p[2]), float(p[3])))
            sig.append(sg)

    for off in offsets:
        amplitude_row(off)
    data = np.array(rows)
    fitted: dict[str, tuple[float, float]] = {}
    notes: list[str] = []
    converged = True
    # the closure point over the detuning offsets: the leakage minimum at the scale nearest one
    s_ref = min(scales, key=lambda s: abs(s - 1.0))
    sig_rows = [v for v in sig if v is not None]
    off_used = float(offsets[int(np.argmin(np.abs(offsets)))])
    if len(offsets) >= 3:
        sel = data[:, 0] == s_ref
        x, y = data[sel, 1], data[sel, 3]
        sg_leak = None
        if len(sig_rows) == len(sig):
            leak_sig = np.array([math.hypot(float(v[1]), float(v[2])) for v in sig_rows])
            sg_leak = leak_sig[sel]
        fit = weighted_fit(
            lambda p, xx: float(p[0]) * (np.asarray(xx) - float(p[1])) ** 2 + float(p[2]),
            [
                max(float(y.max() - y.min()), 1e-9) / max(float(x.max() - x.min()), 1e-9) ** 2 * 4.0,
                float(x[np.argmin(y)]),
                float(y.min()),
            ],
            x,
            y,
            sigma=sg_leak,
        )
        off0, s_off = fit.value(1)
        fitted["closure_offset_hz"] = (off0, s_off)
        fitted["leakage_min"] = fit.value(2)
        if (
            fit.converged
            and fit.params[0] > 0.0
            and not at_scan_edge(off0, float(x.min()), float(x.max()), 0.02)
        ):
            # measure the scale at the fitted offset the table will apply (one extra scan when the grid lacks it)
            off_used = float(off0)
            step = float(np.min(np.diff(np.asarray(offsets)))) if len(offsets) > 1 else 0.0
            if min(abs(off_used - o) for o in offsets) > 1e-3 * max(step, 1.0):
                amplitude_row(off_used)
                data = np.array(rows)
                sig_rows = [v for v in sig if v is not None]
        else:
            converged = False
            notes.append(
                "the leakage parabola over the detuning offsets did not converge or its minimum sits at the scan edge"
            )
    # the entangling amplitude at the closure offset
    sel = data[:, 1] == off_used
    sc, p00, p11 = data[sel, 0], data[sel, 2], data[sel, 4]
    with np.errstate(divide="ignore", invalid="ignore"):
        balance = np.where(p00 + p11 > 0.0, (p00 - p11) / np.where(p00 + p11 > 0.0, p00 + p11, 1.0), 0.0)
    if sc.size >= 3:
        chi_guess = abs(float(waveform.chi_total_rad)) or math.pi / 4.0
        sg_c = None
        if len(sig_rows) == len(sig):
            rows_sel = [v for v, keep in zip(sig_rows, sel) if keep]
            sg_c = np.array([math.hypot(float(v[0]), float(v[3])) for v in rows_sel]) / np.maximum(
                p00 + p11, 1e-9
            )
        fit_c = weighted_fit(
            lambda p, s: np.cos(2.0 * float(p[0]) * np.asarray(s) ** 2),
            [chi_guess],
            sc,
            balance,
            sigma=sg_c,
            bounds=([0.0], [math.pi]),
        )
        chi1, s_chi = fit_c.value(0)
        if fit_c.converged and chi1 > 0.0:
            scale = math.sqrt((math.pi / 4.0) / chi1)
            s_scale = 0.5 * scale * s_chi / chi1
            fitted["chi_unit_rad"] = (chi1, s_chi)
            fitted["closure_scale"] = (scale, s_scale)
            if at_scan_edge(scale, float(sc.min()), float(sc.max()), 0.0):
                converged = False
                notes.append("the closure scale lies outside the scanned amplitudes")
        else:
            converged = False
            notes.append("the cos(2 chi s^2) fit did not converge")
    if "closure_scale" not in fitted:
        diff = p00 - p11
        crossings = np.flatnonzero(np.diff(np.sign(diff)) != 0)
        if crossings.size:
            i = int(crossings[0])
            x0, x1, y0, y1 = sc[i], sc[i + 1], diff[i], diff[i + 1]
            root = float(x0 - y0 * (x1 - x0) / (y1 - y0)) if y1 != y0 else float(x0)
            fitted["closure_scale"] = (root, float(abs(x1 - x0)))
        else:
            converged = False
    fitted["closure_offset_used_hz"] = (off_used, 0.0)
    return MSScan(
        data=data,
        fitted=fitted,
        model="ms_population_scan",
        provenance_id="conv.ms_closure",
        converged=converged,
        notes=tuple(notes),
        sigma=None if any(v is None for v in sig) else np.vstack([v for v in sig if v is not None]),
        requested=ScanParameters(
            {"amplitudes": [float(a) for a in amplitudes], "detunings_hz": [float(d) for d in detunings_hz]}
        ),
        subject={"pair": (int(pair[0]), int(pair[1]))},
    )


def _analysis_beliefs(
    table: Any, pair: tuple[int, int], sq: Mapping[int, Any], kw: dict[str, Any]
) -> tuple[dict[int, float], dict[int, float]]:
    from qutip_trap.control.schedule import carrier_rabi_hz
    from qutip_trap.control.table import usable

    rabi = kw.get("analysis_rabi_hz")
    if rabi is None:
        rabi = {q: carrier_rabi_hz(table, q, sq[q]) for q in pair}
    stark: dict[int, float] = {}
    for q in pair:
        entry = table.stark.get((q, sq[q].table_key_beam))
        stark[q] = float(entry.value) if usable(entry) and entry is not None else 0.0
    return dict(rabi), stark


def parity_scan(
    machine: Machine, pair: tuple[int, int], analysis_phases_rad: Sequence[float], **kw: Any
) -> ExperimentResult:
    """Scan the phase of a pi/2 analysis pulse on both ions after the gate; returns a ``ParityScan`` fitted with
    Pi(phi) = C cos(2 phi + phi_0) + B and the Bell-fidelity bound (P_00 + P_11 + C)/2 (populations without the analysis
    pulse). The analysis pulses use the single-qubit drives at the table's Rabi frequencies and Stark shifts."""
    device, kw = laboratory_kwargs(machine, kw)
    from qutip_trap.calibration.entangling import exact_gate_check, parity_after_analysis_pulse

    waveform, ent, sq, table, _modes, space = _entangling_setup(device, pair, kw)
    obs = _observation(device, kw)
    rabi, stark = _analysis_beliefs(table, pair, sq, kw)
    nbar = {int(k): float(v) for k, v in dict(kw.get("nbar", {})).items()}
    rows: list[tuple[float, float, float, float]] = []
    sig: list[float | None] = []
    for k, phi in enumerate(sorted(float(x) for x in analysis_phases_rad)):
        _par, pops = parity_after_analysis_pulse(
            device,
            waveform,
            pair,
            ent,
            table,
            space=space,
            analysis_phase_rad=phi,
            analysis_rabi_hz=rabi,
            nbar=nbar,
            options=_solver_options(kw),
            builder_options=kw.get("builder_options"),
            sample=kw.get("sample"),
            analysis_drives=sq,
            analysis_stark_hz=stark,
            qubit_shifts_hz=_frame_shifts(kw),
        )
        p, sg = _observe_pops(obs, pops, pair, "parity_scan", k)
        par = float(p[0] + p[3] - p[1] - p[2])
        rows.append((phi, par, float(p[0]), float(p[3])))
        sig.append(None if sg is None else float(np.sqrt(np.sum(sg**2))))
    data = np.array(rows)
    fitted: dict[str, tuple[float, float]] = {}
    converged = True
    if data.shape[0] >= 4:
        phis, parity = data[:, 0], data[:, 1]
        sigma = None if any(s is None for s in sig) else np.array([s for s in sig if s is not None])
        fit = weighted_fit(
            parity_model,
            [float(np.max(parity) - np.min(parity)) / 2.0, 0.0, float(np.mean(parity))],
            phis,
            parity,
            sigma=sigma,
        )
        contrast = abs(float(fit.params[0]))
        fitted = {
            "contrast": (contrast, float(fit.errors[0])),
            "phi0_rad": (
                float(fit.params[1]) + (math.pi if fit.params[0] < 0 else 0.0),
                float(fit.errors[1]),
            ),
            "offset": fit.value(2),
            "chi2_per_dof": (fit.chi2_per_dof, 0.0),
        }
        base, _ = exact_gate_check(
            device,
            waveform,
            pair,
            ent,
            table,
            space=space,
            nbar=nbar,
            options=_solver_options(kw),
            builder_options=kw.get("builder_options"),
            sample=kw.get("sample"),
            qubit_shifts_hz=_frame_shifts(kw),
        )
        pb, sgb = _observe_pops(obs, base.populations, pair, "parity_populations", 0)
        s_pop = 0.0 if sgb is None else float(math.hypot(float(sgb[0]), float(sgb[3])))
        fitted["bell_fidelity_bound"] = (
            0.5 * (float(pb[0]) + float(pb[3]) + contrast),
            0.5 * math.hypot(float(fit.errors[0]), s_pop),
        )
        fitted["P00"] = (float(pb[0]), 0.0 if sgb is None else float(sgb[0]))
        fitted["P11"] = (float(pb[3]), 0.0 if sgb is None else float(sgb[3]))
        converged = fit.converged
    return ParityScan(
        data=data,
        fitted=fitted,
        model="parity_oscillation",
        provenance_id="conv.entangling_angle",
        converged=converged,
        sigma=None if any(s is None for s in sig) else np.array([s for s in sig if s is not None]),
        requested=ScanParameters({"analysis_phases_rad": [float(x) for x in analysis_phases_rad]}),
        chi2=fitted["chi2_per_dof"][0] if "chi2_per_dof" in fitted else None,
        subject={"pair": (int(pair[0]), int(pair[1]))},
    )


def _ideal_parity(
    theta_rad: float, spins: Sequence[tuple[float, float]], analysis_phase_rad: float, internal: Sequence[int]
) -> np.ndarray:
    """Pi(phi) on ideal matrices: MS(phi_0, phi_1, theta) then GPi2(analysis) on both ions from |internal>."""
    from qutip_trap.control.native import gpi2
    from qutip_trap.control.native import ms as native_ms

    ket = np.zeros(4, dtype=complex)
    ket[2 * int(internal[0]) + int(internal[1])] = 1.0
    an = np.kron(gpi2(analysis_phase_rad), gpi2(analysis_phase_rad))
    out = []
    for phi0, phi1 in spins:
        psi = an @ (native_ms(phi0, phi1, theta_rad) @ ket)
        p = np.abs(psi) ** 2
        out.append(float(p[0] + p[3] - p[1] - p[2]))
    return np.array(out)


def _fit_periodic(
    phases: np.ndarray, signal: np.ndarray, sigma: np.ndarray | None, k: int
) -> tuple[float, float, float, bool, float]:
    """(phi_0, sigma_phi_0, contrast, converged, chi2/dof) of C cos(k phi + phi_0) + B, from several starting phases."""

    def model(p: np.ndarray, x: np.ndarray) -> np.ndarray:
        return np.asarray(float(p[0]) * np.cos(k * np.asarray(x) + float(p[1])) + float(p[2]))

    best: tuple[float, float, float, bool, float] | None = None
    c0 = 0.5 * float(signal.max() - signal.min())
    for guess in np.linspace(-math.pi, math.pi, 8, endpoint=False):
        fit = weighted_fit(model, [c0, guess, float(signal.mean())], phases, signal, sigma=sigma)
        phi0 = float(fit.params[1]) + (math.pi if fit.params[0] < 0.0 else 0.0)
        cand = (
            (phi0 + math.pi) % (2.0 * math.pi) - math.pi,
            float(fit.errors[1]),
            abs(float(fit.params[0])),
            fit.converged,
            fit.chi2_per_dof,
        )
        if best is None or cand[4] < best[4]:
            best = cand
    assert best is not None
    return best


def ms_phase_scan(
    machine: Machine, pair: tuple[int, int], spin_phases_rad: Sequence[float], **kw: Any
) -> ExperimentResult:
    """Scan the MS gate's spin phases against a fixed analysis pulse; returns an ``MSScan`` with ``correction_rad[i]``,
    the phase to add to ion i's legs.

    Against the same scan on ideal matrices, the phases (phi, phi) from |00> measure the sum of the ions' frame
    misalignments and (phi, 0) from |01> (``inputs`` including "01") their difference. Data rows (input, phase, parity).
    """
    device, kw = laboratory_kwargs(machine, kw)
    from qutip_trap.calibration.entangling import parity_after_analysis_pulse

    waveform, ent, sq, table, _modes, space = _entangling_setup(device, pair, kw)
    obs = _observation(device, kw)
    rabi, stark = _analysis_beliefs(table, pair, sq, kw)
    nbar = {int(k): float(v) for k, v in dict(kw.get("nbar", {})).items()}
    analysis = float(kw.get("analysis_phase_rad", 0.0))
    theta = 2.0 * abs(float(waveform.chi_total_rad))
    phases = np.array(sorted(float(x) for x in spin_phases_rad))
    inputs = tuple(kw.get("inputs", ("00",)))
    n = device.crystal.n_ions
    rows: list[tuple[float, float, float]] = []
    sig_all: list[float | None] = []
    fitted: dict[str, tuple[float, float]] = {}
    notes: list[str] = []
    converged = True
    offsets: dict[str, tuple[float, float]] = {}
    for inp in inputs:
        if inp not in ("00", "01"):
            raise ValueError("inputs are '00' and/or '01'")
        internal = [0] * n
        if inp == "01":
            internal[pair[1]] = 1
        spins = [(phi, phi) if inp == "00" else (phi, 0.0) for phi in phases]
        signal: list[float] = []
        sig: list[float | None] = []
        for k, sp in enumerate(spins):
            _par, pops = parity_after_analysis_pulse(
                device,
                waveform,
                pair,
                ent,
                table,
                space=space,
                analysis_phase_rad=analysis,
                analysis_rabi_hz=rabi,
                nbar=nbar,
                options=_solver_options(kw),
                builder_options=kw.get("builder_options"),
                sample=kw.get("sample"),
                analysis_drives=sq,
                analysis_stark_hz=stark,
                spin_phases_rad=sp,
                internal=internal,
                qubit_shifts_hz=_frame_shifts(kw),
            )
            p, sg = _observe_pops(obs, pops, pair, f"ms_phase_scan[{inp}]", k)
            par = float(p[0] + p[3] - p[1] - p[2])
            signal.append(par)
            sig.append(None if sg is None else float(np.sqrt(np.sum(sg**2))))
            rows.append((float(inp == "01"), sp[0], par))
        sig_all.extend(sig)
        k_period = 2 if inp == "00" else 1
        ideal = _ideal_parity(theta, spins, analysis, (internal[pair[0]], internal[pair[1]]))
        ideal_shift = _ideal_parity(
            theta,
            [(a + 0.1, b + (0.1 if inp == "00" else 0.0)) for a, b in spins],
            analysis,
            (internal[pair[0]], internal[pair[1]]),
        )
        phi_ideal, _s, _c, _ok, _chi = _fit_periodic(phases, ideal, None, k_period)
        phi_shifted, _s2, _c2, _ok2, _chi2 = _fit_periodic(phases, ideal_shift, None, k_period)
        # d phi_0 / d(common offset): +-k for the 00 scan, +-1 for the 01 scan; the sign of the convention comes from the ideal
        response = ((phi_shifted - phi_ideal + math.pi) % (2.0 * math.pi) - math.pi) / 0.1
        sigma = None if any(s is None for s in sig) else np.array([s for s in sig if s is not None])
        phi_meas, s_phi, contrast, ok, _chi_m = _fit_periodic(phases, np.array(signal), sigma, k_period)
        raw = (phi_meas - phi_ideal + math.pi) % (2.0 * math.pi) - math.pi
        if abs(response) < 0.1:
            converged = False
            notes.append(
                f"input {inp}: the ideal scan does not respond to a spin-phase offset; no alignment measured"
            )
            continue
        # the offset in spin-phase units: sum (00) or difference (01) of the per-ion misalignments
        offset = raw / response * (2.0 if inp == "00" else 1.0)
        s_off = s_phi / abs(response) * (2.0 if inp == "00" else 1.0)
        key = "offset_sum_rad" if inp == "00" else "offset_diff_rad"
        offsets[key] = (offset, s_off)
        fitted[key] = (offset, s_off)
        fitted[f"contrast_{inp}"] = (contrast, 0.0)
        converged = converged and ok
    if "offset_sum_rad" in offsets:
        s_sum, e_sum = offsets["offset_sum_rad"]
        if "offset_diff_rad" in offsets:
            d, e_d = offsets["offset_diff_rad"]
            u_a, u_b = 0.5 * (s_sum + d), 0.5 * (s_sum - d)
            e = 0.5 * math.hypot(e_sum, e_d)
        else:
            u_a = u_b = 0.5 * s_sum
            e = 0.5 * e_sum
            notes.append(
                "one input: the sum offset is split equally between the ions (the difference is not measured)"
            )
        fitted[f"correction_rad[{pair[0]}]"] = (-u_a, e)
        fitted[f"correction_rad[{pair[1]}]"] = (-u_b, e)
    return MSScan(
        data=np.array(rows),
        fitted=fitted,
        model="ms_spin_phase_scan",
        provenance_id="conv.spin_motion_phases",
        converged=converged,
        notes=tuple(notes),
        sigma=None if any(s is None for s in sig_all) else np.array([s for s in sig_all if s is not None]),
        requested=ScanParameters({"spin_phases_rad": [float(x) for x in spin_phases_rad]}),
        subject={"pair": (int(pair[0]), int(pair[1]))},
    )


__all__ = ["ms_phase_scan", "ms_scan", "parity_scan"]
