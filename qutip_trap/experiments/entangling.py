"""Entangling-gate experiments: the MS amplitude and detuning scan, the parity scan and the MS phase scan (PLAN.md Section
7.5 items 3 and 4).

``ms_scan`` plays the pair's waveform from |00> at scaled amplitudes and detuning offsets: the closure point is the offset
of least leakage P_01 + P_10 (open loops; a parabola fit), the entangling amplitude the scale s at which (P_00 - P_11)/
(P_00 + P_11) = cos(2 chi_1 s^2) reaches chi = pi/4. ``parity_scan`` follows the gate with a pi/2 analysis pulse of scanned
phase on both ions; the contrast C of Pi(phi) = C cos(2 phi + phi_0) + B bounds the Bell fidelity (P_00 + P_11 + C)/2
(Wright 2019). ``ms_phase_scan`` scans the gate's spin phases against a fixed analysis pulse of the single-qubit drives:
the parity's phase offset against the same scan on ideal matrices is the misalignment of the entangling axis relative to
the single-qubit frame, from |00> the SUM of the two ions' offsets and from |01> their DIFFERENCE. All three run in the
frame the machine programs (``qubit_shifts_hz``) at the mode frequencies the table believes (``mode_frequencies_hz``), so
the corrections they write carry the frame error ``run`` applies.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, NamedTuple, Unpack

import numpy as np

from qutip_trap.experiments.fitting import at_scan_edge, fit_fringe, weighted_fit, wrap_angle
from qutip_trap.experiments.result import ExperimentResult, ParityScan, ScanParameters
from qutip_trap.experiments.single_ion import _Lab, _LabOptions

if TYPE_CHECKING:
    from qutip_trap.calibration.entangling import GateCheck
    from qutip_trap.control.schedule import GateDrive
    from qutip_trap.control.shaping import GateModes
    from qutip_trap.control.table import CalibrationTable, Waveform
    from qutip_trap.device.model import Device
    from qutip_trap.dynamics.space import HilbertSpace
    from qutip_trap.machine import Machine
    from qutip_trap.options import Physics


class _GateSetup(NamedTuple):
    waveform: Waveform
    entangling: dict[int, GateDrive]
    single: dict[int, GateDrive]
    table: CalibrationTable
    modes: GateModes
    space: HilbertSpace


def _entangling_setup(device: Device, pair: tuple[int, int], kw: Mapping[str, Any]) -> _GateSetup:
    """The pair's waveform from ``kw["table"]``, the device's drives, and the gate modes and space built at the mode
    frequencies the machine believes (``kw["mode_frequencies_hz"]``, never the crystal's hidden truth) unless ``kw["modes"]``
    and ``kw["space"]`` are given; the space follows the run's rule under ``kw["options"]``."""
    from qutip_trap.calibration.entangling import spot_check_space
    from qutip_trap.control.shaping import gate_modes

    table = kw.get("table")
    waveform = table.waveform_for(pair) if table is not None else None
    if table is None or waveform is None:
        raise ValueError(
            "the entangling experiments read the pair's waveform from a CalibrationTable with an ms entry"
        )
    resolved = device.roles.resolve(device)
    ent, sq = dict(resolved.entangling), dict(resolved.gate)
    beams = ent[pair[0]].beams
    if len(beams) != 2:
        raise ValueError("the entangling experiments take a Raman (two-beam) entangling drive")
    nbar = {int(k): float(v) for k, v in dict(kw.get("nbar") or {}).items()}
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
    occupied = replace(modes, nbar=tuple(nbar.get(m, 0.0) for m in modes.modes))
    space = kw.get("space") or spot_check_space(device, occupied, waveform, pair, kw["options"])[0]
    return _GateSetup(waveform, ent, sq, table, modes, space)


def _gate_lab(
    machine: Machine,
    pair: tuple[int, int],
    modes: GateModes | None,
    space: HilbertSpace | None,
    mode_frequencies_hz: Mapping[int, float] | None,
    kw: _LabOptions,
) -> tuple[_Lab, _GateSetup]:
    """The lab of an entangling experiment and the gate setup ``_entangling_setup`` builds for it."""
    lab = _Lab.of(machine, kw)
    setup = {
        "table": lab.table,
        "nbar": lab.nbar,
        "modes": modes,
        "space": space,
        "mode_frequencies_hz": mode_frequencies_hz,
        "options": lab.options,
    }
    return lab, _entangling_setup(lab.device, pair, setup)


def _lab_physics(lab: _Lab) -> Physics:
    """The physics the lab's engines play: the machine's under the call's builder options (``builder_options``)."""
    return replace(lab.physics, builder=lab.builder)


def _gate_check(lab: _Lab, g: _GateSetup, waveform: Waveform, pair: tuple[int, int]) -> GateCheck:
    """The gate played once from |00> on the exact space as a run plays it under the lab's physics
    (``calibration.entangling.exact_gate_check``)."""
    from qutip_trap.calibration.entangling import exact_gate_check

    check, _traces = exact_gate_check(
        lab.device,
        waveform,
        pair,
        g.entangling,
        g.table,
        space=g.space,
        physics=_lab_physics(lab),
        nbar=lab.nbar,
        options=lab.options,
        sample=lab.sample,
        qubit_shifts_hz=lab.qubit_shifts_hz,
        single_qubit_drives=g.single,
    )
    return check


def _parity_populations(
    lab: _Lab,
    g: _GateSetup,
    pair: tuple[int, int],
    analysis_phase_rad: float,
    spin_phases_rad: tuple[float, float] = (0.0, 0.0),
    internal: Sequence[int] | None = None,
) -> np.ndarray:
    """(P00, P01, P10, P11) after the gate and a pi/2 analysis pulse of the single-qubit drives on both ions, as a run plays
    GPi2 after the gate under the lab's physics: at the table's Rabi frequencies, with its Stark shifts and crosstalk
    (``calibration.entangling.parity_after_analysis_pulse``)."""
    from qutip_trap.calibration.entangling import parity_after_analysis_pulse

    _par, pops = parity_after_analysis_pulse(
        lab.device,
        g.waveform,
        pair,
        g.entangling,
        g.table,
        space=g.space,
        physics=_lab_physics(lab),
        analysis_phase_rad=analysis_phase_rad,
        nbar=lab.nbar,
        options=lab.options,
        sample=lab.sample,
        single_qubit_drives=g.single,
        spin_phases_rad=spin_phases_rad,
        internal=internal,
        qubit_shifts_hz=lab.qubit_shifts_hz,
    )
    return _pops4(pops)


def _pops4(pops: Mapping[str, float]) -> np.ndarray:
    return np.array([pops["P00"], pops["P01"], pops["P10"], pops["P11"]], dtype=float)


@dataclass(frozen=True, eq=False)
class OffsetFn:
    """tau -> fn(tau) + offset: a frequency-modulated leg's detuning shifted by a scan offset, picklable when ``fn`` is
    (every coefficient on the pulse path must be, for the parallel solver map)."""

    fn: Callable[[float], float]
    offset: float

    def __call__(self, tau: float) -> float:
        return float(self.fn(tau)) + self.offset


def shift_detuning(waveform: Waveform, offset_hz: float) -> Waveform:
    """Every blue leg + offset, every red leg - offset: the symmetric detuning scan of Section 7.5."""
    from qutip_trap.control.table import Segment

    if offset_hz == 0.0:
        return waveform

    def shift(v: Any, sign: float) -> Any:
        return OffsetFn(v, sign * offset_hz) if callable(v) else float(v) + sign * offset_hz

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
    *,
    modes: GateModes | None = None,
    space: HilbertSpace | None = None,
    mode_frequencies_hz: Mapping[int, float] | None = None,
    **kw: Unpack[_LabOptions],
) -> ExperimentResult:
    """Populations after the pair's waveform at amplitude scale factors ``amplitudes`` and detuning offsets
    ``detunings_hz`` (added to every blue leg's beat note, subtracted from every red one); data columns (scale, offset_hz,
    P00, P01 + P10, P11). ``modes`` and ``space`` replace the ones built at ``mode_frequencies_hz`` (default the crystal's).

    Fitted: ``closure_offset_hz`` and ``leakage_min`` (the leakage parabola over three or more offsets at the scale nearest
    one), ``chi_unit_rad`` (the angle at unit scale from cos(2 chi_1 s^2)) and ``closure_scale`` = sqrt((pi/4)/chi_1), both
    measured at ``closure_offset_used_hz``: the fitted closure offset when the parabola converged (one extra amplitude scan
    when the grid lacks it), so the table applies the two together. Not converged when a fit failed or lands at the edge
    of its scan."""
    from qutip_trap.control.shaping import scaled

    lab, g = _gate_lab(machine, pair, modes, space, mode_frequencies_hz, kw)
    scales = sorted(float(x) for x in amplitudes)
    offsets = sorted(float(x) for x in detunings_hz)
    if any(s <= 0.0 for s in scales):
        raise ValueError("amplitude scale factors are positive")
    rows: list[tuple[float, float, float, float, float]] = []
    sig: list[np.ndarray | None] = []

    def amplitude_row(off: float) -> None:
        """One amplitude scan at detuning offset ``off``, appended to the rows."""
        wf_off = shift_detuning(g.waveform, off)
        for s in scales:
            check = _gate_check(lab, g, scaled(wf_off, s), pair)
            p, sg = lab.obs.joint(_pops4(check.populations), pair, "ms_scan", len(rows))
            rows.append((s, off, float(p[0]), float(p[1] + p[2]), float(p[3])))
            sig.append(sg)

    for off in offsets:
        amplitude_row(off)
    data = np.array(rows)
    fitted: dict[str, tuple[float, float]] = {}
    notes: list[str] = []
    converged = True
    s_ref = min(scales, key=lambda s: abs(s - 1.0))
    off_used = float(offsets[int(np.argmin(np.abs(offsets)))])
    if len(offsets) >= 3:
        # the closure point: the leakage minimum over the detuning offsets at the scale nearest one
        sel = data[:, 0] == s_ref
        x, y = data[sel, 1], data[sel, 3]
        sg_leak = None
        if all(v is not None for v in sig):
            sg_leak = np.array([math.hypot(float(v[1]), float(v[2])) for v in sig if v is not None])[sel]
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
            # the scale is measured at the fitted closure offset, the one the table then applies (one extra amplitude
            # scan when the grid does not already carry it)
            off_used = float(off0)
            step = float(np.min(np.diff(np.asarray(offsets)))) if len(offsets) > 1 else 0.0
            if min(abs(off_used - o) for o in offsets) > 1e-3 * max(step, 1.0):
                amplitude_row(off_used)
                data = np.array(rows)
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
        sg_c = None
        if all(v is not None for v in sig):
            rows_sel = [v for v, keep in zip(sig, sel) if keep and v is not None]
            sg_c = np.array([math.hypot(float(v[0]), float(v[3])) for v in rows_sel]) / np.maximum(
                p00 + p11, 1e-9
            )
        fit_c = weighted_fit(
            lambda p, s: np.cos(2.0 * float(p[0]) * np.asarray(s) ** 2),
            [abs(float(g.waveform.chi_total_rad)) or math.pi / 4.0],
            sc,
            balance,
            sigma=sg_c,
            bounds=([0.0], [math.pi]),
        )
        chi1, s_chi = fit_c.value(0)
        if fit_c.converged and chi1 > 0.0:
            scale = math.sqrt((math.pi / 4.0) / chi1)
            fitted["chi_unit_rad"] = (chi1, s_chi)
            fitted["closure_scale"] = (scale, 0.5 * scale * s_chi / chi1)
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
    return ExperimentResult(
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
        experiment="ms_scan",
    )


def parity_scan(
    machine: Machine,
    pair: tuple[int, int],
    analysis_phases_rad: Sequence[float],
    *,
    modes: GateModes | None = None,
    space: HilbertSpace | None = None,
    mode_frequencies_hz: Mapping[int, float] | None = None,
    **kw: Unpack[_LabOptions],
) -> ExperimentResult:
    """Parity after the gate and a pi/2 analysis pulse of phase phi on both ions (Section 7.9); data columns (phi, parity,
    P00, P11). Fitted: contrast C, phi0_rad and offset of Pi(phi) = C cos(2 phi + phi_0) + B, and the Bell-fidelity bound
    (P_00 + P_11 + C)/2 with the populations P00 and P11 read without the analysis pulse."""
    lab, g = _gate_lab(machine, pair, modes, space, mode_frequencies_hz, kw)
    rows: list[tuple[float, float, float, float]] = []
    sig: list[float | None] = []
    for k, phi in enumerate(sorted(float(x) for x in analysis_phases_rad)):
        p, sg = lab.obs.joint(_parity_populations(lab, g, pair, phi), pair, "parity_scan", k)
        rows.append((phi, float(p[0] + p[3] - p[1] - p[2]), float(p[0]), float(p[3])))
        sig.append(None if sg is None else float(np.sqrt(np.sum(sg**2))))
    data = np.array(rows)
    sigma = None if any(s is None for s in sig) else np.array([s for s in sig if s is not None])
    fitted: dict[str, tuple[float, float]] = {}
    converged = True
    chi2: float | None = None
    if data.shape[0] >= 4:
        fringe = fit_fringe(data[:, 0], data[:, 1], sigma, 2)
        fitted = {"contrast": fringe.contrast, "phi0_rad": fringe.phase_rad, "offset": fringe.offset}
        pops = _gate_check(lab, g, g.waveform, pair).populations
        pb, sgb = lab.obs.joint(_pops4(pops), pair, "parity_populations", 0)
        s_pop = 0.0 if sgb is None else float(math.hypot(float(sgb[0]), float(sgb[3])))
        fitted["bell_fidelity_bound"] = (
            0.5 * (float(pb[0]) + float(pb[3]) + fringe.contrast[0]),
            0.5 * math.hypot(fringe.contrast[1], s_pop),
        )
        fitted["P00"] = (float(pb[0]), 0.0 if sgb is None else float(sgb[0]))
        fitted["P11"] = (float(pb[3]), 0.0 if sgb is None else float(sgb[3]))
        converged, chi2 = fringe.converged, fringe.chi2_per_dof
    return ParityScan(
        data=data,
        fitted=fitted,
        model="parity_oscillation",
        provenance_id="conv.entangling_angle",
        converged=converged,
        sigma=sigma,
        requested=ScanParameters({"analysis_phases_rad": [float(x) for x in analysis_phases_rad]}),
        chi2=chi2,
        subject={"pair": (int(pair[0]), int(pair[1]))},
    )


def _ideal_parity(
    theta_rad: float, spins: Sequence[tuple[float, float]], internal: tuple[int, int]
) -> np.ndarray:
    """Pi(phi) on ideal matrices: MS(phi_0, phi_1, theta) then GPi2(0) on both ions from |internal>."""
    from qutip_trap.control.native import gpi2
    from qutip_trap.control.native import ms as native_ms

    ket = np.zeros(4, dtype=complex)
    ket[2 * internal[0] + internal[1]] = 1.0
    analysis = np.kron(gpi2(0.0), gpi2(0.0))
    out = []
    for phi0, phi1 in spins:
        p = np.abs(analysis @ (native_ms(phi0, phi1, theta_rad) @ ket)) ** 2
        out.append(float(p[0] + p[3] - p[1] - p[2]))
    return np.array(out)


def ms_phase_scan(
    machine: Machine,
    pair: tuple[int, int],
    spin_phases_rad: Sequence[float],
    *,
    inputs: Sequence[str] = ("00",),
    modes: GateModes | None = None,
    space: HilbertSpace | None = None,
    mode_frequencies_hz: Mapping[int, float] | None = None,
    **kw: Unpack[_LabOptions],
) -> ExperimentResult:
    """Section 7.5 step 3: the MS gate's spin phases scanned against a fixed GPi2(0) analysis pulse of the single-qubit
    drives. From ``inputs`` "00" the phases (phi, phi) give C cos(2 phi + phi_0), whose offset against the ideal scan is
    the SUM of the two ions' misalignments; from "01" (a GPi(0) on the second ion first) the phases (phi, 0) give
    C cos(phi + phi_0') and the DIFFERENCE. Fitted offset_sum_rad, contrast_00 and, with both inputs, offset_diff_rad,
    contrast_01; per ion correction_rad[i] to ADD to the ion's leg phases (``control.shaping.phase_shifted``). Data rows
    (input, spin phase, parity)."""
    lab, g = _gate_lab(machine, pair, modes, space, mode_frequencies_hz, kw)
    theta = 2.0 * abs(float(g.waveform.chi_total_rad))
    phases = np.array(sorted(float(x) for x in spin_phases_rad))
    rows: list[tuple[float, float, float]] = []
    sig_all: list[float | None] = []
    fitted: dict[str, tuple[float, float]] = {}
    notes: list[str] = []
    converged = True
    for inp in tuple(inputs):
        if inp not in ("00", "01"):
            raise ValueError("inputs are '00' and/or '01'")
        common = inp == "00"  # a common spin-phase offset from |00>; the second ion's alone from |01>
        k_period = 2 if common else 1
        internal = [0] * lab.device.crystal.n_ions
        internal[pair[1]] = 0 if common else 1
        spins = [(phi, phi if common else 0.0) for phi in phases]
        signal: list[float] = []
        sig: list[float | None] = []
        for k, sp in enumerate(spins):
            pops = _parity_populations(lab, g, pair, 0.0, sp, internal)
            p, sg = lab.obs.joint(pops, pair, f"ms_phase_scan[{inp}]", k)
            parity = float(p[0] + p[3] - p[1] - p[2])
            signal.append(parity)
            sig.append(None if sg is None else float(np.sqrt(np.sum(sg**2))))
            rows.append((float(not common), sp[0], parity))
        sig_all.extend(sig)
        levels = (internal[pair[0]], internal[pair[1]])
        phi_ideal = fit_fringe(phases, _ideal_parity(theta, spins, levels), None, k_period).phase_rad[0]
        shifted = [(a + 0.1, b + (0.1 if common else 0.0)) for a, b in spins]
        phi_shifted = fit_fringe(phases, _ideal_parity(theta, shifted, levels), None, k_period).phase_rad[0]
        # d phi_0/d(offset): +-k for either scan; the sign of the convention comes from the ideal
        response = wrap_angle(phi_shifted - phi_ideal) / 0.1
        sigma = None if any(s is None for s in sig) else np.array([s for s in sig if s is not None])
        measured = fit_fringe(phases, np.array(signal), sigma, k_period)
        if abs(response) < 0.1:
            converged = False
            notes.append(
                f"input {inp}: the ideal scan does not respond to a spin-phase offset; no alignment measured"
            )
            continue
        # the offset in spin-phase units: the sum (00) or the difference (01) of the per-ion misalignments
        key = "offset_sum_rad" if common else "offset_diff_rad"
        fitted[key] = (
            wrap_angle(measured.phase_rad[0] - phi_ideal) / response * k_period,
            measured.phase_rad[1] / abs(response) * k_period,
        )
        fitted[f"contrast_{inp}"] = (measured.contrast[0], 0.0)
        converged = converged and measured.converged
    if "offset_sum_rad" in fitted:
        s_sum, e_sum = fitted["offset_sum_rad"]
        if "offset_diff_rad" in fitted:
            d, e_d = fitted["offset_diff_rad"]
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
    return ExperimentResult(
        data=np.array(rows),
        fitted=fitted,
        model="ms_spin_phase_scan",
        provenance_id="conv.spin_motion_phases",
        converged=converged,
        notes=tuple(notes),
        sigma=None if any(s is None for s in sig_all) else np.array([s for s in sig_all if s is not None]),
        requested=ScanParameters({"spin_phases_rad": [float(x) for x in spin_phases_rad]}),
        subject={"pair": (int(pair[0]), int(pair[1]))},
        experiment="ms_phase_scan",
    )
