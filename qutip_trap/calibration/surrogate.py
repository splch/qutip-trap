"""The surrogate calibration table, the default calibration of PLAN.md Section 7.5.

The derived values of the device become ``seed`` entries, the entangling waveforms come from the closed-form integrals
corrected by the exact spot checks of ``calibration.entangling`` on the resolved-mode space, and the detection threshold
and window come from ``calibration.readout`` on the simulated readout model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from qutip_trap.calibration.entangling import (
    CalibrationRun,
    calibrate_entangling_angle,
    ms_schedule,
    spot_check_space,
)
from qutip_trap.calibration.readout import DetectionCalibration, calibrate_detection
from qutip_trap.control.schedule import GateDrive, resolve_drives
from qutip_trap.control.shaping import (
    CHI_MAXIMAL_RAD,
    ClosureError,
    GateModes,
    ShapedPulse,
    gate_modes,
    solve_amplitude_modulation,
    symmetric_pulse,
)
from qutip_trap.control.table import CalEntry, CalibrationTable, Waveform
from qutip_trap.light.raman import (
    crosstalk_ratios,
    derive_optical_drive,
    derive_raman_drive,
)
from qutip_trap.light.roles import NoDetectionBeamError, detection_beams
from qutip_trap.options import Numerics, Physics
from qutip_trap.prep.recipe import preparation_occupations, recipe_of
from qutip_trap.readout.detection import RecordModel
from qutip_trap.readout.fluorescence import detection_rates_for_ion
from qutip_trap.run.levels import within_budget
from qutip_trap.units import TWO_PI

if TYPE_CHECKING:
    from qutip_trap.device.model import Device
    from qutip_trap.run.space import ModeClass3

CROSSTALK_MIN = 1e-6
"""Rabi ratios below this are not stored (a beam of finite waist gives every ion some light)."""
MU_ABOVE_TOP_FRACTION = 0.35
"""The surrogate's AM beat note sits this fraction of the smallest mode gap above the highest coupled mode."""
AM_CONDITION_MAX = 10.0
"""The largest condition number of its two-body angle (``angle_condition_number``) the surrogate accepts from a segmented
AM solution before it adds segments (PLAN.md Section 7.5).

Choi's 2N + 1 segments leave the 2N real closure conditions a ONE-dimensional null space, whose one direction is scaled to
|chi| = pi/4 however little angle it carries; where it carries almost none, the per-mode angles are large and cancel. On
pair (0, 1) of ``presets.yb171_chain`` (radial 3.0 and 2.9 MHz) Choi's solutions of 2 to 5 ions at 1.0 MHz axial, 6 at
0.6, 8 at 0.4 and 12 at 0.35 have condition numbers 1.20 to 2.30 and need 1.5 to 3.2 times the entangling pair's
full-power carrier Rabi frequency (2 ions: 1.74, 228 kHz against 147 kHz), and 8 ions at 0.5 MHz a condition number of
517 (chi_m of -177, +83, +59, +50, -26, ... rad), a 9.745 MHz peak, 70.7 carriers. Over every pair of 25 linear chains of
2 to 12 ions at 0.3 to 1.5 MHz axial (272 pairs up to mirror symmetry) the median is 1.96, the chains the test suite
calibrates stay below 2.7, and the 25 pairs above 10 each pass at 2N + 3 (one at 2N + 5), with condition numbers below
3.0 and peaks below 4.7 carriers. The condition number rather than the peak against the carrier because the waveform does
not depend on the beam power and the carrier does: a ten times stronger pair would accept the 8-ion solution at 7.1
carriers and a ten times weaker one refuse the 2-ion solution at 15.6."""
AM_SEGMENT_STEP = 2
"""Segments are added two at a time, each one more null-space dimension, so that the count stays odd like Choi's 2N + 1."""
AM_SEGMENTS_ADDED_MAX = 16
"""The most segments the surrogate adds to Choi's 2N + 1 (eight steps, a 17-dimensional null space): every refused pair
of ``AM_CONDITION_MAX``'s survey passes within four, so the cap bounds only the search (about 10 ms a solve for 12 ions at
2N + 17)."""


def _seed(value: float, pid: str, experiment: str, t0_s: float) -> CalEntry:
    return CalEntry(float(value), 0.0, "seed", experiment, pid, float(t0_s), 0)


@dataclass(frozen=True)
class SurrogateReport:
    """What the surrogate calibration did per pair and per ion."""

    table: CalibrationTable
    entangling: dict[tuple[int, int], CalibrationRun]
    mode_classes: dict[tuple[int, int], dict[int, ModeClass3]]
    frozen_chi_rad: dict[tuple[int, int], dict[int, float]]
    """Per pair, the chi_m of the modes frozen in its spot check (the loss the calibration target absorbed)."""
    detection: DetectionCalibration | None
    nbar: dict[int, float]
    notes: tuple[str, ...] = field(default_factory=tuple)


def angle_condition_number(waveform: Waveform) -> float:
    """kappa = sum_m |chi_m| / |sum_m chi_m|: the condition number of the waveform's two-body angle as the sum of its
    per-mode angles, the factor by which the angle amplifies a relative error of one mode's loop area (1 when every mode
    adds to the angle with one sign)."""
    return sum(abs(chi) for chi in waveform.chi_m.values()) / abs(waveform.chi_total_rad)


def _peak_rabi_hz(shaped: ShapedPulse) -> float:
    """The largest per-tone Rabi frequency the pulse plays on any of its ions, Hz."""
    return max(abs(float(x)) for amps in shaped.envelope.amplitude_rad_s.values() for x in amps) / TWO_PI


@dataclass(frozen=True)
class SurrogatePulse:
    """The closed-form pulse ``surrogate_waveform`` solves for one pair: the solution, the modes it closes and, when the
    segment rule refused Choi's 2N + 1 solution, that solution."""

    shaped: ShapedPulse
    modes: GateModes
    refused: ShapedPulse | None = None
    """Choi's 2N + 1 solution when its angle's condition number exceeds ``AM_CONDITION_MAX``, ``shaped`` being what the
    rule kept in its place; None when ``shaped`` is Choi's solution or the one-mode square pulse."""

    @property
    def summary(self) -> str:
        """The pulse as the calibration notes name it: the solver's method and, when the rule refused Choi's count, the
        segments, peak Rabi frequency and condition number of the refused solution and of the kept one."""
        head = f"{self.shaped.method} waveform"
        if self.refused is None:
            return head
        n_min, n = len(self.refused.waveform.segments), len(self.shaped.waveform.segments)
        kappa = angle_condition_number(self.shaped.waveform)
        choi = (
            f"Choi's 2N + 1 = {n_min} need a {_peak_rabi_hz(self.refused) / 1e3:.1f} kHz peak, their angle's condition "
            f"number {angle_condition_number(self.refused.waveform):.3g} above AM_CONDITION_MAX = {AM_CONDITION_MAX:g}"
        )
        kept = f"{_peak_rabi_hz(self.shaped) / 1e3:.1f} kHz at condition number {kappa:.3g}"
        if kappa <= AM_CONDITION_MAX:
            return f"{head} on {n} segments ({choi}; {n} need {kept})"
        return (
            f"{head} on {n} segments ({choi}, and no count up to {n_min + AM_SEGMENTS_ADDED_MAX} comes within it: the "
            f"lowest peak, {kept} on {n}, is kept)"
        )


def surrogate_waveform(
    device: Device,
    pair: tuple[int, int],
    beams: tuple[int, int],
    *,
    nbar: Mapping[int, float],
    duration_s: float = 100e-6,
    mode_frequencies_hz: Mapping[int, float] | None = None,
) -> SurrogatePulse:
    """The closed-form pulse for ``pair``: the symmetric square pulse when the Raman pair couples the pair to one mode,
    else Choi's segmented AM with the beat note ``MU_ABOVE_TOP_FRACTION`` of the smallest gap between the coupled modes
    above the highest one (Landsman's and Chen's placement above the spectrum; at the midpoint of two modes the
    power-optimal direction of the one-dimensional null space flips within tens of hertz). The AM pulse has Choi's 2N + 1
    segments unless their angle is ill-conditioned (``angle_condition_number`` above ``AM_CONDITION_MAX``); then segments
    are added ``AM_SEGMENT_STEP`` at a time, up to ``AM_SEGMENTS_ADDED_MAX``, and the fewest that pass are kept, or the
    lowest peak Rabi frequency when none does. ``mode_frequencies_hz`` solves at the frequencies the table BELIEVES
    instead of the crystal's."""
    modes = gate_modes(device, pair, beams, nbar=nbar, eta_min=1e-9, mode_frequencies_hz=mode_frequencies_hz)
    if modes.n_modes == 1:
        return SurrogatePulse(
            symmetric_pulse(modes, gate_mode=modes.modes[0], loops=1, duration_s=duration_s), modes
        )
    freqs = sorted(w / TWO_PI for w in modes.omega_rad_s)
    gap = min(b - a for a, b in zip(freqs[:-1], freqs[1:]))
    mu = freqs[-1] + MU_ABOVE_TOP_FRACTION * gap
    minimal = solve_amplitude_modulation(modes, mu_hz=mu, duration_s=duration_s)
    if angle_condition_number(minimal.waveform) <= AM_CONDITION_MAX:
        return SurrogatePulse(minimal, modes)
    n_min = len(minimal.waveform.segments)
    tried = [minimal]
    for n in range(n_min + AM_SEGMENT_STEP, n_min + AM_SEGMENTS_ADDED_MAX + 1, AM_SEGMENT_STEP):
        longer = solve_amplitude_modulation(modes, mu_hz=mu, duration_s=duration_s, n_segments=n)
        if angle_condition_number(longer.waveform) <= AM_CONDITION_MAX:
            return SurrogatePulse(longer, modes, refused=minimal)
        tried.append(longer)
    return SurrogatePulse(min(tried, key=_peak_rabi_hz), modes, refused=minimal)


def canonical_pairs(device: Device, pairs: Sequence[Sequence[int]] | None) -> tuple[tuple[int, int], ...]:
    """The entangling pairs of a calibration request in one form: each pair of distinct ions of the crystal as (lower ion,
    higher ion), each once, in increasing order; None is every pair of the crystal. A pair's waveform serves either key
    order (``CalibrationTable.waveform_for``), so two requests that name the same pairs are one request."""
    n = device.crystal.n_ions
    if pairs is None:
        return tuple((a, b) for a in range(n) for b in range(a + 1, n))
    out: set[tuple[int, int]] = set()
    for pair in pairs:
        ions = sorted(int(q) for q in pair)
        if len(ions) != 2 or ions[0] == ions[1] or ions[0] < 0 or ions[1] >= n:
            raise ValueError(
                f"an entangling pair is two distinct ions of the {n}-ion crystal, not {tuple(pair)}"
            )
        out.add((ions[0], ions[1]))
    return tuple(sorted(out))


def surrogate_table(
    device: Device,
    *,
    seed: int = 0,
    t0_s: float = 0.0,
    gate_drives: Mapping[int, GateDrive] | None = None,
    entangling_drives: Mapping[int, GateDrive] | None = None,
    pairs: Sequence[tuple[int, int]] | None = None,
    spot_check: bool = True,
    detection_records: int = 10_000,
    detection_windows_s: Sequence[float] | None = None,
    options: Numerics | None = None,
    physics: Physics | None = None,
) -> SurrogateReport:
    """The surrogate CalibrationTable for ``device``: ``pairs`` restricts the entangling waveforms (default:
    every pair of the crystal; each keyed as ``canonical_pairs`` orders it), ``spot_check=False`` stores the closed-form
    waveforms as seeds, the explicit drive maps override the device's roles; the spot checks integrate with ``options``
    (its caps too) and play the gate a run plays under ``physics`` (the machine's; default ``Physics()``: the Stark
    compensation, the crosstalk echo, the hardware chain and the builder options)."""
    opts = options or Numerics()
    phys = physics if physics is not None else Physics()
    drives, ent = resolve_drives(device, gate_drives, entangling_drives)
    crystal = device.crystal
    n = crystal.n_ions
    notes: list[str] = []
    hint = next(
        ((int(d.beams[0]), int(d.beams[1])) for d in ent.values() if d.kind == "raman" and len(d.beams) == 2),
        None,
    )
    occupations = preparation_occupations(device, recipe_of(device, raman_pair=hint))
    if device.preparation is None:
        notes.append("preparation recipe inferred by prep.recipe.standard_recipe (the device carries none)")
    b_gauss = device.field.B_gauss
    # qubit frequencies, Rabi frequencies, Stark shifts and crosstalk from the derived values
    qubit_freq: dict[int, CalEntry] = {}
    rabi: dict[tuple[int, int], CalEntry] = {}
    stark: dict[tuple[int, int], CalEntry] = {}
    crosstalk: dict[tuple[int, int], CalEntry] = {}
    for i in range(n):
        sp = crystal.species[i]
        f0, _d1, _d2 = sp.transition_frequency_hz(sp.qubit[0], sp.qubit[1], b_gauss)
        qubit_freq[i] = _seed(f0, "conv.frequencies", "derived_transition_frequency", t0_s)
        spec = drives[i]
        key = (i, spec.table_key_beam)
        if spec.kind == "raman":
            dd = derive_raman_drive(device, i, (spec.beams[0], spec.beams[1]), scattering=False)
            pid, experiment = "conv.two_photon_rabi", "derived_raman_drive"
        elif spec.kind in ("optical_E1", "optical_E2"):
            dd = derive_optical_drive(device, i, spec.beams[0], scattering=False)
            pid, experiment = "conv.rabi_frequency", "derived_optical_drive"
        else:
            notes.append(
                f"ion {i}: a {spec.kind} drive has no derivable carrier Rabi frequency; entry left absent"
            )
            continue
        rabi[key] = _seed(dd.carrier_rabi_hz, pid, experiment, t0_s)
        stark[key] = _seed(dd.stark_shift_hz, pid, experiment, t0_s)
        for j, eps in crosstalk_ratios(device, i, spec.beams, kind=spec.kind).items():
            if abs(eps) >= CROSSTALK_MIN:
                crosstalk[(i, j)] = _seed(abs(eps), "conv.crosstalk_ratio", "derived_beam_profile", t0_s)
    # the entangling drives' own carrier Rabi frequencies and Stark shifts (the global pair), keyed by their table beam, so
    # that the scheduler compensates the entangling gate's light shift and the played chain has a belief to convert against;
    # only the ions that HAVE an entangling drive
    for i in sorted(ent):
        spec = ent[i]
        key = (i, spec.table_key_beam)
        if key in rabi or spec.kind != "raman" or len(spec.beams) != 2:
            continue
        dd = derive_raman_drive(device, i, (spec.beams[0], spec.beams[1]), scattering=False)
        rabi[key] = _seed(dd.carrier_rabi_hz, "conv.two_photon_rabi", "derived_raman_drive", t0_s)
        stark[key] = _seed(dd.stark_shift_hz, "conv.two_photon_rabi", "derived_raman_drive", t0_s)
    modes_entries = {
        m: _seed(mode.omega_hz, "conv.mode_index", "derived_crystal_modes", t0_s)
        for m, mode in enumerate(crystal.modes)
    }
    nbar_entries = {
        m: _seed(v, "conv.preparation_stage_order", "preparation_model", t0_s) for m, v in occupations.items()
    }
    # the rate the noise model heats the run at (the correlation length and the spectrum's white level folded in); no rate
    # without field noise
    model_rates = device.noise.heating_rates_quanta_per_s(device)
    heating = {
        m: _seed(float(model_rates.get(m, 0.0)), "conv.electric_field_noise", "derived_heating_rate", t0_s)
        for m in range(len(crystal.modes))
    }
    base = CalibrationTable(
        device_hash=device.hash(),
        seed=int(seed),
        surrogate=True,
        qubit_freq=qubit_freq,
        rabi=rabi,
        stark=stark,
        crosstalk=crosstalk,
        modes=modes_entries,
        nbar=nbar_entries,
        ms={},
        field=_seed(b_gauss, "conv.curvature_naming", "field_value", t0_s),
        micromotion={},
        detection={},
        heating=heating,
        fitted_at_s=float(t0_s),
    )
    # entangling waveforms per pair: closed-form solution, mode classes, exact spot check on the resolved space
    ms: dict[tuple[int, int], Waveform] = {}
    runs: dict[tuple[int, int], CalibrationRun] = {}
    classes_by_pair: dict[tuple[int, int], dict[int, ModeClass3]] = {}
    frozen_chi: dict[tuple[int, int], dict[int, float]] = {}
    for a, b in canonical_pairs(device, pairs):
        if a not in ent:
            notes.append(f"pair {(a, b)}: ion {a} has no entangling drive; no waveform is solved for it")
            continue
        spec_a = ent[a]
        if spec_a.kind != "raman" or len(spec_a.beams) != 2:
            notes.append(f"pair {(a, b)}: the surrogate solves Raman (bichromatic) waveforms only; skipped")
            continue
        try:
            pulse = surrogate_waveform(device, (a, b), (spec_a.beams[0], spec_a.beams[1]), nbar=occupations)
        except ClosureError as exc:
            notes.append(f"pair {(a, b)}: no closed-form waveform ({exc}); skipped")
            continue
        shaped, modes = pulse.shaped, pulse.modes
        space_for = partial(spot_check_space, device, modes, shaped.waveform, (a, b), opts)
        space, classes = space_for()
        classes_by_pair[(a, b)] = classes
        frozen_chi[(a, b)] = {
            m: float(shaped.waveform.chi_m.get(m, 0.0)) for m, cls in classes.items() if cls == "frozen"
        }
        inside, dim, nnz = within_budget(space, opts)
        if spot_check and not inside:
            # the pair's GATE_LOCAL space: the ions the check plays pulses on (the pair, and every spectator a neighbour
            # echo plays on) and the resolved modes, the rest of the chain absent
            played = ms_schedule(
                device,
                shaped.waveform,
                (a, b),
                ent,
                base,
                physics=phys,
                options=opts,
                single_qubit_drives=drives,
            )
            space, _classes_local = space_for(ions=sorted({i for p in played.pulses for i in p.drive.ions}))
            inside, dim_local, nnz_local = within_budget(space, opts)
            notes.append(
                f"pair {(a, b)}: the joint spot-check space (dimension {dim}, {nnz} drive non-zeros) exceeds the JOINT_EXACT "
                f"guards ({opts.joint_dimension_max}, {opts.nnz_max}); the exact check runs on the pair's GATE_LOCAL space "
                f"(dims {space.dims}, dimension {dim_local}, {nnz_local} non-zeros)"
            )
            if not inside:
                notes.append(
                    f"pair {(a, b)}: the GATE_LOCAL spot-check space exceeds the guards too; the closed-form waveform is stored "
                    "as a seed"
                )
        if spot_check and inside:
            run = calibrate_entangling_angle(
                device,
                shaped.waveform,
                (a, b),
                ent,
                base,
                space=space,
                physics=phys,
                chi_target_rad=CHI_MAXIMAL_RAD,
                options=opts,
                single_qubit_drives=drives,
            )
            runs[(a, b)] = run
            ms[(a, b)] = run.waveform
            notes.append(
                f"pair {(a, b)}: {pulse.summary}, resolved modes {tuple(t.mode for t in space.resolved)} "
                f"(dims {space.dims}), surrogate error {run.surrogate_error:.4f}, converged {run.converged}"
            )
        else:
            ms[(a, b)] = shaped.waveform
            notes.append(f"pair {(a, b)}: {pulse.summary} stored as the closed-form seed (no spot check)")
    # detection: the threshold and window on ion 0's simulated readout model (one threshold for the register)
    det_cal: DetectionCalibration | None = None
    detection: dict[str, CalEntry] = {}
    try:
        det_idx = detection_beams(device, 0)
    except NoDetectionBeamError as exc:
        notes.append(f"no detection calibration: {exc}")
        det_idx = ()
    if det_idx:
        rates, scheme, _model = detection_rates_for_ion(
            crystal.species[0],
            b_gauss,
            device.field.direction,
            [device.beams[k] for k in det_idx],
            position_m=tuple(float(x) for x in crystal.positions_m[0]),
        )
        windows = (
            tuple(float(x) for x in detection_windows_s)
            if detection_windows_s is not None
            else tuple(float(x) for x in np.geomspace(0.25, 2.5, 12) * device.detector.window_s)
        )
        det_cal = calibrate_detection(
            RecordModel.from_rates(rates, device.detector),
            scheme,
            windows_s=windows,
            n_records=int(detection_records),
            rng=np.random.default_rng(int(seed)),
            fitted_at_s=t0_s,
            sample_id=0,
        )
        detection = dict(det_cal.entries)
    return SurrogateReport(
        table=replace(base, ms=ms, detection=detection),
        entangling=runs,
        mode_classes=classes_by_pair,
        frozen_chi_rad=frozen_chi,
        detection=det_cal,
        nbar=occupations,
        notes=tuple(notes),
    )
