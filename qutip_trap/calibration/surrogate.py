"""The surrogate calibration table, the default calibration of every run (``method="closed_form"``): derived values as
seeds, closed-form entangling waveforms corrected by exact spot checks, the detection threshold and window."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from qutip_trap.calibration.entangling import CalibrationRun, calibrate_entangling_angle
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
from qutip_trap.hilbert.space import HilbertSpace, ModeTruncation
from qutip_trap.light.raman import (
    crosstalk_ratios,
    derive_optical_drive,
    derive_raman_drive,
)
from qutip_trap.light.roles import detection_beams
from qutip_trap.prep.recipe import PreparationRecipe, preparation_occupations, recipe_of
from qutip_trap.readout.detection import RecordModel
from qutip_trap.readout.fluorescence import detection_rates_for_ion
from qutip_trap.run.levels import within_budget
from qutip_trap.run.space import ModeContribution, cap_for, classify, waveform_contributions
from qutip_trap.trap.heating import heating_rate_quanta_per_s, single_sided_from_two_sided
from qutip_trap.units import TWO_PI

if TYPE_CHECKING:
    from qutip_trap.device.model import Device
    from qutip_trap.dynamics.engine import SolverOptions
    from qutip_trap.dynamics.hamiltonian import BuilderOptions

CROSSTALK_MIN = 1e-6
"""Crosstalk Rabi ratios below this are not stored."""
MU_ABOVE_TOP_FRACTION = 0.35
"""The surrogate's AM beat note sits this fraction of the smallest mode gap above the highest coupled mode."""


def _seed(value: float, pid: str, experiment: str, t0_s: float, unc: float = 0.0) -> CalEntry:
    return CalEntry(float(value), float(unc), "seed", experiment, pid, float(t0_s), 0)


@dataclass(frozen=True)
class SurrogateReport:
    """The surrogate table and what the calibration did per pair and per ion."""

    table: CalibrationTable
    entangling: dict[tuple[int, int], CalibrationRun]
    mode_classes: dict[tuple[int, int], dict[int, str]]
    frozen_chi_rad: dict[tuple[int, int], dict[int, float]]
    """Per pair, chi_m of the modes frozen in its spot check."""
    detection: DetectionCalibration | None
    nbar: dict[int, float]
    notes: tuple[str, ...] = field(default_factory=tuple)


def surrogate_waveform(
    device: Device,
    pair: tuple[int, int],
    beams: tuple[int, int],
    *,
    nbar: Mapping[int, float],
    duration_s: float = 100e-6,
    mu_hz: float | None = None,
    eta_min: float = 1e-9,
    mode_frequencies_hz: Mapping[int, float] | None = None,
) -> tuple[ShapedPulse, GateModes]:
    """The closed-form waveform for ``pair``: a symmetric square pulse when the Raman pair couples to one mode, else Choi's
    segmented AM at ``mu_hz``, by default ``MU_ABOVE_TOP_FRACTION`` of the smallest mode gap above the highest coupled mode
    (between two modes the closure is ill-conditioned); ``mode_frequencies_hz`` overrides the crystal's frequencies."""
    modes = gate_modes(
        device, pair, beams, nbar=nbar, eta_min=eta_min, mode_frequencies_hz=mode_frequencies_hz
    )
    if modes.n_modes == 1:
        return symmetric_pulse(modes, gate_mode=modes.modes[0], loops=1, duration_s=duration_s), modes
    freqs = sorted(w / TWO_PI for w in modes.omega_rad_s)
    gap = min(b - a for a, b in zip(freqs[:-1], freqs[1:]))
    mu = float(mu_hz) if mu_hz is not None else freqs[-1] + MU_ABOVE_TOP_FRACTION * gap
    return solve_amplitude_modulation(modes, mu_hz=mu, duration_s=duration_s), modes


def spot_check_space(
    modes: GateModes,
    contributions: Mapping[int, ModeContribution],
    n_ions: int,
    n_modes_total: int,
    *,
    freeze_alpha_max: float,
    freeze_chi_max_rad: float,
    d_min: int = 6,
    d_max: int = 64,
    caps: Mapping[int, int] | None = None,
    ions: Sequence[int] | None = None,
    tail: float | None = None,
) -> tuple[HilbertSpace, dict[int, str]]:
    """The reduced space of a pair's spot check: the ``resolved`` modes capped by ``cap_for`` at boundary threshold ``tail``
    (default the engine's ``boundary_population_max``), the rest frozen, over every ion (JOINT_EXACT) or ``ions`` alone."""
    from qutip_trap.dynamics.engine import SolverOptions

    boundary = SolverOptions().boundary_population_max if tail is None else float(tail)
    classes: dict[int, str] = {}
    resolved: list[ModeTruncation] = []
    for k, m in enumerate(modes.modes):
        c = contributions[m]
        cls = classify(
            c, coupled=True, freeze_alpha_max=freeze_alpha_max, freeze_chi_max_rad=freeze_chi_max_rad
        )
        classes[m] = cls
        if cls == "resolved":
            tr = cap_for(c.radius, modes.nbar[k], c.eta_max, d_min=d_min, d_max=d_max, tail=boundary)
            d = int(caps[m]) if caps is not None and m in caps else tr.d
            resolved.append(ModeTruncation(m, d, (0, min(tr.expected_n_range[1], d - 1)), tr.eta_max))
    frozen = tuple(m for m in range(n_modes_total) if m not in {t.mode for t in resolved})
    if ions is not None:
        local = tuple(sorted(int(i) for i in ions))
        return HilbertSpace(tuple([2] * len(local)), tuple(resolved), None, frozen, ions=local), classes
    return HilbertSpace(tuple([2] * n_ions), tuple(resolved), None, frozen), classes


def surrogate_table(
    device: Device,
    *,
    seed: int = 0,
    t0_s: float = 0.0,
    gate_drives: Mapping[int, GateDrive] | None = None,
    entangling_drives: Mapping[int, GateDrive] | None = None,
    pairs: Sequence[tuple[int, int]] | None = None,
    recipe: PreparationRecipe | None = None,
    nbar: Mapping[int, float] | None = None,
    spot_check: bool = True,
    detection_records: int = 10_000,
    detection_windows_s: Sequence[float] | None = None,
    ms_duration_s: float = 100e-6,
    ms_mu_hz: float | None = None,
    rabi_hz: Mapping[tuple[int, int], float] | None = None,
    options: SolverOptions | None = None,
    builder_options: BuilderOptions | None = None,
    caps: Mapping[int, int] | None = None,
    tolerance_rad: float = 1e-4,
) -> SurrogateReport:
    """Build the surrogate ``CalibrationTable`` for ``device``: ``pairs`` restricts the entangling waveforms (default every
    pair), ``nbar`` overrides the prepared occupations, ``rabi_hz`` supplies Rabi frequencies the device cannot derive
    (microwave drives), ``spot_check=False`` stores the closed-form waveforms as seeds."""
    from qutip_trap.dynamics.engine import SolverOptions

    opts = options or SolverOptions()
    drives, ent = resolve_drives(device, gate_drives, entangling_drives)
    crystal = device.crystal
    n = crystal.n_ions
    notes: list[str] = []
    hint = next(
        ((int(d.beams[0]), int(d.beams[1])) for d in ent.values() if d.kind == "raman" and len(d.beams) == 2),
        None,
    )
    occupations = (
        dict(nbar)
        if nbar is not None
        else preparation_occupations(device, recipe or recipe_of(device, raman_pair=hint))
    )
    if nbar is None and device.preparation is None and recipe is None:
        notes.append("preparation recipe inferred by prep.recipe.standard_recipe (the device carries none)")
    b_gauss = device.field.B_gauss
    # qubit frequencies, Rabi frequencies, Stark shifts and crosstalk from the derived values (seed)
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
            rabi[key] = _seed(dd.carrier_rabi_hz, "conv.two_photon_rabi", "derived_raman_drive", t0_s)
            stark[key] = _seed(dd.stark_shift_hz, "conv.two_photon_rabi", "derived_raman_drive", t0_s)
            for j, eps in crosstalk_ratios(device, i, spec.beams, kind="raman").items():
                if abs(eps) >= CROSSTALK_MIN:
                    crosstalk[(i, j)] = _seed(abs(eps), "conv.crosstalk_ratio", "derived_beam_profile", t0_s)
        elif spec.kind in ("optical_E1", "optical_E2"):
            dd = derive_optical_drive(device, i, spec.beams[0], scattering=False)
            rabi[key] = _seed(dd.carrier_rabi_hz, "conv.rabi_frequency", "derived_optical_drive", t0_s)
            stark[key] = _seed(dd.stark_shift_hz, "conv.rabi_frequency", "derived_optical_drive", t0_s)
            for j, eps in crosstalk_ratios(device, i, spec.beams, kind=spec.kind).items():
                if abs(eps) >= CROSSTALK_MIN:
                    crosstalk[(i, j)] = _seed(abs(eps), "conv.crosstalk_ratio", "derived_beam_profile", t0_s)
        elif rabi_hz is not None and key in rabi_hz:
            rabi[key] = _seed(rabi_hz[key], "conv.rabi_frequency", "supplied_rabi_frequency", t0_s)
        else:
            notes.append(
                f"ion {i}: a {spec.kind} drive has no derivable carrier Rabi frequency; entry left absent"
            )
    # the entangling drives' Rabi and Stark seeds (the scheduler compensates the gate's light shift), for ions that have one
    for i in sorted(ent):
        spec = ent[i]
        key = (i, spec.table_key_beam)
        if key in rabi or spec.kind != "raman" or len(spec.beams) != 2:
            continue
        dd = derive_raman_drive(device, i, (spec.beams[0], spec.beams[1]), scattering=False)
        rabi[key] = _seed(dd.carrier_rabi_hz, "conv.two_photon_rabi", "derived_raman_drive", t0_s)
        stark[key] = _seed(dd.stark_shift_hz, "conv.two_photon_rabi", "derived_raman_drive", t0_s)
    # modes, occupations, heating (seed)
    modes_entries = {
        m: _seed(mode.omega_hz, "conv.mode_index", "derived_crystal_modes", t0_s)
        for m, mode in enumerate(crystal.modes)
    }
    nbar_entries = {
        m: _seed(v, "conv.preparation_stage_order", "preparation_model", t0_s) for m, v in occupations.items()
    }
    heating: dict[int, CalEntry] = {}
    spec_e = device.noise.S_E
    # the rate the noise model heats the run at (white level and correlation length included), as Device.derived() reports
    model_rates: dict[int, float] = {}
    if not spec_e.is_zero() and device.noise.correlation_length_m is not None:
        model_rates = device.noise.heating_rates_quanta_per_s(device)
    for m, mode in enumerate(crystal.modes):
        w = mode.omega_rad_s
        if m in model_rates:
            rate = float(model_rates[m])
        else:
            # no correlation length declared: the two-sided spectrum at the mode frequency, Device.derived()'s fallback too
            s_two = float(np.interp(w, spec_e.omega_rad_s, spec_e.S, left=0.0, right=0.0))
            ion = int(np.argmax(np.abs(mode.eigenvector)))
            rate = heating_rate_quanta_per_s(
                float(single_sided_from_two_sided(s_two)), float(crystal.masses_kg[ion]), w
            )
        heating[m] = _seed(rate, "conv.electric_field_noise", "derived_heating_rate", t0_s)
    field_entry = _seed(b_gauss, "conv.curvature_naming", "field_value", t0_s)
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
        field=field_entry,
        micromotion={},
        detection={},
        heating=heating,
        fitted_at_s=float(t0_s),
    )
    # entangling waveforms per pair: closed-form solution, mode classes, exact spot check on the resolved space
    ms: dict[tuple[int, int], Waveform] = {}
    runs: dict[tuple[int, int], CalibrationRun] = {}
    classes_by_pair: dict[tuple[int, int], dict[int, str]] = {}
    frozen_chi: dict[tuple[int, int], dict[int, float]] = {}
    wanted = list(pairs) if pairs is not None else [(a, b) for a in range(n) for b in range(a + 1, n)]
    for pair in wanted:
        a, b = int(pair[0]), int(pair[1])
        if a not in ent:
            notes.append(f"pair {(a, b)}: ion {a} has no entangling drive; no waveform is solved for it")
            continue
        spec_a = ent[a]
        if spec_a.kind != "raman" or len(spec_a.beams) != 2:
            notes.append(f"pair {(a, b)}: the surrogate solves Raman (bichromatic) waveforms only; skipped")
            continue
        beams = (spec_a.beams[0], spec_a.beams[1])
        try:
            shaped, modes = surrogate_waveform(
                device, (a, b), beams, nbar=occupations, duration_s=ms_duration_s, mu_hz=ms_mu_hz
            )
        except ClosureError as exc:
            notes.append(f"pair {(a, b)}: no closed-form waveform ({exc}); skipped")
            continue
        contributions = waveform_contributions(shaped.waveform, modes, (a, b))
        space, classes = spot_check_space(
            modes,
            contributions,
            n,
            len(crystal.modes),
            freeze_alpha_max=opts.freeze_alpha_max,
            freeze_chi_max_rad=opts.freeze_chi_max_rad,
            caps=caps,
            tail=opts.boundary_population_max,
        )
        classes_by_pair[(a, b)] = classes
        frozen_chi[(a, b)] = {
            m: float(shaped.waveform.chi_m.get(m, 0.0)) for m, cls in classes.items() if cls == "frozen"
        }
        inside, dim, nnz = within_budget(space, opts)
        if spot_check and not inside:
            # the pair's GATE_LOCAL space: the two ions and the resolved modes, the rest of the chain absent
            space, _classes_local = spot_check_space(
                modes,
                contributions,
                n,
                len(crystal.modes),
                freeze_alpha_max=opts.freeze_alpha_max,
                freeze_chi_max_rad=opts.freeze_chi_max_rad,
                caps=caps,
                ions=(a, b),
                tail=opts.boundary_population_max,
            )
            inside_local, dim_local, nnz_local = within_budget(space, opts)
            notes.append(
                f"pair {(a, b)}: the joint spot-check space (dimension {dim}, {nnz} drive non-zeros) exceeds the JOINT_EXACT "
                f"guards ({opts.joint_dimension_max}, {opts.nnz_max}); the exact check runs on the pair's GATE_LOCAL space "
                f"(dims {space.dims}, dimension {dim_local}, {nnz_local} non-zeros; Section 5.4)"
            )
            inside = inside_local
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
                chi_target_rad=CHI_MAXIMAL_RAD,
                reference="n0",
                tolerance_rad=tolerance_rad,
                options=opts,
                builder_options=builder_options,
            )
            runs[(a, b)] = run
            ms[(a, b)] = run.waveform
            notes.append(
                f"pair {(a, b)}: {shaped.method} waveform, resolved modes {tuple(t.mode for t in space.resolved)} "
                f"(dims {space.dims}), surrogate error {run.surrogate_error:.4f}, converged {run.converged}"
            )
        else:
            ms[(a, b)] = shaped.waveform
            notes.append(
                f"pair {(a, b)}: {shaped.method} waveform stored as the closed-form seed (no spot check)"
            )
    # detection: the threshold and window on ion 0's simulated readout model (one threshold for the register)
    det_cal: DetectionCalibration | None = None
    detection: dict[str, CalEntry] = {}
    try:
        det_idx = detection_beams(device, 0)
    except ValueError as exc:
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
        record_model = RecordModel.from_rates(rates, device.detector)
        windows = (
            tuple(float(x) for x in detection_windows_s)
            if detection_windows_s is not None
            else tuple(float(x) for x in np.geomspace(0.25, 2.5, 12) * device.detector.window_s)
        )
        det_cal = calibrate_detection(
            record_model,
            scheme,
            windows_s=windows,
            n_records=int(detection_records),
            rng=np.random.default_rng(int(seed)),
            fitted_at_s=t0_s,
            sample_id=0,
        )
        detection = dict(det_cal.entries)
    table = CalibrationTable(
        device_hash=base.device_hash,
        seed=base.seed,
        surrogate=True,
        qubit_freq=qubit_freq,
        rabi=rabi,
        stark=stark,
        crosstalk=crosstalk,
        modes=modes_entries,
        nbar=nbar_entries,
        ms=ms,
        field=field_entry,
        micromotion={},
        detection=detection,
        heating=heating,
        fitted_at_s=float(t0_s),
    )
    return SurrogateReport(
        table=table,
        entangling=runs,
        mode_classes=classes_by_pair,
        frozen_chi_rad=frozen_chi,
        detection=det_cal,
        nbar=occupations,
        notes=tuple(notes),
    )
