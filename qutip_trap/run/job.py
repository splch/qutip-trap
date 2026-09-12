"""``run`` and ``prepare``: the orchestration of Section 3.4 (PLAN.md Sections 3.4, 5.3, 5.7, 7.2, 7.3, 8.6; Appendix E; M6).

    Device + Circuit
      -> compile: Circuit -> native gates (phase-tracked)                    [control.compiler]
      -> calibrate (surrogate, cached per Device): CalibrationTable         [calibration.surrogate]
      -> schedule: native gates -> Pulses with absolute times, the measure event   [control.schedule]
      -> space: resolved / frozen / dropped modes with their caps           [run.space]
      -> prepare: Doppler -> sideband -> pump, the initial mixed state      [prep.recipe, prep.sequence]
      -> evolve: every branch of the initial mixture through the pulses     [dynamics.engine]
         (JOINT_EXACT), or the gate-local walk of Section 5.4               [run.gate_local]
      -> readout: the joint outcome, then the photon records or the POVM    [readout.discriminate]
      -> Result

The initial state of Section 5.7, rho(0) = (x)_i rho_int,i (x) (x)_m rho_th,m, is a mixture that is diagonal in the
computational and Fock bases: its branches (one internal level per ion, one Fock state per carried mode with eta != 0) are
evolved as PURE states with ``sesolve`` and recombined with their weights, the Fock-sum path of Section 5.3, exact up to
the branches below ``SolverOptions.branch_weight_min`` whose total weight is reported (``mesolve`` on the joint space is a
reference path for dimensions below about 100 only, and the trajectory path of M7/M9b carries the Lindblad channels). The
frozen spectators' Fock states are part of the same enumeration (Wineland's shot-to-shot Debye-Waller statistics, Section
5.2, as a weighted sum rather than a per-shot draw). With collapse operators present (the device's noise model, M7) every
branch runs through ``mesolve`` up to ``SolverOptions.mesolve_dimension_max`` and through ``SolverOptions.ntraj`` keyed
quantum-jump trajectories above it (Section 5.3).

Shots are distributed over dynamical samples in CONTIGUOUS BLOCKS, not round-robin (Section 3.4 says round-robin; the
deviation is recorded as ``conv.shot_blocks_per_sample``, because one quasi-static draw per contiguous block of the shot clock
is what the hardware does: sample k owns shots [sum M_j, sum M_j + M_k) and is drawn at t0 + (sum_{j<k} M_j) T_rep, so its
Drift values belong to the times its own shots were taken; round-robin would give every sample the same mean time and
destroy the between-sample spread the Ornstein-Uhlenbeck chain over the sample times produces). Default samples =
min(shots, 64) when the noise model has quasi-static or sampled content, 1 when it is quiet; every Drift is an
Ornstein-Uhlenbeck chain over the sample times (Section 7.5), the shots of one sample are drawn from that sample's register
state, and the error bars use the effective sample size Var(p_hat) = Var_total(p_s)/S, since shots of one sample are not
independent draws from the ensemble. The (sample, branch) engine runs of a JOINT_EXACT run are independent
and are spread over the workers of ``SolverOptions.workers`` through QuTiP's map (Section 11.3 item 9; M9b), every branch
starting from the selected space (a cap the truncation monitor grows on one branch is reported, never carried into another, so
the result is the same whatever the worker count); a worker runs its own trajectories in-process. Background-gas collisions
(Section 6.7) are Poisson events per shot:
a heating kick during the cooling stages is heralded and kept (the crystal is recooled), any other event is heralded and the
shot discarded, a reorder permutes the persistent ion order, a loss or dark-ion event flags the ion so that every later shot
reads it dark; the remaining ions' dynamics stay on the nominal crystal (an approximation the notes record). The seeds are
keyed by (sample, trajectory, shot, ion, channel) exactly as Section 3.4 requires.
"""

from __future__ import annotations

import itertools
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import qutip as qt

from qutip_trap.control.compiler import CompileReport
from qutip_trap.control.schedule import CrosstalkSuppression, GateDrive, Schedule
from qutip_trap.dynamics.engine import EngineReport, MotionalModel, SeedSpec, SolverOptions, State, Traces
from qutip_trap.dynamics.evolve import ConvergenceReport, convergence_check
from qutip_trap.dynamics.parallel import map_tasks, worker_count
from qutip_trap.hilbert.operators import thermal_populations
from qutip_trap.hilbert.space import HilbertSpace
from qutip_trap.hilbert.truncation import warn_if_boundary_exceeds
from qutip_trap.noise.collisions import (
    collision_rate_per_ion,
    sample_collisions,
    sample_kick_quanta,
    sample_reorder,
)
from qutip_trap.noise.levels import InternalLevels, internal_levels
from qutip_trap.noise.sampling import KEY_BRANCH_WEIGHT, NoiseSample, key_frozen_n, quiet_sample
from qutip_trap.noise.scattering import scattering_estimates
from qutip_trap.prep.recipe import PreparationRun, recipe_of, run_preparation
from qutip_trap.prep.sequence import prepare_state
from qutip_trap.readout.detection import RecordModel, count_anomaly_band
from qutip_trap.readout.discriminate import (
    POVM,
    Discriminator,
    ReadoutOutcome,
    ThresholdDiscriminator,
    measure,
    povm_for,
    product_povm,
)
from qutip_trap.readout.fluorescence import FluorescenceRates, ReadoutScheme, detection_rates_for_ion
from qutip_trap.run.gate_local import EngineSetup, GateLocalReport, evolve_gate_local
from qutip_trap.run.levels import FidelityLevel, decide_level, within_budget
from qutip_trap.run.pipeline import compile_calibrate_schedule
from qutip_trap.run.results import Diagnostics, Progress, Result, RunState, aggregate, binomial_error_bars
from qutip_trap.run.space import SpaceSelection, select_space
from qutip_trap.units import TWO_PI
from qutip_trap.validation.two_qubit_closed_forms import ballance_thermal_error

if TYPE_CHECKING:
    from qutip_trap.control.compiler import Circuit
    from qutip_trap.control.shaping import GateModes
    from qutip_trap.control.table import CalibrationTable, Waveform
    from qutip_trap.device.model import Device
    from qutip_trap.dynamics.channels import CollapseOp
    from qutip_trap.dynamics.hamiltonian import BuilderOptions
    from qutip_trap.light.beams import Beam

ReadoutMode = Literal["fast", "full"]


class RunError(RuntimeError):
    """The run cannot proceed as asked (a level the release does not implement, a missing calibration entry)."""


# ---- preparation ----------------------------------------------------------------------------------------------------------


def prepare(
    device: Device,
    space: HilbertSpace,
    table: CalibrationTable | None,
    sample: NoiseSample | None = None,
    seeds: SeedSpec | None = None,
    *,
    preparation: PreparationRun | None = None,
    levels: Mapping[int, InternalLevels] | None = None,
) -> State:
    """Doppler -> sideband/EIT -> optical pump, in that order (Section 4.2.6): the Appendix E ``State`` on ``space`` with
    thermal resolved modes at the recipe's occupations (the pumps' recoil included), the frozen modes' nbar, and every ion's
    pumped internal state with its preparation error. The recipe is the device's or the standard one; ``table`` and
    ``sample`` are accepted for the Appendix E signature (the M6 preparation reads the device's physics only); ``levels``
    are the register level maps of ions with d > 2 (M7), else derived from the space."""
    run_prep = preparation if preparation is not None else run_preparation(device, recipe_of(device))
    species = device.crystal.species[0]
    lv = levels if levels is not None else level_maps(device, space)
    return prepare_state(
        space,
        run_prep.sequence,
        qubit_labels=species.qubit,
        extra_nbar=run_prep.pump_heating,
        levels={i: m.labels for i, m in lv.items()} if lv else None,
    )


def level_maps(device: Device, space: HilbertSpace) -> dict[int, InternalLevels]:
    """The register level maps of every ion whose factor has d > 2 (``noise/levels.py``)."""
    out: dict[int, InternalLevels] = {}
    for i, d in enumerate(space.ion_dims):
        if d > 2:
            out[i] = internal_levels(
                device.crystal.species[i],
                int(d),
                device.field.B_gauss,
                (
                    float(device.field.direction[0]),
                    float(device.field.direction[1]),
                    float(device.field.direction[2]),
                ),
            )
    return out


def _raman_pair_hint(drives: Mapping[int, GateDrive]) -> tuple[int, int] | None:
    """The Raman pair of the entangling drives (the global beams), the natural cooling pair of the standard recipe."""
    for spec in drives.values():
        if spec.kind == "raman" and len(spec.beams) == 2:
            return (int(spec.beams[0]), int(spec.beams[1]))
    return None


# ---- the initial mixture as branches --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Branch:
    weight: float
    levels: tuple[int, ...]
    """Internal level per ion."""
    fock: dict[int, int]
    """Fock state per resolved mode and per frozen mode with eta != 0."""


def _truncated_weights(
    probabilities: Sequence[float], weight_min: float, *, what: str = "factor"
) -> list[tuple[int, float]]:
    """The states of one factor whose probability reaches ``weight_min``, with their probabilities.

    An empty result is refused rather than replaced by the most likely state at weight 1: renormalizing a mixture that was
    truncated to nothing would report a pure state and a dropped weight of zero for a distribution the cutoff never
    resolved (M6 fix; the old fallback was ``[(argmax, 1.0)]``)."""
    out = [(k, float(p)) for k, p in enumerate(probabilities) if p >= weight_min]
    if not out:
        best = float(np.max(probabilities)) if len(probabilities) else 0.0
        raise RunError(
            f"SolverOptions.branch_weight_min = {weight_min:g} keeps no state of the {what}: its most likely state has "
            f"probability {best:.3e}. Lower branch_weight_min below that (the Fock sum of Section 5.3 needs at least one "
            "branch), or cool the mode further"
        )
    return out


def enumerate_branches(
    internal_probabilities: Sequence[Sequence[float]],
    mode_nbar: Mapping[int, float],
    weight_min: float,
) -> tuple[list[Branch], float]:
    """Branches of the diagonal initial mixture with weight >= ``weight_min`` (descending), and the dropped weight."""
    ion_options = [
        _truncated_weights(p, weight_min, what=f"internal state of ion {i}")
        for i, p in enumerate(internal_probabilities)
    ]
    mode_options: dict[int, list[tuple[int, float]]] = {}
    for m, nb in mode_nbar.items():
        if nb <= 0.0:
            mode_options[m] = [(0, 1.0)]
            continue
        d = int(math.ceil(math.log(weight_min) / math.log(nb / (1.0 + nb)))) + 2
        mode_options[m] = _truncated_weights(
            [float(x) for x in thermal_populations(nb, max(d, 2))],
            weight_min,
            what=f"thermal state of mode {m} (nbar = {nb:.3g})",
        )
    modes = sorted(mode_options)
    branches: list[Branch] = []
    for ion_choice in itertools.product(*ion_options):
        w_int = float(np.prod([p for _k, p in ion_choice])) if ion_choice else 1.0
        if w_int < weight_min:
            continue
        for mode_choice in itertools.product(*[mode_options[m] for m in modes]):
            w = w_int * float(np.prod([p for _k, p in mode_choice])) if mode_choice else w_int
            if w < weight_min:
                continue
            branches.append(
                Branch(
                    w,
                    tuple(k for k, _p in ion_choice),
                    {m: k for m, (k, _p) in zip(modes, mode_choice)},
                )
            )
    branches.sort(key=lambda b: -b.weight)
    if not branches:
        # every factor kept at least one state, but no PRODUCT of them reaches the cutoff: refuse rather than evolve nothing
        # (the run would otherwise reach the readout with an empty state and fail there with an opaque message; M6 fix)
        best = 1.0
        for options in [*ion_options, *[mode_options[m] for m in modes]]:
            best *= max(p for _k, p in options)
        raise RunError(
            f"SolverOptions.branch_weight_min = {weight_min:g} keeps no branch of the initial mixture: the most likely "
            f"branch of {len(ion_options)} ion factors and {len(modes)} carried modes has weight {best:.3e}. Lower "
            "branch_weight_min below that, cool the modes further, or carry fewer modes (Section 5.3)"
        )
    total = sum(b.weight for b in branches)
    return branches, float(max(1.0 - total, 0.0))


def internal_probabilities(state: State, space: HilbertSpace) -> list[list[float]]:
    """Per ion, the diagonal of its reduced internal state (the pumped populations)."""
    out: list[list[float]] = []
    for i in range(space.n_ions):
        rho = space.marginal(state, (i,))
        out.append([float(np.real(x)) for x in np.diag(np.asarray(rho.full()))])
    return out


# ---- readout stage --------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ReadoutStage:
    rates: tuple[FluorescenceRates, ...]
    schemes: tuple[ReadoutScheme, ...]
    models: tuple[RecordModel, ...]
    discriminator: Discriminator
    povm: POVM | None
    """The form the fast path samples; None when ``readout="full"`` was asked for and no POVM was needed (a Monte-Carlo
    POVM over a time-resolved discriminator costs 2 x n_samples records per ion, Section 8.4)."""
    product: POVM | None
    """The zero-crosstalk product form: the per-ion (eps_B, eps_D) report reads it even when the register form samples."""
    leakage: dict[int, float]
    depumping: dict[int, tuple[float, float]] = field(default_factory=dict)
    """Neighbour distance -> (Delta R_d, Delta R_b) the leaked resonant light of one bright neighbour drives: the
    depumping half of Wineland's crosstalk mechanism (Section 8.5), bounded by I_ion/I_sat = 3 lambda^2/(8 pi^2 x^2)."""
    crosstalk_discrepancy: float | None = None
    """max |register confusion - product POVM| at the configured crosstalk: the "bounded and REPORTED" amount of Sections
    9.5 and 9.17; None at zero crosstalk or when the POVM was not built."""
    micromotion: dict[int, tuple[float, float]] = field(default_factory=dict)
    """Per ion, (beta, Omega_rf) of the detection beam when Section 8.8's J_0^2/J_1^2 factor was applied."""


def readout_stage(
    device: Device,
    table: CalibrationTable,
    *,
    discriminator: Discriminator | None = None,
    levels: Mapping[int, InternalLevels] | None = None,
    povm_samples: int = 20_000,
    seed: int = 0,
    need_povm: bool = True,
) -> ReadoutStage:
    """The per-ion rate objects of the device's detection beams, the table's threshold and window, and the POVM; ``levels``
    extends each scheme's classes to the leakage levels of a d > 2 register factor (M7).

    A threshold discriminator's POVM is exact through the count distributions; any other strategy of Section 8.3
    (time-resolved ML, adaptive, first-photon, camera) has no closed form, so its confusion is estimated from
    ``povm_samples`` sampled records per level and per ion, seeded from ``seed``, and the POVM carries the statistical
    uncertainty it earned. ``need_povm=False`` skips the construction entirely, which is what ``readout="full"`` wants:
    the full record path replaces the POVM rather than preceding it (Section 5.7).
    """
    from qutip_trap.light.roles import detection_beams
    from qutip_trap.readout.discriminate import max_confusion_discrepancy
    from qutip_trap.readout.fluorescence import neighbour_pumping_rates

    rates: list[FluorescenceRates] = []
    schemes: list[ReadoutScheme] = []
    models: list[RecordModel] = []
    micromotion: dict[int, tuple[float, float]] = {}
    for i in range(device.crystal.n_ions):
        keys = detection_beams(device, i)
        beams = [device.beams[k] for k in keys]
        beta, omega_rf = detection_micromotion(device, i, [device.beams[k] for k in keys])
        r, scheme, model = detection_rates_for_ion(
            device.crystal.species[i],
            device.field.B_gauss,
            device.field.direction,
            beams,
            position_m=tuple(float(x) for x in device.crystal.positions_m[i]),
            micromotion_beta=beta,
            omega_rf_rad_s=omega_rf,
        )
        if beta > 0.0:
            micromotion[i] = (beta, omega_rf)
        if levels is not None and i in levels:
            ground, _ = model.resonant_manifold()
            scheme = ReadoutScheme.for_species(device.crystal.species[i], ground, labels=levels[i].labels)
        rates.append(r)
        schemes.append(scheme)
        models.append(RecordModel.from_rates(r, device.detector))
    if discriminator is None:
        thr = table.detection.get("threshold")
        win = table.detection.get("window_s")
        if thr is None or win is None or thr.status == "uncalibrated" or win.status == "uncalibrated":
            raise RunError(
                "the calibration table has no detection threshold and window (Section 7.5 item 5): run calibrate() or pass a "
                "discriminator"
            )
        discriminator = ThresholdDiscriminator(float(thr.value), float(win.value))
    # the time-resolved discriminators of Section 8.3 carry the dark class whose 1/tau is Myerson's decay term; their
    # default "dark" silently drops it on a shelving device (the shelf decays at 1/tau_D, not at R_b), so the scheme
    # states it (audit 2026-09-07 B9)
    dark_classes = {s.dark_class for s in schemes}
    if hasattr(discriminator, "dark_class") and len(dark_classes) == 1:
        wanted = next(iter(dark_classes))
        if discriminator.dark_class != wanted:
            # ignore: Discriminator is a Protocol, so `replace` cannot see that this instance is a dataclass
            discriminator = replace(discriminator, dark_class=wanted)  # type: ignore[type-var]
    leakage = {int(k): float(v) for k, v in device.detector.psf_leakage.items() if v > 0.0}
    # the depumping half of Wineland's crosstalk (Section 8.5): a bright neighbour's leaked resonant light pumps its
    # neighbours between the manifolds at the bound I_ion/I_sat = 3 lambda^2/(8 pi^2 x^2), which is a pure function of the
    # cycling wavelength and the ion spacing; taken at the CLOSEST pair of each distance, the worst case
    depumping: dict[int, tuple[float, float]] = {}
    pos = np.asarray(device.crystal.positions_m, dtype=float)
    for d in sorted(leakage):
        gaps = [float(np.linalg.norm(pos[i + d] - pos[i])) for i in range(len(pos) - d)]
        if not gaps or min(gaps) <= 0.0:
            continue
        i0 = int(np.argmin(gaps))
        keys = detection_beams(device, i0)
        cycling = min((device.beams[k] for k in keys), key=lambda b: b.wavelength_m)
        sp = device.crystal.species[i0]
        s_beam = cycling.intensity_at(pos[i0]) / sp.transition(sp.cycling).i_sat_w_m2
        if s_beam <= 0.0:
            continue
        d_d, d_b = neighbour_pumping_rates(rates[i0], s_beam, cycling.wavelength_m, min(gaps))
        if d_d > 0.0 or d_b > 0.0:
            depumping[d] = (d_d, d_b)
    if not need_povm:
        return ReadoutStage(
            tuple(rates),
            tuple(schemes),
            tuple(models),
            discriminator,
            None,
            None,
            leakage,
            depumping,
            None,
            micromotion,
        )
    exact = isinstance(discriminator, ThresholdDiscriminator)
    n_samples = 0 if exact else int(povm_samples)
    if not exact and n_samples <= 0:
        raise RunError(
            f"the {type(discriminator).__name__} discriminator has no closed-form confusion (Section 8.4): "
            "povm_samples must be positive, or run with readout='full'"
        )
    rng = None if exact else np.random.default_rng(SeedSpec(int(seed)).child(0, 0, 0, 0, "povm_estimate"))
    product = product_povm(models, schemes, discriminator, n_samples=n_samples, rng=rng)
    povm = (
        povm_for(models, schemes, discriminator, leakage, depumping=depumping, n_samples=n_samples, rng=rng)
        if leakage
        else product
    )
    discrepancy = max_confusion_discrepancy(povm, product) if leakage else None
    return ReadoutStage(
        tuple(rates),
        tuple(schemes),
        tuple(models),
        discriminator,
        povm,
        product,
        leakage,
        depumping,
        discrepancy,
        micromotion,
    )


def detection_micromotion(device: Device, ion: int, beams: Sequence[Beam]) -> tuple[float, float]:
    """(beta, Omega_rf) of the detection light on one ion: Section 8.8's "R_o carries J_0(beta)^2 and a first-sideband
    channel J_1(beta)^2, like every other drive", with beta = k . u_1 for the SINGLE-photon wavevector of the shortest
    detection beam (the cycling line, whose photons are the ones counted) and Omega_rf from the trap's rf drive.

    (0, 0) without an rf record or without a detection beam, which is what Section 4.1.1's beta = 0 means: the factor
    is then the identity and no extra Bloch solve is paid for. A trap that DOES carry an rf drive but whose
    ``Trap.micromotion_beta`` cannot be evaluated (an rf phase imbalance with no rod geometry factors R_m and alpha, a
    rod record without its single endcap voltage, a point outside the Mathieu stability region) raises instead of
    reporting beta = 0, the same rule as ``dynamics.hamiltonian.micromotion_index``: a missing geometry factor hidden
    behind beta = 0 is a default-value fallback that silently switches the detection sideband off (M1 audit).
    """
    if device.trap.rf is None or not beams:
        return 0.0, 0.0
    cycling = min(beams, key=lambda b: b.wavelength_m)
    beta = float(
        device.trap.micromotion_beta(device.crystal.species[ion], cycling.k_vector()).as_peak().total
    )
    if beta <= 0.0:
        return 0.0, 0.0
    return beta, float(device.trap.rf.omega_rad_s)


# ---- the intrinsic budget of Section 9.6 -----------------------------------------------------------------------------------


def sideband_lamb_dicke_deficit(modes: GateModes, selection: SpaceSelection) -> float:
    """The sideband matrix element's Lamb-Dicke deficit (Section 4.3.1): how far the exact spin-dependent force falls short
    of its linear-in-eta value at the Fock states the gate actually populates. Reported as its own budget key,
    ``<gate>.sideband_lamb_dicke_deficit``, and NOT summed into the total: its mean over the thermal state is exactly the
    force rescaling the s^2 calibration of Section 4.4.7 (7) absorbs, and its thermal spread is the Debye-Waller term
    (``ballance_thermal_error``) the total already carries, so summing it would double count. It is NOT the "Bessel force
    saturation" Section 9.6 names, which is Roos's carrier saturation (J_0 + J_2)(4 Omega/mu) of ``roos_bessel_saturation``
    below.

    The first blue sideband element is Omega_{n+1,n}/Omega = |<n+1|D(i eta)|n>| = e^{-eta^2/2} (eta/sqrt(n+1)) |L^{(1)}_n(eta^2)|
    (Wineland et al. 1998, NIST J. Res. 103, 259, Eq. 18 with the phase of Eq. 21; Section 4.3.1), whose leading term is
    eta sqrt(n + 1). The relative deficit

        f = 1 - Omega_{n+1,n}/(Omega eta sqrt(n + 1))

    is the fractional loss of force amplitude, taken over the gate's ions and the resolved and frozen modes at the highest
    Fock index each carries (``ModeTruncation.expected_n_range``, else the thermal mean), which is the worst case: f grows
    with n and with eta. It is reported and summed into the budget as a scale, like the off-resonant carrier scale. The
    calibration absorbs the eta- and mode-INDEPENDENT part of f (the s^2 rescaling of Section 4.4.7 (7) is fitted to the
    measured angle), so the number below is an upper bound on what survives calibration, not an estimate of it."""
    from qutip_trap.hilbert.operators import rabi_matrix_element

    worst = 0.0
    populated = {t.mode: t.expected_n_range[1] for t in selection.space.resolved}
    for k, m in enumerate(modes.modes):
        if selection.mode_class.get(m) not in ("resolved", "frozen"):
            continue
        eta = max(abs(modes.eta[i][k]) for i in modes.ions)
        if eta == 0.0:
            continue
        n = int(populated.get(m, int(round(modes.nbar[k]))))
        exact = rabi_matrix_element(n + 1, n, eta)
        linear = eta * math.sqrt(n + 1.0)
        if linear <= 0.0:
            continue
        worst = max(worst, abs(1.0 - exact / linear))
    return float(worst)


def roos_bessel_saturation(waveform: Waveform) -> float:
    """Section 9.6's "Bessel force saturation" term as an infidelity scale: the carrier's strong-drive correction scales the
    spin-dependent force by (J_0 + J_2)(4 Omega/mu) (Roos 2008, New J. Phys. 10, 013002, Eq. 17; Section 4.4.1 "the force
    is saturated by (J_0 + J_2)" at argument 4 Omega/delta), so an uncalibrated pulse solved for chi = pi/4 with the linear
    force reaches chi (1 - f)^2 with f = 1 - (J_0 + J_2), an angle error pi f/2 and a Bell-state infidelity

        1 - F = sin^2(pi f / 2)

    (MS(chi + d) against MS(chi) on |00> overlaps as cos d). It is evaluated at the played waveform's worst point: the
    largest tone amplitude Omega and the smallest tone-to-carrier detuning mu over its segments (an FM segment's detuning is
    sampled along the segment). Only the bichromatic (MS) family has a carrier coupling to saturate; a light-shift or
    gradient waveform has none and contributes zero. Like the other scales it is an upper bound on what survives
    calibration: the s^2 rescaling of Section 4.4.7 (7) absorbs a uniform force deficit entirely."""
    from qutip_trap.validation.two_qubit_closed_forms import roos_force_saturation

    if waveform.kind != "ms" or waveform.segments is None:
        return 0.0
    peak = 0.0
    mu_min = math.inf
    for seg in waveform.segments:
        grid = np.linspace(0.0, seg.duration_s, 101)
        for amp in seg.amplitude_hz.values():
            val = float(np.max(np.abs([amp(x) for x in grid]))) if callable(amp) else abs(float(amp))
            peak = max(peak, val)
        for det in seg.detuning_hz.values():
            vals = [abs(float(det(x))) for x in grid] if callable(det) else [abs(float(det))]
            mu_min = min(mu_min, min(vals))
    if peak == 0.0:
        return 0.0
    if not math.isfinite(mu_min) or mu_min == 0.0:
        raise RunError(
            "an MS waveform carries a tone on the carrier: Roos's saturation (J_0 + J_2)(4 Omega/mu) needs mu != 0"
        )
    f = 1.0 - roos_force_saturation(TWO_PI * peak, TWO_PI * mu_min)
    return math.sin(math.pi * f / 2.0) ** 2


def intrinsic_budget(device: Device, sched: Schedule, selection: SpaceSelection) -> dict[str, float]:
    """The closed-form error scales reported beside the result (Section 9.6): per entangling gate the residual displacement
    eps_ent = sum_{i,m} |alpha_{i,m}|^2 (2 nbar_m + 1) of the played waveform, the n = 0-referenced Debye-Waller loss of its
    modes, the off-resonant carrier scale (Omega_peak/nu_min)^2, Roos's Bessel force saturation sin^2(pi f/2) with
    f = 1 - (J_0 + J_2)(4 Omega/mu), the frozen spectators' chi, and (reported, not summed) the sideband element's
    Lamb-Dicke deficit; per single-qubit pulse the
    sideband scale eta^2 (Omega/nu)^2 of the nearest mode and the addressing crosstalk sum_j sin^2(eps_ij theta/2); and
    their sum."""
    from qutip_trap.control.shaping import waveform_integrals
    from qutip_trap.run.space import gate_modes_for

    out: dict[str, float] = {}
    total = 0.0
    for gate in sched.gates:
        modes = gate_modes_for(device, gate, selection.nbar)
        ints = waveform_integrals(gate.waveform, modes)
        eps_ent = float(ints.residual_error(modes))
        dw = 0.0
        for k, m in enumerate(modes.modes):
            if selection.mode_class.get(m) == "resolved" or selection.mode_class.get(m) == "frozen":
                eta = max(abs(modes.eta[i][k]) for i in modes.ions)
                dw += ballance_thermal_error(eta, modes.nbar[k])
        peak = 0.0
        nu_min = min(modes.omega_rad_s)
        if gate.waveform.segments is not None:
            for seg in gate.waveform.segments:
                for amp in seg.amplitude_hz.values():
                    val = (
                        float(np.max(np.abs([amp(x) for x in np.linspace(0.0, seg.duration_s, 101)])))
                        if callable(amp)
                        else abs(float(amp))
                    )
                    peak = max(peak, val)
        carrier = (2.0 * math.pi * peak / nu_min) ** 2 if nu_min > 0.0 else 0.0
        lamb_dicke = sideband_lamb_dicke_deficit(modes, selection)
        bessel = roos_bessel_saturation(gate.waveform)
        chi_frozen = sum(
            abs(v) for m, v in gate.waveform.chi_m.items() if selection.mode_class.get(m) == "frozen"
        )
        # Roos's spin-axis tilt psi = (4 Omega/mu) sin(zeta) with zeta the beat phase at the gate start (Section 4.4.1): zero
        # when the hardware resets the beat note per gate, part of the gate when the tones run continuously (M6 finding)
        tilt = 0.0
        if device.hardware.phase_continuous and gate.waveform.segments is not None:
            mu0 = gate.waveform.segments[0].detuning_hz.get("blue", 0.0)
            if not callable(mu0) and float(mu0) != 0.0:
                zeta = (2.0 * math.pi * abs(float(mu0)) * gate.t_start_s) % (2.0 * math.pi)
                mean_amp = float(
                    np.mean(
                        [
                            abs(float(a)) if not callable(a) else abs(float(a(0.0)))
                            for seg in gate.waveform.segments
                            for a in seg.amplitude_hz.values()
                        ]
                    )
                )
                psi = 4.0 * mean_amp / abs(float(mu0)) * abs(math.sin(zeta))
                tilt = math.sin(psi) ** 2
        out[f"{gate.gate_id}.residual_displacement"] = eps_ent
        out[f"{gate.gate_id}.debye_waller"] = dw
        out[f"{gate.gate_id}.carrier_scale"] = carrier
        out[f"{gate.gate_id}.bessel_saturation"] = bessel
        out[f"{gate.gate_id}.sideband_lamb_dicke_deficit"] = lamb_dicke
        out[f"{gate.gate_id}.frozen_chi_rad"] = chi_frozen
        out[f"{gate.gate_id}.beat_phase_tilt"] = tilt
        total += eps_ent + dw + carrier + bessel + tilt
    from qutip_trap.light.raman import lamb_dicke_parameters

    for pulse in sched.pulses:
        gid = pulse.gate_id or ""
        if len(pulse.drive.ions) != 1 or not gid.startswith("gpi"):
            continue
        env = pulse.drive.tones[0].envelope_hz
        omega = 2.0 * math.pi * (abs(float(env)) if not callable(env) else abs(float(env(0.0))))
        # addressing crosstalk (Section 3.3): a neighbour j sees the same resonant carrier at eps_ij Omega, a coherent
        # rotation by eps_ij theta whose infidelity is sin^2(eps_ij theta/2) (a Raman ratio is the INTENSITY ratio, M6)
        if pulse.drive.crosstalk:
            theta = omega * pulse.duration_s
            xt = sum(math.sin(abs(eps) * theta / 2.0) ** 2 for eps in pulse.drive.crosstalk.values())
            out[f"{gid}.crosstalk"] = xt
            total += xt
        dk = pulse.drive.delta_k(device.beams)
        if float(np.linalg.norm(dk)) == 0.0:
            continue
        etas, _ = lamb_dicke_parameters(device, pulse.drive.ions[0], dk)
        scale = 0.0
        for m, eta in etas.items():
            if eta != 0.0:
                scale = max(scale, (eta * omega / device.crystal.modes[m].omega_rad_s) ** 2)
        out[f"{gid}.sideband_scale"] = scale
        total += scale
    # photon scattering per pulse (Section 9.7 row 'Scattering'): the estimate reported whether or not the channels are simulated
    for pulse in sched.pulses:
        gid = pulse.gate_id or ""
        for key, val in scattering_estimates(device, pulse).items():
            out[f"{gid}.{key}"] = val
            if key.endswith((".P_raman", ".P_leak", ".rayleigh_dephasing")):
                total += val
    out["total"] = total
    return out


def effective_sample_size(bits_per_sample: Sequence[np.ndarray]) -> float:
    """The effective number of independent shots behind a histogram assembled from several dynamical samples (Section 3.4).

    With S samples of M_s shots and per-sample frequencies p_s of a bitstring, the pooled estimate is p_hat = sum_s M_s p_s /
    sum_s M_s and the one-way random-effects decomposition of its variance is Var(p_hat) = Var(p_s)/S, where Var(p_s) =
    sigma^2_between + p(1 - p)/M is the TOTAL per-sample variance: the within-sample binomial term is already inside the
    spread of the p_s and must NOT be added a second time. Var(p_hat) is estimated directly by the shot-count-weighted form
    sum_s M_s (p_s - p_hat)^2 / [(S - 1) sum_s M_s], which is var(p_s, ddof=1)/S exactly when the M_s are equal, and
    n_eff = p(1 - p)/Var(p_hat), the minimum over the bitstrings seen, capped at the shot count. One sample returns the
    shot count."""
    total = sum(int(b.shape[0]) for b in bits_per_sample)
    non_empty = [b for b in bits_per_sample if b.shape[0] > 0]
    if len(non_empty) <= 1:
        return float(total)
    keys: set[str] = set()
    for b in non_empty:
        keys |= set(aggregate(b)[1])
    n_eff = float(total)
    s_count = len(non_empty)
    for key in keys:
        ps = np.array([aggregate(b)[1].get(key, 0.0) for b in non_empty])
        ms = np.array([b.shape[0] for b in non_empty], dtype=float)
        p = float(np.average(ps, weights=ms))
        if p <= 0.0 or p >= 1.0:
            continue
        # the total per-sample variance (between + within, weighted by the shot counts), divided by S: adding the binomial
        # term on top of np.var(ps) would count it twice and shrink n_eff by up to sqrt(2) (M6 finding)
        var = float(np.sum(ms * (ps - p) ** 2) / ((s_count - 1) * np.sum(ms)))
        if var > 0.0:
            n_eff = min(n_eff, p * (1.0 - p) / var)
    return float(n_eff)


# ---- run --------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunRecord:
    """Everything a run produced besides the Result: what the application of Section 14 replays (kept on ``Result`` by id)."""

    compile: CompileReport
    schedule: Schedule
    selection: SpaceSelection
    preparation: PreparationRun
    branches: tuple[Branch, ...]
    traces: tuple[Traces, ...]
    register_state: qt.Qobj | None
    """The recombined register density matrix; None when GATE_LOCAL carried a pure-state ensemble (Section 5.4)."""
    readout: ReadoutStage
    outcome: ReadoutOutcome
    table: CalibrationTable
    qubit_shifts_hz: dict[int, float]
    notes: tuple[str, ...] = field(default_factory=tuple)
    gate_local: GateLocalReport | None = None
    """The GATE_LOCAL walk's report (M9a), None for a JOINT_EXACT run."""


_LAST_RECORD: dict[int, RunRecord] = {}


def last_record(result: Result) -> RunRecord:
    """The RunRecord behind a Result of this process (compile report, schedule, selection, traces, readout stage)."""
    return _LAST_RECORD[id(result)]


def with_fields(result: Result, **changes: Any) -> Result:
    """``dataclasses.replace`` on a Result that keeps its RunRecord reachable through ``last_record`` (the record is keyed
    by the result's identity, so a plain replace would lose it); ``Machine.run`` stamps ``machine_hash`` with it."""
    out = replace(result, **changes)
    record = _LAST_RECORD.get(id(result))
    if record is not None:
        _LAST_RECORD[id(out)] = record
    return out


def run(
    circuit: Circuit,
    device: Device,
    shots: int,
    *,
    table: CalibrationTable | None = None,
    t0_s: float = 0.0,
    shot_period_s: float | None = None,
    samples: int | None = None,
    level: Literal["JOINT_EXACT", "GATE_LOCAL", "auto"] | FidelityLevel = "auto",
    seed: int = 0,
    options: SolverOptions | None = None,
    gate_drives: Mapping[int, GateDrive] | None = None,
    entangling_drives: Mapping[int, GateDrive] | None = None,
    builder_options: BuilderOptions | None = None,
    readout: ReadoutMode = "fast",
    discriminator: Discriminator | None = None,
    povm_samples: int = 20_000,
    space: HilbertSpace | None = None,
    caps: Mapping[int, int] | None = None,
    channels: Sequence[CollapseOp] = (),
    entangler: Literal["ms", "zz"] = "ms",
    parallel: bool | None = None,
    keep_final_state: bool = False,
    calibrate_kwargs: Mapping[str, Any] | None = None,
    noise: bool = True,
    internal_levels: int = 2,
    crosstalk_suppression: CrosstalkSuppression = "none",
    stark_compensation: bool = True,
    enr_group: tuple[Sequence[int], int] | None = None,
    progress: Callable[[Progress], None] | None = None,
) -> Result:
    """Compile -> calibrate -> schedule -> prepare -> evolve -> readout -> Result (Section 3.4).

    ``table=None`` builds the surrogate table at ``t0_s`` for the pairs the circuit uses; ``shot_period_s=None`` derives
    T_rep from the preparation, the schedule and the detection window. ``readout="fast"`` applies the POVM to the joint
    outcome, ``"full"`` generates every photon record (Section 5.7, never both). ``noise=True`` (M7) draws the device's
    dynamical samples (``NoiseModel.sample_sequence``), assembles its collapse operators (heating, dephasing, and under the
    SolverOptions switches the scattering and intensity-noise operators), applies the hardware chain and the collision
    process; ``noise=False`` runs the quiet nominal sample without channels (the M6 behaviour). ``channels`` are extra
    explicit collapse operators. ``internal_levels`` > 2 gives every ion a register factor with leakage levels
    (``noise/levels.py``) so that scattering out of the qubit pair is simulated and read out by the manifold's class.
    ``crosstalk_suppression`` selects Section 6.6's echo schemes for the MS gates. ``stark_compensation`` (M8) lets the
    scheduler detune every pulse by the table's believed light shift; the pulses the ions see are the scheduler's requests
    converted through the device's derived values by the engine's played chain (``control.played``), so a table fitted by
    simulated experiments (``calibrate(surrogate=False)``) drives the machine with its own errors (Sections 7.3, 7.5). The
    table's CALIBRATED micromotion shims are programmed onto the device the run evolves (the crystal re-solved at that
    setting), so a compensation fitted at one time against a stray field that has since drifted leaves the residual excess
    micromotion a laboratory would have; the reported device hash stays the uncompensated device's. An ``uncalibrated``
    qubit-frequency entry makes the run REFUSE (``RunError``) rather than take the frame at the true transition.
    ``level`` (M9a): ``"auto"`` runs JOINT_EXACT inside the Section 11.5 guards and GATE_LOCAL above them (Section 5.4: every
    gate step on the exact joint space of its ions and resolved modes, the register carried as a density matrix or a pure-state
    ensemble, the motional model tracked between steps, the residual displacement and the channel summaries reported in
    ``Diagnostics.gate_local``); either level can be forced. ``enr_group`` = (modes, N_exc) carries a group of cold modes as
    one excitation-number-restricted factor (Section 11.3 item 1). ``parallel`` defaults to the device's
    ``HardwareChain.parallel_addressing`` (Section 7.3: single-qubit gates run in parallel only if the device model allows
    parallel addressing); entangling gates stay serialized one at a time per crystal either way. ``progress`` (0.2.0) is
    called with a ``Progress`` per integrated pulse and per (sample, branch) engine run when those run in-process, per
    dynamical sample evolved and per sample read out; an exception it raises aborts the run.
    """
    if shots <= 0:
        raise ValueError("shots must be positive")
    started = time.perf_counter()
    created_at = datetime.now(UTC).isoformat(timespec="seconds")
    notify = _Reporter(progress, started)
    # 1 to 3: compile, calibrate (the cached surrogate when no table is given), program the calibrated shims and schedule,
    # the prefix ``Machine.schedule`` shares (run/pipeline.py; docs/api_implementation_plan.md 1.3)
    prefix = compile_calibrate_schedule(
        circuit,
        device,
        table=table,
        seed=seed,
        t0_s=t0_s,
        options=options,
        gate_drives=gate_drives,
        entangling_drives=entangling_drives,
        builder_options=builder_options,
        caps=caps,
        calibrate_kwargs=calibrate_kwargs,
        entangler=entangler,
        parallel=parallel,
        crosstalk_suppression=crosstalk_suppression,
        stark_compensation=stark_compensation,
        internal_levels=internal_levels,
    )
    opts = prefix.options
    ent_drives = prefix.entangling_drives
    notes: list[str] = list(prefix.notes)
    report = prefix.report
    compiled = prefix.compiled
    table = prefix.table
    device = prefix.device
    sched = prefix.schedule
    # 4. preparation (the physics of the recipe) and the space
    cooling_pair = _raman_pair_hint(ent_drives)
    prep_run = run_preparation(device, recipe_of(device, raman_pair=cooling_pair))
    if device.preparation is None:
        notes.append("preparation recipe inferred by prep.recipe.standard_recipe (the device carries none)")
    notes.extend(prep_run.notes)
    n_ions = device.crystal.n_ions
    n_modes = len(device.crystal.modes)
    if internal_levels < 2:
        raise ValueError("internal_levels is at least 2")
    if space is None:
        selection = select_space(
            device,
            sched,
            opts,
            nbar=prep_run.nbar,
            caps=caps,
            ion_dims=[int(internal_levels)] * n_ions,
            enr=enr_group,
        )
    else:
        if enr_group is not None:
            raise ValueError("give the ENR group inside the supplied space or as enr_group, not both")
        selection = SpaceSelection(
            space,
            {m: _supplied_class(space, m) for m in range(n_modes)},
            {},
            {m: float(prep_run.nbar.get(m, 0.0)) for m in range(n_modes)},
            ("space supplied by the caller",),
            budget=within_budget(space, opts),
        )
    joint_space = selection.space
    levels = level_maps(device, joint_space)
    if levels:
        notes.append(
            "register factors with leakage levels: "
            + "; ".join(f"ion {i}: {', '.join(m.labels)}" for i, m in levels.items())
        )
    # the Section 11.5 verdict the selection already reached on its DECLARATION, before any operator was allocated
    # (M9b audit B2: HilbertSpace.check() used to build an O(D) identity, so the guard could not refuse what it had built)
    _ok, dim, nnz = selection.budget
    # PLAN.md Appendix E: level="auto" "resolves through resolve_level(device, circuit, options)". It used to be dead code
    # while run() inlined within_budget (M9a audit B6/E18); the run's actual space makes the guards exact instead of the
    # estimate resolve_level falls back to without one. The decision carries the numbers it compared, which the result
    # reports as Diagnostics.level_reason (docs/api_implementation_plan.md 1.2).
    decision = decide_level(device, compiled, opts, space=joint_space)
    ok = decision.level is FidelityLevel.JOINT_EXACT
    requested = FidelityLevel(level)
    run_level: FidelityLevel = decision.level if requested is FidelityLevel.AUTO else requested
    level_reason = (
        decision.reason
        if requested is FidelityLevel.AUTO
        else f"{run_level.value} forced by the caller; level='auto' would choose {decision.reason}"
    )
    if run_level == "JOINT_EXACT" and not ok:
        # Section 11.5: the monitor "refuses to build joint spaces above a configurable dimension"; the knobs ARE the
        # configuration, so an explicit JOINT_EXACT above them is refused rather than built. Before 2026-09-08 it went ahead
        # with a note: the ENR end-to-end test then built a 37752-dimensional space ([2, 2, 11, 13, 66]) that took 20 GB
        # and an hour before QobjEvo rejected its term (ledger conv.space_declaration_before_allocation)
        raise RunError(
            f"level='JOINT_EXACT' asks for a joint space of dimension {dim} with {nnz} drive non-zeros, above the Section 11.5 "
            f"guards (joint_dimension_max = {opts.joint_dimension_max}, nnz_max = {opts.nnz_max}); raise them in SolverOptions "
            "to build it deliberately, reduce the caps or the resolved modes, or let level='auto' route the run to GATE_LOCAL"
        )
    if run_level == "GATE_LOCAL":
        notes.append(
            f"GATE_LOCAL (Section 5.4): the joint space would have dimension {dim} and {nnz} drive non-zeros against the "
            f"guards ({opts.joint_dimension_max}, {opts.nnz_max})"
            + ("" if not ok else "; requested below the guards")
        )
        if space is not None:
            notes.append(
                "GATE_LOCAL builds its own gate-local spaces; the supplied space sets the mode classes reported"
            )
    seeds = SeedSpec(int(seed))
    # the prepared state: on the joint space for JOINT_EXACT, on the register alone for GATE_LOCAL (whose joint space is the
    # one that did not fit; the motional model starts from the recipe's occupations, Section 5.4)
    prep_space = (
        joint_space
        if run_level == "JOINT_EXACT"
        else HilbertSpace(tuple(joint_space.ion_dims), (), None, tuple(range(n_modes)))
    )
    state0 = prepare(device, prep_space, table, quiet_sample(0), seeds, preparation=prep_run, levels=levels)
    # 5. the branches of the initial mixture (JOINT_EXACT: the Fock-sum path of Section 5.3)
    probs_int = internal_probabilities(state0, prep_space)
    branches: list[Branch] = []
    dropped_weight = 0.0
    if run_level == "JOINT_EXACT":
        from qutip_trap.light.raman import lamb_dicke_parameters

        coupled_frozen: set[int] = set()
        for pulse in sched.pulses:
            dk = pulse.drive.delta_k(device.beams)
            if float(np.linalg.norm(dk)) == 0.0:
                continue
            for ion in pulse.drive.ions:
                etas, _ = lamb_dicke_parameters(device, ion, dk)
                # only FROZEN spectators are enumerated as Fock branches (their Debye-Waller factor is the physics); a
                # dropped mode is not modelled at all, so it costs no branch (Section 5.2; M6 fix)
                coupled_frozen.update(
                    m for m in joint_space.frozen if abs(etas[m]) > 1e-12 and m not in joint_space.dropped
                )
        mode_nbar = {m.mode: float(state0.motional.nbar.get(m.mode, 0.0)) for m in joint_space.resolved}
        enr_modes = list(joint_space.enr_group[0]) if joint_space.enr_group is not None else []
        mode_nbar.update({m: float(state0.motional.nbar.get(m, 0.0)) for m in enr_modes})
        mode_nbar.update({m: float(state0.motional.nbar.get(m, 0.0)) for m in sorted(coupled_frozen)})
        branches, dropped_weight = enumerate_branches(probs_int, mode_nbar, opts.branch_weight_min)
        if joint_space.enr_group is not None:
            # an ENR Fock tuple lives inside the excitation cap; branches above it are dropped and reported (Section 5.1)
            n_exc = joint_space.enr_group[1]
            kept = [b for b in branches if sum(b.fock.get(m, 0) for m in enr_modes) <= n_exc]
            over = sum(b.weight for b in branches) - sum(b.weight for b in kept)
            if over > 0.0:
                notes.append(
                    f"initial-mixture branches above the ENR cap N_exc = {n_exc} dropped: weight {over:.3e} (renormalized)"
                )
                dropped_weight += over
            branches = kept
        if dropped_weight > 0.0:
            notes.append(
                f"initial-mixture branches below branch_weight_min = {opts.branch_weight_min:g} dropped: total weight "
                f"{dropped_weight:.3e} (renormalized)"
            )
    # 6. the qubit-frequency shifts: the true transition minus the table's frame (Section 7.3; M2 hand-off)
    shifts: dict[int, float] = {}
    for i in range(n_ions):
        sp = device.crystal.species[i]
        f_true, _d1, _d2 = sp.transition_frequency_hz(sp.qubit[0], sp.qubit[1], device.field.B_gauss)
        entry = table.qubit_freq.get(i)
        if entry is not None and entry.status == "uncalibrated":
            # ``shifts`` is the ONLY channel by which the table's frequency error reaches the physics, so a zero here would
            # put the frame exactly on the true transition and make an uncalibrated qubit frequency error-free by
            # construction - the fallback Section 7.3's last clause forbids ("an entry the calibration could not establish
            # is uncalibrated and refuses to schedule rather than falling back to them").
            raise RunError(
                f"ion {i}: the calibration table's qubit frequency is uncalibrated (fitted by {entry.experiment!r}); the "
                "frame cannot be programmed and the run refuses rather than taking it at the true transition "
                "(Section 7.3). Re-calibrate the ion's Ramsey-frequency experiment or pass a table that carries it."
            )
        if entry is None:
            # a device with no qubit-frequency seed at all (never the surrogate, which always seeds one): there is no
            # believed frame to be wrong about, so the frame is the transition itself and the run says so
            shifts[i] = 0.0
            notes.append(
                f"ion {i}: the table has no qubit frequency; the frame is taken at the true transition"
            )
        else:
            shifts[i] = float(f_true - entry.value)
    # 7. timing (Section 7.5) and the dynamical samples (Section 3.4; M7)
    stage = readout_stage(
        device,
        table,
        discriminator=discriminator,
        levels=levels or None,
        povm_samples=povm_samples,
        seed=seed,
        need_povm=(readout == "fast"),
    )
    window = float(stage.discriminator.window_s)
    t_rep = (
        float(shot_period_s)
        if shot_period_s is not None
        else prep_run.duration_s + sched.pulses_end_s + window + float(device.hardware.dead_time_s)
    )
    quiet = (not noise) or device.noise.is_quiet(device)
    if quiet:
        n_samples = 1
        if samples not in (None, 1):
            notes.append(
                f"samples={samples} requested but the noise model has no quasi-static or sampled content (or noise=False): "
                "one nominal sample"
            )
    else:
        n_samples = int(samples) if samples is not None else min(int(shots), 64)
        n_samples = max(1, min(n_samples, int(shots)))
    counts_per_sample = [shots // n_samples + (1 if k < shots % n_samples else 0) for k in range(n_samples)]
    first_shots = [sum(counts_per_sample[:k]) for k in range(n_samples)]
    sample_times = [t0_s + f * t_rep for f in first_shots]
    if quiet:
        samples_seq: tuple[NoiseSample, ...] = tuple(quiet_sample(k, t) for k, t in enumerate(sample_times))
    else:
        rng_noise = np.random.default_rng(seeds.child(0, 0, 0, 0, "noise_samples"))
        samples_seq = device.noise.sample_sequence(
            rng_noise, sample_times, device=device, duration_s=sched.pulses_end_s, t0_s=t0_s
        )
    # 8. evolve every sample: every branch of the initial mixture on the joint space (JOINT_EXACT), or the gate-local walk
    setup = EngineSetup(
        builder_options=builder_options,
        channels=tuple(channels),
        qubit_shifts_hz=shifts,
        device_channels=bool(noise),
        levels_by_ion=levels or None,
        hardware_chain=True,
        table=table,
    )
    dims_int = [list(joint_space.ion_dims), list(joint_space.ion_dims)]
    d_int = int(np.prod(joint_space.ion_dims))
    traces_all: list[Traces] = []
    boundary: dict[int, float] = {m.mode: 0.0 for m in joint_space.resolved}
    margin_reached: dict[int, int] = {}
    populated_max: dict[int, int] = {}
    integrators: list[str] = []
    approximations: list[str] = []
    methods: list[str] = []
    n_traj_max = 1
    register_states: list[list[tuple[float, qt.Qobj]]] = []
    gl_report: GateLocalReport | None = None
    space_initial = joint_space
    kernel_kinds: set[str] = set()
    workers_used = 1
    propagator_hits = 0
    convergence: ConvergenceReport | None = None
    if run_level == "JOINT_EXACT":
        engine = setup.engine()
        total_weight = sum(b.weight for b in branches)
        # every (sample, branch) run is independent: prepare the initial states on the selected space, run them all through
        # the map of Section 11.3 item 9, then accumulate in order (M9b)
        payloads: list[tuple[int, int, State, NoiseSample]] = []
        for s_idx, smp in enumerate(samples_seq):
            for k, br in enumerate(branches):
                fock_res = {
                    m: n for m, n in br.fock.items() if joint_space.mode_class(m) in ("resolved", "enr")
                }
                thermal_frozen = {m: float(state0.motional.nbar.get(m, 0.0)) for m in joint_space.frozen}
                st = joint_space.initial_state(
                    list(br.levels),
                    fock=fock_res,
                    thermal={m: v for m, v in thermal_frozen.items()},
                    provenance=tuple(state0.provenance) + (f"m6.branch[{k}]",),
                )
                values = dict(smp.values)
                values.update(
                    {
                        key_frozen_n(m): float(n)
                        for m, n in br.fock.items()
                        if joint_space.mode_class(m) == "frozen"
                    }
                )
                values[KEY_BRANCH_WEIGHT] = float(br.weight / total_weight)
                sample_b = NoiseSample(
                    sample_id=smp.sample_id, values=values, ou_grids=dict(smp.ou_grids), t_s=smp.t_s
                )
                payloads.append((s_idx, k, st, sample_b))
        results, workers_used = _run_engine_tasks(
            engine, device, sched, joint_space, payloads, seeds, opts, report=notify
        )
        if opts.convergence_check:
            # Section 5.5's second bullet: repeat the evolution with atol and rtol tightened by ten and report the change in
            # the FIRST sample's register populations, which are deterministic where the sampled histogram is not (M2's
            # dynamics.evolve.convergence_check; the plumbing is M9's). Three passes, as the option's docstring says.
            def _register_populations(o: SolverOptions) -> dict[str, np.ndarray]:
                res, _w = _run_engine_tasks(engine, device, sched, joint_space, payloads, seeds, o)
                rho = np.zeros((d_int, d_int), dtype=complex)
                for (p_idx, k, _st, _sb), (tr_c, _rep_c) in zip(payloads, res):
                    if p_idx != 0:
                        continue
                    rho += (branches[k].weight / total_weight) * np.asarray(tr_c.final.internal.full())
                return {"register_populations": np.real(np.diag(rho))}

            convergence = convergence_check(_register_populations, opts)
            notes.append(convergence.summary())
        grown_space = joint_space
        for s_idx in range(len(samples_seq)):
            rho_int = np.zeros((d_int, d_int), dtype=complex)
            for (p_idx, k, _st, _sb), (tr, rep) in zip(payloads, results):
                if p_idx != s_idx:
                    continue
                br = branches[k]
                traces_all.append(tr)
                rho_int += (br.weight / total_weight) * np.asarray(tr.final.internal.full())
                for m, v in tr.boundary_population.items():
                    boundary[m] = max(boundary.get(m, 0.0), float(v))
                for seg in rep.segments:
                    if seg.integrator not in integrators:
                        integrators.append(seg.integrator)
                for a in rep.approximations:
                    if a not in approximations:
                        approximations.append(a)
                for n in rep.notes:
                    if n not in notes:
                        notes.append(n)
                if rep.method not in methods:
                    methods.append(rep.method)
                n_traj_max = max(n_traj_max, rep.trajectories)
                for m, v in rep.margin_reached.items():
                    margin_reached[m] = min(margin_reached.get(m, int(v)), int(v))
                for m, v in rep.populated_n_max.items():
                    populated_max[m] = max(populated_max.get(m, 0), int(v))
                if rep.kernel != "none":
                    kernel_kinds.add(rep.kernel)
                propagator_hits += rep.propagator_cache_hits
                if rep.space != grown_space and rep.space.dimension > grown_space.dimension:
                    # the truncation monitor grew the caps on this branch (Section 5.5): the diagnostics report the largest space
                    grown_space = rep.space
            register_states.append([(1.0, qt.Qobj(rho_int, dims=dims_int))])
            notify("sample", s_idx + 1, len(samples_seq))
        joint_space = grown_space
        dims_int = [list(joint_space.ion_dims), list(joint_space.ion_dims)]
    else:
        register_states, gl_report, _models = evolve_gate_local(
            device,
            sched,
            samples_seq,
            seeds,
            opts,
            register0=state0.internal,
            nbar0=state0.motional.nbar,
            ion_dims=joint_space.ion_dims,
            setup=setup,
            caps=caps,
            progress=(lambda done, total: notify("sample", done, total)) if notify.active else None,
        )
        for step in gl_report.steps:
            for m, v in step.boundary_population.items():
                boundary[m] = max(boundary.get(m, 0.0), float(v))
            for m, v in step.margin_reached.items():
                margin_reached[m] = min(margin_reached.get(m, int(v)), int(v))
            for integ in step.integrators:
                if integ not in integrators:
                    integrators.append(integ)
            if step.method not in methods:
                methods.append(step.method)
            n_traj_max = max(n_traj_max, step.n_traj)
            for n in step.notes:
                if n not in notes:
                    notes.append(n)
        notes.extend(n for n in gl_report.notes if n not in notes)
        # the workers the walk ACTUALLY used, not worker_count(opts): the tomography runs its inputs in-process whenever the
        # local space has no resolved mode (every carrier step) and the quasi-static samples are iterated serially, so a
        # GPi2 GATE_LOCAL run used to report N workers while nothing ran in parallel (M9b audit B10)
        workers_used = gl_report.workers
        if opts.convergence_check:

            def _gate_local_populations(o: SolverOptions) -> dict[str, np.ndarray]:
                states, _rep, _mods = evolve_gate_local(
                    device,
                    sched,
                    samples_seq[:1],
                    seeds,
                    o,
                    register0=state0.internal,
                    nbar0=state0.motional.nbar,
                    ion_dims=joint_space.ion_dims,
                    setup=setup,
                    caps=caps,
                )
                rho = sum(w * np.asarray((st if st.isoper else qt.ket2dm(st)).full()) for w, st in states[0])
                return {"register_populations": np.real(np.diag(np.asarray(rho)))}

            # no cache clearing: fingerprint_options keys the extraction cache on atol and rtol, so the tightened pass
            # misses and the base-tolerance pass of convergence_check hits what the primary walk already computed
            convergence = convergence_check(_gate_local_populations, opts)
            notes.append(convergence.summary())
        approximations.append(
            f"GATE_LOCAL: {len([s for s in gl_report.steps if s.kind == 'gate'])} gate steps and "
            f"{len([s for s in gl_report.steps if s.kind == 'idle'])} idle steps through exact gate-local spaces (largest dimension "
            f"{gl_report.largest_local_dimension}); spin-motion and mode-mode correlations traced out between steps, the residual "
            f"displacement bound {gl_report.residual_bound_total:.2e}, the frozen excitation bound {gl_report.frozen_excitation_total:.2e}, "
            f"the dropped crosstalk {gl_report.dropped_crosstalk_total:.2e}, the dropped motional branches' bound "
            f"{gl_report.branch_error_total:.2e} and the keyed tolerance's convergence change {gl_report.tolerance_change_total:.2e} "
            "reported (Section 5.4)"
        )
    cap_growth = {
        t.mode: t.d - space_initial.truncation(t.mode).d
        for t in joint_space.resolved
        if t.d != space_initial.truncation(t.mode).d
    }
    if cap_growth:
        notes.append(
            "the truncation monitor raised the caps (Section 5.5): "
            + ", ".join(f"mode {m} by {add} level(s)" for m, add in sorted(cap_growth.items()))
        )
    # 1.9: a boundary population the retries left above the threshold is said out loud as well as reported
    warn_if_boundary_exceeds(boundary, opts.boundary_population_max)
    dm_states = [[(w, st) for w, st in members if st.isoper] for members in register_states]
    rho_register: qt.Qobj | None = None
    if all(len(ms) == len(all_) for ms, all_ in zip(dm_states, register_states)):
        acc = sum((w * st for members in dm_states for w, st in members), 0.0 * register_states[0][0][1])
        rho_register = acc / len(register_states)
    # 9. readout per sample on its register state(s), the collision process per shot (Section 6.7)
    register_space = HilbertSpace(tuple(joint_space.ion_dims), (), None, ())
    run_state = RunState.nominal(n_ions)
    collisions = device.noise.collisions if noise else None
    coll_rates: dict[int, float] = {}
    if collisions is not None and collisions.pressure_pa > 0.0:
        coll_rates = {
            i: collision_rate_per_ion(collisions, float(device.crystal.masses_kg[i])) for i in range(n_ions)
        }
    # the measured set is the circuit's targets UNIONED with every trailing measure operation, exactly the set the scheduler
    # puts in its terminal event (control.schedule); reading Circuit.measure alone made run() histogram all N ions on a
    # circuit whose only measurement was a trailing `measure` op (M6 fix)
    declared: list[int] = list(compiled.measure)
    for op in compiled.ops:
        if op.name == "measure":
            declared.extend(q for q in op.qubits if q not in declared)
    measured = tuple(sorted(declared)) if declared else tuple(range(n_ions))
    bits_kept: list[np.ndarray] = []
    levels_kept: list[np.ndarray] = []
    heralds_kept: list[int] = []
    posteriors_kept: list[np.ndarray] = []
    records_kept: list[list[int]] = []
    sub_bins_kept: list[np.ndarray] = []
    arrivals_kept: list[tuple[np.ndarray, ...]] = []
    anomaly_bands = (
        [
            count_anomaly_band(stage.models[q], window, ("bright", stage.schemes[q].dark_class))
            for q in measured
        ]
        if readout == "full"
        else []
    )
    bits_per_sample: list[np.ndarray] = []
    discarded = 0
    prep_duration = prep_run.duration_s
    # the kick heats the SOFTEST mode most (delta nbar = E_kick/(hbar omega_m)), so that mode sets the reported quanta
    soft_mode_omega = (
        2.0 * math.pi * min(float(m.omega_hz) for m in device.crystal.modes) if device.crystal.modes else 0.0
    )
    kick_quanta: list[float] = []
    reorders = 0
    outcome: ReadoutOutcome | None = None
    for s_idx, (smp, members) in enumerate(zip(samples_seq, register_states)):
        n_s = counts_per_sample[s_idx]
        if n_s == 0:
            bits_per_sample.append(np.zeros((0, len(measured)), dtype=np.uint8))
            continue
        sample_bits: list[np.ndarray] = []
        shot = first_shots[s_idx]
        for (_w, rho_s), n_k in zip(members, _allocate(n_s, [w for w, _s in members])):
            if n_k == 0:
                continue
            reg_state = State(
                internal=rho_s,
                motional=MotionalModel(reduced={}, nbar={}, frozen=()),
                joint=rho_s,
                provenance=tuple(state0.provenance) + ("m6.register_mixture",),
            )
            outcome = measure(
                register_space,
                reg_state,
                stage.schemes,
                stage.models,
                stage.discriminator,
                seeds,
                shots=n_k,
                sample=smp.sample_id,
                trajectory=0,
                first_shot=shot,
                leakage=stage.leakage or None,
                depumping=stage.depumping or None,
                mode=readout,
                povm=stage.povm if readout == "fast" else None,
                keep_records=(readout == "full"),
            )
            for j in range(n_k):
                herald = 0
                keep = True
                if coll_rates:
                    rng_c = np.random.default_rng(seeds.child(smp.sample_id, 0, shot, 0, "collisions"))
                    assert collisions is not None
                    for ev in sample_collisions(rng_c, collisions, coll_rates, t_rep):
                        herald |= 1
                        run_state = RunState(
                            run_state.order,
                            run_state.dark,
                            run_state.lost,
                            run_state.events + ((shot, f"collision:{ev.outcome}:ion{ev.ion}"),),
                        )
                        if ev.outcome == "heating_kick":
                            # the kick's energy scale is the neutral's thermal energy times the mass ratio (Section 6.7);
                            # drawn per event and recorded in quanta of the softest mode, the one it heats most
                            kick = 0.0
                            if soft_mode_omega > 0.0:
                                kick = sample_kick_quanta(
                                    rng_c,
                                    collisions,
                                    float(device.crystal.masses_kg[ev.ion]),
                                    soft_mode_omega,
                                )
                                kick_quanta.append(kick)
                            run_state = RunState(
                                run_state.order,
                                run_state.dark,
                                run_state.lost,
                                run_state.events[:-1]
                                + ((shot, f"collision:heating_kick:ion{ev.ion}:dnbar={kick:.3g}"),),
                            )
                            if ev.time_s >= prep_duration:
                                keep = False  # the crystal melted during the sequence; the Doppler stage recools only before it
                        elif ev.outcome == "reorder":
                            keep = False
                            run_state = RunState(
                                sample_reorder(rng_c, collisions, run_state.order, ev.ion),
                                run_state.dark,
                                run_state.lost,
                                run_state.events,
                            )
                            reorders += 1
                        elif ev.outcome == "loss":
                            keep = False
                            run_state = RunState(
                                run_state.order, run_state.dark, run_state.lost | {ev.ion}, run_state.events
                            )
                        else:  # dark_ion
                            keep = False
                            run_state = RunState(
                                run_state.order, run_state.dark | {ev.ion}, run_state.lost, run_state.events
                            )
                row = np.asarray(outcome.bits[j, list(measured)], dtype=np.uint8).copy()
                unusable = run_state.dark | run_state.lost
                if unusable:
                    herald |= 2
                    for col, q in enumerate(measured):
                        if q in unusable:
                            row[col] = stage.schemes[q].bit_of_class("dark")
                if outcome.records is not None and any(
                    not (lo <= outcome.records[j][q].total <= hi)
                    for q, (lo, hi) in zip(measured, anomaly_bands)
                ):
                    herald |= 4
                shot += 1
                if not keep:
                    discarded += 1
                    continue
                bits_kept.append(row)
                levels_kept.append(np.asarray(outcome.levels[j, list(measured)], dtype=np.uint8).copy())
                sample_bits.append(row)
                heralds_kept.append(herald)
                if outcome.posteriors is not None:
                    posteriors_kept.append(np.asarray(outcome.posteriors[j]))
                if readout == "full" and outcome.records is not None:
                    recs_j = outcome.records[j]
                    records_kept.append([recs_j[q].total for q in measured])
                    if all(recs_j[q].sub_bins is not None for q in measured):
                        sub_bins_kept.append(
                            np.stack([np.asarray(recs_j[q].sub_bins, dtype=int) for q in measured])
                        )
                    if all(recs_j[q].arrivals_s is not None for q in measured):
                        arrivals_kept.append(
                            tuple(np.asarray(recs_j[q].arrivals_s, dtype=float) for q in measured)
                        )
        bits_per_sample.append(
            np.asarray(sample_bits, dtype=np.uint8).reshape(-1, len(measured))
            if sample_bits
            else np.zeros((0, len(measured)), dtype=np.uint8)
        )
        notify("readout", s_idx + 1, len(samples_seq))
    assert outcome is not None
    bits = np.asarray(bits_kept, dtype=np.uint8).reshape(-1, len(measured))
    counts, probabilities = aggregate(bits)
    n_eff = effective_sample_size(bits_per_sample)
    error_bars = binomial_error_bars(probabilities, n_eff)
    spam: dict[str, tuple[float, float]] = {}
    if stage.product is not None:
        model_errors = stage.product.per_ion_errors()
    else:
        # readout="full" replaced the POVM rather than preceding it (Section 5.7), so the (eps_B, eps_D) report is
        # estimated from the records this run actually generated (Section 8.6, "estimated from calibration runs"); a
        # level the circuit never populated reports nan, never a silent zero
        lv = np.asarray(levels_kept, dtype=np.uint8).reshape(-1, len(measured))
        errs: list[tuple[float, float]] = []
        for col, q in enumerate(measured):
            bright = stage.schemes[q].bright_level
            pair: list[float] = []
            for lev, wrong_bit in ((bright, 1 - bright), (1 - bright, bright)):
                mask = lv[:, col] == lev
                pair.append(float(np.mean(bits[mask, col] == wrong_bit)) if np.any(mask) else math.nan)
            errs.append((pair[0], pair[1]))
        model_errors = tuple(errs)
    for i, (eps_b, eps_d) in enumerate(model_errors):
        spam[f"q{i}"] = (float(eps_b), float(eps_d))
        spam[f"q{i}.state_preparation"] = (float(prep_run.preparation_error(i)), 0.0)
    photon_records = np.array(records_kept, dtype=int) if (readout == "full" and records_kept) else None
    sub_bin_records = (
        np.stack(sub_bins_kept) if (sub_bins_kept and len(sub_bins_kept) == len(bits_kept)) else None
    )
    arrival_times = tuple(arrivals_kept) if (arrivals_kept and len(arrivals_kept) == len(bits_kept)) else None
    posteriors = np.array(posteriors_kept) if posteriors_kept else None
    if not noise:
        approximations.append(
            "noise: the nominal sample without the device's collapse operators (noise=False)"
        )
    elif quiet and any(m != "sesolve" for m in methods):
        approximations.append(
            "noise: the nominal sample (no quasi-static or sampled content) with the device's collapse operators, solver "
            + "/".join(methods)
        )
    elif quiet:
        approximations.append("noise: the nominal sample; the noise model is quiet")
    else:
        approximations.append(
            f"noise: {n_samples} dynamical samples at the shot clock (T_rep = {t_rep:.4g} s), solver {'/'.join(methods)}"
        )
    if run_state.dark or run_state.lost:
        approximations.append(
            "collisions: after a dark-ion or loss event the remaining ions' dynamics stay on the nominal crystal; the flagged "
            "ions read dark, and the crystal_image experiment of Section 6.7 detects the event for a recalibration on the "
            "reduced device (a Device with N - 1 ions is a different device and gets its own table, Section 7.5)"
        )
    if kick_quanta:
        approximations.append(
            f"collisions: {len(kick_quanta)} heating kick(s) drawn from the Section 6.7 energy distribution "
            f"(k_B T m_gas/m_ion, mean {sum(kick_quanta) / len(kick_quanta):.3g} quanta of the softest mode, max "
            f"{max(kick_quanta):.3g}); a kick landing before the preparation is recooled by the cooling stage and the shot "
            "is kept, but a kick landing after it DISCARDS the shot rather than adding its nbar to the already-evolved "
            "sample, because the dynamical samples are propagated before the shot loop (the melt and recrystallization "
            "dynamics are a Section 1.3 non-goal)"
        )
    if reorders:
        approximations.append(
            f"collisions: {reorders} reorder event(s) permuted RunState.order from the configured permutation "
            "distribution and DISCARDED the shot; the permutation does not re-derive b_{i,m}, eta_{i,m} or the pair sign "
            "s of Section 7.7 for the shots that follow, so Section 6.7's 'runs the rest of the sequence with the wrong "
            "mode structure' is met at the herald/discard level only (the sample loop is not re-entered after a reorder)"
        )
    if noise:
        sentence = device.noise.provenance_sentence()
        if sentence:
            approximations.append(f"noise provenance (Section 6.1): {sentence}")
    disc_name = type(stage.discriminator).__name__
    if stage.product is not None:
        approximations.append(
            f"SPAM definition: readout (eps_B, eps_D) of the {disc_name} discriminator's product POVM at zero crosstalk "
            "(Section 13 row 'Readout figure of merit'); state preparation 1 - P(target) of the optical pump (Section 4.2.6)"
        )
        if stage.product.uncertainty > 0.0:
            approximations.append(
                f"readout POVM: {disc_name} has no closed-form confusion, so it was estimated from {povm_samples} sampled "
                f"records per level per ion; every POVM entry carries a statistical uncertainty of "
                f"{stage.product.uncertainty:.2e} (Section 8.4)"
            )
    else:
        approximations.append(
            f"SPAM definition: readout (eps_B, eps_D) of the {disc_name} discriminator estimated from this run's own "
            "photon records (readout='full' replaces the POVM rather than preceding it, Section 5.7); a qubit level the "
            "circuit never populated reports nan; state preparation 1 - P(target) of the optical pump (Section 4.2.6)"
        )
    if readout == "fast":
        approximations.append(
            f"readout fast path: the {'register-wide confusion' if stage.leakage else 'product POVM'} applied to the joint outcome "
            "(Section 5.7)"
        )
    else:
        approximations.append(
            f"readout full path: one photon record per ion per shot generated from the {disc_name} discriminator's window "
            f"({stage.discriminator.window_s:.4g} s) and discriminated, the neighbour coupling of Section 8.5 "
            f"{'applied at the configured PSF leakage' if stage.leakage else 'inactive (no PSF leakage configured)'} "
            "(Section 5.7)"
        )
    if stage.depumping:
        approximations.append(
            "readout crosstalk (depumping half, Section 8.5): one bright neighbour raises an ion's (R_d, R_b) by "
            + ", ".join(
                f"d={d}: ({dd:.4g}, {db:.4g}) s^-1" for d, (dd, db) in sorted(stage.depumping.items())
            )
            + " through the leaked resonant light bounded by I_ion/I_sat = 3 lambda^2/(8 pi^2 x^2), the neighbours' "
            "classes frozen at their start (the first-order form)"
        )
    if stage.crosstalk_discrepancy is not None:
        approximations.append(
            f"readout crosstalk: the register-wide confusion at the configured PSF leakage {stage.leakage} differs from the "
            f"product POVM by at most {stage.crosstalk_discrepancy:.4f} in a per-ion declared-bright probability (the "
            "bounded, reported discrepancy of Sections 9.5 and 9.17)"
        )
    for i, (beta, omega_rf) in sorted(stage.micromotion.items()):
        approximations.append(
            f"readout micromotion: ion {i}'s detection rates carry J_0({beta:.4g})^2 on the carrier and J_1^2 on each "
            f"first sideband at Omega_rf/2pi = {omega_rf / (2.0 * math.pi):.4g} Hz (Section 8.8)"
        )
    approximations.extend(device.hardware.describe())
    if selection.guard_violations:
        notes.extend(v for v in selection.guard_violations if v not in notes)
    diagnostics = Diagnostics(
        level=_diagnostics_level(run_level),
        space=joint_space,
        mode_class=dict(selection.mode_class),
        run_state=run_state,
        wall_clock_span_s=float((shots - 1) * t_rep),
        boundary_population=boundary,
        margin_levels={m.mode: m.margin_levels for m in joint_space.resolved},
        dropped_modes=selection.dropped_modes,
        frozen_contribution=selection.frozen_contribution,
        integrator=",".join(integrators) if integrators else "none",
        tolerances=(opts.atol, opts.rtol),
        samples=n_samples,
        trajectories=(len(branches) if branches else 1) * n_traj_max,
        branches=len(branches) if branches else 1,
        shots_per_sample=int(shots // n_samples),
        shots_per_sample_realized=tuple(int(m) for m in counts_per_sample),
        effective_sample_size=n_eff,
        root_seed=int(seed),
        calibration=table,
        approximations=tuple(approximations) + tuple(notes) + tuple(selection.notes),
        intrinsic_budget=intrinsic_budget(device, sched, selection),
        dropped_branch_weight=dropped_weight,
        frozen_excitation_bound=dict(selection.frozen_excitation),
        dropped_contribution=selection.dropped_contribution,
        margin_reached=margin_reached,
        populated_n_max=populated_max,
        cap_growth=cap_growth,
        gate_local=gl_report,
        kernel=_kernel_kind(kernel_kinds),
        workers=workers_used,
        propagator_cache_hits=propagator_hits,
        convergence=convergence,
        level_reason=level_reason,
    )
    result = Result(
        bitstrings=bits,
        bit_order="qubit0_lsb",
        counts=counts,
        probabilities=probabilities,
        error_bars=error_bars,
        photon_records=photon_records,
        posteriors=posteriors,
        noise_samples=tuple(samples_seq),
        heralds=np.asarray(heralds_kept, dtype=np.uint8),
        discarded_shots=int(discarded),
        run_state=run_state,
        spam=spam,
        final_state=rho_register if keep_final_state else None,
        diagnostics=diagnostics,
        sub_bin_records=sub_bin_records,
        arrival_times_s=arrival_times,
        qubits=tuple(int(q) for q in measured),
        registers=dict(compiled.registers),
        created_at=created_at,
        duration_s=float(time.perf_counter() - started),
    )
    _LAST_RECORD[id(result)] = RunRecord(
        compile=report,
        schedule=sched,
        selection=selection,
        preparation=prep_run,
        branches=tuple(branches),
        traces=tuple(traces_all),
        register_state=rho_register,
        readout=stage,
        outcome=outcome,
        table=table,
        qubit_shifts_hz=shifts,
        notes=tuple(notes),
        gate_local=gl_report,
    )
    return result


def _diagnostics_level(level: FidelityLevel) -> Literal["JOINT_EXACT", "GATE_LOCAL"]:
    """The level a run ran at, as the Appendix E literal ``Diagnostics.level`` carries (never the AUTO policy)."""
    if level is FidelityLevel.JOINT_EXACT:
        return "JOINT_EXACT"
    if level is FidelityLevel.GATE_LOCAL:
        return "GATE_LOCAL"
    raise ValueError("a run reports the level it ran at, not the AUTO policy")


def _kernel_kind(kinds: set[str]) -> str:
    if not kinds:
        return "none"
    if kinds == {"assembled"}:
        return "assembled"
    if kinds == {"factorized"}:
        return "factorized"
    return "mixed"


def _engine_task(
    payload: tuple[Any, Device, Schedule, State, HilbertSpace, NoiseSample, SeedSpec, SolverOptions],
) -> tuple[Traces, EngineReport]:
    """One (sample, branch) engine run as a map task (module-level so that it pickles under ``map="parallel"``)."""
    engine, device, sched, state, space, smp, seeds, opts = payload
    traces = engine.run_pulses(device, sched, state, space, smp, seeds, opts)
    rep = engine.last_report
    assert rep is not None
    return traces, rep


class _Reporter:
    """The ``progress`` callback of one run (docs/api_implementation_plan.md 1.8) with the run's own clock: ``report(stage,
    done, total)`` builds the ``Progress`` and hands it on; a callback of None makes every report a no-op."""

    def __init__(self, callback: Callable[[Progress], None] | None, started: float) -> None:
        self.callback = callback
        self.started = started

    @property
    def active(self) -> bool:
        return self.callback is not None

    def __call__(self, stage: str, done: int, total: int) -> None:
        if self.callback is not None:
            self.callback(Progress(stage, int(done), int(total), time.perf_counter() - self.started))


def _run_engine_tasks(
    engine: Any,
    device: Device,
    sched: Schedule,
    space: HilbertSpace,
    payloads: Sequence[tuple[int, int, State, NoiseSample]],
    seeds: SeedSpec,
    opts: SolverOptions,
    report: _Reporter | None = None,
) -> tuple[list[tuple[Traces, EngineReport]], int]:
    """The JOINT_EXACT engine runs of ``run()``: in-process on one engine when the map is serial, one worker is available or
    there is a single run (the engine's trajectory map then takes the workers), else spread over the workers with the
    trajectories of every run in-process (Section 11.3 item 9; M9b). Returns the (traces, report) pairs in order and the
    worker count the maps used. In-process runs report every pulse (counted across the runs) and every run to ``report``;
    a parallel map reports its runs once, when it returns."""
    workers = worker_count(opts)
    n_runs = len(payloads)
    if opts.map == "serial" or workers <= 1 or n_runs < 2:
        out: list[tuple[Traces, EngineReport]] = []
        for k, (_s_idx, _k, st, smp) in enumerate(payloads):
            if report is not None and report.active:

                def per_pulse(p: Progress, k: int = k) -> None:
                    assert report is not None
                    report("pulse", k * p.total + p.done, n_runs * p.total)

                engine.progress = per_pulse
            traces = engine.run_pulses(device, sched, st, space, smp, seeds, opts)
            rep = engine.last_report
            assert rep is not None
            out.append((traces, rep))
            if report is not None:
                report("branch", k + 1, n_runs)
        engine.progress = None
        used = max((r.workers for _t, r in out), default=1)
        return out, used
    inner = replace(opts, map="serial")
    items = [(engine, device, sched, st, space, smp, seeds, inner) for _s_idx, _k, st, smp in payloads]
    results = map_tasks(_engine_task, items, map_kind=opts.map, workers=workers)
    if report is not None:
        report("branch", n_runs, n_runs)
    return results, min(workers, n_runs)


def _supplied_class(space: HilbertSpace, mode: int) -> Literal["resolved", "frozen", "dropped", "enr"]:
    cls = space.mode_class(mode)
    if cls == "resolved":
        return "resolved"
    if cls == "enr":
        return "enr"
    return "frozen"


def _allocate(total: int, weights: Sequence[float]) -> list[int]:
    """Largest-remainder allocation of ``total`` shots over members with the given weights (one member takes them all)."""
    if len(weights) == 1:
        return [int(total)]
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    raw = w * total
    base = np.floor(raw).astype(int)
    rest = int(total - base.sum())
    order = np.argsort(-(raw - base))
    for k in order[:rest]:
        base[k] += 1
    return [int(x) for x in base]


def to_register_order(vec: np.ndarray, n_qubits: int) -> np.ndarray:
    """Reorder a ket from the compiler's bit order (qubit 0 the LEAST-significant index bit, Section 13) to the register
    state's tensor order (ion 0 the FIRST factor of ``qutip.tensor``, the most-significant index bit)."""
    arr = np.asarray(vec, dtype=complex).reshape([2] * n_qubits)  # axes (q_{n-1}, ..., q_0)
    return np.asarray(np.transpose(arr, axes=list(reversed(range(n_qubits)))).reshape(-1))


def ideal_register_state(result_or_circuit: Result | Circuit) -> np.ndarray:
    """The ideal register ket the run's state is compared with: U_compiled |0...0> of the COMPILED circuit, whose residual
    virtual-Z frame is absorbed (the frame the measurement discards, Section 7.6), rotated by the SCHEDULER's final frame (the
    virtual-Z updates it made for the compensated light shifts and the echo schemes, M8), for a Result; U |0...0> of the circuit
    itself for a Circuit. Returned in the REGISTER order of ``Result.final_state`` (ion 0 the first tensor factor), converted
    from the compiler's qubit-0-least-significant order by ``to_register_order``."""
    from qutip_trap.control.compiler import Circuit as _Circuit
    from qutip_trap.control.compiler import circuit_unitary
    from qutip_trap.control.native import rz

    if isinstance(result_or_circuit, _Circuit):
        circuit = result_or_circuit
        frame: dict[int, float] = {}
    else:
        rec = last_record(result_or_circuit)
        circuit = rec.compile.circuit
        frame = dict(rec.schedule.phase_frame)
    u = circuit_unitary(circuit)
    ket = np.asarray(u[:, 0], dtype=complex)
    # the scheduler's own frame (the compensated light shifts' RZ(2 pi int delta dt), an echo scheme's Z(pi)): the physical state
    # carries these rotations and the frame absorbs them (Section 7.6), so the ideal state is rotated by them too; qubit q is
    # the q-th bit (least significant first) of the compiler's index
    n = circuit.n_qubits
    for q, theta in frame.items():
        if theta == 0.0:
            continue
        op = rz(
            -theta
        )  # a frame offset theta leaves the state as RZ(-theta) times the ideal one (Section 7.6)
        for idx in range(2**n):
            ket[idx] *= op[(idx >> q) & 1, (idx >> q) & 1]
    return to_register_order(ket, n)


def register_fidelity(result: Result, target: np.ndarray | qt.Qobj | None = None) -> float:
    """<target| rho |target> of the run's recombined register state (``keep_final_state=True``) against an ideal ket in the
    register order (ion 0 the first tensor factor); the default target is ``ideal_register_state(result)``, the compiled
    circuit's state with its frame absorbed."""
    rho = result.final_state
    if rho is None:
        raise ValueError("run with keep_final_state=True to compare the register state")
    n = result.n_qubits
    tgt = ideal_register_state(result) if target is None else target
    vec = np.asarray(tgt.full() if isinstance(tgt, qt.Qobj) else tgt, dtype=complex).ravel()
    dims = [int(x) for x in rho.dims[0]]
    if all(d == 2 for d in dims):
        ket = qt.Qobj(vec.reshape(-1, 1), dims=[[2] * n, [1] * n])
    else:
        # a qudit register (leakage levels, M7): the ideal ket lives on the qubit levels 0 and 1 of every factor
        full = np.zeros(dims, dtype=complex)
        full[tuple(slice(0, 2) for _ in dims)] = vec.reshape([2] * n)
        ket = qt.Qobj(full.reshape(-1, 1), dims=[dims, [1] * n])
    return float(np.real(qt.expect(rho, ket)))


__all__ = [
    "Branch",
    "ReadoutStage",
    "RunError",
    "RunRecord",
    "detection_micromotion",
    "effective_sample_size",
    "enumerate_branches",
    "ideal_register_state",
    "intrinsic_budget",
    "internal_probabilities",
    "last_record",
    "level_maps",
    "prepare",
    "readout_stage",
    "register_fidelity",
    "run",
    "to_register_order",
]
