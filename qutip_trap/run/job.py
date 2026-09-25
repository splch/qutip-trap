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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np
import qutip as qt

from qutip_trap.control.compiler import CompileReport
from qutip_trap.control.schedule import GateDrive, Schedule
from qutip_trap.dynamics.engine import SeedSpec, SolverOptions, State, Traces
from qutip_trap.hilbert.operators import thermal_populations
from qutip_trap.hilbert.space import HilbertSpace
from qutip_trap.noise.levels import InternalLevels, internal_levels
from qutip_trap.noise.sampling import NoiseSample
from qutip_trap.noise.scattering import scattering_estimates
from qutip_trap.options import Numerics, Physics, Readout
from qutip_trap.prep.recipe import PreparationRun, recipe_of, run_preparation
from qutip_trap.prep.sequence import prepare_state
from qutip_trap.readout.detection import RecordModel
from qutip_trap.readout.discriminate import (
    POVM,
    Discriminator,
    ReadoutOutcome,
    ThresholdDiscriminator,
    povm_for,
    product_povm,
)
from qutip_trap.readout.fluorescence import FluorescenceRates, ReadoutScheme, detection_rates_for_ion
from qutip_trap.run.gate_local import GateLocalReport
from qutip_trap.run.levels import FidelityLevel
from qutip_trap.run.results import Progress, Result, aggregate
from qutip_trap.run.space import SpaceSelection
from qutip_trap.units import TWO_PI
from qutip_trap.validation.two_qubit_closed_forms import ballance_thermal_error

if TYPE_CHECKING:
    from qutip_trap.control.compiler import Circuit
    from qutip_trap.control.shaping import GateModes
    from qutip_trap.control.table import CalibrationTable, Waveform
    from qutip_trap.device.model import Device
    from qutip_trap.light.beams import Beam
    from qutip_trap.machine import Machine

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
    """The readout of every KEPT shot, in ``Result.bitstrings`` row order, over every ion (a discarded shot is left out, as
    it is from the Result): the declared bits, the sampled levels behind them, the detection time used and the posteriors
    and photon records where the discriminator produced them. The pipeline reads the register out once per (sample, branch)
    batch; this is the concatenation of those batches."""
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


LEGACY_RUN_KEYWORDS: Final[dict[str, str]] = {
    "t0_s": "Pass physics=Physics(t0_s=...), or set it on the Machine.",
    "shot_period_s": "Pass physics=Physics(shot_period_s=...), or set it on the Machine.",
    "noise": "Pass physics=Physics(noise=...), or set it on the Machine.",
    "internal_levels": "Pass physics=Physics(internal_levels=...), or set it on the Machine.",
    "crosstalk_suppression": "Pass physics=Physics(crosstalk_suppression=...), or set it on the Machine.",
    "stark_compensation": "Pass physics=Physics(stark_compensation=...), or set it on the Machine.",
    "entangler": "Pass physics=Physics(entangler=...), or set it on the Machine.",
    "channels": "Pass physics=Physics(extra_channels=...), or set it on the Machine.",
    "builder_options": "Pass physics=Physics(builder=...), or set it on the Machine.",
    "options": (
        "Pass numerics=Numerics.from_solver_options(options) and physics=Physics.from_solver_options(options, ...), or "
        "build the Machine with them."
    ),
    "space": "Pass numerics=Numerics(truncation=Truncation(space=...)), or set it on the Machine.",
    "caps": "Pass numerics=Numerics(truncation=Truncation(caps=...)), or set it on the Machine.",
    "enr_group": "Pass numerics=Numerics(truncation=Truncation(enr_group=...)), or set it on the Machine.",
    "samples": "Pass numerics=Numerics(parallel=Parallel(samples=...)), or set it on the Machine.",
    "parallel": "Pass numerics=Numerics(parallel=Parallel(addressing=...)), or set it on the Machine.",
    "readout": "Pass readout=Readout(mode=...), or set it on the Machine.",
    "discriminator": "Pass readout=Readout(discriminator=...), or set it on the Machine.",
    "povm_samples": "Pass readout=Readout(povm_samples=...), or set it on the Machine.",
    "gate_drives": "Declare the drives on the device: dataclasses.replace(device, roles=BeamRoles(gate=...)).",
    "entangling_drives": "Declare the drives on the device: dataclasses.replace(device, roles=BeamRoles(entangling=...)).",
    "calibrate_kwargs": "Pin the table on the Machine: Machine(device).calibrated(seed=..., **calibrate_kwargs).run(...).",
}
"""The keyword arguments of ``run`` deprecated in 0.3.0 (docs/api_implementation_plan.md 2.1), each with the sentence that
names its new home; ``machine_with_run_kwargs`` rewrites a call that passes one and warns once per keyword and call site.
Removable in v0.5 at the earliest (docs/deprecations.md)."""

LEGACY_DEADLINE: Final[str] = "v0.5"

LEGACY_CALL_KEYWORDS: Final[dict[str, str]] = {
    "table": "Pin the table on the Machine: machine.calibrated(...) or dataclasses.replace(machine, table=...).",
    "level": "Set the level on the Machine: dataclasses.replace(machine, level=FidelityLevel.GATE_LOCAL).",
    "keep_final_state": "Remove it: the benchmark decides which of its runs keep the final state.",
}
"""The ``run`` keywords a benchmark's ``**run_kwargs`` forwarded that are not option-object fields: deprecated on the
benchmarks in 0.3.0 (the machine carries the table and the level; ``keep_final_state`` is the benchmark's own choice)."""

_PHYSICS_FIELDS: Final[dict[str, str]] = {
    "t0_s": "t0_s",
    "shot_period_s": "shot_period_s",
    "noise": "noise",
    "internal_levels": "internal_levels",
    "crosstalk_suppression": "crosstalk_suppression",
    "stark_compensation": "stark_compensation",
    "entangler": "entangler",
    "channels": "extra_channels",
    "builder_options": "builder",
}
_TRUNCATION_FIELDS: Final[dict[str, str]] = {"space": "space", "caps": "caps", "enr_group": "enr_group"}
_PARALLEL_FIELDS: Final[dict[str, str]] = {"samples": "samples", "parallel": "addressing"}
_READOUT_FIELDS: Final[dict[str, str]] = {
    "readout": "mode",
    "discriminator": "discriminator",
    "povm_samples": "povm_samples",
}


def machine_with_run_kwargs(
    base: Machine | Device,
    kwargs: Mapping[str, Any],
    *,
    caller: str = "qutip_trap.run.job.run",
    circuit: Circuit | None = None,
    seed: int = 0,
    table: CalibrationTable | None = None,
    level: str | FidelityLevel | None = None,
    physics: Physics | Mapping[str, Any] | None = None,
    numerics: Numerics | Mapping[str, Any] | None = None,
    readout: Readout | Mapping[str, Any] | None = None,
    stacklevel: int = 1,
    call_keywords: bool = False,
) -> Machine:
    """The ``Machine`` a call in the 0.1.0 shape asked for: ``base`` (a Device becomes ``Machine(device)``) with the table,
    level and option objects given explicitly, then every keyword of ``LEGACY_RUN_KEYWORDS`` in ``kwargs`` rewritten onto
    the machine with a :class:`~qutip_trap._compat.QutipTrapDeprecationWarning` naming the keyword, ``caller`` and the new
    home. A keyword the table does not list is refused like any unexpected keyword argument. A field-level keyword
    (``noise``, ``caps``, ``povm_samples``, ...) passed beside the object it lives on sets that one field of the object, the
    keyword winning; ``options`` (a whole ``SolverOptions``) beside ``numerics`` or ``physics`` is refused, since the two
    would overlap on every field. ``gate_drives`` and
    ``entangling_drives`` become the device's roles; ``calibrate_kwargs`` pins the table the compile-calibrate-schedule
    prefix would have built for ``circuit`` at ``seed`` (the same cached surrogate, so the run is the same run). The warnings
    are attributed ``stacklevel`` frames above this function's caller: 1 names the caller's line, 2 (what ``run`` passes) the
    line that called the caller. With ``call_keywords`` (the benchmarks) ``table``, ``level`` and ``keep_final_state``
    are accepted in ``kwargs`` as well (``LEGACY_CALL_KEYWORDS``): the first two move onto the machine, the third is
    dropped, each with its warning; a bare ``Device`` base warns there too (the laboratory's 0.4.0 rule,
    ``machine.warn_bare_device``), while ``run`` itself takes a device by design and never warns for it."""
    from qutip_trap._compat import message, warn
    from qutip_trap.device.model import Device as _Device
    from qutip_trap.machine import Machine as _Machine
    from qutip_trap.machine import warn_bare_device

    kwargs = dict(kwargs)
    machine = _Machine(base) if isinstance(base, _Device) else base
    if call_keywords and isinstance(base, _Device):
        warn_bare_device(caller, stacklevel=stacklevel + 1)
    if call_keywords:
        for key, fix in LEGACY_CALL_KEYWORDS.items():
            if key in kwargs:
                warn(
                    message(f"the {key!r} argument of {caller}", LEGACY_DEADLINE, fix),
                    stacklevel=stacklevel + 1,
                )
                value = kwargs.pop(key)
                if key == "table" and value is not None:
                    machine = replace(machine, table=value)
                elif key == "level" and value is not None:
                    machine = replace(machine, level=FidelityLevel(value))
    unknown = sorted(k for k in kwargs if k not in LEGACY_RUN_KEYWORDS)
    if unknown:
        raise TypeError(f"{caller}() got unexpected keyword argument(s) {unknown}")
    given_physics = physics is not None
    given_numerics = numerics is not None
    if table is not None:
        machine = replace(machine, table=table)
    if level is not None:
        machine = replace(machine, level=FidelityLevel(level))
    if physics is not None:
        machine = replace(
            machine, physics=Physics.from_mapping(physics) if isinstance(physics, Mapping) else physics
        )
    if numerics is not None:
        machine = replace(
            machine, numerics=Numerics.from_mapping(numerics) if isinstance(numerics, Mapping) else numerics
        )
    if readout is not None:
        machine = replace(
            machine, readout=Readout.from_mapping(readout) if isinstance(readout, Mapping) else readout
        )
    for key in kwargs:  # one warning per keyword, attributed to the line the caller names
        warn(
            message(f"the {key!r} argument of {caller}", LEGACY_DEADLINE, LEGACY_RUN_KEYWORDS[key]),
            stacklevel=stacklevel + 1,
        )

    phys = machine.physics
    num = machine.numerics
    read = machine.readout
    device = machine.device
    if "options" in kwargs:
        if given_numerics or given_physics:
            raise TypeError(
                f"{caller}() got the deprecated keyword 'options' beside "
                f"{'numerics' if given_numerics else 'physics'}=; pass one form (the new home: "
                f"{LEGACY_RUN_KEYWORDS['options']})"
            )
        opts = kwargs["options"] or SolverOptions()
        num = Numerics.from_solver_options(opts)
        phys = replace(
            phys,
            scattering="channels" if opts.scattering_channels else "estimate",
            scattering_recoil=opts.scattering_recoil,
            intensity_noise_channels=opts.intensity_noise_channels,
            hardware_chain=opts.hardware_chain,
        )
    for key, field_name in _PHYSICS_FIELDS.items():
        if key in kwargs:
            phys = replace(phys, **{field_name: kwargs[key]})
    truncation = num.truncation
    for key, field_name in _TRUNCATION_FIELDS.items():
        if key in kwargs:
            truncation = replace(truncation, **{field_name: kwargs[key]})
    parallel = num.parallel
    for key, field_name in _PARALLEL_FIELDS.items():
        if key in kwargs:
            parallel = replace(parallel, **{field_name: kwargs[key]})
    num = replace(num, truncation=truncation, parallel=parallel)
    for key, field_name in _READOUT_FIELDS.items():
        if key in kwargs:
            read = replace(read, **{field_name: kwargs[key]})
    if "gate_drives" in kwargs or "entangling_drives" in kwargs:
        roles = device.roles
        if "gate_drives" in kwargs and kwargs["gate_drives"] is not None:
            roles = replace(roles, gate=dict(kwargs["gate_drives"]))
        if "entangling_drives" in kwargs and kwargs["entangling_drives"] is not None:
            roles = replace(roles, entangling=dict(kwargs["entangling_drives"]))
        device = replace(device, roles=roles)
    machine = replace(machine, device=device, physics=phys, numerics=num, readout=read)
    if kwargs.get("calibrate_kwargs") is not None and machine.table is None:
        if circuit is None:
            raise TypeError(
                f"{caller}(): calibrate_kwargs needs the circuit whose pairs the table is built for"
            )
        from qutip_trap.run.pipeline import compile_calibrate_schedule

        prefix = compile_calibrate_schedule(
            circuit,
            machine.device,
            seed=seed,
            calibrate_kwargs=dict(kwargs["calibrate_kwargs"]),
            **machine._prefix_kwargs(),
        )
        machine = replace(machine, table=prefix.table)
    return machine


def run(
    circuit: Circuit,
    device: Device,
    shots: int,
    *,
    table: CalibrationTable | None = None,
    level: Literal["JOINT_EXACT", "GATE_LOCAL", "auto"] | FidelityLevel = "auto",
    seed: int = 0,
    keep_final_state: bool = False,
    progress: Callable[[Progress], None] | None = None,
    physics: Physics | Mapping[str, Any] | None = None,
    numerics: Numerics | Mapping[str, Any] | None = None,
    readout: Readout | Mapping[str, Any] | ReadoutMode | None = None,
    **deprecated: Any,
) -> Result:
    """Compile -> calibrate -> schedule -> prepare -> evolve -> readout -> Result (Section 3.4): ``Machine(device, table=table,
    physics=physics, numerics=numerics, readout=readout, level=level).run(circuit, shots, seed=seed, ...)``.

    Since 0.3.0 (docs/api_implementation_plan.md 2.1) the run is ``qutip_trap.run.pipeline.execute`` on a ``Machine``; this
    function builds that machine. ``physics`` (which effects are simulated), ``numerics`` (how the integration is done) and
    ``readout`` (how the record is read) are the option objects of ``qutip_trap.options`` or mappings of their fields; None
    means their defaults, which are what ``run`` did in 0.1.0. ``table=None`` builds the surrogate table at the machine's
    ``t0_s`` for the pairs the circuit uses; ``level`` is the Section 11.5 policy (``"auto"``, or either level forced);
    ``seed`` roots every keyed stream; ``progress`` (0.2.0) is called with a ``Progress`` per pulse, branch, sample and
    readout. The keyword arguments of 0.1.0 (``LEGACY_RUN_KEYWORDS``: ``t0_s``, ``options``, ``noise``, ``readout="fast"``
    as a string, ``gate_drives``, ...) are still accepted: each warns once per call site with its new home and is rewritten
    onto the machine, so the old call and the new call produce the same result; they are removable in v0.5 at the earliest
    (docs/deprecations.md). The stages, their notes and the ``Diagnostics`` are documented on ``execute``.
    """
    if isinstance(
        readout, str
    ):  # the 0.1.0 form: readout="fast" | "full" shares the name of the object parameter
        deprecated = {**deprecated, "readout": readout}
        readout = None
    machine = machine_with_run_kwargs(
        device,
        deprecated,
        circuit=circuit,
        seed=seed,
        table=table,
        level=level,
        physics=physics,
        numerics=numerics,
        readout=readout,
        stacklevel=2,
    )
    return machine.run(circuit, shots, seed=seed, keep_final_state=keep_final_state, progress=progress)


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
    circuit's state with its frame absorbed.

    The register state spans every ion of the crystal, while the ideal ket spans the circuit's qubits (which address the
    leading ions, qubit q on ion q) and is the same whether the circuit measures all of them or a subset. A circuit on fewer
    qubits than ions leaves the remaining ions in the prepared state |0>, so its ket is embedded on the register with |0> on
    those trailing factors before the overlap is taken; a qudit register (leakage levels, M7) places the ket on the qubit
    levels 0 and 1 of every factor."""
    rho = result.final_state
    if rho is None:
        raise ValueError("run with keep_final_state=True to compare the register state")
    tgt = ideal_register_state(result) if target is None else target
    vec = np.asarray(tgt.full() if isinstance(tgt, qt.Qobj) else tgt, dtype=complex).ravel()
    dims = [int(x) for x in rho.dims[0]]
    n_ions = len(dims)
    n = int(round(math.log2(vec.size))) if vec.size else 0
    if vec.size == 0 or 2**n != vec.size:
        raise ValueError(f"the target ket has {vec.size} amplitudes, not a power of two")
    if n > n_ions:
        raise ValueError(f"the target ket spans {n} qubits but the register holds {n_ions} ions")
    if n < n_ions:
        idle = np.zeros(2 ** (n_ions - n), dtype=complex)
        idle[0] = 1.0
        vec = np.kron(vec, idle)  # ion 0 is the first factor, so the circuit's qubits are the leading ions
    if all(d == 2 for d in dims):
        ket = qt.Qobj(vec.reshape(-1, 1), dims=[dims, [1] * n_ions])
    else:
        full = np.zeros(dims, dtype=complex)
        full[tuple(slice(0, 2) for _ in dims)] = vec.reshape([2] * n_ions)
        ket = qt.Qobj(full.reshape(-1, 1), dims=[dims, [1] * n_ions])
    return float(np.real(qt.expect(rho, ket)))
