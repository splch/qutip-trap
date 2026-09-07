"""``run`` and ``prepare``: the orchestration of Section 3.4 (PLAN.md Sections 3.4, 5.3, 5.7, 7.2, 7.3, 8.6; Appendix E; M6).

    Device + Circuit
      -> compile: Circuit -> native gates (phase-tracked)                    [control.compiler]
      -> calibrate (surrogate, cached per Device): CalibrationTable         [calibration.surrogate]
      -> schedule: native gates -> Pulses with absolute times, the measure event   [control.schedule]
      -> space: resolved / frozen / dropped modes with their caps           [run.space]
      -> prepare: Doppler -> sideband -> pump, the initial mixed state      [prep.recipe, prep.sequence]
      -> evolve: every branch of the initial mixture through the pulses     [dynamics.engine]
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

Shots are distributed round-robin over dynamical samples (Section 3.4: default samples = min(shots, 64) when the noise model
has quasi-static or sampled content, 1 when it is quiet): sample k is drawn at the shot clock t0 + k_first T_rep with every
Drift an Ornstein-Uhlenbeck chain over the sample times (Section 7.5), the shots of one sample are drawn from that sample's
register state, and the error bars use the effective sample size of the between/within-sample decomposition, since shots of
one sample are not independent draws from the ensemble. Background-gas collisions (Section 6.7) are Poisson events per shot:
a heating kick during the cooling stages is heralded and kept (the crystal is recooled), any other event is heralded and the
shot discarded, a reorder permutes the persistent ion order, a loss or dark-ion event flags the ion so that every later shot
reads it dark; the remaining ions' dynamics stay on the nominal crystal (an approximation the notes record). The seeds are
keyed by (sample, trajectory, shot, ion, channel) exactly as Section 3.4 requires.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import qutip as qt

from qutip_trap.control.compiler import CompileReport, compile_with_report
from qutip_trap.control.schedule import (
    CrosstalkSuppression,
    GateDrive,
    Schedule,
    default_gate_drives,
    schedule,
)
from qutip_trap.dynamics.engine import JointExactEngine, MotionalModel, SeedSpec, SolverOptions, State, Traces
from qutip_trap.hilbert.operators import thermal_populations
from qutip_trap.hilbert.space import HilbertSpace
from qutip_trap.noise.collisions import collision_rate_per_ion, sample_collisions
from qutip_trap.noise.levels import InternalLevels, internal_levels
from qutip_trap.noise.sampling import NoiseSample, key_frozen_n, quiet_sample
from qutip_trap.noise.scattering import scattering_estimates
from qutip_trap.prep.recipe import PreparationRun, recipe_of, run_preparation
from qutip_trap.prep.sequence import prepare_state
from qutip_trap.readout.detection import RecordModel
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
from qutip_trap.run.levels import FidelityLevel, within_budget
from qutip_trap.run.results import Diagnostics, Result, RunState, aggregate, binomial_error_bars
from qutip_trap.run.space import SpaceSelection, select_space
from qutip_trap.validation.two_qubit_closed_forms import ballance_thermal_error

if TYPE_CHECKING:
    from qutip_trap.control.compiler import Circuit
    from qutip_trap.control.table import CalibrationTable
    from qutip_trap.device.model import Device
    from qutip_trap.dynamics.channels import CollapseOp
    from qutip_trap.dynamics.hamiltonian import BuilderOptions

M9A = "milestone M9a (GATE_LOCAL, PLAN.md Section 5.4)"
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


def _truncated_weights(probabilities: Sequence[float], weight_min: float) -> list[tuple[int, float]]:
    out = [(k, float(p)) for k, p in enumerate(probabilities) if p >= weight_min]
    return out or [(int(np.argmax(probabilities)), 1.0)]


def enumerate_branches(
    internal_probabilities: Sequence[Sequence[float]],
    mode_nbar: Mapping[int, float],
    weight_min: float,
) -> tuple[list[Branch], float]:
    """Branches of the diagonal initial mixture with weight >= ``weight_min`` (descending), and the dropped weight."""
    ion_options = [_truncated_weights(p, weight_min) for p in internal_probabilities]
    mode_options: dict[int, list[tuple[int, float]]] = {}
    for m, nb in mode_nbar.items():
        if nb <= 0.0:
            mode_options[m] = [(0, 1.0)]
            continue
        d = int(math.ceil(math.log(weight_min) / math.log(nb / (1.0 + nb)))) + 2
        mode_options[m] = _truncated_weights(
            [float(x) for x in thermal_populations(nb, max(d, 2))], weight_min
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
    povm: POVM
    product: POVM
    """The zero-crosstalk product form: the per-ion (eps_B, eps_D) report reads it even when the register form samples."""
    leakage: dict[int, float]


def readout_stage(
    device: Device,
    table: CalibrationTable,
    *,
    discriminator: Discriminator | None = None,
    levels: Mapping[int, InternalLevels] | None = None,
) -> ReadoutStage:
    """The per-ion rate objects of the device's detection beams, the table's threshold and window, and the POVM; ``levels``
    extends each scheme's classes to the leakage levels of a d > 2 register factor (M7)."""
    from qutip_trap.light.roles import detection_beams

    rates: list[FluorescenceRates] = []
    schemes: list[ReadoutScheme] = []
    models: list[RecordModel] = []
    for i in range(device.crystal.n_ions):
        beams = [device.beams[k] for k in detection_beams(device, i)]
        r, scheme, model = detection_rates_for_ion(
            device.crystal.species[i],
            device.field.B_gauss,
            device.field.direction,
            beams,
            position_m=tuple(float(x) for x in device.crystal.positions_m[i]),
        )
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
    leakage = {int(k): float(v) for k, v in device.detector.psf_leakage.items() if v > 0.0}
    product = product_povm(models, schemes, discriminator)
    povm = povm_for(models, schemes, discriminator, leakage or None) if leakage else product
    return ReadoutStage(tuple(rates), tuple(schemes), tuple(models), discriminator, povm, product, leakage)


# ---- the intrinsic budget of Section 9.6 -----------------------------------------------------------------------------------


def intrinsic_budget(device: Device, sched: Schedule, selection: SpaceSelection) -> dict[str, float]:
    """The closed-form error scales reported beside the result (Section 9.6): per entangling gate the residual displacement
    eps_ent = sum_{i,m} |alpha_{i,m}|^2 (2 nbar_m + 1) of the played waveform, the n = 0-referenced Debye-Waller loss of its
    modes, the off-resonant carrier scale (Omega_peak/nu_min)^2 and the frozen spectators' chi; per single-qubit pulse the
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
        out[f"{gate.gate_id}.frozen_chi_rad"] = chi_frozen
        out[f"{gate.gate_id}.beat_phase_tilt"] = tilt
        total += eps_ent + dw + carrier + tilt
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
    """The effective number of independent shots behind a histogram assembled from several dynamical samples (Section 3.4):
    with S samples of M_s shots and per-sample frequencies p_s of a bitstring, Var(p_hat) = s^2_between/S + mean[p_s(1 - p_s)/M_s]/S
    and n_eff = p(1 - p)/Var(p_hat); the minimum over the bitstrings seen. One sample returns the shot count."""
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
        between = float(np.var(ps, ddof=1)) / s_count
        within = float(np.mean(ps * (1.0 - ps) / ms)) / s_count
        var = between + within
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
    register_state: qt.Qobj
    readout: ReadoutStage
    outcome: ReadoutOutcome
    table: CalibrationTable
    qubit_shifts_hz: dict[int, float]
    notes: tuple[str, ...] = field(default_factory=tuple)


_LAST_RECORD: dict[int, RunRecord] = {}


def last_record(result: Result) -> RunRecord:
    """The RunRecord behind a Result of this process (compile report, schedule, selection, traces, readout stage)."""
    return _LAST_RECORD[id(result)]


def run(
    circuit: Circuit,
    device: Device,
    shots: int,
    *,
    table: CalibrationTable | None = None,
    t0_s: float = 0.0,
    shot_period_s: float | None = None,
    samples: int | None = None,
    level: Literal["JOINT_EXACT", "GATE_LOCAL", "auto"] = "auto",
    seed: int = 0,
    options: SolverOptions | None = None,
    gate_drives: Mapping[int, GateDrive] | None = None,
    entangling_drives: Mapping[int, GateDrive] | None = None,
    builder_options: BuilderOptions | None = None,
    readout: ReadoutMode = "fast",
    discriminator: Discriminator | None = None,
    space: HilbertSpace | None = None,
    caps: Mapping[int, int] | None = None,
    channels: Sequence[CollapseOp] = (),
    entangler: Literal["ms", "zz"] = "ms",
    parallel: bool = False,
    keep_final_state: bool = False,
    calibrate_kwargs: Mapping[str, Any] | None = None,
    noise: bool = True,
    internal_levels: int = 2,
    crosstalk_suppression: CrosstalkSuppression = "none",
    stark_compensation: bool = True,
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
    simulated experiments (``calibrate(surrogate=False)``) drives the machine with its own errors (Sections 7.3, 7.5).
    """
    if shots <= 0:
        raise ValueError("shots must be positive")
    opts = options or SolverOptions()
    drives = dict(gate_drives) if gate_drives is not None else default_gate_drives(device)
    ent_drives = dict(entangling_drives) if entangling_drives is not None else drives
    notes: list[str] = []
    # 1. compile
    report = compile_with_report(circuit, device, entangler=entangler)
    compiled = report.circuit
    # 2. calibrate (surrogate, cached per device and seed, Section 7.5) when no table is given
    if table is None:
        from qutip_trap.calibration.cache import cached_surrogate

        kw = dict(calibrate_kwargs or {})
        kw.setdefault("pairs", compiled.entangling_pairs())
        sur = cached_surrogate(
            device,
            seed=seed,
            t0_s=t0_s,
            gate_drives=drives,
            entangling_drives=ent_drives,
            options=opts,
            builder_options=builder_options,
            caps=caps,
            **kw,
        )
        table = sur.table
        notes.extend(sur.notes)
    elif not table.is_current_for(device.hash()):
        notes.append(
            "calibration table fitted for another device configuration (hash mismatch): played as given, never regenerated "
            "silently (Section 7.5)"
        )
    # 3. schedule
    sched = schedule(
        compiled,
        device,
        table,
        gate_drives=drives,
        entangling_drives=ent_drives,
        t0_s=0.0,
        parallel=parallel,
        crosstalk_suppression=crosstalk_suppression,
        stark_compensation=stark_compensation,
    )
    # 4. preparation (the physics of the recipe) and the space
    cooling_pair = _raman_pair_hint(ent_drives)
    prep_run = run_preparation(device, recipe_of(device, raman_pair=cooling_pair))
    if device.preparation is None:
        notes.append("preparation recipe inferred by prep.recipe.standard_recipe (the device carries none)")
    notes.extend(prep_run.notes)
    n_ions = device.crystal.n_ions
    if internal_levels < 2:
        raise ValueError("internal_levels is at least 2")
    if space is None:
        selection = select_space(
            device, sched, opts, nbar=prep_run.nbar, caps=caps, ion_dims=[int(internal_levels)] * n_ions
        )
    else:
        selection = SpaceSelection(
            space,
            {
                m: ("resolved" if space.mode_class(m) == "resolved" else "frozen")
                for m in range(len(device.crystal.modes))
            },
            {},
            {m: float(prep_run.nbar.get(m, 0.0)) for m in range(len(device.crystal.modes))},
            ("space supplied by the caller",),
        )
    joint_space = selection.space
    levels = level_maps(device, joint_space)
    if levels:
        notes.append(
            "register factors with leakage levels: "
            + "; ".join(f"ion {i}: {', '.join(m.labels)}" for i, m in levels.items())
        )
    ok, dim, nnz = within_budget(joint_space, opts)
    resolved_level: FidelityLevel = "JOINT_EXACT" if ok else "GATE_LOCAL"
    if level == "GATE_LOCAL" or (level == "auto" and resolved_level == "GATE_LOCAL"):
        raise NotImplementedError(
            f"joint dimension {dim} and drive non-zeros {nnz} against the guards ({opts.joint_dimension_max}, {opts.nnz_max}): "
            f"the run routes to GATE_LOCAL, which is {M9A}"
        )
    if level == "JOINT_EXACT" and not ok:
        notes.append(f"JOINT_EXACT forced above the Section 11.5 guards (dimension {dim}, non-zeros {nnz})")
    seeds = SeedSpec(int(seed))
    state0 = prepare(device, joint_space, table, quiet_sample(0), seeds, preparation=prep_run, levels=levels)
    # 5. the branches of the initial mixture
    probs_int = internal_probabilities(state0, joint_space)
    from qutip_trap.light.raman import lamb_dicke_parameters

    coupled_frozen: set[int] = set()
    for pulse in sched.pulses:
        dk = pulse.drive.delta_k(device.beams)
        if float(np.linalg.norm(dk)) == 0.0:
            continue
        for ion in pulse.drive.ions:
            etas, _ = lamb_dicke_parameters(device, ion, dk)
            coupled_frozen.update(m for m in joint_space.frozen if abs(etas[m]) > 1e-12)
    mode_nbar = {m.mode: float(state0.motional.nbar.get(m.mode, 0.0)) for m in joint_space.resolved}
    mode_nbar.update({m: float(state0.motional.nbar.get(m, 0.0)) for m in sorted(coupled_frozen)})
    branches, dropped_weight = enumerate_branches(probs_int, mode_nbar, opts.branch_weight_min)
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
        if entry is None or entry.status == "uncalibrated":
            shifts[i] = 0.0
            notes.append(
                f"ion {i}: the table has no qubit frequency; the frame is taken at the true transition"
            )
        else:
            shifts[i] = float(f_true - entry.value)
    # 7. timing (Section 7.5) and the dynamical samples (Section 3.4; M7)
    stage = readout_stage(device, table, discriminator=discriminator, levels=levels or None)
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
    # 8. evolve every branch of every sample
    engine = JointExactEngine(
        builder_options=builder_options,
        store_per_segment=2,
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
    integrators: list[str] = []
    approximations: list[str] = []
    methods: list[str] = []
    n_traj_max = 1
    total_weight = sum(b.weight for b in branches)
    register_states: list[qt.Qobj] = []
    for smp in samples_seq:
        rho_int = np.zeros((d_int, d_int), dtype=complex)
        for k, br in enumerate(branches):
            fock_res = {m: n for m, n in br.fock.items() if joint_space.mode_class(m) == "resolved"}
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
            sample_b = NoiseSample(
                sample_id=smp.sample_id, values=values, ou_grids=dict(smp.ou_grids), t_s=smp.t_s
            )
            tr = engine.run_pulses(device, sched, st, joint_space, sample_b, seeds, opts)
            traces_all.append(tr)
            rho_int += (br.weight / total_weight) * np.asarray(tr.final.internal.full())
            for m, v in tr.boundary_population.items():
                boundary[m] = max(boundary.get(m, 0.0), float(v))
            rep = engine.last_report
            if rep is not None:
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
        register_states.append(qt.Qobj(rho_int, dims=dims_int))
    rho_register = sum((r for r in register_states[1:]), register_states[0]) / len(register_states)
    # 9. readout per sample on its register state, the collision process per shot (Section 6.7)
    register_space = HilbertSpace(tuple(joint_space.ion_dims), (), None, ())
    run_state = RunState.nominal(n_ions)
    collisions = device.noise.collisions if noise else None
    coll_rates: dict[int, float] = {}
    if collisions is not None and collisions.pressure_pa > 0.0:
        coll_rates = {
            i: collision_rate_per_ion(collisions, float(device.crystal.masses_kg[i])) for i in range(n_ions)
        }
    measured = tuple(sorted(compiled.measure)) if compiled.measure else tuple(range(n_ions))
    bits_kept: list[np.ndarray] = []
    heralds_kept: list[int] = []
    posteriors_kept: list[np.ndarray] = []
    records_kept: list[list[int]] = []
    bits_per_sample: list[np.ndarray] = []
    discarded = 0
    prep_duration = prep_run.duration_s
    for s_idx, (smp, rho_s) in enumerate(zip(samples_seq, register_states)):
        n_s = counts_per_sample[s_idx]
        if n_s == 0:
            bits_per_sample.append(np.zeros((0, len(measured)), dtype=np.uint8))
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
            shots=n_s,
            sample=smp.sample_id,
            trajectory=0,
            first_shot=first_shots[s_idx],
            leakage=stage.leakage or None,
            mode=readout,
            povm=stage.povm if readout == "fast" else None,
            keep_records=(readout == "full"),
        )
        sample_bits: list[np.ndarray] = []
        for j in range(n_s):
            shot = first_shots[s_idx] + j
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
                        if ev.time_s >= prep_duration:
                            keep = False  # the crystal melted during the sequence; the Doppler stage recools only before it
                    elif ev.outcome == "reorder":
                        keep = False
                        order = list(run_state.order)
                        pos = order.index(ev.ion) if ev.ion in order else 0
                        other = pos + 1 if pos + 1 < n_ions else pos - 1
                        if 0 <= other < n_ions and other != pos:
                            order[pos], order[other] = order[other], order[pos]
                        run_state = RunState(tuple(order), run_state.dark, run_state.lost, run_state.events)
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
            if not keep:
                discarded += 1
                continue
            bits_kept.append(row)
            sample_bits.append(row)
            heralds_kept.append(herald)
            if outcome.posteriors is not None:
                posteriors_kept.append(np.asarray(outcome.posteriors[j]))
            if readout == "full" and outcome.records is not None:
                records_kept.append([rec.total for rec in outcome.records[j]])
        bits_per_sample.append(
            np.asarray(sample_bits, dtype=np.uint8).reshape(-1, len(measured))
            if sample_bits
            else np.zeros((0, len(measured)), dtype=np.uint8)
        )
    bits = np.asarray(bits_kept, dtype=np.uint8).reshape(-1, len(measured))
    counts, probabilities = aggregate(bits)
    n_eff = effective_sample_size(bits_per_sample)
    error_bars = binomial_error_bars(probabilities, n_eff)
    spam: dict[str, tuple[float, float]] = {}
    for i, (eps_b, eps_d) in enumerate(stage.product.per_ion_errors()):
        spam[f"q{i}"] = (float(eps_b), float(eps_d))
        spam[f"q{i}.state_preparation"] = (float(prep_run.preparation_error(i)), 0.0)
    photon_records = np.array(records_kept, dtype=int) if (readout == "full" and records_kept) else None
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
    approximations.append(
        "SPAM definition: readout (eps_B, eps_D) of the threshold discriminator at zero crosstalk (Section 13 row "
        "'Readout figure of merit'); state preparation 1 - P(target) of the optical pump (Section 4.2.6)"
    )
    if readout == "fast":
        approximations.append(
            f"readout fast path: the {'register-wide confusion' if stage.leakage else 'product POVM'} applied to the joint outcome "
            "(Section 5.7)"
        )
    approximations.extend(device.hardware.describe())
    diagnostics = Diagnostics(
        level="JOINT_EXACT",
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
        trajectories=len(branches) * n_traj_max,
        shots_per_sample=int(shots // n_samples),
        effective_sample_size=n_eff,
        root_seed=int(seed),
        calibration=table,
        approximations=tuple(approximations) + tuple(notes) + tuple(selection.notes),
        intrinsic_budget=intrinsic_budget(device, sched, selection),
        dropped_branch_weight=dropped_weight,
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
    )
    return result


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
