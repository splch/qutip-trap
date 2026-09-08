"""GATE_LOCAL: the fidelity level above the joint-exact range (PLAN.md Sections 5.4, 9.8, 11.3 item 8, 11.5; milestone M9a).

For each step of the schedule, a group of time-overlapping pulses (one entangling gate's segments on its pair, one carrier pulse
with its crosstalk neighbours, parallel single-qubit gates) or an idle interval, the executor

1. builds the exact joint space of the addressed ions plus every neighbour receiving crosstalk light above
   ``SolverOptions.crosstalk_threshold`` and of the modes the contribution criterion of Section 11.3 resolves for that step
   (``step_space``: the same classes as JOINT_EXACT's ``run.space``, from the closed-form integrals of the played waveforms
   evaluated at the TRACKED occupations, with caps from the loop radius, the tracked state's populated range and the Section
   5.1.1 margin; the remaining modes are frozen spectators whose Fock populations enter as weighted Debye-Waller branches);
2. initializes the motional part from the tracked motional model (the reduced density matrix of every mode a previous step
   resolved, the thermal state of the others), evolves every one of the prod_i d_i^2 tomography inputs exactly through the
   JOINT_EXACT engine (``dynamics.tomography``), reconstructs the Choi matrix, projects it onto CP and TP and applies the map to
   the register: by Kraus operators on the register density matrix (N <= ``register_dm_max_qubits`` qubits) or by Kraus sampling
   on each member of a pure-state ensemble beyond that (Section 5.4);
3. updates the motional model: the reduced density matrix of every resolved mode as the register-weighted combination of the
   tomography outputs (the outputs are linear in the input, and the register's local marginal is expanded in the input basis),
   the untouched occupations of the others, and reports the residual displacement |alpha_m| per spin eigenstate, the purity
   deficit of the reduced motional state, the frozen modes' off-resonant excitation bound and the crosstalk it dropped;
4. treats idle intervals exactly as JOINT_EXACT does, through the same engine: the one-qubit channel of every ion (its
   quasi-static and sampled offsets, its dephasing) by tomography of the idle schedule, the heating of every tracked mode by
   the master equation on its own factor, and nbar + ndot t for the modes tracked by their occupation alone.

The approximation is what Section 5.4 states: spin-motion and mode-mode correlations left after a step are traced out rather than
carried to the next; every step reports the bound Section 9.8 compares against, sum_m |alpha_m|^2 (2 nbar_m + 1) over its
resolved modes, and the comparison itself is the test suite's. Extractions are cached by (device, step fingerprint, local space,
motional-model fingerprint, noise sample, solver options): the same gate on the same motional state under the same sample costs
nothing twice, and the cost of a fresh one is the prod d_i^2 x branches x n_traj engine runs of Section 11.2.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import qutip as qt

from qutip_trap.control.schedule import GateTarget, PlayedGate, Schedule
from qutip_trap.dynamics.engine import (
    ChannelSummary,
    JointExactEngine,
    MotionalModel,
    SeedSpec,
    SolverOptions,
)
from qutip_trap.dynamics.tomography import (
    TomographyRecord,
    apply_kraus_dm,
    apply_kraus_ket,
    fingerprint_model,
    fingerprint_options,
    fingerprint_pulse,
    fingerprint_sample,
    local_ideal,
)
from qutip_trap.hashing import canonical_digest
from qutip_trap.hilbert.operators import required_margin
from qutip_trap.hilbert.space import HilbertSpace, ModeTruncation
from qutip_trap.run.space import (
    ModeClass3,
    ModeContribution,
    best_contributions,
    cap_for,
    classify,
    frozen_excitation_bounds,
)
from qutip_trap.units import TWO_PI

if TYPE_CHECKING:
    from qutip_trap.control.pulses import Pulse
    from qutip_trap.control.table import CalibrationTable
    from qutip_trap.device.model import Device
    from qutip_trap.dynamics.channels import CollapseOp
    from qutip_trap.dynamics.hamiltonian import BuilderOptions
    from qutip_trap.noise.levels import InternalLevels
    from qutip_trap.noise.sampling import NoiseSample

M9A = "milestone M9a (run/gate_local.py, PLAN.md Section 5.4)"
RegisterKind = Literal["density_matrix", "ensemble"]


# ---- steps ------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GateStep:
    """One unit of the GATE_LOCAL walk: a group of time-overlapping pulses, or an idle interval between groups."""

    kind: Literal["gate", "idle"]
    t_start_s: float
    t_end_s: float
    pulses: tuple[Pulse, ...]
    targets: tuple[GateTarget, ...]
    """The ideal unitaries of the gate pieces whose pulses all lie in this step."""
    ions: tuple[int, ...]
    """The addressed ions (every pulse's ``drive.ions``), sorted."""
    played: tuple[PlayedGate, ...]
    """The entangling gates (as played) whose pulses lie in this step: what the mode selection reads."""

    @property
    def gate_id(self) -> str:
        if self.kind == "idle":
            return f"idle[{self.t_start_s:.9g}, {self.t_end_s:.9g}]"
        ids = [t.gate_id for t in self.targets] or sorted({p.gate_id or "" for p in self.pulses})
        return "+".join(dict.fromkeys(ids))

    @property
    def duration_s(self) -> float:
        return self.t_end_s - self.t_start_s


def gate_steps(schedule: Schedule, *, tolerance_s: float = 1e-12) -> tuple[GateStep, ...]:
    """Split a schedule into gate steps and the idle intervals between them, from the schedule's start (``t0_s``, else min(0,
    the first start) as the engine takes it) to ``pulses_end_s``.

    A step is one gate piece as the scheduler recorded it (every pulse of one ``GateTarget``: the segments of a
    Mølmer-Sørensen waveform on both ions stay together, since tracing the motion out between segments would discard exactly
    the spin-motion correlations the gate builds), merged with every other piece it overlaps in time (parallel single-qubit
    gates on distinct ions); pulses without a target are pieces of their own."""
    ordered = sorted(schedule.pulses, key=lambda p: (p.t_start_s, p.t_end_s))
    target_of: dict[str, int] = {}
    for k, tg in enumerate(schedule.targets):
        for pid in tg.pulse_ids:
            target_of[pid] = k
    pieces: dict[object, list[Pulse]] = {}
    for p in ordered:
        key: object = target_of.get(p.gate_id or "", None)
        if key is None:
            key = ("pulse", p.gate_id, p.t_start_s, tuple(p.drive.ions))
        pieces.setdefault(key, []).append(p)
    spans = sorted(
        ((min(q.t_start_s for q in ps), max(q.t_end_s for q in ps), ps) for ps in pieces.values()),
        key=lambda t: (t[0], t[1]),
    )
    groups: list[list[Pulse]] = []
    group_end = -math.inf
    for s0, s1, ps in spans:
        if groups and s0 < group_end - tolerance_s:
            groups[-1].extend(ps)
            group_end = max(group_end, s1)
        else:
            groups.append(list(ps))
            group_end = s1
    starts = [p.t_start_s for p in schedule.pulses] + [a for a, _ in schedule.idle]
    t0 = float(schedule.t0_s) if schedule.t0_s is not None else min(starts + [0.0])
    t_end = max(schedule.pulses_end_s, t0)
    steps: list[GateStep] = []
    t = t0
    for g in groups:
        gs = min(p.t_start_s for p in g)
        ge = max(p.t_end_s for p in g)
        if gs > t + tolerance_s:
            steps.append(GateStep("idle", t, gs, (), (), (), ()))
        ids = {p.gate_id for p in g if p.gate_id is not None}
        targets = tuple(tg for tg in schedule.targets if tg.pulse_ids and set(tg.pulse_ids) <= ids)
        played = tuple(
            pg
            for pg in schedule.gates
            if any(pid == pg.gate_id or pid.startswith(pg.gate_id + "/") for pid in ids)
        )
        ions = tuple(sorted({int(i) for p in g for i in p.drive.ions}))
        steps.append(GateStep("gate", gs, ge, tuple(g), targets, ions, played))
        t = max(t, ge)
    if t_end > t + tolerance_s:
        steps.append(GateStep("idle", t, t_end, (), (), (), ()))
    return tuple(steps)


# ---- the register -------------------------------------------------------------------------------------------------------------


class Register:
    """The N-qubit register of a GATE_LOCAL run: a density matrix over the ion dimensions, or a pure-state ensemble (Section 5.4).

    A service object with state (the walk advances it step by step), not one of the API's frozen records."""

    def __init__(
        self, dims: Sequence[int], *, dm: np.ndarray | None = None, kets: list[np.ndarray] | None = None
    ) -> None:
        if (dm is None) == (kets is None):
            raise ValueError("a Register is a density matrix or an ensemble of kets")
        self.dims = tuple(int(d) for d in dims)
        self.dm = dm
        self.kets = kets
        self.kraus_draws: list[int] = []

    @property
    def kind(self) -> RegisterKind:
        return "density_matrix" if self.dm is not None else "ensemble"

    @property
    def size(self) -> int:
        return 1 if self.kets is None else len(self.kets)

    @classmethod
    def from_density_matrix(cls, rho: qt.Qobj | np.ndarray, dims: Sequence[int]) -> Register:
        arr = np.asarray(rho.full() if isinstance(rho, qt.Qobj) else rho, dtype=complex)
        return cls(dims, dm=arr)

    @classmethod
    def ensemble_from_populations(
        cls,
        populations: Sequence[np.ndarray | Sequence[float]],
        dims: Sequence[int],
        size: int,
        rng: np.random.Generator,
    ) -> Register:
        """``size`` product basis states drawn from the ions' (diagonal) prepared populations: the pure-state ensemble of a
        register too large for a density matrix."""
        total = int(np.prod(dims))
        kets: list[np.ndarray] = []
        for _ in range(size):
            idx = 0
            for d, p in zip(dims, populations):
                pr = np.asarray(p, dtype=float)
                pr = pr / pr.sum()
                idx = idx * int(d) + int(rng.choice(int(d), p=pr))
            v = np.zeros(total, dtype=complex)
            v[idx] = 1.0
            kets.append(v)
        return cls(dims, kets=kets)

    def marginal(self, factors: Sequence[int]) -> np.ndarray:
        """The reduced density matrix over ``factors`` (in the given order), averaged over the ensemble when it is one."""
        n = len(self.dims)
        fac = [int(f) for f in factors]
        rest = [f for f in range(n) if f not in fac]
        d_loc = int(np.prod([self.dims[f] for f in fac]))
        d_rest = int(np.prod(self.dims)) // d_loc
        if self.dm is not None:
            arr = self.dm.reshape(list(self.dims) + list(self.dims))
            perm = fac + rest + [n + f for f in fac] + [n + f for f in rest]
            a = np.transpose(arr, perm).reshape(d_loc, d_rest, d_loc, d_rest)
            return np.asarray(np.einsum("axbx->ab", a))
        assert self.kets is not None
        out = np.zeros((d_loc, d_loc), dtype=complex)
        for k in self.kets:
            v = np.transpose(k.reshape(list(self.dims)), fac + rest).reshape(d_loc, -1)
            out += v @ v.conj().T
        return np.asarray(out / len(self.kets))

    def apply(
        self,
        kraus: Sequence[np.ndarray],
        factors: Sequence[int],
        rngs: Sequence[np.random.Generator] | None = None,
    ) -> None:
        if self.dm is not None:
            self.dm = apply_kraus_dm(self.dm, kraus, self.dims, factors)
            return
        assert self.kets is not None
        if rngs is None or len(rngs) != len(self.kets):
            raise ValueError("Kraus sampling needs one generator per ensemble member")
        new: list[np.ndarray] = []
        for k, rng in zip(self.kets, rngs):
            v, a = apply_kraus_ket(k, kraus, self.dims, factors, rng)
            new.append(v)
            self.kraus_draws.append(a)
        self.kets = new

    def states(self) -> list[tuple[float, qt.Qobj]]:
        """Weighted states for the readout stage: one density matrix, or the ensemble's kets at equal weight."""
        dims_q = [list(self.dims), list(self.dims)]
        if self.dm is not None:
            return [(1.0, qt.Qobj(self.dm, dims=dims_q))]
        assert self.kets is not None
        w = 1.0 / len(self.kets)
        return [
            (w, qt.Qobj(k.reshape(-1, 1), dims=[list(self.dims), [1] * len(self.dims)])) for k in self.kets
        ]

    def density_matrix(self) -> qt.Qobj:
        """The register density matrix (the ensemble average for an ensemble; large for many qubits)."""
        if self.dm is not None:
            return qt.Qobj(self.dm, dims=[list(self.dims), list(self.dims)])
        assert self.kets is not None
        acc = np.zeros((len(self.kets[0]), len(self.kets[0])), dtype=complex)
        for k in self.kets:
            acc += np.outer(k, k.conj())
        return qt.Qobj(acc / len(self.kets), dims=[list(self.dims), list(self.dims)])


# ---- the gate-local space of a step -----------------------------------------------------------------------------------------------


def _pulse_area_rad(pulse: Pulse) -> float:
    peak = 0.0
    for tone in pulse.drive.tones:
        env = tone.envelope_hz
        if callable(env):
            grid = np.linspace(0.0, pulse.duration_s, 101)
            peak = max(peak, float(np.max(np.abs([float(env(x)) for x in grid]))))
        elif isinstance(env, np.ndarray):
            peak = max(peak, float(np.max(np.abs(env))))
        else:
            peak = max(peak, abs(float(env)))
    return TWO_PI * peak * pulse.duration_s


def _coupled(device: Device, pulses: Sequence[Pulse], ions: Sequence[int]) -> set[int]:
    from qutip_trap.light.raman import lamb_dicke_parameters

    out: set[int] = set()
    for pulse in pulses:
        dk = pulse.drive.delta_k(device.beams)
        if float(np.linalg.norm(dk)) == 0.0:
            continue
        for ion in ions:
            if ion in pulse.drive.ions or ion in pulse.drive.crosstalk:
                etas, _ = lamb_dicke_parameters(device, ion, dk)
                out.update(m for m, e in etas.items() if abs(e) > 1e-12)
    return out


def _populated_of(rho: qt.Qobj, tail: float) -> int:
    p = np.real(np.diag(np.asarray(rho.full())))
    above = np.cumsum(p[::-1])[::-1]
    for n in range(p.size):
        if n + 1 >= p.size or above[n + 1] < tail:
            return n
    return int(p.size - 1)


@dataclass(frozen=True)
class StepSpace:
    space: HilbertSpace
    mode_class: dict[int, ModeClass3]
    contribution: dict[int, ModeContribution]
    frozen_coupled: tuple[int, ...]
    dropped_crosstalk: float
    """sum over the dropped neighbours of sin^2(|eps| theta/2): the rotation the leaked light would have produced."""
    notes: tuple[str, ...]


def step_space(
    device: Device,
    step: GateStep,
    model: MotionalModel,
    options: SolverOptions,
    ion_dims: Mapping[int, int],
    *,
    caps: Mapping[int, int] | None = None,
    d_min: int = 6,
    d_max: int | None = None,
) -> StepSpace:
    """The gate-local space of a gate step (Section 5.4): the addressed ions plus the neighbours above the crosstalk threshold,
    and the modes the Section 11.3 criterion resolves for the step's played gates at the TRACKED occupations; every other mode is
    frozen. The cap of a resolved mode follows the loop radius and the tracked state's populated range plus the Section 5.1.1
    margin.

    ``d_max`` = None reads ``options.mode_dimension_max`` (default 64) and a clamp is named in ``notes`` with the range the
    rule asked for (M9a audit D1)."""
    addressed = set(step.ions)
    neighbours: set[int] = set()
    dropped_xt = 0.0
    for p in step.pulses:
        theta = _pulse_area_rad(p)
        for j, eps in p.drive.crosstalk.items():
            if abs(complex(eps)) >= options.crosstalk_threshold:
                neighbours.add(int(j))
            else:
                dropped_xt += math.sin(abs(complex(eps)) * theta / 2.0) ** 2
    ions_local = tuple(sorted(addressed | neighbours))
    d_ceiling = int(options.mode_dimension_max if d_max is None else d_max)
    n_modes = len(device.crystal.modes)
    nbar_now = {m: float(model.nbar.get(m, 0.0)) for m in range(n_modes)}
    best = best_contributions(device, step.played, nbar_now)
    coupled = _coupled(device, step.pulses, ions_local)
    classes: dict[int, ModeClass3] = {}
    for m in range(n_modes):
        classes[m] = classify(
            best.get(m),
            coupled=m in coupled,
            freeze_alpha_max=options.freeze_alpha_max,
            freeze_chi_max_rad=options.freeze_chi_max_rad,
        )
    notes: list[str] = []
    resolved: list[ModeTruncation] = []
    for m in range(n_modes):
        if classes[m] != "resolved":
            continue
        c = best[m]
        # the same boundary threshold the engine's margin check reads (Section 5.5's per-test override; M9a audit B4)
        tail = float(options.boundary_population_max)
        tr = cap_for(c.radius, nbar_now[m], c.eta_max, d_min=d_min, d_max=d_ceiling, tail=tail)
        n_hi = tr.expected_n_range[1]
        tracked = model.reduced.get(m)
        if tracked is not None:
            # the tracked state's populated range (not thermal any more after a gate) plus the coherent excursion of this gate
            n_tracked = _populated_of(tracked, options.boundary_population_max)
            excursion = int(math.ceil(c.radius**2 + 2.0 * c.radius)) if c.radius > 0.0 else 0
            n_hi = max(n_hi, n_tracked + excursion)
        d_want = max(tr.d, n_hi + 1 + required_margin(c.eta_max), d_min)
        d = min(d_want, d_ceiling)
        if caps is not None and m in caps:
            d = int(caps[m])
        elif d_want > d:
            notes.append(
                f"mode {m}: the cap rule asks for d = {d_want} (expected occupation up to n = {n_hi}) but "
                f"mode_dimension_max = {d_ceiling} clamps it to d = {d} (declared range up to n = {min(n_hi, d - 1)}); the "
                "Section 5.1.1 oracle check and the Section 5.5 margin check are evaluated over the clamped range"
            )
        resolved.append(ModeTruncation(m, d, (0, min(n_hi, d - 1)), tr.eta_max))
        notes.append(
            f"mode {m}: resolved at d = {d} (|alpha|^2(2n+1) = {c.alpha2_weighted:.2e}, |chi| = {c.chi_rad:.3e} rad, radius "
            f"{c.radius:.3f}, nbar {nbar_now[m]:.3g})"
        )
    frozen = tuple(m for m in range(n_modes) if classes[m] != "resolved")
    space = HilbertSpace(
        tuple(int(ion_dims[i]) for i in ions_local), tuple(resolved), None, frozen, ions=ions_local
    )
    frozen_coupled = tuple(sorted(m for m in coupled if classes[m] != "resolved"))
    if neighbours:
        notes.append(
            f"crosstalk neighbours {sorted(neighbours)} join the gate-local space (|eps| >= {options.crosstalk_threshold:g})"
        )
    if dropped_xt > 0.0:
        notes.append(f"crosstalk below the threshold dropped: rotation bound {dropped_xt:.2e}")
    return StepSpace(space, classes, best, frozen_coupled, dropped_xt, tuple(notes))


# ---- the report -----------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GateLocalStep:
    """What one step of the GATE_LOCAL walk did (the first dynamical sample's walk is reported step by step)."""

    gate_id: str
    kind: Literal["gate", "idle"]
    t_start_s: float
    t_end_s: float
    ions: tuple[int, ...]
    """The gate-local space's ions (addressed plus crosstalk neighbours)."""
    space_dims: tuple[int, ...]
    resolved: tuple[int, ...]
    frozen_coupled: tuple[int, ...]
    n_inputs: int
    n_branches: int
    n_traj: int
    method: str
    integrators: tuple[str, ...]
    engine_runs: int
    cache_hit: bool
    cp_residual: float
    tp_residual: float
    summary: ChannelSummary | None
    """The Section 6.8 summary against the step's ideal unitary (None for an idle step)."""
    residual_displacement: dict[int, float]
    """Per resolved mode, max over the spin eigenstates of |<a_m>| after the step (Section 5.4)."""
    residual_bound: float
    """sum_m |alpha_m|^2 (2 nbar_m + 1) over the resolved modes: what Section 9.8 compares the JOINT_EXACT disagreement against."""
    purity_deficit: dict[int, float]
    """1 - Tr rho_m^2 of the reduced motional state the step leaves, per resolved mode."""
    nbar_after: dict[int, float]
    frozen_excitation: dict[int, float]
    """Per frozen coupled mode, the Section 5.2 off-resonant excitation bound of this step."""
    dropped_crosstalk: float
    boundary_population: dict[int, float]
    margin_reached: dict[int, int]
    notes: tuple[str, ...]
    workers: int = 1
    """Processes this step's engine runs actually used (M9b audit B10): the tomography inputs spread over a parallel map, or
    the trajectories inside one engine run. 1 = in-process, which is every carrier step (no resolved mode in its space)."""


@dataclass(frozen=True)
class GateLocalReport:
    """The GATE_LOCAL diagnostics of a run (Section 5.4: the approximation made measurable and visible)."""

    steps: tuple[GateLocalStep, ...]
    """The first sample's walk, step by step."""
    residual_bound_total: float
    """sum over the gate steps of the residual-displacement bound (the first sample)."""
    frozen_excitation_total: float
    dropped_crosstalk_total: float
    register: RegisterKind
    ensemble_size: int
    largest_local_dimension: int
    engine_runs: int
    """Engine calls over every sample and step (cache misses only)."""
    cache_hits: int
    summaries: dict[str, ChannelSummary]
    """Per gate step id (the first sample), the Section 6.8 channel summary."""
    motional_after: dict[str, dict[int, float]]
    """Per step id (the first sample), nbar per tracked mode after the step."""
    notes: tuple[str, ...] = field(default_factory=tuple)
    workers: int = 1
    """The largest number of processes any step of any sample actually used (M9b audit B10). The walk itself iterates its
    quasi-static samples serially (Section 11.3 item 9 asks for them to be spread too; that is unimplemented and this number
    says so), so a run whose every step is a carrier reports 1 however many workers were configured."""

    @property
    def discrepancy_bound(self) -> float:
        """The bound a JOINT_EXACT comparison of the final probabilities is held to (Section 9.8): the residual displacement,
        the frozen spectators' off-resonant excitation and the dropped crosstalk, summed over the steps."""
        return self.residual_bound_total + self.frozen_excitation_total + self.dropped_crosstalk_total


# ---- the executor ---------------------------------------------------------------------------------------------------------------


_TOMOGRAPHY_CACHE: dict[str, TomographyRecord] = {}
_CACHE_MAX = 512


def clear_gate_local_cache() -> None:
    _TOMOGRAPHY_CACHE.clear()


@dataclass(frozen=True)
class EngineSetup:
    """What the JOINT_EXACT engines of the walk are built with (the same knobs ``run`` gives its joint engine)."""

    builder_options: BuilderOptions | None = None
    channels: tuple[CollapseOp, ...] = ()
    qubit_shifts_hz: dict[int, float] = field(default_factory=dict)
    device_channels: bool = False
    levels_by_ion: dict[int, InternalLevels] | None = None
    hardware_chain: bool = True
    table: CalibrationTable | None = None

    def engine(self) -> JointExactEngine:
        return JointExactEngine(
            builder_options=self.builder_options,
            store_per_segment=2,
            channels=tuple(self.channels),
            qubit_shifts_hz=dict(self.qubit_shifts_hz),
            device_channels=self.device_channels,
            levels_by_ion=self.levels_by_ion,
            hardware_chain=self.hardware_chain,
            table=self.table,
        )

    def fingerprint(self) -> tuple[object, ...]:
        table_fp: object = None
        if self.table is not None:
            table_fp = (
                self.table.device_hash,
                self.table.seed,
                dict(self.table.rabi),
                dict(self.table.stark),
                dict(self.table.crosstalk),
                dict(self.table.crosstalk_phase),
            )
        return (
            self.builder_options,
            tuple(c.channel for c in self.channels),
            tuple(sorted(self.qubit_shifts_hz.items())),
            self.device_channels,
            None if self.levels_by_ion is None else {i: m.labels for i, m in self.levels_by_ion.items()},
            self.hardware_chain,
            table_fp,
        )


def _cache_key(
    device: Device,
    step: GateStep,
    space: HilbertSpace,
    model: MotionalModel,
    frozen_coupled: Sequence[int],
    sample: NoiseSample,
    options: SolverOptions,
    setup: EngineSetup,
) -> str:
    modes = [t.mode for t in space.resolved] + list(frozen_coupled)
    return canonical_digest(
        (
            device.hash(),
            tuple(fingerprint_pulse(p) for p in step.pulses),
            (step.t_start_s, step.t_end_s),
            (space.ion_dims, tuple((t.mode, t.d) for t in space.resolved), space.frozen, space.ion_labels),
            fingerprint_model(model, modes),
            fingerprint_sample(sample),
            fingerprint_options(options),
            setup.fingerprint(),
        )
    )


def _purity_deficit(rho: np.ndarray) -> float:
    return float(max(1.0 - np.real(np.trace(rho @ rho)), 0.0))


def _nbar(rho: np.ndarray) -> float:
    return float(np.real(np.sum(np.arange(rho.shape[0]) * np.real(np.diag(rho)))))


def _idle_step(
    device: Device,
    step: GateStep,
    register: Register,
    model: MotionalModel,
    sample: NoiseSample,
    seeds: SeedSpec,
    options: SolverOptions,
    setup: EngineSetup,
    ion_dims: Sequence[int],
    heating_rates: Mapping[int, float],
    step_index: int,
) -> tuple[MotionalModel, GateLocalStep, int]:
    """Free evolution over an idle interval: per ion its exact one-qubit channel, per tracked mode its master equation, per
    occupation-tracked mode nbar + ndot t (the heating that applies whether or not a mode is carried, Section 11.3 item 2)."""
    n_modes = len(device.crystal.modes)
    sched = Schedule((), ((step.t_start_s, step.t_end_s),), (), {}, t0_s=step.t_start_s)
    runs = 0
    workers = 1
    integrators: list[str] = []
    method = "sesolve"
    notes: list[str] = []
    for q, d in enumerate(ion_dims):
        space_q = HilbertSpace((int(d),), (), None, tuple(range(n_modes)), ions=(q,))
        engine = setup.engine()
        rec = engine.tomography(device, sched, space_q, model, sample, seeds, options)
        runs += rec.engine_runs
        workers = max(workers, rec.workers)
        for i in rec.integrators:
            if i not in integrators:
                integrators.append(i)
        if rec.method != "sesolve":
            method = rec.method
        rngs = None
        if register.kind == "ensemble":
            rngs = [
                np.random.default_rng(seeds.child(sample.sample_id, k, 0, 0, f"kraus[{step_index}][{q}]"))
                for k in range(register.size)
            ]
        register.apply(rec.kraus(), (q,), rngs)
    reduced: dict[int, qt.Qobj] = {}
    nbar: dict[int, float] = dict(model.nbar)
    dt = step.duration_s
    for m in range(n_modes):
        rho_m = model.reduced.get(m)
        if rho_m is None:
            if setup.device_channels and heating_rates.get(m, 0.0) > 0.0:
                nbar[m] = nbar.get(m, 0.0) + float(heating_rates[m]) * dt
            continue
        d_m = int(rho_m.shape[0])
        space_m = HilbertSpace(
            (int(ion_dims[0]),),
            (ModeTruncation(m, d_m, (0, d_m - 1), 1e-3),),
            None,
            tuple(k for k in range(n_modes) if k != m),
            ions=(0,),
        )
        state = space_m.initial_state(qt.basis(int(ion_dims[0]), 0), states={m: rho_m}, thermal={})
        engine = setup.engine()
        traces = engine.run_pulses(device, sched, state, space_m, sample, seeds, options)
        runs += 1
        rep = engine.last_report
        assert rep is not None
        workers = max(workers, rep.workers)
        for seg in rep.segments:
            if seg.integrator not in integrators:
                integrators.append(seg.integrator)
        if rep.method != "sesolve":
            method = rep.method
        out = traces.final.motional.reduced[m]
        reduced[m] = out
        nbar[m] = float(np.real(qt.expect(qt.num(d_m), out)))
    new_model = MotionalModel(
        reduced=reduced, nbar=nbar, frozen=tuple(m for m in range(n_modes) if m not in reduced)
    )
    report = GateLocalStep(
        gate_id=step.gate_id,
        kind="idle",
        t_start_s=step.t_start_s,
        t_end_s=step.t_end_s,
        ions=tuple(range(len(ion_dims))),
        space_dims=tuple(int(d) for d in ion_dims),
        resolved=tuple(sorted(reduced)),
        frozen_coupled=(),
        n_inputs=sum(int(d) ** 2 for d in ion_dims),
        n_branches=1,
        n_traj=1,
        method=method,
        integrators=tuple(integrators),
        engine_runs=runs,
        cache_hit=False,
        cp_residual=0.0,
        tp_residual=0.0,
        summary=None,
        residual_displacement={},
        residual_bound=0.0,
        purity_deficit={m: _purity_deficit(np.asarray(r.full())) for m, r in reduced.items()},
        nbar_after={m: nbar[m] for m in sorted(nbar)},
        frozen_excitation={},
        dropped_crosstalk=0.0,
        boundary_population={},
        margin_reached={},
        notes=tuple(notes),
        workers=workers,
    )
    return new_model, report, runs


def _gate_step(
    device: Device,
    step: GateStep,
    register: Register,
    model: MotionalModel,
    sample: NoiseSample,
    seeds: SeedSpec,
    options: SolverOptions,
    setup: EngineSetup,
    ion_dims: Sequence[int],
    caps: Mapping[int, int] | None,
    step_index: int,
) -> tuple[MotionalModel, GateLocalStep, int, bool, HilbertSpace]:
    """One gate step: the local space, the (cached) tomography, the map on the register and the motional update."""
    n_modes = len(device.crystal.modes)
    dims_by_ion = {i: int(d) for i, d in enumerate(ion_dims)}
    sel = step_space(device, step, model, options, dims_by_ion, caps=caps)
    space = sel.space
    key = _cache_key(device, step, space, model, sel.frozen_coupled, sample, options, setup)
    rec = _TOMOGRAPHY_CACHE.get(key)
    hit = rec is not None
    if rec is None:
        engine = setup.engine()
        rec = engine.tomography(device, tuple(step.pulses), space, model, sample, seeds, options)
        if len(_TOMOGRAPHY_CACHE) >= _CACHE_MAX:
            _TOMOGRAPHY_CACHE.clear()
        _TOMOGRAPHY_CACHE[key] = rec
    factors = tuple(int(q) for q in space.ion_labels)
    rho_local = register.marginal(factors)
    rngs = None
    if register.kind == "ensemble":
        rngs = [
            np.random.default_rng(seeds.child(sample.sample_id, k, 0, 0, f"kraus[{step_index}]"))
            for k in range(register.size)
        ]
    register.apply(rec.kraus(), factors, rngs)
    # the motional update: the register-weighted combination of the tomography outputs (Section 5.4 (b))
    mot, alpha = rec.motional_for(rho_local)
    reduced: dict[int, qt.Qobj] = dict(model.reduced)
    nbar: dict[int, float] = dict(model.nbar)
    purity: dict[int, float] = {}
    for m, arr in mot.items():
        reduced[m] = qt.Qobj(arr, dims=[[arr.shape[0]], [arr.shape[0]]])
        nbar[m] = _nbar(arr)
        purity[m] = _purity_deficit(arr)
    resid = rec.residual_displacement()
    # Section 9.8's bound sum_m |alpha_m|^2 (2 nbar_m + 1) at the occupation the mode HAD when the step started: the
    # post-step nbar of the same step's own output made the reported bound depend on the gate's heating (conservative for a
    # heating gate, an under-estimate for a cooling one) and so not reproducible from the step's inputs (M9a audit B12)
    bound = float(sum(v**2 * (2.0 * float(model.nbar.get(m, 0.0)) + 1.0) for m, v in resid.items()))
    excitation, guard = frozen_excitation_bounds(device, step.pulses, sel.frozen_coupled, nbar)
    ideal = local_ideal(space.ion_labels, space.ion_dims, step.targets)
    summary = rec.summary(ideal)
    notes = list(sel.notes) + list(rec.notes) + list(guard)
    if ideal is None:
        notes.append(
            "no GateTarget covers this step's pulses: the channel summary has no ideal (infidelities NaN)"
        )
    new_model = MotionalModel(
        reduced=reduced, nbar=nbar, frozen=tuple(m for m in range(n_modes) if m not in reduced)
    )
    report = GateLocalStep(
        gate_id=step.gate_id,
        kind="gate",
        t_start_s=step.t_start_s,
        t_end_s=step.t_end_s,
        ions=space.ion_labels,
        space_dims=tuple(rec.space.dims),
        resolved=tuple(t.mode for t in rec.space.resolved),
        frozen_coupled=sel.frozen_coupled,
        n_inputs=len(rec.labels),
        n_branches=rec.branches,
        n_traj=rec.n_traj,
        method=rec.method,
        integrators=rec.integrators,
        engine_runs=rec.engine_runs,
        cache_hit=hit,
        cp_residual=rec.cp_residual,
        tp_residual=rec.tp_residual,
        summary=summary,
        residual_displacement=resid,
        residual_bound=bound,
        purity_deficit=purity,
        nbar_after={m: nbar[m] for m in sorted(nbar)},
        frozen_excitation=excitation,
        dropped_crosstalk=sel.dropped_crosstalk,
        boundary_population=dict(rec.boundary_population),
        margin_reached=dict(rec.margin_reached),
        notes=tuple(dict.fromkeys(notes)),
        # a cache hit ran nothing, so it used no worker (M9b audit B10)
        workers=1 if hit else rec.workers,
    )
    return new_model, report, (0 if hit else rec.engine_runs), hit, rec.space


def evolve_gate_local(
    device: Device,
    sched: Schedule,
    samples: Sequence[NoiseSample],
    seeds: SeedSpec,
    options: SolverOptions,
    *,
    register0: qt.Qobj,
    nbar0: Mapping[int, float],
    ion_dims: Sequence[int],
    setup: EngineSetup,
    caps: Mapping[int, int] | None = None,
) -> tuple[list[list[tuple[float, qt.Qobj]]], GateLocalReport, list[MotionalModel]]:
    """The GATE_LOCAL walk of Section 5.4 over every dynamical sample: (per sample the weighted register states for the readout
    stage, the report, per sample the final motional model)."""
    n_ions = device.crystal.n_ions
    n_modes = len(device.crystal.modes)
    steps = gate_steps(sched)
    heating = device.noise.heating_rates_quanta_per_s(device) if setup.device_channels else {}
    kind: RegisterKind = "density_matrix" if n_ions <= options.register_dm_max_qubits else "ensemble"
    populations = [np.real(np.diag(np.asarray(qt.ptrace(register0, [q]).full()))) for q in range(n_ions)]
    out_states: list[list[tuple[float, qt.Qobj]]] = []
    models: list[MotionalModel] = []
    first_steps: list[GateLocalStep] = []
    summaries: dict[str, ChannelSummary] = {}
    motional_after: dict[str, dict[int, float]] = {}
    runs_total = 0
    hits_total = 0
    workers_max = 1
    largest = 0
    bound_total = 0.0
    frozen_total = 0.0
    xt_total = 0.0
    notes: list[str] = []
    for s_idx, smp in enumerate(samples):
        if kind == "density_matrix":
            register = Register.from_density_matrix(register0, ion_dims)
        else:
            rng = np.random.default_rng(seeds.child(smp.sample_id, 0, 0, 0, "register_ensemble"))
            register = Register.ensemble_from_populations(
                populations, ion_dims, options.register_ensemble, rng
            )
        model = MotionalModel(
            reduced={},
            nbar={m: float(nbar0.get(m, 0.0)) for m in range(n_modes)},
            frozen=tuple(range(n_modes)),
        )
        for k, step in enumerate(steps):
            if step.kind == "idle":
                model, rep, runs = _idle_step(
                    device, step, register, model, smp, seeds, options, setup, ion_dims, heating, k
                )
                runs_total += runs
                workers_max = max(workers_max, rep.workers)
            else:
                model, rep, runs, hit, space_used = _gate_step(
                    device, step, register, model, smp, seeds, options, setup, ion_dims, caps, k
                )
                runs_total += runs
                hits_total += int(hit)
                workers_max = max(workers_max, rep.workers)
                largest = max(largest, space_used.dimension)
                if s_idx == 0:
                    bound_total += rep.residual_bound
                    frozen_total += sum(v for v in rep.frozen_excitation.values() if math.isfinite(v))
                    xt_total += rep.dropped_crosstalk
                    if rep.summary is not None:
                        summaries[rep.gate_id] = rep.summary
            if s_idx == 0:
                first_steps.append(rep)
                motional_after[rep.gate_id] = dict(rep.nbar_after)
        out_states.append(register.states())
        models.append(model)
    if kind == "ensemble":
        notes.append(
            f"register carried as a pure-state ensemble of {options.register_ensemble} members ({n_ions} qubits above "
            f"register_dm_max_qubits = {options.register_dm_max_qubits}); maps applied by Kraus sampling (Section 5.4)"
        )
    report = GateLocalReport(
        steps=tuple(first_steps),
        residual_bound_total=bound_total,
        frozen_excitation_total=frozen_total,
        dropped_crosstalk_total=xt_total,
        register=kind,
        ensemble_size=options.register_ensemble if kind == "ensemble" else 1,
        largest_local_dimension=largest,
        engine_runs=runs_total,
        cache_hits=hits_total,
        summaries=summaries,
        motional_after=motional_after,
        notes=tuple(notes),
        workers=workers_max,
    )
    return out_states, report, models


__all__ = [
    "M9A",
    "EngineSetup",
    "GateLocalReport",
    "GateLocalStep",
    "GateStep",
    "Register",
    "StepSpace",
    "clear_gate_local_cache",
    "evolve_gate_local",
    "gate_steps",
    "step_space",
]
