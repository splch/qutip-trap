"""GATE_LOCAL: the fidelity level above the joint-exact range (PLAN.md Section 5.4).

The schedule is split into steps: a group of time-overlapping pulses (one entangling gate's segments on its pair, a carrier
pulse, parallel single-qubit gates) or an idle interval. For each gate step the walk

1. builds the exact joint space of the addressed ions, every neighbour whose crosstalk reaches ``crosstalk_threshold``, and
   the modes the contribution criterion resolves at the TRACKED occupations (``step_space``); the rest are frozen;
2. starts the motional part from the tracked model (the reduced state of every mode a previous step resolved, thermal
   otherwise), extracts the step's channel by tomography through the JOINT_EXACT engine, projects it onto CPTP and applies
   it to the register: a density matrix up to ``register_dm_max_qubits`` qubits, a Kraus-sampled pure-state ensemble above;
3. updates the motional model from the register-weighted tomography outputs and reports the residual displacement, the
   purity deficit, the frozen modes' off-resonant excitation and the dropped crosstalk.

An idle step applies each ion's exact one-qubit channel and heats the tracked modes. Spin-motion and mode-mode correlations
left after a step are traced out (Section 5.4's approximation); every step reports the bound Section 9.8 compares against.
Extractions are cached by (device, step, local space, motional model, noise sample, options, engine setup), the one-ion idle
channels by (device, ion, duration, sample, options, setup), and one engine serves the whole walk.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal

import numpy as np
import qutip as qt

from qutip_trap.control.pulses import fingerprint_pulse
from qutip_trap.control.schedule import GateTarget, PlayedGate, Schedule
from qutip_trap.dynamics.engine import (
    MARGIN_LEAKAGE_FRACTION,
    ChannelSummary,
    JointExactEngine,
    MotionalModel,
    SeedSpec,
    SolverOptions,
    required_margin_under,
)
from qutip_trap.dynamics.tomography import (
    TomographyRecord,
    TomographyRoute,
    apply_kraus_dm,
    apply_kraus_ket,
    fingerprint_model,
    fingerprint_options,
    fingerprint_sample,
    local_ideal,
)
from qutip_trap.hashing import canonical_digest
from qutip_trap.hilbert.operators import displacement_leakage
from qutip_trap.hilbert.space import HilbertSpace, ModeTruncation
from qutip_trap.hilbert.truncation import warn_cap_clamped
from qutip_trap.run.space import (
    _D_MIN,
    ModeClass3,
    ModeContribution,
    _peak_rabi_rad_s,
    best_contributions,
    cap_for,
    classify,
    frozen_excitation_bounds,
)

if TYPE_CHECKING:
    from qutip_trap.control.pulses import Pulse
    from qutip_trap.control.table import CalibrationTable
    from qutip_trap.device.model import Device
    from qutip_trap.dynamics.channels import CollapseOp
    from qutip_trap.dynamics.hamiltonian import BuilderOptions
    from qutip_trap.noise.levels import InternalLevels
    from qutip_trap.noise.sampling import NoiseSample

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
    """Split a schedule into gate steps and the idle intervals between them, from the schedule's start (``t0_s``, else
    min(0, the first start)) to ``pulses_end_s``. A step is one gate piece as the scheduler recorded it (every pulse of one
    ``GateTarget``: a Mølmer-Sørensen waveform's segments stay together, since tracing the motion out between them would
    discard the spin-motion correlations the gate builds), merged with every piece it overlaps in time."""
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
    """The register of a GATE_LOCAL run: a density matrix over the ion dimensions, or a pure-state ensemble (Section 5.4).
    A service object with state (the walk advances it step by step), not a record."""

    def __init__(
        self, dims: Sequence[int], *, dm: np.ndarray | None = None, kets: list[np.ndarray] | None = None
    ) -> None:
        if (dm is None) == (kets is None):
            raise ValueError("a Register is a density matrix or an ensemble of kets")
        self.dims = tuple(int(d) for d in dims)
        self.dm = dm
        self.kets = kets

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
        """``size`` product basis states drawn from the ions' (diagonal) prepared populations."""
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
        if self.dm is not None:
            # one einsum over the 2n-axis view: a traced factor shares its row and column letter, so only the diagonal of
            # the traced part is read
            letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            if 2 * n > len(letters):
                raise ValueError(f"a register of {n} factors exceeds the einsum alphabet")
            row = [letters[f] for f in range(n)]
            col = [letters[f] if f in rest else letters[n + f] for f in range(n)]
            kept = "".join(row[f] for f in fac) + "".join(col[f] for f in fac)
            arr = self.dm.reshape(list(self.dims) + list(self.dims))
            return np.asarray(np.einsum("".join(row) + "".join(col) + "->" + kept, arr).reshape(d_loc, d_loc))
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
        self.kets = [apply_kraus_ket(k, kraus, self.dims, factors, rng)[0] for k, rng in zip(self.kets, rngs)]

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


def _coupled(device: Device, pulses: Sequence[Pulse], ions: Sequence[int]) -> set[int]:
    """The modes a pulse couples to (|eta| > 1e-12) on any of ``ions`` it addresses or leaks crosstalk onto."""
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
) -> StepSpace:
    """The gate-local space of a gate step (Section 5.4): the addressed ions plus the neighbours above the crosstalk
    threshold, and the modes the Section 11.3 criterion resolves for the step's played gates at the TRACKED occupations,
    with caps from the loop radius, the tracked state's populated range and the margin; every other mode is frozen. A cap
    above ``options.mode_dimension_max`` is clamped, warned and named in ``notes``."""
    addressed = set(step.ions)
    neighbours: set[int] = set()
    dropped_xt = 0.0
    for p in step.pulses:
        theta = _peak_rabi_rad_s(p) * p.duration_s
        for j, eps in p.drive.crosstalk.items():
            if abs(complex(eps)) >= options.crosstalk_threshold:
                neighbours.add(int(j))
            else:
                dropped_xt += math.sin(abs(complex(eps)) * theta / 2.0) ** 2
    ions_local = tuple(sorted(addressed | neighbours))
    d_ceiling = int(options.mode_dimension_max)
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
    tail = float(options.boundary_population_max)
    for m in range(n_modes):
        if classes[m] != "resolved":
            continue
        c = best[m]
        tr = cap_for(c.radius, nbar_now[m], c.eta_max, d_min=_D_MIN, d_max=d_ceiling, tail=tail)
        n_hi = tr.expected_n_range[1]
        tracked = model.reduced.get(m)
        if tracked is not None:
            # the tracked state's populated range (not thermal any more after a gate) plus this gate's coherent excursion
            n_tracked = _populated_of(tracked, options.boundary_population_max)
            excursion = int(math.ceil(c.radius**2 + 2.0 * c.radius)) if c.radius > 0.0 else 0
            n_hi = max(n_hi, n_tracked + excursion)
        # the margin above the populated range: the Section 5.1.1 fixture, or the margin derived for the declared element
        # tolerance (``margin_element_tol``; the engine's margin check reads the same rule)
        margin = required_margin_under(c.eta_max, options, n_hi)
        d_want = max(n_hi + 1 + margin, _D_MIN)
        d = min(d_want, d_ceiling)
        if caps is not None and m in caps:
            d = int(caps[m])
        elif d_want > d:
            warn_cap_clamped(m, d_want, n_hi, d, d_ceiling)
            notes.append(
                f"mode {m}: the cap rule asks for d = {d_want} (expected occupation up to n = {n_hi}) but "
                f"mode_dimension_max = {d_ceiling} clamps it to d = {d} (declared range up to n = {min(n_hi, d - 1)}); the "
                "Section 5.1.1 oracle check and the Section 5.5 margin check are evaluated over the clamped range"
            )
        resolved.append(
            ModeTruncation(m, d, (0, min(n_hi, d - 1)), tr.eta_max, element_tol=options.margin_element_tol)
        )
        notes.append(
            f"mode {m}: resolved at d = {d} (|alpha|^2(2n+1) = {c.alpha2_weighted:.2e}, |chi| = {c.chi_rad:.3e} rad, radius "
            f"{c.radius:.3f}, nbar {nbar_now[m]:.3g})"
        )
        if options.margin_element_tol is not None:
            leak = displacement_leakage(c.eta_max, n_hi, d - 1 - min(n_hi, d - 1))
            notes.append(
                f"mode {m}: margin {d - 1 - min(n_hi, d - 1)} above n = {min(n_hi, d - 1)} derived for interior elements exact "
                f"to {options.margin_element_tol:.0e} (the Section 5.1.1 fixture is {required_margin_under(c.eta_max, SolverOptions(), n_hi)}); "
                f"one displacement from n = {min(n_hi, d - 1)} leaks {leak:.1e} past the cap, below "
                f"{tail * MARGIN_LEAKAGE_FRACTION:.0e}"
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


REGISTER_STORE_DIM_MAX = 256
"""``GateLocalStep.register_after`` is stored up to this register dimension (eight qubits, one megabyte per step)."""


@dataclass(frozen=True)
class AppliedChannel:
    """One channel the walk applied to the register: the projected Choi matrix (trace 1; ``dynamics.tomography.
    kraus_operators`` gives the Kraus form applied) and the register factors it acts on, in the Choi matrix's factor order."""

    ions: tuple[int, ...]
    choi: np.ndarray


@dataclass(frozen=True)
class GateLocalStep:
    """What one step of the first dynamical sample's walk did."""

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
    """sum_m |alpha_m|^2 (2 nbar_m + 1) over the resolved modes at their occupation before the step (Section 9.8)."""
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
    """Processes this step's engine runs used (1 = in-process, every carrier step and every cache hit)."""
    route: TomographyRoute = "states"
    """How the channel was extracted (``TomographyRecord.route``): "propagator" on an internal-state-only space, "isometry"
    on a unitary step with resolved modes, "states" on a dissipative step."""
    branch_error_bound: float = 0.0
    """2 x the dropped motional-branch weight of the step's tomography: the diamond-norm bound of the branch floor and the
    tail rule (``tomography_dropped_weight_max``)."""
    tolerance_change: float = 0.0
    """The channel's change when its dominant branch is re-integrated ten times tighter than the map-accuracy-keyed
    tolerance (``tomography_tolerance_keyed``); 0 where the tolerance was not keyed."""
    tolerances: tuple[float, float] = (SolverOptions.atol, SolverOptions.rtol)
    """(atol, rtol) the step's engine runs integrated at."""
    element_error: dict[int, float] = field(default_factory=dict)
    """Per resolved mode, the measured maximum error of the exponential's interior elements over the declared range (the
    Section 5.1.1 oracle), the number ``margin_element_tol`` bounds."""
    register_after: np.ndarray | None = None
    """The register density matrix after this step, ion 0 the first factor; None for a pure-state ensemble or above
    ``REGISTER_STORE_DIM_MAX``. The last step's equals ``RunRecord.register_state``."""
    channels: tuple[AppliedChannel, ...] = ()
    """The channels this step applied, in order: a gate step's one map on its local ions, an idle step's one per ion."""


@dataclass(frozen=True)
class GateLocalReport:
    """The GATE_LOCAL diagnostics of a run (Section 5.4)."""

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
    """Gate steps whose extraction came from the cache."""
    summaries: dict[str, ChannelSummary]
    """Per gate step id (the first sample), the Section 6.8 channel summary."""
    notes: tuple[str, ...] = field(default_factory=tuple)
    workers: int = 1
    """The most processes any step used; the quasi-static samples are walked serially."""
    idle_cache_hits: int = 0
    """One-ion idle channels served from the cache over every sample and idle step."""
    branch_error_total: float = 0.0
    """sum over the gate steps of ``GateLocalStep.branch_error_bound`` (the first sample)."""
    tolerance_change_total: float = 0.0
    """sum over the gate steps of ``GateLocalStep.tolerance_change`` (the first sample)."""

    @property
    def discrepancy_bound(self) -> float:
        """The bound a JOINT_EXACT comparison of the final probabilities is held to (Section 9.8): the residual displacement,
        the frozen spectators' excitation, the dropped crosstalk, the dropped motional branches and the keyed tolerance's
        change, summed over the steps."""
        return (
            self.residual_bound_total
            + self.frozen_excitation_total
            + self.dropped_crosstalk_total
            + self.branch_error_total
            + self.tolerance_change_total
        )


# ---- the executor ---------------------------------------------------------------------------------------------------------------


_TOMOGRAPHY_CACHE: dict[str, TomographyRecord] = {}
_CACHE_MAX = 512


def clear_gate_local_cache() -> None:
    _TOMOGRAPHY_CACHE.clear()


@dataclass(frozen=True)
class EngineSetup:
    """What the JOINT_EXACT engines of the walk are built with (the knobs a run gives its joint engine)."""

    builder_options: BuilderOptions | None = None
    channels: tuple[CollapseOp, ...] = ()
    qubit_shifts_hz: dict[int, float] = field(default_factory=dict)
    device_channels: bool = False
    levels_by_ion: dict[int, InternalLevels] | None = None
    table: CalibrationTable | None = None

    def engine(self) -> JointExactEngine:
        return JointExactEngine(
            builder_options=self.builder_options,
            store_per_segment=2,
            channels=tuple(self.channels),
            qubit_shifts_hz=dict(self.qubit_shifts_hz),
            device_channels=self.device_channels,
            levels_by_ion=self.levels_by_ion,
            hardware_chain=True,
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


def _idle_key(
    device: Device,
    ion: int,
    ion_dim: int,
    step: GateStep,
    sample: NoiseSample,
    seeds: SeedSpec,
    options: SolverOptions,
    setup: EngineSetup,
) -> str:
    """The cache key of one ion's idle channel. An idle schedule has no pulse, so nothing motional enters the one-ion
    Hamiltonian and the channel depends on the ion, the duration (rounded to 1e-15 s like the engine's propagator cache),
    the engine setup, the sample's offsets and the options; a sample with a fast trajectory (an OU grid in absolute time)
    adds the absolute window."""
    return canonical_digest(
        (
            "idle-channel",
            device.hash(),
            int(ion),
            int(ion_dim),
            round(float(step.duration_s), 15),
            (step.t_start_s, step.t_end_s) if sample.ou_grids else None,
            fingerprint_sample(sample),
            fingerprint_options(options),
            setup.fingerprint(),
            seeds.root,
        )
    )


def _purity_deficit(rho: np.ndarray) -> float:
    return float(max(1.0 - np.real(np.trace(rho @ rho)), 0.0))


def _nbar(rho: np.ndarray) -> float:
    return float(np.real(np.sum(np.arange(rho.shape[0]) * np.real(np.diag(rho)))))


def _register_snapshot(register: Register) -> np.ndarray | None:
    """``GateLocalStep.register_after``: a copy of the register density matrix when it is one and small enough to keep."""
    if register.dm is None or int(np.prod(register.dims)) > REGISTER_STORE_DIM_MAX:
        return None
    return np.array(register.dm, dtype=complex, copy=True)


def _idle_step(
    device: Device,
    step: GateStep,
    register: Register,
    model: MotionalModel,
    sample: NoiseSample,
    seeds: SeedSpec,
    options: SolverOptions,
    setup: EngineSetup,
    engine: JointExactEngine,
    ion_dims: Sequence[int],
    heating_rates: Mapping[int, float],
    step_index: int,
    snapshot: bool = True,
) -> tuple[MotionalModel, GateLocalStep, int, int]:
    """Free evolution over an idle interval: per ion its exact one-qubit channel (cached, ``_idle_key``), per tracked mode its
    master equation, per occupation-tracked mode nbar + ndot t. Returns (model, report, engine runs, cache hits)."""
    n_modes = len(device.crystal.modes)
    sched = Schedule((), ((step.t_start_s, step.t_end_s),), (), {}, t0_s=step.t_start_s)
    runs = 0
    hits = 0
    workers = 1
    integrators: list[str] = []
    method = "sesolve"
    route: TomographyRoute = "propagator"
    applied: list[AppliedChannel] = []
    for q, d in enumerate(ion_dims):
        space_q = HilbertSpace((int(d),), (), None, tuple(range(n_modes)), ions=(q,))
        key = _idle_key(device, q, int(d), step, sample, seeds, options, setup)
        rec = _TOMOGRAPHY_CACHE.get(key)
        if rec is None:
            rec = engine.tomography(device, sched, space_q, model, sample, seeds, options)
            runs += rec.engine_runs
            workers = max(workers, rec.workers)
            if len(_TOMOGRAPHY_CACHE) >= _CACHE_MAX:
                _TOMOGRAPHY_CACHE.clear()
            _TOMOGRAPHY_CACHE[key] = rec
        else:
            hits += 1
        for i in rec.integrators:
            if i not in integrators:
                integrators.append(i)
        if rec.method != "sesolve":
            method = rec.method
        if rec.route != "propagator":
            route = rec.route
        rngs = None
        if register.kind == "ensemble":
            rngs = [
                np.random.default_rng(seeds.child(sample.sample_id, k, 0, 0, f"kraus[{step_index}][{q}]"))
                for k in range(register.size)
            ]
        register.apply(rec.kraus(), (q,), rngs)
        applied.append(AppliedChannel((q,), np.asarray(rec.choi, dtype=complex)))
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
        # every one-ion channel came from the idle cache (the per-mode master-equation runs are never cached)
        cache_hit=hits == len(ion_dims),
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
        notes=(),
        workers=workers,
        route=route,
        register_after=_register_snapshot(register) if snapshot else None,
        channels=tuple(applied),
    )
    return new_model, report, runs, hits


def _gate_step(
    device: Device,
    step: GateStep,
    register: Register,
    model: MotionalModel,
    sample: NoiseSample,
    seeds: SeedSpec,
    options: SolverOptions,
    setup: EngineSetup,
    engine: JointExactEngine,
    ion_dims: Sequence[int],
    caps: Mapping[int, int] | None,
    step_index: int,
    snapshot: bool = True,
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
    mot, _alpha = rec.motional_for(rho_local)
    reduced: dict[int, qt.Qobj] = dict(model.reduced)
    nbar: dict[int, float] = dict(model.nbar)
    purity: dict[int, float] = {}
    for m, arr in mot.items():
        reduced[m] = qt.Qobj(arr, dims=[[arr.shape[0]], [arr.shape[0]]])
        nbar[m] = _nbar(arr)
        purity[m] = _purity_deficit(arr)
    resid = rec.residual_displacement()
    # Section 9.8's bound at the occupation each mode HAD when the step started, so it is reproducible from the inputs
    bound = float(sum(v**2 * (2.0 * float(model.nbar.get(m, 0.0)) + 1.0) for m, v in resid.items()))
    excitation, guard = frozen_excitation_bounds(device, step.pulses, sel.frozen_coupled, nbar)
    ideal = local_ideal(space.ion_labels, space.ion_dims, step.targets)
    summary = rec.summary(ideal)
    notes = list(sel.notes) + list(rec.notes) + list(guard)
    # the Section 5.1.1 oracle's measured element error per resolved mode, at the eta the cap was derived for
    element_error: dict[int, float] = {}
    for tr in rec.space.resolved:
        c_m = sel.contribution.get(tr.mode)
        eta_m = min(c_m.eta_max, tr.eta_max) if c_m is not None else tr.eta_max
        _asserted, diff, _tol = rec.space.oracle_status(tr.mode, eta_m)
        element_error[tr.mode] = float(diff)
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
        workers=1 if hit else rec.workers,
        route=rec.route,
        branch_error_bound=float(rec.branch_error_bound),
        tolerance_change=float(rec.tolerance_change or 0.0),
        tolerances=rec.tolerances,
        element_error=element_error,
        register_after=_register_snapshot(register) if snapshot else None,
        channels=(AppliedChannel(factors, np.asarray(rec.choi, dtype=complex)),),
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
    progress: Callable[[int, int], None] | None = None,
) -> tuple[list[list[tuple[float, qt.Qobj]]], GateLocalReport, list[MotionalModel]]:
    """The GATE_LOCAL walk over every dynamical sample: (per sample the weighted register states for the readout stage, the
    report, per sample the final motional model); ``progress(done, total)`` is called after every sample's walk."""
    n_ions = device.crystal.n_ions
    n_modes = len(device.crystal.modes)
    steps = gate_steps(sched)
    # the step spaces' margin is derived for the map accuracy unless the caller declared an element tolerance: the cap
    # rule, the tomography's engine and the extraction cache all read the same options
    if options.margin_element_tol is None:
        options = replace(options, margin_element_tol=options.map_accuracy * 1e-5)
    engine = setup.engine()
    heating = device.noise.heating_rates_quanta_per_s(device) if setup.device_channels else {}
    kind: RegisterKind = "density_matrix" if n_ions <= options.register_dm_max_qubits else "ensemble"
    populations = [np.real(np.diag(np.asarray(qt.ptrace(register0, [q]).full()))) for q in range(n_ions)]
    out_states: list[list[tuple[float, qt.Qobj]]] = []
    models: list[MotionalModel] = []
    first_steps: list[GateLocalStep] = []
    summaries: dict[str, ChannelSummary] = {}
    runs_total = 0
    hits_total = 0
    idle_hits_total = 0
    workers_max = 1
    largest = 0
    bound_total = 0.0
    frozen_total = 0.0
    xt_total = 0.0
    branch_total = 0.0
    tolerance_total = 0.0
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
                model, rep, runs, idle_hits = _idle_step(
                    device,
                    step,
                    register,
                    model,
                    smp,
                    seeds,
                    options,
                    setup,
                    engine,
                    ion_dims,
                    heating,
                    k,
                    snapshot=s_idx == 0,
                )
                runs_total += runs
                idle_hits_total += idle_hits
                workers_max = max(workers_max, rep.workers)
            else:
                model, rep, runs, hit, space_used = _gate_step(
                    device,
                    step,
                    register,
                    model,
                    smp,
                    seeds,
                    options,
                    setup,
                    engine,
                    ion_dims,
                    caps,
                    k,
                    snapshot=s_idx == 0,
                )
                runs_total += runs
                hits_total += int(hit)
                workers_max = max(workers_max, rep.workers)
                largest = max(largest, space_used.dimension)
                if s_idx == 0:
                    bound_total += rep.residual_bound
                    frozen_total += sum(v for v in rep.frozen_excitation.values() if math.isfinite(v))
                    xt_total += rep.dropped_crosstalk
                    branch_total += rep.branch_error_bound
                    tolerance_total += rep.tolerance_change
                    if rep.summary is not None:
                        summaries[rep.gate_id] = rep.summary
            if s_idx == 0:
                first_steps.append(rep)
        out_states.append(register.states())
        models.append(model)
        if progress is not None:
            progress(s_idx + 1, len(samples))
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
        notes=tuple(notes),
        workers=workers_max,
        idle_cache_hits=idle_hits_total,
        branch_error_total=branch_total,
        tolerance_change_total=tolerance_total,
    )
    return out_states, report, models
