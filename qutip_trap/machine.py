"""``Machine``: a trapped-ion computer as a client sees it (docs/api_proposal.md Section 4.2; docs/api_implementation_plan.md
1.3; 0.2.0).

The executor of the ladder: the physical ``Device`` (rung 4) with the roles its beams play, the ``CalibrationTable`` it runs
on (None: the closed-form surrogate of Section 7.5, cached per device), the three option objects ``Physics``, ``Numerics``
and ``Readout`` (``qutip_trap.options``) and the level policy, in one frozen record whose methods walk down the rungs:
``run`` (a ``Result``), ``compile`` (rung 1), ``schedule`` (rung 2: compile, calibrate and schedule without integrating,
IonQ's dry run), ``engine`` (rung 3), ``calibrated`` (the same machine with its table pinned), ``estimate`` (the level, the
space and a wall-time guess before anything is integrated) and ``hash`` (the identity a run record stores). ``run``
is ``qutip_trap.run.pipeline.execute`` on the machine (the pipeline ``run`` delegates to since 0.3.0); ``submit`` (0.4.0)
is the same run in a worker process behind a ``Job``, with ``spec`` the ``RunSpec`` it records; ``error_model`` (0.3.0)
the inverse direction. Variants are ``dataclasses.replace(machine, ...)``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

from qutip_trap.hashing import canonical_digest
from qutip_trap.options import Numerics, Physics, Readout
from qutip_trap.run.levels import FidelityLevel, LevelDecision, decide_level
from qutip_trap.run.space import ModeClass3

if TYPE_CHECKING:
    from qutip_trap.benchmarks.error_model import ErrorModel
    from qutip_trap.control.compiler import Circuit, CompileReport
    from qutip_trap.control.schedule import Schedule
    from qutip_trap.control.table import CalibrationTable
    from qutip_trap.device.model import Device
    from qutip_trap.dynamics.engine import JointExactEngine
    from qutip_trap.hilbert.space import HilbertSpace
    from qutip_trap.run.results import Progress, Result
    from qutip_trap.run.spec import Job, RunSpec

CalibrationMethod = Literal["closed_form", "experiments"]
"""``Machine.calibrated`` and ``calibration.calibrate``: the closed-form surrogate or the simulated experiments."""

COST_FIXED_S = 0.02
"""Section 11.2's fitted per-segment constant a of cost = a + b x elements x evaluations."""
COST_PER_NONZERO_S = 0.58e-9
"""Section 11.2's CSR constant b: seconds per drive-operator non-zero per right-hand-side evaluation."""
EVALUATIONS_PER_PULSE_SECOND = 1.5e9
"""Section 11.2: 1 to 2 x 10^5 right-hand-side evaluations per 100 us pulse with dop853 (the midpoint)."""
TOMOGRAPHY_INPUTS_PER_STEP = 4
"""GATE_LOCAL: the Pi d_i basis kets of a two-ion step the isometry route propagates (Section 5.4)."""


@dataclass(frozen=True)
class Estimate:
    """What a run would do before anything is integrated (``Machine.estimate``): the level and why, the declared joint
    space with the class of every mode, its dimension and drive-operator non-zeros, the pulse counts and the schedule's
    length, and a wall-time guess from the Section 11.2 cost model (an order of magnitude: the constants predate the
    factorized kernel and the rotating frame of the 2026-09-09 performance pass, so a run is usually faster)."""

    level: FidelityLevel
    reason: str
    space: HilbertSpace
    mode_class: dict[int, ModeClass3]
    dimension: int
    nnz: int
    n_pulses: int
    n_entangling: int
    duration_s: float
    """The schedule's length, seconds."""
    wall_time_s: float
    notes: tuple[str, ...] = ()


def _wall_time_guess(dimension: int, nnz: int, sched: Schedule, level: FidelityLevel) -> float:
    """Section 11.2: cost = a + b x (elements per evaluation) x evaluations per segment, CSR elements = the drive operator's
    non-zeros, summed over the pulses; a GATE_LOCAL walk plays each gate on a two-ion local space (the same resolved modes,
    the pair's 2^2 factor) and propagates the tomography inputs, so its guess is per gate on that space."""
    total = 0.0
    n_ions_joint = max(1, round(dimension / max(1, nnz / max(1, dimension)) ** 0 if False else 1))
    del n_ions_joint  # the joint numbers below already carry the ion count
    for pulse in sched.pulses:
        duration = max(0.0, pulse.t_end_s - pulse.t_start_s)
        evaluations = EVALUATIONS_PER_PULSE_SECOND * duration
        if level is FidelityLevel.GATE_LOCAL:
            k = len(pulse.drive.ions)
            local_nnz = nnz * (k * 2**k) / max(1, _ions_of(dimension, nnz))
            total += TOMOGRAPHY_INPUTS_PER_STEP * (
                COST_FIXED_S + COST_PER_NONZERO_S * local_nnz * evaluations
            )
        else:
            total += COST_FIXED_S + COST_PER_NONZERO_S * nnz * evaluations
    return float(total)


def _ions_of(dimension: int, nnz: int) -> float:
    """N 2^N from the joint numbers: nnz = N 2^N Pi d_m^2 and dimension = 2^N Pi d_m (Section 11.2), so nnz/dimension =
    N Pi d_m; the ratio is what the local-space scaling divides by."""
    return max(1.0, nnz / max(1, dimension)) if dimension else 1.0


@dataclass(frozen=True)
class Machine:
    """A trapped-ion computer as a client sees it: the physical device with the roles its beams play, the calibration it
    runs on and the policy that turns a circuit into a ``Result``. Immutable; derive variants with ``dataclasses.replace``
    (``replace(machine, level=FidelityLevel.GATE_LOCAL)``, ``replace(machine, physics=Physics(noise=False))``)."""

    device: Device
    """Rung 4: the apparatus, with ``Device.roles`` naming which beams play which part."""
    table: CalibrationTable | None = None
    """The calibration the scheduler reads; None builds the closed-form surrogate at run time, cached per device and seed."""
    physics: Physics = Physics()
    """Which effects are simulated."""
    numerics: Numerics = Numerics()
    """How the integration is done."""
    readout: Readout = Readout()
    """How the photon record is read."""
    level: FidelityLevel = FidelityLevel.AUTO
    """JOINT_EXACT inside the Section 11.5 guards, GATE_LOCAL above them, or either forced."""
    name: str = ""
    """A label for the record; not part of ``hash()``."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "level", FidelityLevel(self.level))
        if isinstance(self.physics, Mapping):
            object.__setattr__(self, "physics", Physics.from_mapping(self.physics))
        if isinstance(self.numerics, Mapping):
            object.__setattr__(self, "numerics", Numerics.from_mapping(self.numerics))
        if isinstance(self.readout, Mapping):
            object.__setattr__(self, "readout", Readout.from_mapping(self.readout))

    # ---- rung 0 -------------------------------------------------------------------------------------------------------

    def run(
        self,
        circuit: Circuit,
        shots: int,
        *,
        seed: int = 0,
        keep_final_state: bool = False,
        progress: Callable[[Progress], None] | None = None,
    ) -> Result:
        """Compile, calibrate, schedule, prepare, evolve and read out ``circuit`` for ``shots`` (Section 3.4):
        ``qutip_trap.run.pipeline.execute`` on this machine, with its table, level and option objects; ``seed`` is the root
        of every keyed stream and ``progress`` is called per pulse, per branch, per sample and per readout (``Progress``).
        The function ``qutip_trap.run.job.run`` is this method with the machine built from its keyword arguments."""
        from qutip_trap.run.pipeline import execute

        return execute(self, circuit, shots, seed=seed, keep_final_state=keep_final_state, progress=progress)

    def submit(
        self,
        circuit: Circuit,
        shots: int,
        *,
        seed: int = 0,
        keep_final_state: bool = False,
        label: str = "",
    ) -> Job:
        """``run`` in a worker process behind a ``Job`` (docs/api_implementation_plan.md 3.1; 0.4.0): ``job.status()``,
        ``job.progress`` (the latest ``Progress``), ``job.result()`` (the same ``Result`` ``run`` returns at this seed),
        ``job.record()`` (the ``RunRecord`` behind it) and ``job.cancel()`` (stops within one pulse when the engines run
        in-process); ``job.spec`` is the ``RunSpec`` of the call, with ``label`` the caller's name for it."""
        from qutip_trap.run.spec import submit

        return submit(self, circuit, shots, seed=seed, keep_final_state=keep_final_state, label=label)

    def spec(
        self,
        circuit: Circuit,
        shots: int,
        *,
        seed: int = 0,
        keep_final_state: bool = False,
        label: str = "",
    ) -> RunSpec:
        """The ``RunSpec`` of ``run(circuit, shots, seed=seed, keep_final_state=keep_final_state)`` on this machine (0.4.0):
        the frozen, JSON-serialisable record of the request and the policy, with this machine's hash."""
        from qutip_trap.run.spec import RunSpec

        return RunSpec.of(self, circuit, shots, seed=seed, keep_final_state=keep_final_state, label=label)

    # ---- rung 1 and 2 -------------------------------------------------------------------------------------------------

    def compile(self, circuit: Circuit) -> CompileReport:
        """Standard gates to native gates with phase tracking, every block and the whole circuit verified (Section 7.2)."""
        from qutip_trap.control.compiler import compile_report

        return compile_report(circuit, self.device, entangler=self.physics.entangler)

    def schedule(self, circuit: Circuit, *, seed: int = 0) -> Schedule:
        """The compile-calibrate-schedule prefix of ``run`` (``run/pipeline.py``): the pulses with absolute times, the played
        gates and the measurement event, nothing integrated; the table is this machine's, or the cached surrogate at ``seed``."""
        from qutip_trap.run.pipeline import compile_calibrate_schedule

        return compile_calibrate_schedule(circuit, self.device, seed=seed, **self._prefix_kwargs()).schedule

    def _prefix_kwargs(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "t0_s": self.physics.t0_s,
            "options": self.numerics.to_solver_options(self.physics),
            "builder_options": self.physics.builder,
            "caps": self.numerics.truncation.caps,
            "entangler": self.physics.entangler,
            "parallel": self.numerics.parallel.addressing,
            "crosstalk_suppression": self.physics.crosstalk_suppression,
            "stark_compensation": self.physics.stark_compensation,
            "internal_levels": self.physics.internal_levels,
        }

    def calibrated(
        self, method: CalibrationMethod = "closed_form", *, seed: int = 0, **scans: Any
    ) -> Machine:
        """This machine with its table pinned: ``calibration.calibrate(self, method=method, seed=seed, **scans)`` (the
        closed-form surrogate with exact spot checks, Section 7.5's default, or the simulated experiments, M8) and its
        report's table on the record; ``scans`` are that function's scan settings (``pairs``, ``detection_records``,
        ``detection_windows_s``, ``experiments``, ...). ``calibrate`` itself returns the whole ``CalibrationReport``."""
        from qutip_trap.calibration import calibrate

        report = calibrate(self, method=method, seed=seed, **scans)
        return replace(self, table=report.table)

    # ---- before running ------------------------------------------------------------------------------------------------

    def estimate(self, circuit: Circuit, *, seed: int = 0) -> Estimate:
        """What a run would cost before anything is integrated: the schedule, the space Section 5.2 would declare for it,
        the level the guards resolve to (and why), and the Section 11.2 wall-time guess; the app's budgets and a caller
        deciding between running at once and a background job read this."""
        from qutip_trap.prep.recipe import recipe_of, run_preparation
        from qutip_trap.run.job import _raman_pair_hint
        from qutip_trap.run.pipeline import compile_calibrate_schedule
        from qutip_trap.run.space import select_space

        prefix = compile_calibrate_schedule(circuit, self.device, seed=seed, **self._prefix_kwargs())
        device = prefix.device
        opts = prefix.options
        prep_run = run_preparation(
            device, recipe_of(device, raman_pair=_raman_pair_hint(prefix.entangling_drives))
        )
        tr = self.numerics.truncation
        notes: list[str] = list(prefix.notes)
        if tr.space is None:
            selection = select_space(
                device,
                prefix.schedule,
                opts,
                nbar=prep_run.nbar,
                caps=tr.caps,
                ion_dims=[int(self.physics.internal_levels)] * device.crystal.n_ions,
                enr=tr.enr_group,
            )
            space = selection.space
            mode_class = dict(selection.mode_class)
            notes.extend(selection.notes)
        else:
            space = tr.space
            mode_class = {m: _class_of(space, m) for m in range(len(device.crystal.modes))}
            notes.append("space supplied by the caller")
        decision: LevelDecision = decide_level(device, prefix.compiled, opts, space=space)
        level = decision.level if self.level is FidelityLevel.AUTO else self.level
        reason = (
            decision.reason
            if self.level is FidelityLevel.AUTO
            else f"{level.value} forced by the caller; level='auto' would choose {decision.reason}"
        )
        sched = prefix.schedule
        return Estimate(
            level=level,
            reason=reason,
            space=space,
            mode_class=mode_class,
            dimension=decision.dimension,
            nnz=decision.nnz,
            n_pulses=prefix.report.n_pulses,
            n_entangling=prefix.report.n_entangling,
            duration_s=float(sched.pulses_end_s),
            wall_time_s=_wall_time_guess(decision.dimension, decision.nnz, sched, level),
            notes=tuple(notes),
        )

    # ---- rung 3 -------------------------------------------------------------------------------------------------------

    @property
    def engine(self) -> JointExactEngine:
        """The JOINT_EXACT engine a run of this machine builds (Section 5.4): its Hamiltonian builder options, extra
        channels, device channels and hardware chain from ``physics``, its table from the machine; ``run_pulses`` takes the
        ``SolverOptions`` of ``numerics.to_solver_options(physics)``."""
        from qutip_trap.dynamics.engine import JointExactEngine

        return JointExactEngine(
            builder_options=self.physics.builder,
            channels=tuple(self.physics.extra_channels),
            device_channels=bool(self.physics.noise),
            hardware_chain=bool(self.physics.hardware_chain),
            table=self.table,
        )

    # ---- identity and the later phases -----------------------------------------------------------------------------------

    def hash(self) -> str:
        """The identity a run record stores: the device digest, the roles (which the device digest leaves out), the table's
        digest and the three option objects with the level; ``name`` is not part of it."""
        table = None if self.table is None else canonical_digest(self.table)
        return canonical_digest(
            (
                "Machine",
                self.device.hash(),
                self.device.roles,
                table,
                self.physics,
                self.numerics,
                self.readout,
                self.level.value,
            )
        )

    def error_model(self, *, qubits: Sequence[int] | None = None) -> ErrorModel:
        """The phenomenological summary of this machine (``benchmarks.error_model``; docs/api_implementation_plan.md 2.6):
        per native gate kind the average gate infidelity of its GATE_LOCAL channel and its duration, the depolarizing
        weights, the SPAM errors and the noise rates, with the exporters to IonQ's, Quantinuum's and the QDK estimator's
        vocabularies; ``qubits`` restricts the characterised ions."""
        from qutip_trap.benchmarks.error_model import error_model

        return error_model(self, qubits=qubits)

    def specs(self) -> str:
        """The derived quantities of the device as a readable report with their provenance ids (``Device.specs``), then
        the roles the machine resolved (which beams play the gates), whether a table is pinned and the level policy."""
        roles = self.device.roles.resolve(self.device)
        lines = [self.device.specs(), "", "machine"]
        lines.append(f"  gate drives = {dict(sorted(roles.gate.items()))}")
        lines.append(f"  entangling drives = {dict(sorted(roles.entangling.items()))}")
        lines.append(
            f"  detection beam = {roles.detection}"
            + (f"  (inferred: {', '.join(roles.inferred)})" if roles.inferred else "")
        )
        lines.append(
            f"  table = {'pinned (' + self.table.device_hash[:12] + ', seed ' + str(self.table.seed) + ')' if self.table is not None else 'the cached closed-form surrogate'}"
        )
        lines.append(f"  level = {self.level.value}")
        return "\n".join(lines)


def laboratory_kwargs(machine: Machine, kw: Mapping[str, Any]) -> tuple[Device, dict[str, Any]]:
    """The device and the keyword arguments an experiment reads for a call on ``machine``: the machine supplies ``table``
    (its pinned table), ``options`` (``numerics.to_solver_options(physics)``) and ``builder_options`` (``physics.builder``)
    wherever the call did not pass them."""
    out = dict(kw)
    if machine.table is not None:
        out.setdefault("table", machine.table)
    out.setdefault("options", machine.numerics.to_solver_options(machine.physics))
    if machine.physics.builder is not None:
        out.setdefault("builder_options", machine.physics.builder)
    return machine.device, out


def _class_of(space: HilbertSpace, mode: int) -> ModeClass3:
    cls = space.mode_class(mode)
    if cls == "resolved":
        return "resolved"
    if cls == "enr":
        return "enr"
    return "dropped" if mode in space.dropped else "frozen"
