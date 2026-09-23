"""``Machine``: a trapped-ion computer as a client sees it.

One frozen record holds the ``Device``, the ``CalibrationTable`` it runs on (None: the cached closed-form surrogate),
the ``Physics``, ``Numerics`` and ``Readout`` options and the level policy; its methods walk down the rungs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from qutip_trap.hashing import canonical_digest
from qutip_trap.options import Numerics, Physics, Readout
from qutip_trap.run.levels import FidelityLevel, LevelDecision, decide_level
from qutip_trap.run.space import ModeClass3

if TYPE_CHECKING:
    from qutip_trap.benchmarks.error_model import ErrorModel
    from qutip_trap.calibration import CalibrationMethod
    from qutip_trap.control.compiler import Circuit, CompileReport
    from qutip_trap.control.schedule import Schedule
    from qutip_trap.control.table import CalibrationTable
    from qutip_trap.device.model import Device
    from qutip_trap.dynamics.engine import JointExactEngine
    from qutip_trap.hilbert.space import HilbertSpace
    from qutip_trap.run.results import Progress, Result
    from qutip_trap.run.spec import Job, RunSpec

COST_FIXED_S = 0.02
"""The fitted per-segment constant a (s) of the cost model cost = a + b x elements x evaluations."""
COST_PER_NONZERO_S = 0.58e-9
"""The CSR constant b: seconds per drive-operator non-zero per right-hand-side evaluation."""
EVALUATIONS_PER_PULSE_SECOND = 1.5e9
"""Right-hand-side evaluations per second of pulse with dop853 (the midpoint of 1 to 2 x 10^5 per 100 us)."""
TOMOGRAPHY_INPUTS_PER_STEP = 4
"""GATE_LOCAL: the Pi d_i basis kets of a two-ion step that the isometry route propagates."""


@dataclass(frozen=True)
class Estimate:
    """What a run would do before anything is integrated (``Machine.estimate``); ``wall_time_s`` is an
    order-of-magnitude guess from the cost model, usually an overestimate."""

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
    """cost = a + b x (drive-operator non-zeros) x (right-hand-side evaluations), summed over the pulses; under
    GATE_LOCAL each pulse is costed on its local space, once per tomography input."""
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
    """nnz / dimension (N Pi d_m for nnz = N 2^N Pi d_m^2, dimension = 2^N Pi d_m), the local scaling's divisor."""
    return max(1.0, nnz / max(1, dimension)) if dimension else 1.0


@dataclass(frozen=True)
class Machine:
    """A trapped-ion computer as a client sees it: the device, the calibration it runs on and the policy that turns a
    circuit into a ``Result``. Immutable; derive variants with ``dataclasses.replace``."""

    device: Device
    """The apparatus; ``Device.roles`` names which beams play which part."""
    table: CalibrationTable | None = None
    """The calibration the scheduler reads; None builds the closed-form surrogate at run time, cached per device and seed."""
    physics: Physics = Physics()
    """Which effects are simulated."""
    numerics: Numerics = Numerics()
    """How the integration is done."""
    readout: Readout = Readout()
    """How the photon record is read."""
    level: FidelityLevel = FidelityLevel.AUTO
    """AUTO (JOINT_EXACT inside the size guards, GATE_LOCAL above them) or a forced level."""
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
        """Compile, calibrate, schedule, prepare, evolve and read out ``circuit``; ``seed`` roots every keyed stream and
        ``progress`` is called per pulse, branch, sample and readout."""
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
        """``run`` in a worker process behind a ``Job`` (the same ``Result`` at the same seed); ``label`` names it."""
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
        """The frozen, JSON-serialisable ``RunSpec`` of this ``run`` call on this machine, with the machine's hash."""
        from qutip_trap.run.spec import RunSpec

        return RunSpec.of(self, circuit, shots, seed=seed, keep_final_state=keep_final_state, label=label)

    # ---- rung 1 and 2 -------------------------------------------------------------------------------------------------

    def compile(self, circuit: Circuit) -> CompileReport:
        """Standard gates to native gates with phase tracking, every block and the whole circuit verified."""
        from qutip_trap.control.compiler import compile_report

        return compile_report(circuit, self.device, entangler=self.physics.entangler)

    def schedule(self, circuit: Circuit, *, seed: int = 0) -> Schedule:
        """The compile-calibrate-schedule prefix of ``run``, nothing integrated; the table is this machine's, or the
        cached surrogate at ``seed``."""
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
        """This machine with the table of ``calibration.calibrate(self, method=method, seed=seed, **scans)`` pinned."""
        from qutip_trap.calibration import calibrate

        report = calibrate(self, method=method, seed=seed, **scans)
        return replace(self, table=report.table)

    # ---- before running ------------------------------------------------------------------------------------------------

    def estimate(self, circuit: Circuit, *, seed: int = 0) -> Estimate:
        """What a run would cost before anything is integrated: the space, the level (and why) and a wall-time guess."""
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
        """The JOINT_EXACT engine a run of this machine builds (``run_pulses`` takes ``numerics.to_solver_options``)."""
        from qutip_trap.dynamics.engine import JointExactEngine

        return JointExactEngine(
            builder_options=self.physics.builder,
            channels=tuple(self.physics.extra_channels),
            device_channels=bool(self.physics.noise),
            hardware_chain=bool(self.physics.hardware_chain),
            table=self.table,
        )

    # ---- identity and summaries ---------------------------------------------------------------------------------------

    def hash(self) -> str:
        """The identity a run record stores: device digest, roles, table digest, options and level (not ``name``)."""
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
        """The phenomenological error model of this machine over ``qubits`` (None: every ion)."""
        from qutip_trap.benchmarks.error_model import error_model

        return error_model(self, qubits=qubits)

    def specs(self) -> str:
        """``Device.specs``, then the resolved beam roles, the table and the level, as a readable report."""
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


DRIVE_KEYWORDS = ("gate_drive", "gate_drives", "entangling_drives")
"""Keywords the laboratory refuses: a drive is declared once, in ``Device.roles``."""


def laboratory_kwargs(machine: Machine, kw: Mapping[str, Any]) -> tuple[Device, dict[str, Any]]:
    """The device and keyword arguments an experiment reads for a call on ``machine``: the machine fills ``table``,
    ``options`` and ``builder_options`` the call left out."""
    out = dict(kw)
    for key in DRIVE_KEYWORDS:
        if key in out:
            raise TypeError(f"{key!r} is not an experiment argument: declare the drive in Device.roles")
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


__all__ = [
    "COST_FIXED_S",
    "COST_PER_NONZERO_S",
    "DRIVE_KEYWORDS",
    "EVALUATIONS_PER_PULSE_SECOND",
    "TOMOGRAPHY_INPUTS_PER_STEP",
    "Estimate",
    "Machine",
    "laboratory_kwargs",
]
