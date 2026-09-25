"""``Machine``: a Device with the roles its beams play, the calibration table it runs on, the three option objects and the
level policy; ``run`` takes a circuit to a ``Result`` (PLAN.md Section 3.4)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from qutip_trap.hashing import canonical_digest
from qutip_trap.options import Numerics, Physics, Readout
from qutip_trap.run.levels import FidelityLevel, decide_level
from qutip_trap.run.space import ModeClass3

if TYPE_CHECKING:
    from qutip_trap.benchmarks.error_model import ErrorModel
    from qutip_trap.calibration import CalibrationMethod
    from qutip_trap.control.compiler import Circuit, CompileReport
    from qutip_trap.control.schedule import Schedule
    from qutip_trap.control.table import CalibrationTable
    from qutip_trap.device.model import Device
    from qutip_trap.dynamics.engine import JointExactEngine
    from qutip_trap.dynamics.space import HilbertSpace
    from qutip_trap.run.results import Progress, Result
    from qutip_trap.run.spec import Job, RunSpec


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
    space with the class of every mode, its dimension and drive-operator non-zeros, the pulse counts, the schedule's length
    and a wall-time guess from the Section 11.2 cost model (an order of magnitude; a run is usually faster)."""

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
    """Section 11.2: cost = a + b x (drive-operator non-zeros) x evaluations, summed over the pulses; a GATE_LOCAL walk plays
    each gate on a local space of its ions and propagates the tomography inputs."""
    total = 0.0
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
    """nnz/dimension, the scale the local-space guess divides the joint non-zeros by."""
    return max(1.0, nnz / max(1, dimension)) if dimension else 1.0


@dataclass(frozen=True)
class Machine:
    """A trapped-ion computer as a client sees it. Immutable; variants are ``dataclasses.replace(machine, ...)``."""

    device: Device
    """The apparatus, with ``Device.roles`` naming which beams play which part."""
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

    def run(
        self,
        circuit: Circuit,
        shots: int,
        *,
        seed: int = 0,
        keep_final_state: bool = False,
        progress: Callable[[Progress], None] | None = None,
    ) -> Result:
        """Compile, calibrate, schedule, prepare, evolve and read out ``circuit`` for ``shots`` (Section 3.4); ``seed`` is the
        root of every keyed stream and ``progress`` is called per pulse, branch, sample and readout."""
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
        """``run`` in a worker process behind a ``Job`` (status, progress, result, record, cancel); ``job.spec`` is the
        ``RunSpec`` of the call, ``label`` the caller's name for it."""
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
        """The JSON-serialisable ``RunSpec`` of ``run(circuit, shots, seed=seed, keep_final_state=keep_final_state)``."""
        from qutip_trap.run.spec import RunSpec

        return RunSpec.of(self, circuit, shots, seed=seed, keep_final_state=keep_final_state, label=label)

    def compile(self, circuit: Circuit) -> CompileReport:
        """Standard gates to native gates with phase tracking, every block and the whole circuit verified (Section 7.2)."""
        from qutip_trap.control.compiler import compile_report

        return compile_report(circuit, entangler=self.physics.entangler)

    def schedule(self, circuit: Circuit, *, seed: int = 0) -> Schedule:
        """The compile-calibrate-schedule prefix of ``run``: the pulses with absolute times, the played gates and the
        measurement event, nothing integrated; the table is this machine's, or the cached surrogate at ``seed``."""
        from qutip_trap.run.pipeline import compile_calibrate_schedule

        return compile_calibrate_schedule(self, circuit, seed=seed).schedule

    def calibrated(
        self, method: CalibrationMethod = "closed_form", *, seed: int = 0, **scans: Any
    ) -> Machine:
        """This machine with the table of ``calibration.calibrate(self, method=method, seed=seed, **scans)`` pinned; ``scans``
        are that function's scan settings (``pairs``, ``detection_records``, ``detection_windows_s``, ...)."""
        from qutip_trap.calibration import calibrate

        report = calibrate(self, method=method, seed=seed, **scans)
        return replace(self, table=report.table)

    def estimate(self, circuit: Circuit, *, seed: int = 0) -> Estimate:
        """What a run would cost before anything is integrated: the schedule, the space Section 5.2 declares for it, the level
        the guards resolve to (and why), and the Section 11.2 wall-time guess."""
        from qutip_trap.prep.recipe import recipe_of, run_preparation
        from qutip_trap.run.job import _raman_pair_hint
        from qutip_trap.run.pipeline import compile_calibrate_schedule
        from qutip_trap.run.space import SpaceSelection, select_space

        prefix = compile_calibrate_schedule(self, circuit, seed=seed)
        device = prefix.device
        opts = self.numerics
        prep_run = run_preparation(
            device, recipe_of(device, raman_pair=_raman_pair_hint(prefix.entangling_drives))
        )
        if opts.space is None:
            selection = select_space(
                device,
                prefix.schedule,
                opts,
                nbar=prep_run.nbar,
                ion_dims=[int(self.physics.internal_levels)] * device.crystal.n_ions,
            )
        else:
            selection = SpaceSelection.supplied(opts.space, opts, prep_run.nbar, len(device.crystal.modes))
        decision = decide_level(selection.budget, opts, self.level)
        sched = prefix.schedule
        return Estimate(
            level=decision.level,
            reason=decision.reason,
            space=selection.space,
            mode_class=dict(selection.mode_class),
            dimension=decision.dimension,
            nnz=decision.nnz,
            n_pulses=prefix.report.n_pulses,
            n_entangling=prefix.report.n_entangling,
            duration_s=float(sched.pulses_end_s),
            wall_time_s=_wall_time_guess(decision.dimension, decision.nnz, sched, decision.level),
            notes=prefix.notes + selection.notes,
        )

    @property
    def engine(self) -> JointExactEngine:
        """A ``JointExactEngine`` configured from the machine: the builder options, extra channels, device channels, hardware
        chain and channel switches of ``physics`` and the machine's table; ``run_pulses`` takes ``numerics``. Not the engine a
        run builds, which also carries the qubit-frequency shifts and the leakage level maps."""
        from qutip_trap.dynamics.engine import JointExactEngine

        phys = self.physics
        return JointExactEngine(
            builder_options=phys.builder,
            channels=tuple(phys.extra_channels),
            device_channels=bool(phys.noise),
            hardware_chain=bool(phys.hardware_chain),
            scattering_channels=phys.scattering == "channels",
            scattering_recoil=phys.scattering_recoil,
            intensity_noise_channels=bool(phys.intensity_noise_channels),
            table=self.table,
        )

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
        """The phenomenological summary of this machine (``benchmarks.error_model``): per native gate kind the average gate
        infidelity of its GATE_LOCAL channel and its duration, the SPAM errors and the noise rates, with exporters to IonQ's,
        Quantinuum's and the QDK estimator's vocabularies; ``qubits`` restricts the characterised ions."""
        from qutip_trap.benchmarks.error_model import error_model

        return error_model(self, qubits=qubits)

    def specs(self) -> str:
        """``Device.specs`` followed by the roles the machine resolved, whether a table is pinned and the level policy."""
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
