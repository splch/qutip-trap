"""qutip-trap: a first-principles trapped-ion quantum computer simulator built on QuTiP.

    import qutip_trap as trap

    machine = trap.presets.yb171_chain(2)
    result = machine.run(trap.Circuit(2).h(0).cnot(0, 1), shots=2000)
    result.counts, result.diagnostics.level

Everything else is imported from the module that defines it.
"""

__version__ = "0.4.0"
"""The release, recorded by ``Result.to_dict()`` as ``qutip_trap_version``."""

from qutip_trap import presets  # noqa: E402
from qutip_trap.control.compiler import Circuit, Operation  # noqa: E402
from qutip_trap.device.model import BeamRoles, Device  # noqa: E402
from qutip_trap.machine import Estimate, Machine, as_machine  # noqa: E402
from qutip_trap.options import Numerics, Physics, Readout  # noqa: E402
from qutip_trap.run.levels import FidelityLevel, LevelDecision, decide_level  # noqa: E402
from qutip_trap.run.results import Diagnostics, Progress, Result  # noqa: E402
from qutip_trap.run.spec import Job, JobCancelled, JobError, JobStatus, RunSpec  # noqa: E402

__all__ = [
    "BeamRoles",
    "Circuit",
    "Device",
    "Diagnostics",
    "Estimate",
    "FidelityLevel",
    "Job",
    "JobCancelled",
    "JobError",
    "JobStatus",
    "LevelDecision",
    "Machine",
    "Numerics",
    "Operation",
    "Physics",
    "Progress",
    "Readout",
    "Result",
    "RunSpec",
    "__version__",
    "as_machine",
    "decide_level",
    "presets",
]
