"""qutip-trap: a first-principles trapped-ion quantum computer simulator built on QuTiP.

    import qutip_trap as trap

    machine = trap.presets.yb171_chain(2)
    result = machine.run(trap.Circuit(2).h(0).cnot(0, 1), shots=2000)
    result.counts, result.diagnostics.level

Every other name is imported from the module that defines it.
"""

import warnings

with warnings.catch_warnings():
    # QuTiP warns on import when matplotlib is absent ("Graphics will not work"); nothing here draws through QuTiP, so the
    # warning would only mislead. The filter is scoped to this first import and leaves the caller's filters untouched.
    warnings.filterwarnings("ignore", message="matplotlib not found", category=UserWarning)
    import qutip  # noqa: F401

from qutip_trap import presets
from qutip_trap.control.compiler import Circuit
from qutip_trap.machine import Machine
from qutip_trap.options import Numerics, Physics, Readout
from qutip_trap.run.levels import FidelityLevel
from qutip_trap.run.results import Result

__version__ = "0.4.0"
"""The release, recorded by ``Result.to_dict()`` as ``qutip_trap_version``."""

__all__ = [
    "Circuit",
    "FidelityLevel",
    "Machine",
    "Numerics",
    "Physics",
    "Readout",
    "Result",
    "__version__",
    "presets",
]
