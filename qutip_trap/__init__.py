"""qutip-trap: a first-principles trapped-ion quantum computer simulator built on QuTiP.

The specification is ``PLAN.md`` at the repository root; the frozen Appendix E surface is :mod:`qutip_trap.api`. Release
0.1.0 completed milestones M0 to M10 of PLAN.md Section 10 (the atomic layer, the trap and crystal, the one Hamiltonian
builder, cooling and preparation, entangling gates, readout, end-to-end circuits, noise channels, calibration emulation,
the two scaling milestones and the benchmark emulation); 0.2.0 adds the ladder of ``docs/api_proposal.md``. This module is
rung 0, the machine: ``Machine``, ``Circuit``, ``Result`` and the option objects, with ``presets`` for the example machines.
``qutip_trap.circuit``, ``qutip_trap.schedule``, ``qutip_trap.dynamics`` and ``qutip_trap.physics`` are the rungs below,
``qutip_trap.io`` the wire formats and ``qutip_trap.interop`` the adapters to other SDKs. Every name here is imported on
first use through a module ``__getattr__``, so ``import qutip_trap`` stays free of numpy and qutip
(``tests/test_import_time.py``). The recommended alias is ``import qutip_trap as trap`` (docs/conventions.md, "Vocabulary").

    import qutip_trap as trap

    machine = trap.presets.yb171_chain(2)
    result = machine.run(trap.Circuit(2).h(0).cnot(0, 1), shots=2000)
    result.counts, result.diagnostics.level
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "0.2.0"
"""The release, recorded by ``Result.to_dict()`` as ``qutip_trap_version``."""

_RUNG_0: dict[str, tuple[str, str | None]] = {
    "Machine": ("qutip_trap.machine", "Machine"),
    "Estimate": ("qutip_trap.machine", "Estimate"),
    "Circuit": ("qutip_trap.control.compiler", "Circuit"),
    "Operation": ("qutip_trap.control.compiler", "Operation"),
    "Result": ("qutip_trap.run.results", "Result"),
    "Diagnostics": ("qutip_trap.run.results", "Diagnostics"),
    "Progress": ("qutip_trap.run.results", "Progress"),
    "FidelityLevel": ("qutip_trap.run.levels", "FidelityLevel"),
    "LevelDecision": ("qutip_trap.run.levels", "LevelDecision"),
    "decide_level": ("qutip_trap.run.levels", "decide_level"),
    "Physics": ("qutip_trap.options", "Physics"),
    "Numerics": ("qutip_trap.options", "Numerics"),
    "Readout": ("qutip_trap.options", "Readout"),
    "Device": ("qutip_trap.device.model", "Device"),
    "BeamRoles": ("qutip_trap.device.model", "BeamRoles"),
    "presets": ("qutip_trap.presets", None),
    "circuit": ("qutip_trap.circuit", None),
    "schedule": ("qutip_trap.schedule", None),
    "dynamics": ("qutip_trap.dynamics", None),
    "physics": ("qutip_trap.physics", None),
    "io": ("qutip_trap.io", None),
    "interop": ("qutip_trap.interop", None),
}
"""The rung-0 names: each -> (the module that defines it, its attribute there; None for the module itself)."""

__all__ = [
    "BeamRoles",
    "Circuit",
    "Device",
    "Diagnostics",
    "Estimate",
    "FidelityLevel",
    "LevelDecision",
    "Machine",
    "Numerics",
    "Operation",
    "Physics",
    "Progress",
    "Readout",
    "Result",
    "__version__",
    "circuit",
    "decide_level",
    "dynamics",
    "interop",
    "io",
    "physics",
    "presets",
    "schedule",
]


def __getattr__(name: str) -> Any:
    """Import a rung-0 name on first use and cache it on the module (PEP 562)."""
    try:
        module_name, attribute = _RUNG_0[name]
    except KeyError:
        raise AttributeError(f"module 'qutip_trap' has no attribute {name!r}") from None
    module = importlib.import_module(module_name)
    value: Any = module if attribute is None else getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*__all__, *(name for name in globals() if name.startswith("__"))})


if TYPE_CHECKING:
    # the lazy names, spelled out for type checkers and editors
    from qutip_trap import circuit, dynamics, interop, io, physics, presets, schedule
    from qutip_trap.control.compiler import Circuit, Operation
    from qutip_trap.device.model import BeamRoles, Device
    from qutip_trap.machine import Estimate, Machine
    from qutip_trap.options import Numerics, Physics, Readout
    from qutip_trap.run.levels import FidelityLevel, LevelDecision, decide_level
    from qutip_trap.run.results import Diagnostics, Progress, Result
