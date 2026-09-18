"""Rung 3 of the ladder, the dynamics (docs/api_proposal.md Section 4.5; docs/api_implementation_plan.md 1.6): the ONE
Hamiltonian builder and what it builds, the channels, the solvers, the matrix-free kernel, the parallel maps and the pulse
engine (PLAN.md Sections 4.3.1, 5, 11; M2 onward, M9b for the kernel and the maps). ``Machine.engine`` is the way here from
rung 2: ``JointExactEngine.run_pulses`` returns ``Traces`` without any readout, ``build_hamiltonian`` the terms a segment
integrates (``BuiltHamiltonian`` with one ``DriveRecord`` per pulse and ion), ``prepare`` the initial ``State`` on a
``HilbertSpace``, ``gate_channel`` the Section 6.8 summary of one native gate kind, and the tomography records the GATE_LOCAL
walk extracts; the ``Numerics`` groups (``Integration``, ``Truncation``, ``Trajectories``, ``GateLocal``, ``Parallel``) are the
knobs; ``NoiseSample`` with the key builders (``KEY_BRANCH_WEIGHT``, ``key_frozen_n``, ``key_qubit_offset_hz``,
``key_mode_offset_hz``) names what a run stamps on every branch and sample (0.4.0). The names that sit above this package
in the layering (``prepare``, ``gate_channel``, the option groups) are imported on first use. The tomography internals
``choi_least_squares`` and ``project_cptp`` moved to ``qutip_trap.experimental`` in 0.4.0; importing them from here warns
(docs/deprecations.md)."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from qutip_trap.dynamics.channels import CollapseOp
from qutip_trap.dynamics.engine import (  # noqa: I001
    ChannelSummary,
    EngineReport,
    JointExactEngine,
    MotionalModel,
    PulseEngine,
    SeedSpec,
    SegmentReport,
    SolverOptions,
    State,
    Traces,
)
from qutip_trap.dynamics.evolve import ConvergenceReport, convergence_check
from qutip_trap.dynamics.kernels import FactorizedOperator, apply_drive_kernel, factorized_qobj, is_factorized
from qutip_trap.dynamics.parallel import map_tasks, worker_count
from qutip_trap.hilbert.space import CachedOperators, HilbertSpace, ModeTruncation

_LAZY: dict[str, tuple[str, str]] = {
    "BuilderOptions": ("qutip_trap.dynamics.hamiltonian", "BuilderOptions"),
    "BuiltHamiltonian": ("qutip_trap.dynamics.hamiltonian", "BuiltHamiltonian"),
    "DriveRecord": ("qutip_trap.dynamics.hamiltonian", "DriveRecord"),
    "build_hamiltonian": ("qutip_trap.dynamics.hamiltonian", "build_hamiltonian"),
    "prepare": ("qutip_trap.run.job", "prepare"),
    "TomographyRecord": ("qutip_trap.dynamics.tomography", "TomographyRecord"),
    "input_states": ("qutip_trap.dynamics.tomography", "input_states"),
    "kraus_operators": ("qutip_trap.dynamics.tomography", "kraus_operators"),
    "NoiseSample": ("qutip_trap.noise.sampling", "NoiseSample"),
    "KEY_BRANCH_WEIGHT": ("qutip_trap.noise.sampling", "KEY_BRANCH_WEIGHT"),
    "key_frozen_n": ("qutip_trap.noise.sampling", "key_frozen_n"),
    "key_mode_offset_hz": ("qutip_trap.noise.sampling", "key_mode_offset_hz"),
    "key_qubit_offset_hz": ("qutip_trap.noise.sampling", "key_qubit_offset_hz"),
    "gate_channel": ("qutip_trap.benchmarks.budget", "gate_channel"),
    "GateChannel": ("qutip_trap.benchmarks.budget", "GateChannel"),
    "Numerics": ("qutip_trap.options", "Numerics"),
    "Integration": ("qutip_trap.options", "Integration"),
    "Truncation": ("qutip_trap.options", "Truncation"),
    "Trajectories": ("qutip_trap.options", "Trajectories"),
    "GateLocal": ("qutip_trap.options", "GateLocal"),
    "Parallel": ("qutip_trap.options", "Parallel"),
}
"""Names imported on first use to keep the import graph acyclic: the modules above this package in the layering, and the
tomography and noise-sampling modules, which reach the noise package (``control.schedule`` imports ``dynamics.frames``
while it is itself being imported, and ``noise.scattering`` imports ``control.schedule``, so this package's own
initialization must need neither)."""


_MOVED_TO_EXPERIMENTAL: dict[str, str] = {
    "choi_least_squares": "Import choi_least_squares from qutip_trap.experimental.",
    "project_cptp": "Import project_cptp from qutip_trap.experimental.",
}
"""The tomography internals this rung exported in 0.2.0 and 0.3.0: deprecated here since 0.4.0, the names live in
``qutip_trap.experimental`` (docs/api_implementation_plan.md 3.3), where the stability guarantee does not reach."""


def __getattr__(name: str) -> Any:
    if name in _MOVED_TO_EXPERIMENTAL:
        from qutip_trap._compat import message, warn

        warn(message(f"qutip_trap.dynamics.{name}", "v0.6", _MOVED_TO_EXPERIMENTAL[name]), stacklevel=2)
        return getattr(importlib.import_module("qutip_trap.dynamics.tomography"), name)
    try:
        module_name, attribute = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module 'qutip_trap.dynamics' has no attribute {name!r}") from None
    value: Any = getattr(importlib.import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY, *_MOVED_TO_EXPERIMENTAL})


if TYPE_CHECKING:
    from qutip_trap.benchmarks.budget import GateChannel, gate_channel
    from qutip_trap.dynamics.hamiltonian import (
        BuilderOptions,
        BuiltHamiltonian,
        DriveRecord,
        build_hamiltonian,
    )
    from qutip_trap.dynamics.tomography import TomographyRecord, input_states, kraus_operators
    from qutip_trap.noise.sampling import (
        KEY_BRANCH_WEIGHT,
        NoiseSample,
        key_frozen_n,
        key_mode_offset_hz,
        key_qubit_offset_hz,
    )
    from qutip_trap.options import GateLocal, Integration, Numerics, Parallel, Trajectories, Truncation
    from qutip_trap.run.job import prepare

__all__ = [
    "KEY_BRANCH_WEIGHT",
    "BuilderOptions",
    "BuiltHamiltonian",
    "CachedOperators",
    "ChannelSummary",
    "CollapseOp",
    "ConvergenceReport",
    "DriveRecord",
    "EngineReport",
    "FactorizedOperator",
    "GateChannel",
    "GateLocal",
    "HilbertSpace",
    "Integration",
    "JointExactEngine",
    "ModeTruncation",
    "MotionalModel",
    "NoiseSample",
    "Numerics",
    "Parallel",
    "PulseEngine",
    "SeedSpec",
    "SegmentReport",
    "SolverOptions",
    "State",
    "TomographyRecord",
    "Traces",
    "Trajectories",
    "Truncation",
    "apply_drive_kernel",
    "build_hamiltonian",
    "convergence_check",
    "factorized_qobj",
    "gate_channel",
    "input_states",
    "is_factorized",
    "key_frozen_n",
    "key_mode_offset_hz",
    "key_qubit_offset_hz",
    "kraus_operators",
    "map_tasks",
    "prepare",
    "worker_count",
]
