"""Hamiltonian builder, channels, solvers, the matrix-free kernel, the parallel maps and the pulse engine (PLAN.md Sections
4.3.1, 5, 11; M2 onward, M9b for the kernel and the maps)."""

from __future__ import annotations

from qutip_trap.dynamics.channels import CollapseOp
from qutip_trap.dynamics.engine import (  # noqa: I001
    ChannelSummary,
    MotionalModel,
    PulseEngine,
    SeedSpec,
    SolverOptions,
    State,
    Traces,
)
from qutip_trap.dynamics.kernels import FactorizedOperator, apply_drive_kernel, factorized_qobj, is_factorized
from qutip_trap.dynamics.parallel import map_tasks, worker_count

__all__ = [
    "ChannelSummary",
    "CollapseOp",
    "FactorizedOperator",
    "MotionalModel",
    "PulseEngine",
    "SeedSpec",
    "SolverOptions",
    "State",
    "Traces",
    "apply_drive_kernel",
    "factorized_qobj",
    "is_factorized",
    "map_tasks",
    "worker_count",
]
