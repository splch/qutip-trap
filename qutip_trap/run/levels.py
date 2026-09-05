"""Fidelity levels JOINT_EXACT and GATE_LOCAL and their budget (PLAN.md Sections 5.4, 11.5; Appendix E)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from qutip_trap.control.compiler import Circuit
    from qutip_trap.device.model import Device
    from qutip_trap.dynamics.engine import SolverOptions

FidelityLevel = Literal["JOINT_EXACT", "GATE_LOCAL"]
M9A = "milestone M9a (run/levels.py, PLAN.md Section 5.4)"


def resolve_level(device: Device, circuit: Circuit, options: SolverOptions) -> FidelityLevel:
    """JOINT_EXACT when the joint dimension <= options.joint_dimension_max and the drive-operator non-zeros
    <= options.nnz_max, else GATE_LOCAL (Sections 5.4, 11.5)."""
    raise NotImplementedError(f"resolve_level is {M9A}")


__all__ = ["FidelityLevel", "resolve_level"]
