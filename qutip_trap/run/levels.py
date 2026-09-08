"""Fidelity levels JOINT_EXACT and GATE_LOCAL and their budget (PLAN.md Sections 5.4, 11.5; Appendix E; M6 for the budget,
M9a for GATE_LOCAL itself)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from qutip_trap.control.compiler import Circuit
    from qutip_trap.device.model import Device
    from qutip_trap.dynamics.engine import SolverOptions
    from qutip_trap.hilbert.space import HilbertSpace

FidelityLevel = Literal["JOINT_EXACT", "GATE_LOCAL"]
M9A = "milestone M9a (GATE_LOCAL: state-based process tomography and the motional model, PLAN.md Section 5.4)"

ESTIMATE_RESOLVED_MODES = 2
ESTIMATE_MODE_DIMENSION = 12
"""Without a schedule the level is estimated for the Section 9.6 fixture rule: two resolved modes at d_m = 12."""


def within_budget(space: HilbertSpace, options: SolverOptions) -> tuple[bool, int, int]:
    """(inside the Section 11.5 guards, joint dimension, drive-operator non-zero estimate) of a DECLARED space.

    Both numbers are arithmetic in the declaration's ion dimensions, resolved caps and ENR group, and a ``HilbertSpace``
    allocates no operator when it is constructed, so this decides whether to build anything at all - which is what Section
    11.5's "refuses to build" requires (M9b audit B2)."""
    from qutip_trap.run.space import drive_operator_nonzeros

    dim = space.dimension
    nnz = drive_operator_nonzeros(space)
    return dim <= options.joint_dimension_max and nnz <= options.nnz_max, dim, nnz


def resolve_level(
    device: Device, circuit: Circuit, options: SolverOptions, *, space: HilbertSpace | None = None
) -> FidelityLevel:
    """JOINT_EXACT when the joint dimension <= options.joint_dimension_max and the drive-operator non-zeros <= options.nnz_max,
    else GATE_LOCAL (Sections 5.4, 11.5). With ``space`` (the run's actual selection) the guards are exact; without one they
    are estimated for N ions with two resolved modes at d_m = 12, the Section 9.6 fixture rule."""
    if space is not None:
        ok, _dim, _nnz = within_budget(space, options)
        return "JOINT_EXACT" if ok else "GATE_LOCAL"
    n = device.crystal.n_ions
    has_entangling = any(len(op.qubits) == 2 and not op.is_non_unitary for op in circuit.ops)
    n_res = ESTIMATE_RESOLVED_MODES if has_entangling else 1
    dim = (2**n) * ESTIMATE_MODE_DIMENSION**n_res
    nnz = n * (2**n) * (ESTIMATE_MODE_DIMENSION**2) ** n_res
    return "JOINT_EXACT" if dim <= options.joint_dimension_max and nnz <= options.nnz_max else "GATE_LOCAL"


__all__ = [
    "ESTIMATE_MODE_DIMENSION",
    "ESTIMATE_RESOLVED_MODES",
    "FidelityLevel",
    "M9A",
    "resolve_level",
    "within_budget",
]
