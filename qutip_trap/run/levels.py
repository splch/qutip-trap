"""Fidelity levels JOINT_EXACT and GATE_LOCAL, the AUTO policy and the Section 11.5 budget that decides between them (PLAN.md
Sections 5.4, 11.5; Appendix E; M6 for the budget, M9a for GATE_LOCAL itself; docs/api_implementation_plan.md 1.2 for the
enum and the reason a decision carries)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qutip_trap.control.compiler import Circuit
    from qutip_trap.device.model import Device
    from qutip_trap.dynamics.engine import SolverOptions
    from qutip_trap.hilbert.space import HilbertSpace


class FidelityLevel(StrEnum):
    """The level a run integrates at (Section 5.4), or the policy that picks one. ``JOINT_EXACT`` evolves every branch of
    the initial mixture on the joint space of the ions and the resolved modes; ``GATE_LOCAL`` walks the circuit gate by gate
    on exact gate-local spaces and carries the register between them; ``AUTO`` takes JOINT_EXACT inside the Section 11.5
    guards (``SolverOptions.joint_dimension_max`` and ``nnz_max``) and GATE_LOCAL above them. The members equal the strings
    the code accepted before 0.2.0 (``"auto"``, ``"JOINT_EXACT"``, ``"GATE_LOCAL"``), so ``Diagnostics.level == "GATE_LOCAL"``
    still holds; docs/conventions.md ("Vocabulary") records why the enum is not named ``Level``."""

    AUTO = "auto"
    JOINT_EXACT = "JOINT_EXACT"
    GATE_LOCAL = "GATE_LOCAL"


M9A = "milestone M9a (GATE_LOCAL: state-based process tomography and the motional model, PLAN.md Section 5.4)"

ESTIMATE_RESOLVED_MODES = 2
ESTIMATE_MODE_DIMENSION = 12
"""Without a schedule the level is estimated for the Section 9.6 fixture rule: two resolved modes at d_m = 12."""


@dataclass(frozen=True)
class LevelDecision:
    """Why a run integrates at the level it does (Section 11.5): the level, the joint dimension and the drive-operator
    non-zero count that were compared, the two guards they were compared against, and whether the numbers are the declared
    space's or the Section 9.6 estimate; ``reason`` is the sentence ``Diagnostics.level_reason`` carries."""

    level: FidelityLevel
    dimension: int
    nnz: int
    joint_dimension_max: int
    nnz_max: int
    estimated: bool
    """True when no space was given and the numbers are the fixture estimate (two resolved modes at d_m = 12)."""

    @property
    def reason(self) -> str:
        how = "estimated" if self.estimated else "declared"
        dim_sign = "<=" if self.dimension <= self.joint_dimension_max else ">"
        nnz_sign = "<=" if self.nnz <= self.nnz_max else ">"
        return (
            f"{self.level.value}: {how} joint dimension {self.dimension} {dim_sign} joint_dimension_max = "
            f"{self.joint_dimension_max}, drive-operator non-zeros {self.nnz} {nnz_sign} nnz_max = {self.nnz_max} "
            "(Section 11.5)"
        )


def within_budget(space: HilbertSpace, options: SolverOptions) -> tuple[bool, int, int]:
    """(inside the Section 11.5 guards, joint dimension, drive-operator non-zero estimate) of a DECLARED space.

    Both numbers are arithmetic in the declaration's ion dimensions, resolved caps and ENR group, and a ``HilbertSpace``
    allocates no operator when it is constructed, so this decides whether to build anything at all - which is what Section
    11.5's "refuses to build" requires (M9b audit B2)."""
    from qutip_trap.run.space import drive_operator_nonzeros

    dim = space.dimension
    nnz = drive_operator_nonzeros(space)
    return dim <= options.joint_dimension_max and nnz <= options.nnz_max, dim, nnz


def decide_level(
    device: Device, circuit: Circuit, options: SolverOptions, *, space: HilbertSpace | None = None
) -> LevelDecision:
    """The Section 11.5 decision with its numbers: JOINT_EXACT when the joint dimension <= ``options.joint_dimension_max``
    and the drive-operator non-zeros <= ``options.nnz_max``, else GATE_LOCAL (Sections 5.4, 11.5). With ``space`` (the run's
    actual selection) the guards are exact; without one they are estimated for N ions with two resolved modes at d_m = 12,
    the Section 9.6 fixture rule."""
    if space is not None:
        ok, dim, nnz = within_budget(space, options)
        estimated = False
    else:
        n = device.crystal.n_ions
        has_entangling = any(len(op.qubits) == 2 and not op.is_non_unitary for op in circuit.ops)
        n_res = ESTIMATE_RESOLVED_MODES if has_entangling else 1
        dim = (2**n) * ESTIMATE_MODE_DIMENSION**n_res
        nnz = n * (2**n) * (ESTIMATE_MODE_DIMENSION**2) ** n_res
        ok = dim <= options.joint_dimension_max and nnz <= options.nnz_max
        estimated = True
    return LevelDecision(
        level=FidelityLevel.JOINT_EXACT if ok else FidelityLevel.GATE_LOCAL,
        dimension=int(dim),
        nnz=int(nnz),
        joint_dimension_max=int(options.joint_dimension_max),
        nnz_max=int(options.nnz_max),
        estimated=estimated,
    )


def resolve_level(
    device: Device, circuit: Circuit, options: SolverOptions, *, space: HilbertSpace | None = None
) -> FidelityLevel:
    """The level of :func:`decide_level` alone (Appendix E's signature): JOINT_EXACT inside the Section 11.5 guards, GATE_LOCAL
    above them, exact with ``space`` and estimated without one; the member compares equal to the strings."""
    return decide_level(device, circuit, options, space=space).level
