"""Fidelity levels and the Section 11.5 size guards that choose between them (PLAN.md Sections 5.4, 11.5)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from qutip_trap.dynamics.space import HilbertSpace
    from qutip_trap.options import Numerics


class FidelityLevel(StrEnum):
    """The level a run integrates at, or the policy that picks one. ``JOINT_EXACT`` evolves every branch of the initial
    mixture on the joint space of the ions and the resolved modes; ``GATE_LOCAL`` walks the circuit gate by gate on exact
    gate-local spaces and carries the register between them (Section 5.4); ``AUTO`` takes JOINT_EXACT inside the Section
    11.5 guards (``joint_dimension_max``, ``nnz_max``) and GATE_LOCAL above them."""

    AUTO = "auto"
    JOINT_EXACT = "JOINT_EXACT"
    GATE_LOCAL = "GATE_LOCAL"


class Budget(NamedTuple):
    """The Section 11.5 guards on a declared space: both hold, the joint dimension, the drive-operator non-zeros."""

    inside: bool
    dimension: int
    nnz: int


@dataclass(frozen=True)
class LevelDecision:
    """The level a run integrates at under a policy, with the numbers the guards compared; ``reason`` is the sentence
    ``Diagnostics.level_reason`` carries."""

    level: FidelityLevel
    inside: bool
    """Both guards hold on the declared space: the AUTO verdict."""
    forced: bool
    """The policy named the level rather than AUTO."""
    dimension: int
    nnz: int
    joint_dimension_max: int
    nnz_max: int

    @property
    def reason(self) -> str:
        auto = FidelityLevel.JOINT_EXACT if self.inside else FidelityLevel.GATE_LOCAL
        dim_sign = "<=" if self.dimension <= self.joint_dimension_max else ">"
        nnz_sign = "<=" if self.nnz <= self.nnz_max else ">"
        why = (
            f"{auto.value}: declared joint dimension {self.dimension} {dim_sign} joint_dimension_max = "
            f"{self.joint_dimension_max}, drive-operator non-zeros {self.nnz} {nnz_sign} nnz_max = {self.nnz_max} "
            "(Section 11.5)"
        )
        return (
            f"{self.level.value} forced by the caller; level='auto' would choose {why}"
            if self.forced
            else why
        )


def within_budget(space: HilbertSpace, options: Numerics) -> Budget:
    """The guards on a DECLARED space: both numbers are arithmetic on its dimensions and a ``HilbertSpace`` allocates no
    operator when constructed, so the verdict comes before anything is built."""
    from qutip_trap.run.space import drive_operator_nonzeros

    dim = space.dimension
    nnz = drive_operator_nonzeros(space)
    return Budget(dim <= options.joint_dimension_max and nnz <= options.nnz_max, dim, nnz)


def decide_level(budget: Budget, options: Numerics, policy: FidelityLevel) -> LevelDecision:
    """The level ``policy`` runs at on a space with this ``budget``: the forced level, or AUTO's verdict."""
    auto = FidelityLevel.JOINT_EXACT if budget.inside else FidelityLevel.GATE_LOCAL
    forced = policy is not FidelityLevel.AUTO
    return LevelDecision(
        level=policy if forced else auto,
        inside=budget.inside,
        forced=forced,
        dimension=budget.dimension,
        nnz=budget.nnz,
        joint_dimension_max=int(options.joint_dimension_max),
        nnz_max=int(options.nnz_max),
    )
