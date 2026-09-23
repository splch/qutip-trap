"""Fidelity levels JOINT_EXACT and GATE_LOCAL, the AUTO policy and the size budget that decides between them."""

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
    """The level a run integrates at, or the policy that picks one: ``JOINT_EXACT`` evolves every branch of the initial
    mixture on the joint space of ions and resolved modes, ``GATE_LOCAL`` walks the circuit gate by gate on exact
    gate-local spaces, ``AUTO`` takes JOINT_EXACT inside the size guards. Members equal their strings."""

    AUTO = "auto"
    JOINT_EXACT = "JOINT_EXACT"
    GATE_LOCAL = "GATE_LOCAL"


M9A = "milestone M9a (GATE_LOCAL: state-based process tomography and the motional model, PLAN.md Section 5.4)"

ESTIMATE_RESOLVED_MODES = 2
ESTIMATE_MODE_DIMENSION = 12
"""The Fock dimension of each resolved mode in the level estimate made without a space."""


@dataclass(frozen=True)
class LevelDecision:
    """Why a run integrates at the level it does: the numbers compared, the guards and whether the numbers are the
    declared space's or an estimate; ``reason`` is the sentence ``Diagnostics.level_reason`` carries."""

    level: FidelityLevel
    dimension: int
    nnz: int
    joint_dimension_max: int
    nnz_max: int
    estimated: bool
    """True when no space was given and the numbers are the estimate."""

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
    """(inside the guards, joint dimension, drive-operator non-zero estimate) of a declared space: arithmetic on the
    declaration, so it decides whether to build before anything is allocated."""
    from qutip_trap.run.space import drive_operator_nonzeros

    dim = space.dimension
    nnz = drive_operator_nonzeros(space)
    return dim <= options.joint_dimension_max and nnz <= options.nnz_max, dim, nnz


def decide_level(
    device: Device, circuit: Circuit, options: SolverOptions, *, space: HilbertSpace | None = None
) -> LevelDecision:
    """The level decision with its numbers: JOINT_EXACT when the joint dimension and the drive-operator non-zeros are
    within ``options.joint_dimension_max`` and ``nnz_max``, else GATE_LOCAL. Exact with ``space``; estimated without one
    (``ESTIMATE_RESOLVED_MODES`` modes of ``ESTIMATE_MODE_DIMENSION``, one mode without an entangling gate)."""
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
    """The level of :func:`decide_level` alone."""
    return decide_level(device, circuit, options, space=space).level
