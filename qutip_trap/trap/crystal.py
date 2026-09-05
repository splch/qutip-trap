"""Ion crystal and normal modes (PLAN.md Sections 4.1.2, 4.1.3, 4.1.7; Appendix E; milestone M1).

Mode addressing (Section 13, row "Mode index"): every ``mode: int`` anywhere in the package is a position in
``Crystal.modes``, ordered axial, transverse_1, transverse_2, ascending frequency within a family, with
unit-norm mass-weighted eigenvectors whose last component is positive. The Lamb-Dicke parameter
eta_{i,m} = (delta_k . e_hat) c_{i,m} sqrt(hbar/(2 m_i omega_m)) times C0 is computed HERE and nowhere else
(Section 5.7), with the ion's own mass and the mass-weighted eigenvector component (Section 4.1.7).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from qutip_trap.species.model import Species
    from qutip_trap.trap.mathieu import MathieuParameters

M1 = "milestone M1 (trap/crystal.py, PLAN.md Section 4.1.3)"

FAMILY_ORDER: tuple[str, ...] = ("axial", "transverse_1", "transverse_2")


@dataclass(frozen=True)
class Mode:
    family: Literal["axial", "transverse_1", "transverse_2"]
    index: int
    """Position within the family, ascending frequency."""
    omega_hz: float
    e_hat: tuple[float, float, float]
    eigenvector: np.ndarray
    """Mass-weighted c_{i,m}, unit norm, last component positive (Section 4.1.3)."""

    def __post_init__(self) -> None:
        if self.omega_hz <= 0.0:
            raise ValueError("mode frequency must be positive")
        v = np.asarray(self.eigenvector, dtype=float)
        if v.ndim != 1:
            raise ValueError("eigenvector must be one-dimensional (one component per ion)")
        if not np.isclose(float(np.linalg.norm(v)), 1.0, rtol=0.0, atol=1e-9):
            raise ValueError("eigenvector must have unit norm")
        if v[-1] < 0.0:
            raise ValueError("eigenvector sign gauge: last component must be positive (Section 13)")


@dataclass(frozen=True)
class Crystal:
    species: tuple[Species, ...]
    """One entry per ion (mixed species allowed)."""
    positions_m: np.ndarray
    """(N, 3) equilibrium positions."""
    modes: tuple[Mode, ...]
    """3N modes in ONE canonical order: axial, transverse_1, transverse_2, ascending frequency within a family."""

    def __post_init__(self) -> None:
        n = len(self.species)
        pos = np.asarray(self.positions_m)
        if pos.shape != (n, 3):
            raise ValueError(f"positions_m must have shape ({n}, 3), got {pos.shape}")
        if len(self.modes) != 3 * n:
            raise ValueError(f"a crystal of {n} ions has 3N = {3 * n} modes, got {len(self.modes)}")
        fam_rank = {f: k for k, f in enumerate(FAMILY_ORDER)}
        keys = [(fam_rank[m.family], m.omega_hz) for m in self.modes]
        if keys != sorted(keys):
            raise ValueError(
                "modes must be ordered axial, transverse_1, transverse_2 and ascending in frequency"
            )
        for m in self.modes:
            if len(m.eigenvector) != n:
                raise ValueError("every mode eigenvector has one component per ion")

    @property
    def n_ions(self) -> int:
        return len(self.species)

    def lamb_dicke(
        self, ion: int, mode: int, delta_k: np.ndarray, *, micromotion: MathieuParameters | None
    ) -> float:
        """eta_{i,m} = (delta_k . e_hat) c_{i,m} sqrt(hbar/(2 m_i omega_m)) times C0 from ``micromotion`` (Section 4.1.1)."""
        raise NotImplementedError(f"Crystal.lamb_dicke is {M1}")


def solve_crystal(*args: object, **kwargs: object) -> Crystal:
    """Equilibrium positions by Newton iteration, mass-weighted Hessian, normal modes (Sections 4.1.2, 4.1.3, 4.1.7)."""
    raise NotImplementedError(f"solve_crystal is {M1}")


__all__ = ["FAMILY_ORDER", "Crystal", "Mode", "solve_crystal"]
