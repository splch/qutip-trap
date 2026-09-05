"""The composite space: per-ion qudit x modes, with the ENR option (PLAN.md Sections 5.1, 5.4; Appendix E).

Each ion is a qudit of dimension d (2 by default, 2 + leakage levels when scattering is modelled); each
resolved mode a truncated oscillator of d_m = n_max + 1 Fock levels; an optional ENR group of cold undriven
modes shares one excitation cap N_exc; frozen spectators contribute their chi_m and alpha_m analytically
(Section 5.2). The tensor order is ions then modes; operators are built once per space and cached (M2).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, prod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qutip import Qobj

    from qutip_trap.control.schedule import Schedule
    from qutip_trap.device.model import Device
    from qutip_trap.dynamics.engine import SolverOptions, State

M2 = "milestone M2 (hilbert/, PLAN.md Section 5.1)"
M9A = "milestone M9a (PLAN.md Sections 5.4, 11.3)"


def enr_dimension(n_modes: int, n_exc: int) -> int:
    """Number of Fock tuples of ``n_modes`` modes with total excitation <= n_exc: C(n_modes + n_exc, n_exc)."""
    if n_modes <= 0 or n_exc < 0:
        raise ValueError("an ENR group has at least one mode and a non-negative cap")
    return comb(n_modes + n_exc, n_exc)


@dataclass(frozen=True)
class ModeTruncation:
    mode: int
    d: int
    """d = n_max + 1 Fock levels."""
    expected_n_range: tuple[int, int]
    eta_max: float

    def __post_init__(self) -> None:
        if self.d < 2:
            raise ValueError("a resolved mode needs at least two Fock levels")
        lo, hi = self.expected_n_range
        if lo < 0 or hi < lo:
            raise ValueError("expected_n_range must be ordered and non-negative")
        if hi > self.d - 1:
            raise ValueError(f"expected occupation up to n = {hi} exceeds n_max = {self.d - 1}")
        if self.eta_max < 0.0:
            raise ValueError("eta_max is a magnitude")

    @property
    def n_max(self) -> int:
        return self.d - 1


@dataclass(frozen=True)
class CachedOperators:
    """sigma operators per ion, ladder operators per mode and displacement operators by (ion, mode) (M2)."""

    sigma_plus: tuple[Qobj, ...]
    sigma_minus: tuple[Qobj, ...]
    sigma_z: tuple[Qobj, ...]
    a: tuple[Qobj, ...]
    displacement: dict[tuple[int, int], Qobj]


@dataclass(frozen=True)
class HilbertSpace:
    ion_dims: tuple[int, ...]
    """2, or 2 + leakage levels, per ion."""
    resolved: tuple[ModeTruncation, ...]
    """Product-space modes."""
    enr_group: tuple[tuple[int, ...], int] | None
    """(modes, N_exc), optional."""
    frozen: tuple[int, ...]
    """Frozen spectators."""

    def __post_init__(self) -> None:
        self.check()

    def check(self) -> None:
        """Bookkeeping invariants of Section 5.1; the Qobj shape assertions join in M2 with ``operators()``."""
        if not self.ion_dims or any(d < 2 for d in self.ion_dims):
            raise ValueError("every ion is a qudit of dimension >= 2")
        resolved = [m.mode for m in self.resolved]
        enr_modes = list(self.enr_group[0]) if self.enr_group is not None else []
        frozen = list(self.frozen)
        all_modes = resolved + enr_modes + frozen
        if len(set(all_modes)) != len(all_modes):
            raise ValueError("a mode is resolved, in the ENR group or frozen, never in two classes")
        if self.enr_group is not None and (not enr_modes or self.enr_group[1] < 0):
            raise ValueError("an ENR group names at least one mode and a non-negative excitation cap")

    @property
    def dims(self) -> list[int]:
        """Tensor factors in order: ions, resolved modes, then the ENR group as one factor."""
        d = list(self.ion_dims) + [m.d for m in self.resolved]
        if self.enr_group is not None:
            d.append(enr_dimension(len(self.enr_group[0]), self.enr_group[1]))
        return d

    @property
    def dimension(self) -> int:
        """The joint dimension 2^N-like product that Section 11's cost model and the Section 5.4 budget guard."""
        return prod(self.dims)

    def operators(self) -> CachedOperators:
        """sigma_i, a_m, D_i(delta_k) by expm with the analytic Laguerre oracle checks (Section 5.1.1)."""
        raise NotImplementedError(f"HilbertSpace.operators is {M2}")

    def marginal(self, state: State | Qobj, keep: tuple[int, ...]) -> Qobj:
        """Reduced state over ions/modes; built from enr_state_dictionaries when an ENR group is present."""
        raise NotImplementedError(f"HilbertSpace.marginal is {M2}")

    @classmethod
    def for_(cls, device: Device, schedule: Schedule, options: SolverOptions) -> HilbertSpace:
        """The resolved-mode selection and truncation policy of Sections 5.5 and 11.3."""
        raise NotImplementedError(f"HilbertSpace.for_ is {M9A}")


__all__ = ["CachedOperators", "HilbertSpace", "ModeTruncation", "enr_dimension"]
