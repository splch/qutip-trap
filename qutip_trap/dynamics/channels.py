"""Collapse operators from noise parameters (PLAN.md Sections 5.7, 6; Appendix E; milestone M7).

Convention rows that bind here (Section 13): heating rates Gamma(N_bar + 1) on a and Gamma N_bar on
a^dagger; qubit dephasing L = sqrt(gamma_phi/2) sigma_z so that the coherence decays at gamma_phi = 1/T2;
motional dephasing L = a^dagger a sqrt(2/tau); the Rayleigh operator (1/2) sqrt(Gamma_el) sigma_z, whose
dissipator prefactor is Gamma_el/4 while the coherence decays at Gamma_el/2, both correct; recoil-resolved
emission channels one operator per (decay channel, recoil class), never summed coherently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qutip import Qobj


@dataclass(frozen=True)
class CollapseOp:
    op: Qobj
    rate_hz: float
    """The rate the operator carries under its root, as an ordinary frequency for reporting."""
    channel: str
    ion: int | None
    mode: int | None


__all__ = ["CollapseOp"]
