"""Beams, Raman drives, multi-level Bloch solver, scattering and recoil (PLAN.md Sections 4.2.8, 4.3, 4.5; M2, M3a)."""

from __future__ import annotations

from qutip_trap.light.beams import Beam, PolarizationModulation, PolGradientBeams
from qutip_trap.light.comb import CombSpec

__all__ = ["Beam", "CombSpec", "PolGradientBeams", "PolarizationModulation"]
# light.bloch (the scattering-rate object of Section 13) and light.recoil (the emission kernel of Section 4.2.8) are
# imported by path; they pull in QuTiP, which the Appendix E data model keeps out of module import time.
