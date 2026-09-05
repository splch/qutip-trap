"""Laser cooling and state preparation (PLAN.md Section 4.2; milestone M3 on the M3a Bloch builder).

Order (Section 4.2.6): Doppler cooling first, then sideband or EIT cooling with its own repump, then a final
optical pump immediately before the circuit.
"""

from __future__ import annotations

M3 = "milestone M3 (prep/, PLAN.md Section 4.2)"


def doppler_cooling(*args: object, **kwargs: object) -> object:
    """Section 4.2.1 rate coefficients with W from light/bloch.py; force model as a guarded cross-check only."""
    raise NotImplementedError(f"doppler_cooling is {M3}")


def sideband_cooling(*args: object, **kwargs: object) -> object:
    raise NotImplementedError(f"sideband_cooling is {M3}")


def eit_cooling(*args: object, **kwargs: object) -> object:
    raise NotImplementedError(f"eit_cooling is {M3}")


def optical_pumping(*args: object, **kwargs: object) -> object:
    raise NotImplementedError(f"optical_pumping is {M3}")


__all__ = ["doppler_cooling", "eit_cooling", "optical_pumping", "sideband_cooling"]
