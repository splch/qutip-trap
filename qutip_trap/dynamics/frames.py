"""Interaction pictures and phase bookkeeping between pulses (PLAN.md Sections 4.3.4, 5.2; milestone M2).

Sign rule (Section 13, "Spin and motion phases"): phi_s = (phi_b + phi_r)/2 and phi_m = (phi_b - phi_r)/2 of
the tone phases, the force axis carrying a further pi/2 from the i of the sideband coupling that the frame
alignment absorbs.
"""

from __future__ import annotations

M2 = "milestone M2 (dynamics/frames.py, PLAN.md Section 5.2)"


def interaction_picture(*args: object, **kwargs: object) -> object:
    raise NotImplementedError(f"interaction_picture is {M2}")


__all__ = ["interaction_picture"]
