"""AM/FM/PM pulse solvers for multi-mode Molmer-Sorensen closure (PLAN.md Section 4.4.3; milestone M4).

Conventions to implement (Section 13): Choi's displacement integral with sin(mu t) inside the integrand and
bare e^{i omega_m t}, or delta_m = omega_m - mu with a slow envelope, never mixed; the entangling angle uses
the SYMMETRIZED two-body kernel (Choi's factor 2 holds only for proportional envelopes); closure
eta Omega/eps = 1/(2 sqrt K) in the plan's per-tone (hbar Omega/2) convention (``check_ms_closure.py``).
"""

from __future__ import annotations

M4 = "milestone M4 (control/shaping.py, PLAN.md Section 4.4.3)"


def solve_amplitude_modulation(*args: object, **kwargs: object) -> object:
    raise NotImplementedError(f"solve_amplitude_modulation is {M4}")


def solve_frequency_modulation(*args: object, **kwargs: object) -> object:
    raise NotImplementedError(f"solve_frequency_modulation is {M4}")


__all__ = ["solve_amplitude_modulation", "solve_frequency_modulation"]
