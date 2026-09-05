"""The ONE Hamiltonian builder (PLAN.md Sections 3.1, 4.3.1, 5.2, 5.7; milestone M2; multi-level mode M3a).

H(t) = H_mot + H_int + sum_i H_drive,i(t) + sum_i H_Stark,i(t) + H_anh + H_curv, every term switchable and
every switch recorded in the run's diagnostics (Section 5.7). Approximations (Lamb-Dicke expansion,
rotating wave on the sidebands) are options defaulting to none (Section 3.1).
"""

from __future__ import annotations

M2 = "milestone M2 (dynamics/hamiltonian.py, PLAN.md Section 4.3.1)"


def build_hamiltonian(*args: object, **kwargs: object) -> object:
    raise NotImplementedError(f"build_hamiltonian is {M2}")


__all__ = ["build_hamiltonian"]
