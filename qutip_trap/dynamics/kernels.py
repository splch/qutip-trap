"""Mode-factorized application of sigma_+ (x) prod_m D_m for matrix-free propagation (PLAN.md Section 11.3; M9b).

Section 11.1 (v4 benchmark) showed the coefficient structure, not sparsity, sets the cost at the current
sizes; the mode-factorized kernel is the scaling remedy and its acceptance test is the Section 11.1 table.
"""

from __future__ import annotations

M9B = "milestone M9b (dynamics/kernels.py, PLAN.md Section 11.3)"


def apply_drive_kernel(*args: object, **kwargs: object) -> object:
    raise NotImplementedError(f"apply_drive_kernel is {M9B}")


__all__ = ["apply_drive_kernel"]
