"""Cached operators and the exact displacement operator (PLAN.md Section 5.1.1; milestone M2).

The drive is built from the exact displacement operator D(i eta) = exp[i eta (a + a^dagger)] by matrix
exponential, asserted against the analytic Laguerre elements over the populated range with the modulus
|<n'|D|n>| in (i eta)^{|n' - n|} form (Section 9.13), never from the Lamb-Dicke expansion by default.
"""

from __future__ import annotations

M2 = "milestone M2 (hilbert/operators.py, PLAN.md Section 5.1.1)"


def displacement_operator(*args: object, **kwargs: object) -> object:
    raise NotImplementedError(f"displacement_operator is {M2}")


def displacement_element_analytic(*args: object, **kwargs: object) -> complex:
    """<n'|D(i eta)|n> in closed form (associated Laguerre polynomials), the oracle for the exponential."""
    raise NotImplementedError(f"displacement_element_analytic is {M2}")


__all__ = ["displacement_element_analytic", "displacement_operator"]
