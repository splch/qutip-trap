"""qutip-trap: a first-principles trapped-ion quantum computer simulator built on QuTiP.

The specification is ``PLAN.md`` at the repository root. Milestone M0 (PLAN.md Section 10) delivers
the scaffolding and the public interfaces of Appendix E, re-exported by :mod:`qutip_trap.api`; the
physics arrives milestone by milestone, and every stub in this package names the milestone that
implements it in its ``NotImplementedError``.
"""

from __future__ import annotations

__version__ = "0.0.1"

__all__ = ["__version__"]
