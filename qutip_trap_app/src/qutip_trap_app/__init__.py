"""qutip-trap-app: the zoomable application of PLAN.md Section 14 (milestone M11).

Milestone M11.1 (this package's first increment) is the layer under the screens: the run record and its storage policy
(``record``, ``storage``), on-demand re-simulation of a zoomed pulse with caching (``resim``), the provenance index generated
from Part II of the plan (``provenance``), and the pure-Python view-models over the record (``viewmodel``), where the
Section 9.11 consistency tests run. Nothing here imports Flet; the screens of M11.2 to M11.4 are thin views over these
modules. The one place the core is imported is ``core``.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
