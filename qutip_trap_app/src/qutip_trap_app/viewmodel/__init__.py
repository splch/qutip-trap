"""View-models: pure Python over the run record, no Flet import (PLAN.md Section 14.6 "Layering").

``viewmodel/`` "turns run records into view data, performs the coarse-graining, and resolves provenance; it is where the
correctness tests run". One module per zoom level (``machine`` Level 0, ``circuit`` Level 1, ``schedule`` Level 2,
``dynamics`` Level 3), the numerics panel every level carries (``numerics``), the catalogue of displayed quantities with
their ledger ids (``catalogue``) and the learning layer (``learn``). The Level 4 pages read the device card of the record
and the device model directly; their view-models arrive with M11.3.
"""

from __future__ import annotations

__all__: list[str] = []
