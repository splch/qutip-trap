"""Flet views: one module per level over the pure view-models (PLAN.md Section 14.6 "Layering"; DESIGN.md Section 5).

``views/`` is the only place Flet is imported. Every number a view shows is a ``Shown`` from the view-models rendered by
``common.shown`` with its provenance chip; nothing here computes physics.
"""

from __future__ import annotations

__all__: list[str] = []
