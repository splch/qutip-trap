"""A displayed value with its label, its unit and a line of detail for the hover text."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Shown:
    label: str
    value: float | int | str | bool | None
    unit: str = ""
    detail: str = ""
