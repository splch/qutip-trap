"""The discrimination drills of the Learn view, generated from the current run record (DESIGN.md Section 3).

Interleaving helps when the goal is telling similar things apart (pooled effect g = 0.42 for inductive learning), so the
drills mix four discriminations in immediate succession: is this device-card number an estimate, calibrated, measured or a
device parameter; what does this number's chip say was checked; was this mode resolved, frozen or dropped; does this bar sit
inside two error bars of its target. Every answer is read from the record or the provenance index, so a drill never
disagrees with the screen it is about.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

from qutip_trap_app.provenance import TAG_GLYPH, TAGS, ProvenanceIndex
from qutip_trap_app.record import Record
from qutip_trap_app.viewmodel.catalogue import CATALOGUE
from qutip_trap_app.viewmodel.machine import device_card_view, histogram

DrillKind = Literal["status", "chip_tag", "mode_class", "bar_within"]

STATUS_OPTIONS: tuple[str, ...] = ("estimate", "calibrated", "measured", "derived", "device parameter")
MODE_OPTIONS: tuple[str, ...] = ("resolved", "frozen", "dropped")
BAR_OPTIONS: tuple[str, ...] = ("inside two error bars", "outside two error bars")
N_DRILLS = 8

CONCEPT_OF: dict[DrillKind, str] = {
    "status": "calibration",
    "chip_tag": "provenance_tags",
    "mode_class": "truncation_and_convergence",
    "bar_within": "target_vs_simulated",
}


@dataclass(frozen=True)
class Drill:
    id: str
    kind: DrillKind
    question: str
    options: tuple[str, ...]
    answer: str
    concept_id: str
    where: str
    """Where on the screens the answer can be checked, in plain words."""


def _value_text(value: object, unit: str) -> str:
    return (f"{value:.4g} {unit}" if isinstance(value, float) else f"{value} {unit}").strip()


def drills_for(record: Record, index: ProvenanceIndex) -> tuple[Drill, ...]:
    """Up to ``N_DRILLS`` drills over ``record``, the four kinds interleaved, in an order fixed by the record's key (the same
    set on re-render, another set on another record)."""
    rng = random.Random(record.key())
    card = device_card_view(record)
    pools: dict[DrillKind, list[Drill]] = {k: [] for k in CONCEPT_OF}
    for row in card.rows + card.spam + card.gate_errors:
        word = next((w for w in STATUS_OPTIONS if row.status.lower().startswith(w)), None)
        if word is None:
            continue
        q = CATALOGUE[row.value.quantity]
        pools["status"].append(
            Drill(
                f"status:{row.label}",
                "status",
                f"On the device card, {row.label} reads {_value_text(row.value.value, q.unit)}. Is that number",
                STATUS_OPTIONS,
                word,
                CONCEPT_OF["status"],
                "the status word beside the number on the device card (Level 0)",
            )
        )
    seen: set[str] = set()
    for row in card.rows + card.spam:
        q = CATALOGUE[row.value.quantity]
        if q.ledger_id in seen:
            continue
        seen.add(q.ledger_id)
        chip = index.chip(q.ledger_id)
        pools["chip_tag"].append(
            Drill(
                f"chip:{q.id}",
                "chip_tag",
                f"The chip on '{q.label}' ({row.label}). What checking does it record?",
                tuple(f"{TAG_GLYPH[t]} {t}" for t in TAGS),
                f"{TAG_GLYPH[chip.tag]} {chip.tag}",
                CONCEPT_OF["chip_tag"],
                "hover the chip beside the number (Level 0, device card)",
            )
        )
    for m, cls in sorted(record.space.mode_class.items()):
        mode = next((x for x in record.device_card.modes if x.index == m), None)
        if cls not in MODE_OPTIONS or mode is None:
            continue
        pools["mode_class"].append(
            Drill(
                f"mode:{m}",
                "mode_class",
                f"Mode {m} ({mode.family} {mode.family_index} at {mode.omega_hz / 1e6:.3f} MHz). In this run it was",
                MODE_OPTIONS,
                cls,
                CONCEPT_OF["mode_class"],
                "the mode classes in the numerics strip, or the closure table on Level 2",
            )
        )
    for bar in histogram(record).bars:
        dev = bar.deviation_in_error_bars
        if dev is None:
            continue
        pools["bar_within"].append(
            Drill(
                f"bar:{bar.key}",
                "bar_within",
                f"The {bar.key} bar is {bar.probability.value:.3f} against a target of {bar.target_probability.value:.3f}, "
                f"with an error bar of {bar.error_bar.value:.3f}. It sits",
                BAR_OPTIONS,
                BAR_OPTIONS[0] if abs(dev) <= 2.0 else BAR_OPTIONS[1],
                CONCEPT_OF["bar_within"],
                "the histogram and its table on Level 0",
            )
        )
    for pool in pools.values():
        rng.shuffle(pool)
    out: list[Drill] = []
    # interleaved: one of each kind in turn, so confusable discriminations follow each other in immediate succession
    while len(out) < N_DRILLS and any(pools.values()):
        for pool in pools.values():
            if pool and len(out) < N_DRILLS:
                out.append(pool.pop())
    return tuple(out)
