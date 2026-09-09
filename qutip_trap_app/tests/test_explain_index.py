"""The explain drawer's content for every Part II subsection (PLAN.md Section 14.5 "Explain panel"; M11.4): the provenance
index carries every section's own Markdown (Part II and the rest of the plan the screens cite), every concept, catalogue
quantity and preset resolves to a section whose text (or whose subsections) the drawer can show, and a chip knows which
section to open."""

from __future__ import annotations

from qutip_trap_app.provenance import TEXT_SECTIONS, ProvenanceIndex
from qutip_trap_app.viewmodel import learn
from qutip_trap_app.viewmodel.catalogue import CATALOGUE
from qutip_trap_app.viewmodel.presets import PRESETS
from qutip_trap_app.views.shell import LEVEL_SECTIONS


def test_every_part_ii_section_has_its_text() -> None:
    idx = ProvenanceIndex.load()
    part_ii = idx.part_ii()
    assert len(part_ii) > 60
    without = []
    for info in part_ii:
        text = idx.section_text(info.number)
        if not text:
            # a heading with no text of its own (Section 4, 4.1, ...): the drawer lists its subsections instead
            assert idx.subsections(info.number), info.number
            without.append(info.number)
            continue
        assert len(text) > 40, info.number
    assert set(without) <= {"4", "4.1", "5", "6", "7", "8"}, without
    assert TEXT_SECTIONS is None, "every section ships its text: the governing sections reach outside Part II"
    assert idx.section_text("14.5").startswith("- **Provenance chips**"), (
        "the app's own section, for the chips concept"
    )
    assert idx.section_text("3.4") and idx.section_text("11.1") and idx.section_text("13")


def test_every_governing_section_resolves_to_text() -> None:
    idx = ProvenanceIndex.load()
    numbers = set(LEVEL_SECTIONS.values()) | set(learn.PAGE_SECTIONS.values())
    numbers |= {c.section for c in learn.CONCEPTS.values()}
    numbers |= {q.section for q in CATALOGUE.values()}
    numbers |= {p.section for p in PRESETS.values()}
    for n in sorted(numbers):
        idx.section(n)
        nearest = idx.nearest_section_with_text(n)
        assert idx.section_text(nearest) or idx.subsections(nearest), (n, nearest)


def test_chips_open_a_section_with_text() -> None:
    idx = ProvenanceIndex.load()
    for q in list(CATALOGUE.values())[:80]:
        section = idx.section_for_chip(q.ledger_id)
        assert idx.section_text(section), (q.id, section)
    assert idx.section_for_chip("conv.lamb_dicke").startswith("4.1")
    assert idx.on_open_section is None, "headless: chips are hover-only until a session installs the opener"
