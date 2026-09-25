"""Every quantity in the view-model catalogue resolves to a tagged Part II item
(a static test over the catalogue; the build fails on an untagged quantity). Also the Section 14.6 core contract: the
application imports the core in one module only."""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from qutip_trap_app import device_layer, provenance, resim
from qutip_trap_app.record import LiveRun, Record
from qutip_trap_app.viewmodel import catalogue, learn
from qutip_trap_app.viewmodel import presets as presets_vm
from qutip_trap_app.viewmodel.catalogue import CATALOGUE, Shown
from qutip_trap_app.viewmodel.circuit import compile_report, phase_register, register_after, timeline
from qutip_trap_app.viewmodel.drills import CONCEPT_OF, drills_for
from qutip_trap_app.viewmodel.dynamics import pulse_dynamics, recorded_zoom
from qutip_trap_app.viewmodel.machine import device_card_view, histogram, shot
from qutip_trap_app.viewmodel.numerics import numerics_panel
from qutip_trap_app.viewmodel.physics import (
    cooling_view,
    crystal_view,
    gate_rows,
    hamiltonian_view,
    layer_card_view,
    light_view,
    noise_view,
    readout_view,
    species_view,
    trap_view,
)
from qutip_trap_app.viewmodel.schedule import closure, pulse_view, time_axis

SRC = Path(__file__).resolve().parents[1] / "src" / "qutip_trap_app"


def test_every_catalogue_and_concept_id_is_in_the_ledger() -> None:
    idx = provenance.ProvenanceIndex.load()
    assert not idx.missing(catalogue.ledger_ids())
    assert not idx.missing(learn.ledger_ids())
    assert not idx.missing(presets_vm.ledger_ids())
    assert presets_vm.catalogue_ids() <= set(CATALOGUE)
    assert set(CONCEPT_OF.values()) <= set(learn.CONCEPTS)
    for q in CATALOGUE.values():
        chip = idx.chip(q.ledger_id)
        assert chip.tag in provenance.TAGS and chip.label.startswith(chip.glyph)
        idx.section(q.section)  # the explain panel's target exists
    for c in learn.CONCEPTS.values():
        idx.section(c.section)
    assert len(idx.part_ii()) > 60


def test_chip_carries_section_source_and_corrected_form() -> None:
    idx = provenance.ProvenanceIndex.load()
    chip = idx.chip("conv.lamb_dicke")
    assert chip.tag == "verified" and "Wineland" in chip.source and "eta" in chip.equation
    assert "4.1.7" in chip.sections and idx.sections_for("conv.lamb_dicke")[0].part_ii
    corrected = idx.chip("conv.micromotion_correction")
    assert corrected.tag == "corrected" and corrected.corrected_form
    assert provenance.sections_named("9.16 row 4.5-7; 4.5.5 (M4 audit 2026-09-07)") == ("9.16", "4.5.5")


def test_the_packaged_asset_loads_the_index_of_the_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(provenance, "ASSET_PATH", tmp_path / "provenance_index.json")
    provenance.main()
    monkeypatch.setattr(
        provenance, "repository_root", lambda: tmp_path
    )  # a packaged build: no PLAN.md beside it
    shipped = provenance.ProvenanceIndex.load()
    source = provenance.ProvenanceIndex(provenance.generate_index())
    assert shipped.chip("conv.lamb_dicke") == source.chip("conv.lamb_dicke")
    assert shipped.part_ii() == source.part_ii() and shipped.section_text("14.5") == source.section_text(
        "14.5"
    )


def _shown(obj: object, out: list[Shown], depth: int = 0) -> None:
    if isinstance(obj, Shown):
        out.append(obj)
        return
    if depth > 9:
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _shown(v, out, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _shown(v, out, depth + 1)
    elif dataclasses.is_dataclass(obj):
        for f in dataclasses.fields(obj):
            _shown(getattr(obj, f.name), out, depth + 1)


def test_every_displayed_quantity_of_the_bell_record_has_a_chip(bell: tuple[Record, LiveRun]) -> None:
    record, _ = bell
    idx = provenance.ProvenanceIndex.load()
    views: list[object] = [
        histogram(record),
        device_card_view(record),
        shot(record, 0),
        timeline(record),
        register_after(record, 0),
        phase_register(record),
        compile_report(record),
        time_axis(record),
        pulse_view(record, 0),
        closure(record, next(g.gate_id for g in record.schedule.gates)),
        numerics_panel(record),
    ]
    shown: list[Shown] = []
    for v in views:
        _shown(v, shown)
    assert len(shown) > 80
    for s in shown:
        q = CATALOGUE[s.quantity]
        assert idx.chip(q.ledger_id), s.quantity
    # the drills are generated from the same record and answer from it (DESIGN.md Section 3)
    drills = drills_for(record, idx)
    assert len(drills) >= 4 and len({d.kind for d in drills}) == 4, "four discriminations, interleaved"
    assert all(d.answer in d.options and d.concept_id in learn.CONCEPTS for d in drills)
    assert drills == drills_for(record, idx), "deterministic for one record"
    assert [d.kind for d in drills[:4]] == ["status", "chip_tag", "mode_class", "bar_within"]


def test_every_displayed_quantity_of_levels_3_and_4_has_a_chip(bell: tuple[Record, LiveRun]) -> None:
    """The Level 3 dynamics of the recorded trace and the Hamiltonian record, and every Level 4 page over the device layer
    of the record's own device."""
    record, live = bell
    idx = provenance.ProvenanceIndex.load()
    step = next(s.index for s in record.schedule.steps if s.gate_id.startswith("ms"))
    record, ham = resim.hamiltonian_record(record, live, step)
    layer = device_layer.derive_device_layer(record.job.device.build(), table=live.table, sweeps=False)
    views: list[object] = [
        pulse_dynamics(record, recorded_zoom(record, step)),
        hamiltonian_view(record, ham),
        species_view(layer),
        trap_view(layer),
        crystal_view(layer),
        light_view(layer),
        noise_view(layer),
        cooling_view(layer),
        readout_view(layer, record.table),
        gate_rows(layer, record.table),
        layer_card_view(layer, record.table),
    ]
    shown: list[Shown] = []
    for v in views:
        _shown(v, shown, 0)
    assert len(shown) > 200
    for s in shown:
        q = CATALOGUE[s.quantity]
        assert idx.chip(q.ledger_id), s.quantity


def test_core_is_imported_in_one_module_only() -> None:
    offenders: list[str] = []
    for path in SRC.rglob("*.py"):
        if path.name == "core.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            if any(n == "qutip_trap" or n.startswith("qutip_trap.") for n in names):
                offenders.append(str(path.relative_to(SRC)))
    assert not offenders, f"qutip_trap imported outside core.py: {offenders}"


def test_viewmodels_do_not_import_flet() -> None:
    for path in (SRC / "viewmodel").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not any(a.name.startswith("flet") for a in node.names), path
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("flet"), path
