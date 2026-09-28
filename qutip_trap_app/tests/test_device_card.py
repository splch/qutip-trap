"""The Level 0 device card names its gate errors by gate kind and ions, entangling gate first, one row per kind, ions and
source, with no compiler piece id in a label."""

from __future__ import annotations

from qutip_trap_app.record import LiveRun, Record
from qutip_trap_app.viewmodel.machine import device_card_view


def test_the_card_names_its_gate_errors_by_gate_and_ions_the_entangling_gate_first(
    bell: tuple[Record, LiveRun],
) -> None:
    record, _live = bell
    card = device_card_view(record)
    rows = card.gate_errors
    assert rows and rows[0].label == "MS on ions 0 and 1", [r.label for r in rows]
    assert len({(r.label, r.status) for r in rows}) == len(rows), "one row per gate kind, ions and source"
    assert all("[" not in r.label for r in rows), "no compiler piece id such as gpi2[3]"
    single = [r for r in rows if r.label.startswith("GPi2")]
    assert single and all(r.label in ("GPi2 on ion 0", "GPi2 on ion 1") for r in single)
    assert any(r.label == "qubit frequency, ion 0" for r in card.rows)
