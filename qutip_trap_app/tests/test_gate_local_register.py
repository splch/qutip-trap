"""Level 1 over a GATE_LOCAL record: the record stores no trace, so the register after each gate is the walk's own register
after the gate's step; it agrees with the run's final register exactly, and a record with no register says why."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from qutip_trap_app.record import LiveRun, Record
from qutip_trap_app.viewmodel.circuit import (
    RegisterUnavailable,
    fidelity_to_ket,
    gate_local_register_after,
    register_after,
    target_ket_after,
    timeline,
)
from qutip_trap_app.views.shell import child_route
from qutip_trap_app.views.state import Store


def test_the_register_of_a_gate_local_run_is_the_walks_own(bell_gate_local: tuple[Record, LiveRun]) -> None:
    record, live = bell_gate_local
    assert record.diagnostics.level == "GATE_LOCAL" and not record.traces and record.gate_local is not None
    assert all(st.register_after is not None for st in record.gate_local.steps)
    gates = timeline(record)
    assert gates
    for g in gates:
        reg = register_after(record, g.index)
        assert "GATE_LOCAL" in reg.note
        assert abs(sum(reg.populations.values()) - 1.0) < 1e-9
        assert 0.0 < float(reg.purity.value or 0.0) <= 1.0 + 1e-9
        assert 0.0 <= float(reg.fidelity.value or 0.0) <= 1.0 + 1e-9
        assert set(reg.bloch) == set(range(record.n_ions))
    last = len(gates) - 1
    rho = gate_local_register_after(record, last)
    derived = fidelity_to_ket(rho, target_ket_after(record, last))
    recorded = record.results.register_fidelity
    assert recorded is not None and abs(derived - recorded) < 1e-9, (derived, recorded)
    assert live.run.gate_local is not None
    core_last = live.run.gate_local.steps[-1].register_after
    assert core_last is not None and np.max(np.abs(core_last - rho)) < 1e-12, (
        "the record copies the core's register"
    )
    # zooming in from the machine lands on a gate whose register renders
    store = Store()
    store.records, store.current = {record.key: record}, record.key
    route = child_route(store, f"/job/{record.key}")
    assert route is not None and route.startswith(f"/job/{record.key}/circuit/")
    register_after(record, next(g.index for g in gates if g.gate_id == route.rsplit("/", 1)[1]))


def test_a_record_with_no_register_says_why(bell_gate_local: tuple[Record, LiveRun]) -> None:
    record, _live = bell_gate_local
    with pytest.raises(RegisterUnavailable, match="neither traces nor"):
        register_after(dataclasses.replace(record, gate_local=None), 0)
    assert record.gate_local is not None
    bare = dataclasses.replace(
        record,
        gate_local=dataclasses.replace(
            record.gate_local,
            steps=tuple(dataclasses.replace(st, register_after=None) for st in record.gate_local.steps),
        ),
    )
    with pytest.raises(RegisterUnavailable, match="kept no register"):
        register_after(bare, 0)
