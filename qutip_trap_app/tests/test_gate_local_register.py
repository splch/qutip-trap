"""Level 1 over a GATE_LOCAL record (the level the core picks above its joint-dimension guard, Section 11.5): the record
stores no trace, so the register after each gate is the walk's own register after the gate's step
(``GateLocalStep.register_after``); it agrees with the run's final register exactly, and a record with no register source
raises the typed error the view catches."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from qutip_trap_app.record import LiveRun, Record
from qutip_trap_app.viewmodel.circuit import (
    GATE_LOCAL_REGISTER_NOTE,
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
    assert record.diagnostics.level == "GATE_LOCAL" and not record.traces and record.replay is None
    assert record.gate_local is not None and all(
        st.register_after is not None for st in record.gate_local.steps
    )
    gates = timeline(record)
    assert gates, "the compiled timeline is there"
    for g in gates:
        reg = register_after(record, g.index)
        assert reg.weights_note == GATE_LOCAL_REGISTER_NOTE
        assert abs(sum(v.value for v in reg.populations.values() if isinstance(v.value, float)) - 1.0) < 1e-9
        assert 0.0 < float(reg.purity.value or 0.0) <= 1.0 + 1e-9
        assert 0.0 <= float(reg.fidelity.value or 0.0) <= 1.0 + 1e-9
        assert set(reg.bloch) == set(range(record.n_qubits))
    # the run's own final register: the last gate's register is the walk's, idle channels included, so the two agree exactly
    last = len(gates) - 1
    rho = gate_local_register_after(record, last)
    assert abs(float(np.real(np.trace(rho))) - 1.0) < 1e-9
    derived = fidelity_to_ket(rho, target_ket_after(record, last))
    recorded = record.results.register_fidelity
    assert recorded is not None and abs(derived - recorded) < 1e-9, (derived, recorded)
    assert live.core_record.gate_local is not None
    core_last = live.core_record.gate_local.steps[-1].register_after
    assert core_last is not None and np.max(np.abs(core_last - rho)) < 1e-12, (
        "the record copies the core's register"
    )
    # zoom in from Level 0 lands on a gate whose register renders
    store = Store()
    key = record.key()
    store.records, store.current = {key: record}, key
    route = child_route(store, f"/job/{key}")
    assert route is not None and route.startswith(f"/job/{key}/circuit/")
    gate = route.rsplit("/", 1)[1]
    register_after(record, next(g.index for g in gates if g.gate_id == gate))


def test_a_record_with_no_register_source_says_so(bell_gate_local: tuple[Record, LiveRun]) -> None:
    record, _live = bell_gate_local
    bare = dataclasses.replace(record, gate_local=None)
    with pytest.raises(RegisterUnavailable):
        register_after(bare, 0)
    assert record.gate_local is not None
    uncapped = dataclasses.replace(
        record,
        gate_local=dataclasses.replace(
            record.gate_local,
            steps=tuple(dataclasses.replace(st, register_after=None) for st in record.gate_local.steps),
        ),
    )
    # a register above the core's store cap, or a pure-state ensemble, stores no register after its steps
    with pytest.raises(RegisterUnavailable):
        register_after(uncapped, 0)
