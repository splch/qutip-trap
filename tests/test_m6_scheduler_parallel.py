"""Section 7.3's serialization rule under parallel addressing (M6 audit item P1-3).

"MS gates are serialized (one at a time per crystal in the first release)" and single-qubit gates "run in parallel if the
device model allows parallel addressing". Before the fix an entangling gate under ``parallel=True`` started at
``max(clock[a], clock[b])`` and never touched the crystal-wide clock, so two MS gates on disjoint pairs of a four-ion chain
were scheduled on top of each other on the SAME global Raman pair - a waveform one AWG cannot play, which
``Schedule.__post_init__`` accepts because it only forbids overlap on a shared ion."""

from __future__ import annotations

import dataclasses
import math

import pytest

from qutip_trap.api import Circuit, Operation
from qutip_trap.control.schedule import ScheduleError, schedule
from qutip_trap.control.shaping import gate_modes, symmetric_pulse
from tests.m4_fixtures import derived_seeds, table_with_waveform
from tests.m6_fixtures import circuit_fixture

MS = math.pi / 2.0


@pytest.fixture(scope="module")
def four_ion():  # type: ignore[no-untyped-def]
    """A four-ion chain with one calibrated symmetric waveform served for both disjoint pairs (0, 1) and (2, 3)."""
    fx = circuit_fixture(4, with_recipe=False)
    waveforms = {}
    for pair in ((0, 1), (2, 3)):
        modes = gate_modes(fx.device, pair, (0, 1))
        waveforms[pair] = symmetric_pulse(
            modes, gate_mode=modes.modes[-1], loops=1, epsilon_hz=20e3, pair=pair
        ).waveform
    rabi, stark = derived_seeds(fx.device, fx.gate_drives)
    table = table_with_waveform((0, 1), waveforms[(0, 1)], rabi_hz=rabi, stark_hz=stark)
    table = dataclasses.replace(table, ms=waveforms)
    return fx, table


def _span(sch, gate_id_prefix: str) -> tuple[float, float]:  # type: ignore[no-untyped-def]
    ps = [p for p in sch.pulses if (p.gate_id or "").startswith(gate_id_prefix)]
    assert ps, f"no pulses with gate id {gate_id_prefix!r}"
    return min(p.t_start_s for p in ps), max(p.t_end_s for p in ps)


@pytest.mark.parametrize("parallel_addressing", [False, True])
def test_two_ms_gates_on_disjoint_pairs_never_overlap(four_ion, parallel_addressing: bool) -> None:  # type: ignore[no-untyped-def]
    """Section 7.3: one entangling gate at a time per crystal, whether or not the chain addresses in parallel."""
    fx, table = four_ion
    dev = dataclasses.replace(
        fx.device,
        hardware=dataclasses.replace(fx.device.hardware, parallel_addressing=parallel_addressing),
    )
    circ = Circuit(
        4,
        (Operation("ms", (0, 1), (0.0, 0.0, MS)), Operation("ms", (2, 3), (0.0, 0.0, MS))),
        (0, 1, 2, 3),
    )
    sch = schedule(circ, dev, table, gate_drives=fx.gate_drives, entangling_drives=fx.entangling_drives)
    first, second = _span(sch, "ms[0]"), _span(sch, "ms[1]")
    assert first[1] <= second[0] + 1e-15, (
        f"parallel_addressing={parallel_addressing}: ms[0] {first} overlaps ms[1] {second}"
    )
    # they are separated by exactly the hardware dead time, not by more
    assert second[0] == pytest.approx(first[1] + dev.hardware.dead_time_s)
    assert len(sch.gates) == 2 and {g.pair for g in sch.gates} == {(0, 1), (2, 3)}
    assert sch.gates[0].t_end_s <= sch.gates[1].t_start_s + 1e-15
    # every entangling pulse of the two gates draws on the same global Raman pair, which is why they cannot share a slot
    assert {g.beams for g in sch.gates} == {(0, 1)}


def test_parallel_addressing_overlaps_single_qubit_gates_but_the_serial_default_does_not(four_ion) -> None:  # type: ignore[no-untyped-def]
    """The other half of Section 7.3: with the device model allowing it, carrier pulses on distinct ions do run together."""
    fx, table = four_ion
    circ = Circuit(4, (Operation("gpi2", (0,), (0.0,)), Operation("gpi2", (2,), (0.0,))), (0, 1, 2, 3))
    serial = schedule(circ, fx.device, table, gate_drives=fx.gate_drives)
    a, b = _span(serial, "gpi2[0]"), _span(serial, "gpi2[1]")
    assert a[1] <= b[0], "the serial default sequences single-qubit gates"
    dev_par = dataclasses.replace(
        fx.device, hardware=dataclasses.replace(fx.device.hardware, parallel_addressing=True)
    )
    par = schedule(circ, dev_par, table, gate_drives=fx.gate_drives)
    a2, b2 = _span(par, "gpi2[0]"), _span(par, "gpi2[1]")
    assert a2 == b2, "distinct addressing beams play the two carrier pulses in the same window"


def test_a_single_qubit_gate_after_an_ms_still_waits_for_it_under_parallel_addressing(four_ion) -> None:  # type: ignore[no-untyped-def]
    """The pair clocks must move too: a carrier pulse on a gate ion cannot run inside the entangling pulse it follows."""
    fx, table = four_ion
    dev = dataclasses.replace(
        fx.device, hardware=dataclasses.replace(fx.device.hardware, parallel_addressing=True)
    )
    circ = Circuit(
        4,
        (
            Operation("ms", (0, 1), (0.0, 0.0, MS)),
            Operation("gpi2", (1,), (0.0,)),
            Operation("gpi2", (3,), (0.0,)),
        ),
        (0, 1, 2, 3),
    )
    sch = schedule(circ, dev, table, gate_drives=fx.gate_drives, entangling_drives=fx.entangling_drives)
    ms_span = _span(sch, "ms[0]")
    on_gate_ion = _span(sch, "gpi2[1]")
    spectator = _span(sch, "gpi2[2]")
    assert on_gate_ion[0] >= ms_span[1], "ion 1 took part in the MS gate"
    # ion 3's beam is free during the MS gate, so a parallel chain starts its carrier pulse at t0
    assert spectator[0] == pytest.approx(0.0)


def test_parallel_true_is_refused_when_the_device_model_does_not_allow_it(four_ion) -> None:  # type: ignore[no-untyped-def]
    """Section 7.3 conditions parallelism on the device model, so the flag cannot contradict ``HardwareChain``."""
    fx, table = four_ion
    circ = Circuit(4, (Operation("gpi2", (0,), (0.0,)),), (0, 1, 2, 3))
    assert fx.device.hardware.parallel_addressing is False
    with pytest.raises(ScheduleError, match="parallel_addressing"):
        schedule(circ, fx.device, table, gate_drives=fx.gate_drives, parallel=True)
    # explicitly serial on a parallel chain is allowed (the caller may want the serial path, e.g. crosstalk echoes)
    dev = dataclasses.replace(
        fx.device, hardware=dataclasses.replace(fx.device.hardware, parallel_addressing=True)
    )
    assert schedule(circ, dev, table, gate_drives=fx.gate_drives, parallel=False).pulses
