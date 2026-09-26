"""The preparation sequence and its hand-off ``State``: the last stage that addressed a mode sets its occupation, the
pumped qubit is the register factor, and undefined modes or unpumped ions are refused."""

from __future__ import annotations

import math

import numpy as np
import pytest
import qutip as qt

from qutip_trap.device.presets import secular_trap, yb171_chain
from qutip_trap.dynamics.multilevel import MultiLevelOptions
from qutip_trap.dynamics.space import HilbertSpace, ModeTruncation
from qutip_trap.light.bloch import BlochModel, beam_for_transition
from qutip_trap.prep.doppler import doppler_cooling
from qutip_trap.prep.pumping import PumpingResult, optical_pumping
from qutip_trap.prep.recipe import HAND_OFF_TAIL, recipe_of, run_preparation
from qutip_trap.prep.sequence import (
    PreparationSequence,
    PreparationStage,
    doppler_stage,
    prepare_state,
    pulsed_sideband_stage,
    pump_stage,
)
from qutip_trap.prep.sideband import thermal_distribution
from qutip_trap.species import species
from qutip_trap.species.polarization import linear_polarization
from qutip_trap.species.raman import AtomicStructure
from qutip_trap.trap.crystal import solve_crystal
from qutip_trap.units import TWO_PI
from tests.fixtures import (
    TWO_LEVEL_EXCITED_PLUS,
    TWO_LEVEL_GROUND,
    gamma_rad_s,
    sigma_plus_beam,
    structure,
    two_level_atom,
)

YB = species("171Yb+")
QUBIT = ("S1/2 F=0 mF=0", "S1/2 F=1 mF=0")


def _pump() -> PumpingResult:
    st = AtomicStructure(YB, 5.0, (0.0, 0.0, 1.0))
    line = YB.transition("S1/2-P1/2")
    waist = 20e-6
    magic = tuple(linear_polarization((1.0, 0.0, 0.0), math.acos(1.0 / math.sqrt(3.0)), (0.0, 0.0, 1.0)))
    power = 0.5 * line.i_sat_w_m2 * math.pi * waist**2 / 2.0
    beam = beam_for_transition(
        st, "S1/2 F=1 mF=0", "P1/2 F=1 mF=0", 0.0, (1.0, 0.0, 0.0), magic, power_w=power, waist_m=waist
    )
    model = BlochModel(st, [beam], levels=("S1/2", "P1/2"), options=MultiLevelOptions(leak="renormalize"))
    return optical_pumping(model, [QUBIT[0]], duration_s=20e-6, samples=2001)


def _doppler(nbar: dict[int, float]) -> PreparationStage:
    return PreparationStage("doppler", "prep.doppler", nbar=nbar, ions=(0,))


def test_final_nbar_takes_the_last_stage_that_addressed_each_mode() -> None:
    seq = PreparationSequence(
        (
            _doppler({0: 10.0, 1: 8.0, 2: 3.0}),
            pulsed_sideband_stage({2: 0.05}, (0,)),
            pulsed_sideband_stage({1: 0.2}, (0,)),
        )
    )
    assert seq.final_nbar() == {0: 10.0, 1: 0.2, 2: 0.05}
    assert seq.final_pump(0) is None


def test_prepare_state_builds_the_state_with_thermal_modes_and_the_pumped_qubit() -> None:
    pump = _pump()
    seq = PreparationSequence(
        (
            _doppler({0: 10.0, 1: 8.0, 2: 3.0}),
            pulsed_sideband_stage({2: 0.05}, (0,)),
            pump_stage(pump, (0,)),
        )
    )
    space = HilbertSpace(
        ion_dims=(2,), resolved=(ModeTruncation(2, 12, (0, 3), 0.1),), enr_group=None, frozen=(0, 1)
    )
    state = prepare_state(space, seq, qubit_labels=QUBIT)
    assert state.provenance == ("prep.doppler", "prep.sideband.pulsed", "prep.pumping")
    assert state.motional.nbar[0] == 10.0 and state.motional.nbar[1] == 8.0
    assert state.motional.nbar[2] == pytest.approx(0.05, rel=1e-6)
    assert state.motional.frozen == (0, 1)
    rho_int = state.internal
    assert rho_int.shape == (2, 2)
    assert float(np.real(rho_int.full()[0, 0])) == pytest.approx(1.0 - pump.preparation_error, abs=1e-12)
    assert pump.preparation_error < 1e-4
    assert state.joint is not None and state.joint.isoper
    # the register factor is the pumped qubit: population of |0> in the joint state
    p0 = qt.expect(qt.tensor(qt.basis(2, 0).proj(), qt.qeye(12)), state.joint)
    assert float(np.real(p0)) == pytest.approx(1.0 - pump.preparation_error, abs=1e-9)
    # the pumps' recoil heating adds to the last cooling stage
    heated = prepare_state(space, seq, qubit_labels=QUBIT, extra_nbar={2: 0.01})
    assert heated.motional.nbar[2] == pytest.approx(0.06, rel=1e-6)


def test_the_pulsed_sideband_hand_off_keeps_the_mean_and_its_note_says_how_far_from_thermal_it_is() -> None:
    """On the default 171Yb+ chain the pulsed cooling of the two x modes leaves the mean the thermal hand-off keeps, with
    <n^2> 7.33 and 5.64 times the thermal state's (1e-2) and at most 1e-4 of the population above n = 20 and 15 where the
    thermal state's stops at n = 2; the per-mode preparation note, which a run carries into its approximations, says so."""
    dev = yb171_chain(2).device
    prep = run_preparation(dev, recipe_of(dev))
    assert set(prep.sideband_populations) == set(prep.sideband_nbar) == {2, 3}
    for m, ratio, top in ((2, 7.33, 20), (3, 5.64, 15)):
        p = prep.sideband_populations[m]
        n = np.arange(p.size, dtype=float)
        assert p.sum() == pytest.approx(1.0)
        assert float(n @ p) == pytest.approx(prep.sideband_nbar[m], rel=1e-12)
        thermal = thermal_distribution(prep.sideband_nbar[m], p.size)
        assert float(n**2 @ p) / float(n**2 @ thermal) == pytest.approx(ratio, rel=1e-2)
        assert [int(np.searchsorted(np.cumsum(q), 1.0 - HAND_OFF_TAIL)) for q in (p, thermal)] == [top, 2]
        (note,) = [x for x in prep.notes if x.startswith(f"mode {m}:")]
        assert "handed off as the thermal state of that nbar" in note
        assert f"above n = 2, against the pulsed distribution's {float(n**2 @ p):.3g} and n = {top}" in note


def test_prepare_state_refuses_undefined_modes_and_unpumped_ions() -> None:
    pump = _pump()
    seq = PreparationSequence((_doppler({0: 10.0}), pump_stage(pump, (0,))))
    space = HilbertSpace(
        ion_dims=(2,), resolved=(ModeTruncation(1, 8, (0, 3), 0.1),), enr_group=None, frozen=(0,)
    )
    with pytest.raises(ValueError, match="no cooling stage addressed"):
        prepare_state(space, seq, qubit_labels=QUBIT)
    space2 = HilbertSpace(
        ion_dims=(2, 2), resolved=(ModeTruncation(0, 8, (0, 3), 0.1),), enr_group=None, frozen=()
    )
    seq2 = PreparationSequence((_doppler({0: 1.0}), pump_stage(pump, (0,))))
    with pytest.raises(ValueError, match="never pumped"):
        prepare_state(space2, seq2, qubit_labels=QUBIT)


def test_doppler_stage_wraps_the_rate_result() -> None:
    sp = two_level_atom()
    g = gamma_rad_s()
    trap = secular_trap((2.5e6, 2.6e6, 0.05 * g / TWO_PI))
    crystal = solve_crystal(trap, (sp,))
    st = structure(sp)
    beam = sigma_plus_beam(st, TWO_LEVEL_GROUND, TWO_LEVEL_EXCITED_PLUS, 0.05 * g, -0.5 * g)
    res = doppler_cooling(st, [beam], crystal, modes=[0])
    stage = doppler_stage(res)
    assert stage.kind == "doppler" and stage.nbar is not None
    assert stage.nbar[0] == pytest.approx(res.mode(0).nbar)
    assert stage.ions == (0,)
