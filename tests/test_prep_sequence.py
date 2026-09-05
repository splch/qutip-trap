"""The preparation sequence, its stage-order guard and the hand-off ``State`` (PLAN.md Sections 4.2.6, 4.2.7, 5.7; Section 9.17 row
"Preparation stage order"; milestone M3)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import qutip as qt

from qutip_trap.api import HilbertSpace, ModeTruncation
from qutip_trap.dynamics.multilevel import MultiLevelOptions
from qutip_trap.light.bloch import BlochModel, beam_for_transition
from qutip_trap.prep.pumping import optical_pumping
from qutip_trap.prep.sequence import (
    PreparationSequence,
    PreparationStage,
    StageOrderError,
    doppler_stage,
    prepare_state,
    pulsed_sideband_stage,
    pump_stage,
    sideband_stage,
)
from qutip_trap.species import species
from qutip_trap.species.polarization import linear_polarization
from qutip_trap.species.raman import AtomicStructure

YB = species("171Yb+")
QUBIT = ("S1/2 F=0 mF=0", "S1/2 F=1 mF=0")


def _pump():  # type: ignore[no-untyped-def]
    st = AtomicStructure(YB, 5.0, (0.0, 0.0, 1.0))
    line = YB.transition("S1/2-P1/2")
    waist = 20e-6
    magic = tuple(linear_polarization((1.0, 0.0, 0.0), math.acos(1.0 / math.sqrt(3.0)), (0.0, 0.0, 1.0)))
    power = 0.5 * line.i_sat_w_m2 * math.pi * waist**2 / 2.0
    beam = beam_for_transition(
        st, "S1/2 F=1 mF=0", "P1/2 F=1 mF=0", 0.0, (1.0, 0.0, 0.0), magic, power_w=power, waist_m=waist
    )  # type: ignore[arg-type]
    model = BlochModel(st, [beam], levels=("S1/2", "P1/2"), options=MultiLevelOptions(leak="renormalize"))
    return optical_pumping(model, [QUBIT[0]], duration_s=20e-6, samples=2001)


def _cooling(kind: str, nbar: dict[int, float]) -> PreparationStage:
    if kind == "doppler":
        return PreparationStage("doppler", "prep.doppler", nbar=nbar, ions=(0,))
    if kind == "sideband":
        return sideband_stage(nbar, (0,))
    return pulsed_sideband_stage(nbar, (0,))


def test_stage_order_is_enforced() -> None:
    pump = pump_stage(_pump(), (0,))
    doppler = _cooling("doppler", {0: 10.0, 1: 8.0, 2: 3.0})
    sideband = _cooling("sideband", {2: 0.05})
    PreparationSequence((doppler, sideband, pump))
    PreparationSequence((doppler, pump, doppler, pump))
    PreparationSequence((doppler,))  # cooling only, no pump yet
    with pytest.raises(StageOrderError):
        PreparationSequence((pump, doppler, pump))
    with pytest.raises(StageOrderError):
        PreparationSequence((sideband, pump))
    with pytest.raises(StageOrderError):
        PreparationSequence((doppler, pump, sideband))
    with pytest.raises(StageOrderError):
        PreparationSequence(())
    with pytest.raises(ValueError):
        PreparationStage("pump", "x")
    with pytest.raises(ValueError):
        PreparationStage("doppler", "x")


def test_final_nbar_takes_the_last_stage_that_addressed_each_mode() -> None:
    seq = PreparationSequence(
        (
            _cooling("doppler", {0: 10.0, 1: 8.0, 2: 3.0}),
            _cooling("sideband", {2: 0.05}),
            _cooling("pulsed", {1: 0.2}),
        )
    )
    assert seq.final_nbar() == {0: 10.0, 1: 0.2, 2: 0.05}
    assert seq.kinds == ("doppler", "sideband", "pulsed_sideband") and not seq.ends_with_pump
    assert seq.final_pump(0) is None


def test_prepare_state_builds_the_appendix_e_state_with_thermal_modes_and_the_pumped_qubit() -> None:
    pump = _pump()
    seq = PreparationSequence(
        (
            _cooling("doppler", {0: 10.0, 1: 8.0, 2: 3.0}),
            _cooling("sideband", {2: 0.05}),
            pump_stage(pump, (0,)),
        )
    )
    space = HilbertSpace(
        ion_dims=(2,), resolved=(ModeTruncation(2, 12, (0, 3), 0.1),), enr_group=None, frozen=(0, 1)
    )
    state = prepare_state(space, seq, qubit_labels=QUBIT)
    assert state.provenance == ("prep.doppler", "prep.sideband", "prep.pumping")
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


def test_prepare_state_refuses_undefined_modes_and_unpumped_ions() -> None:
    pump = _pump()
    seq = PreparationSequence((_cooling("doppler", {0: 10.0}), pump_stage(pump, (0,))))
    space = HilbertSpace(
        ion_dims=(2,), resolved=(ModeTruncation(1, 8, (0, 3), 0.1),), enr_group=None, frozen=(0,)
    )
    with pytest.raises(ValueError):
        prepare_state(space, seq, qubit_labels=QUBIT)
    space2 = HilbertSpace(
        ion_dims=(2, 2), resolved=(ModeTruncation(0, 8, (0, 3), 0.1),), enr_group=None, frozen=()
    )
    seq2 = PreparationSequence((_cooling("doppler", {0: 1.0}), pump_stage(pump, (0,))))
    with pytest.raises(ValueError):
        prepare_state(space2, seq2, qubit_labels=QUBIT)
    # a coolant ion (never pumped) with an explicitly prepared qubit neighbour: sympathetic bookkeeping
    state = prepare_state(space2, seq2, qubit_labels=QUBIT, internal={1: qt.basis(2, 1)})
    assert state.internal.shape == (4, 4)
    assert float(np.real(state.internal.full()[1, 1])) == pytest.approx(
        1.0 - pump.preparation_error, abs=1e-9
    )


def test_doppler_stage_wraps_the_rate_result() -> None:
    from qutip_trap.api import Trap
    from qutip_trap.prep.doppler import doppler_cooling
    from qutip_trap.trap.crystal import solve_crystal
    from tests.bloch_fixtures import (
        TWO_LEVEL_EXCITED_PLUS,
        TWO_LEVEL_GROUND,
        gamma_rad_s,
        sigma_plus_beam,
        structure,
        two_level_atom,
    )

    sp = two_level_atom()
    g = gamma_rad_s()
    trap = Trap(
        omega_hz=(2.5e6, 2.6e6, 0.05 * g / 6.283185307179586),
        axis_angle_rad=0.0,
        rf=None,
        dc=None,
        geometry=None,
        stray_field_v_per_m=(0.0, 0.0, 0.0),
        shim_voltages_v={},
    )
    crystal = solve_crystal(trap, (sp,))
    st = structure(sp)
    beam = sigma_plus_beam(st, TWO_LEVEL_GROUND, TWO_LEVEL_EXCITED_PLUS, 0.05 * g, -0.5 * g)
    res = doppler_cooling(st, [beam], crystal, modes=[0])
    stage = doppler_stage(res)
    assert (
        stage.kind == "doppler"
        and stage.nbar is not None
        and stage.nbar[0] == pytest.approx(res.mode(0).nbar)
    )
    assert stage.ions == (0,) and stage.rates is res
