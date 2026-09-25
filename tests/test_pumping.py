"""Optical pumping: the 171Yb+ pump into |0,0> with its preparation error, duration, photon count and recoil heating with
the ion's participation, the qubit hand-off, and Zeeman pumping on a spin-zero fixture."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap.device.presets import secular_trap
from qutip_trap.dynamics.multilevel import MultiLevelOptions
from qutip_trap.light.bloch import BlochModel, beam_for_transition
from qutip_trap.light.recoil import angular_factor, emission_lamb_dicke
from qutip_trap.prep.pumping import optical_pumping, scrambled_initial_state
from qutip_trap.species import species
from qutip_trap.species.polarization import linear_polarization
from qutip_trap.species.raman import AtomicStructure
from qutip_trap.trap.crystal import solve_crystal
from qutip_trap.units import TWO_PI
from tests.fixtures import spin_zero_like

YB = species("171Yb+")
DARK = "S1/2 F=0 mF=0"
BRIGHT = tuple(f"S1/2 F=1 mF={m}" for m in (-1, 0, 1))
WAIST = 20e-6


def yb_pump_model(s0: float = 0.5) -> BlochModel:
    st = AtomicStructure(YB, 5.0, (0.0, 0.0, 1.0))
    line = YB.transition("S1/2-P1/2")
    magic = tuple(linear_polarization((1.0, 0.0, 0.0), math.acos(1.0 / math.sqrt(3.0)), (0.0, 0.0, 1.0)))
    power = s0 * line.i_sat_w_m2 * math.pi * WAIST**2 / 2.0
    beam = beam_for_transition(
        st, "S1/2 F=1 mF=0", "P1/2 F=1 mF=0", 0.0, (1.0, 0.0, 0.0), magic, power_w=power, waist_m=WAIST
    )  # type: ignore[arg-type]
    return BlochModel(st, [beam], levels=("S1/2", "P1/2"), options=MultiLevelOptions(leak="renormalize"))


def test_yb171_pump_error_time_photons_and_recoil_heating_with_participation() -> None:
    """Pumping 171Yb+ into |0,0> at s_o = 0.5 scatters 3.00 +- 0.02 photons and reaches its 2.5e-6 residual (10 %)
    within 10 us, and the recoil heats the 1 MHz and 3 MHz modes by 8.5e-3 and 2.8e-3 quanta (5 %)."""
    model = yb_pump_model()
    trap = secular_trap((3.0e6, 2.9e6, 1.0e6))
    crystal = solve_crystal(trap, (YB,))
    res = optical_pumping(model, [DARK], duration_s=30e-6, samples=6001, crystal=crystal, ion=0)
    assert res.photons_scattered == pytest.approx(3.0, abs=0.02)
    assert res.preparation_error < 1e-5 and res.steady_state_error == pytest.approx(2.5e-6, rel=0.1)
    assert res.preparation_error == pytest.approx(res.steady_state_error, rel=0.05)
    assert res.time_to_reach_s is not None and 1e-6 < res.time_to_reach_s < 10e-6
    assert res.motional_heating_quanta[0] == pytest.approx(8.5e-3, rel=0.05)
    assert res.motional_heating_quanta[1] == pytest.approx(2.8e-3, rel=0.05)
    # the heating is photons x (mean alpha over the emitted polarizations) x eta_em^2 with the ion's participation (c = 1 here)
    eta_em = emission_lamb_dicke(crystal, 0, TWO_PI / YB.transition("S1/2-P1/2").wavelength_vac_m, 0)
    lo = 3.0 * angular_factor(0, 1.0) * eta_em**2
    hi = 3.0 * angular_factor(1, 1.0) * eta_em**2
    assert lo < res.motional_heating_quanta[0] < hi
    rho = res.qubit_density_matrix((DARK, "S1/2 F=1 mF=0"))
    assert rho.shape == (2, 2) and float(np.real(rho.tr())) == pytest.approx(1.0)
    assert float(np.real(rho.full()[0, 0])) == pytest.approx(1.0 - res.preparation_error, abs=1e-12)


def test_scrambled_initial_state_is_the_resonant_ground_manifold() -> None:
    model = yb_pump_model()
    rho = scrambled_initial_state(model)
    pops = model.build.populations(rho)
    assert all(pops[lab] == pytest.approx(1.0 / 3.0) for lab in BRIGHT) and pops[DARK] == 0.0
    rho2 = scrambled_initial_state(model, [DARK, BRIGHT[0]])
    assert model.build.populations(rho2)[DARK] == pytest.approx(0.5)
    res = optical_pumping(model, [DARK], duration_s=20e-6, samples=2001, initial=rho2)
    assert res.preparation_error < 1e-5


def test_zeeman_pumping_on_the_spin_zero_fixture_takes_three_photons() -> None:
    """sigma+ light pumps a spin-zero S1/2 |-1/2> into the dark |+1/2> with 3.00 +- 0.01 photons (a 1/3 branching into
    the target) and a residual below 1e-6."""
    sp = spin_zero_like()
    st = AtomicStructure(sp, 1.0, (0.0, 0.0, 1.0))
    line = sp.transition("S1/2-P1/2")
    pol = (-1.0 / math.sqrt(2.0) + 0j, -1j / math.sqrt(2.0), 0j)
    power = 0.05 * line.i_sat_w_m2 * math.pi * WAIST**2 / 2.0
    beam = beam_for_transition(
        st, "S1/2 mJ=-1/2", "P1/2 mJ=1/2", 0.0, (0.0, 0.0, 1.0), pol, power_w=power, waist_m=WAIST
    )
    model = BlochModel(st, [beam], levels=("S1/2", "P1/2"))
    assert model.build.static
    res = optical_pumping(model, ["S1/2 mJ=1/2"], duration_s=40e-6, samples=8001, initial=["S1/2 mJ=-1/2"])
    assert res.photons_scattered == pytest.approx(3.0, abs=0.01)
    assert res.preparation_error < 1e-6 and res.time_to_reach_s is not None
    rho = res.qubit_density_matrix(("S1/2 mJ=1/2", "S1/2 mJ=-1/2"))
    assert float(np.real(rho.full()[0, 0])) == pytest.approx(1.0, abs=1e-6)
