"""Drives derived from the beams: Raman geometry and Rabi frequency, Stark shift, scattering, crosstalk, microwaves."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap.device.model import Field
from qutip_trap.light.beams import Beam
from qutip_trap.light.microwave import (
    ac_zeeman_shift_hz,
    coupling_rad_s,
    effective_detuning_hz,
    rabi_frequency_hz,
    square_microwave_drive,
)
from qutip_trap.light.raman import (
    crosstalk_ratios,
    derive_raman_drive,
    differential_stark_shift_hz,
    scattering_budget,
    square_drive,
)
from qutip_trap.species import species
from qutip_trap.species.raman import AtomicStructure
from qutip_trap.species.zeeman import g_I_steck
from qutip_trap.units import H_J_S, MU_B_J_PER_T, TWO_PI
from tests.fixtures import single_ion_raman_device, two_ion_raman_device


def test_raman_drive_geometry_and_lamb_dicke() -> None:
    """Delta k = k_1 - k_2: 2k counter-propagating with eta on the x mode only; a co-propagating pair has Delta k = 0 and no
    motional coupling; eta equals Crystal.lamb_dicke and the 171Yb+ 355 nm anchor scaled to 3.0 MHz."""
    dev = single_ion_raman_device()
    dd = derive_raman_drive(dev, 0, (0, 1))
    k = TWO_PI / 355e-9
    assert np.allclose(dd.delta_k, [2 * k, 0.0, 0.0])
    assert dd.kind == "raman" and dd.beams == (0, 1) and not dd.c0_applied
    assert dd.etas[0] == 0.0 and dd.etas[2] == 0.0
    assert dd.etas[1] == pytest.approx(dev.crystal.lamb_dicke(0, 1, dd.delta_k, micromotion=None))
    assert dd.etas[1] == pytest.approx(0.1103011 * math.sqrt(3.045 / 3.0), rel=2e-3)
    assert dd.carrier_rabi_hz > 1e3 and dd.pi_time_s() == pytest.approx(0.5 / dd.carrier_rabi_hz)
    b1, _ = dev.beams
    co = Beam(355e-9, (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), 20e-6, 10e-3, (0.0, 0.0, 0.0))
    dd_co = derive_raman_drive(
        dev.__class__(**{**dev.__dict__, "beams": (b1, co)}), 0, (0, 1), scattering=False
    )
    assert np.allclose(dd_co.delta_k, 0.0) and all(e == 0.0 for e in dd_co.etas.values())
    assert dd_co.micromotion is None


def test_raman_rabi_frequency_scales_as_the_field_product_and_matches_the_species_api() -> None:
    dev = single_ion_raman_device()
    dd = derive_raman_drive(dev, 0, (0, 1), scattering=False)
    b1, b2 = dev.beams
    b1_double = Beam(b1.wavelength_m, b1.k_hat, b1.polarization, b1.waist_m, 2 * b1.power_w, b1.pointing_m)
    dd2 = derive_raman_drive(
        dev.__class__(**{**dev.__dict__, "beams": (b1_double, b2)}), 0, (0, 1), scattering=False
    )
    assert dd2.carrier_rabi_hz / dd.carrier_rabi_hz == pytest.approx(math.sqrt(2.0), rel=1e-9)
    yb = species("171Yb+")
    lower, upper = yb.qubit
    assert abs(yb.raman_coupling_hz(lower, upper, b1, b2, dev.field)) == pytest.approx(
        dd.carrier_rabi_hz, rel=1e-9
    )
    # pi-pi polarizations cannot drive the m_F = 0 <-> 0 clock transition (the vector light shift vanishes)
    pi_pair = (
        Beam(355e-9, (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), 20e-6, 10e-3, (0.0, 0.0, 0.0)),
        Beam(355e-9, (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), 20e-6, 10e-3, (0.0, 0.0, 0.0)),
    )
    dev_pi = dev.__class__(**{**dev.__dict__, "beams": pi_pair, "field": Field(5.0, (0.0, 0.0, 1.0))})
    assert derive_raman_drive(dev_pi, 0, (0, 1), scattering=False).carrier_rabi_hz < 1e-6 * dd.carrier_rabi_hz


def test_stark_shift_is_the_differential_light_shift() -> None:
    """delta_St = delta(up) - delta(down) summed over the beams. 355 nm is blue of P1/2 (+33.2 THz) and red of P3/2
    (-66.6 THz), so each level is pushed up while the scalar shift nearly cancels and the differential shift, set by the
    hyperfine splittings in the denominators, is 1.3e-2 of it."""
    dev = single_ion_raman_device()
    dd = derive_raman_drive(dev, 0, (0, 1), scattering=False)
    yb = species("171Yb+")
    lower, upper = yb.qubit
    expected = sum(
        yb.light_shift_hz(upper, b, dev.field) - yb.light_shift_hz(lower, b, dev.field) for b in dev.beams
    )
    assert dd.stark_shift_hz == pytest.approx(expected, rel=1e-9)
    assert differential_stark_shift_hz(dev, 0, (0, 1)) == pytest.approx(expected, rel=1e-9)
    assert yb.light_shift_hz(lower, dev.beams[0], dev.field) > 0.0, "blue-detuned light raises the level"
    assert abs(dd.stark_shift_hz) < 3e-2 * yb.light_shift_hz(lower, dev.beams[0], dev.field)
    drive = square_drive(dd, detuning_hz=1e3, phase_rad=0.2)
    assert drive.stark_shift_hz == pytest.approx(dd.stark_shift_hz)
    assert drive.tones[0].detuning_hz == 1e3 and drive.tones[0].phase_rad == 0.2
    assert float(drive.tones[0].envelope_hz) == pytest.approx(dd.carrier_rabi_hz)  # type: ignore[arg-type]
    assert square_drive(dd, include_stark=False).stark_shift_hz == 0.0


def test_scattering_budget_and_per_pulse_error() -> None:
    """Rates per second from both qubit states under both beams; the d = 2 per-pulse error is the Raman probability during
    the pulse, of order Ozeri's P_Raman ~ pi gamma/omega_f ~ 1e-6 for 171Yb+ at 355 nm."""
    dev = single_ion_raman_device()
    dd = derive_raman_drive(dev, 0, (0, 1))
    assert dd.scattering is not None
    b = dd.scattering
    for a in species("171Yb+").qubit:
        assert b.rayleigh_per_s[a] > 0.0 and b.raman_spin_flip_per_s[a] > 0.0 and b.leakage_per_s[a] >= 0.0
    assert 1e-7 < b.per_pulse_error(dd.pi_time_s()) < 1e-4
    assert b.rayleigh_dephasing_per_s >= 0.0 and b.residual_excited_population < 1e-6
    assert scattering_budget(dev, 0, (0, 1)).rayleigh_per_s == b.rayleigh_per_s


def test_crosstalk_ratios_follow_the_beam_profile() -> None:
    """eps_ij = Omega_j/Omega_i is a Rabi (amplitude) ratio: for beams pointed at ion 0, the neighbour at distance s along z
    sees the product of two Gaussian FIELD factors exp(-s^2/w^2)."""
    dev = two_ion_raman_device(waist_m=10e-6)
    ratios = crosstalk_ratios(dev, 0, (0, 1))
    assert set(ratios) == {1}
    s = float(np.linalg.norm(dev.crystal.positions_m[1] - dev.crystal.positions_m[0]))
    assert abs(ratios[1]) == pytest.approx(math.exp(-2 * s**2 / (10e-6) ** 2), rel=1e-6)
    assert 0.0 < abs(ratios[1]) < 1.0


def test_microwave_rabi_frequency_closed_form_and_ac_zeeman_sign() -> None:
    """The clock transition |0,0> <-> |1,0> is driven by the pi component of B_1: Omega = mu_B B_1 (g_J - g_I)/(2 hbar) for
    B_1 || B, zero for B_1 perpendicular to B; a pi drive has no spectator coupling; delta_eff = delta - delta_ac."""
    yb = species("171Yb+")
    field = Field(5.0, (0.0, 0.0, 1.0))
    b1 = 1e-6  # tesla
    st = AtomicStructure(yb, field.B_gauss, field.direction)
    lower, upper = yb.qubit
    closed = (
        MU_B_J_PER_T
        * b1
        * (yb.level("S1/2").g_J - g_I_steck(yb.mu_I_nuclear_magnetons, yb.nuclear_spin))
        / (2.0 * H_J_S)
    )
    assert abs(rabi_frequency_hz(yb, field, (0.0, 0.0, b1))) == pytest.approx(closed, rel=1e-6)
    assert abs(rabi_frequency_hz(yb, field, (b1, 0.0, 0.0))) < 1e-9 * closed
    assert abs(coupling_rad_s(st, lower, upper, (0.0, 0.0, b1))) == pytest.approx(TWO_PI * closed, rel=1e-6)
    f0 = yb.transition_frequency_hz(lower, upper, 5.0)[0]
    assert ac_zeeman_shift_hz(yb, field, (0.0, 0.0, b1), f0) == pytest.approx(0.0, abs=1e-12)
    # a perpendicular component couples |0,0> to |1,+-1> at +-omega_Z: the documented sum term by term
    shift = ac_zeeman_shift_hz(yb, field, (b1, 0.0, 0.0), f0 + 1e5)
    labels = [s.full_label for s in st.states_of("S1/2")]
    pair = {st.state(upper).full_label, st.state(lower).full_label}
    manual = {}
    for a in (lower, upper):
        tot = 0.0
        for lab in labels:
            if lab == st.state(a).full_label or {st.state(a).full_label, lab} == pair:
                continue
            m = coupling_rad_s(st, a, lab, (b1, 0.0, 0.0))
            w = TWO_PI * (st.state(a).energy_hz - st.state(lab).energy_hz)
            omega = TWO_PI * (f0 + 1e5)
            tot += abs(m) ** 2 / (4 * (w + omega)) + abs(m) ** 2 / (4 * (w - omega))
        manual[a] = tot
    assert shift == pytest.approx((manual[upper] - manual[lower]) / TWO_PI, rel=1e-9)
    assert effective_detuning_hz(4.5, -1.0) == pytest.approx(5.5), "Harty 2014: +4.5 Hz - (-1.0 Hz)"
    drive = square_microwave_drive(0, 2e4, detuning_hz=4.5)
    assert drive.kind == "microwave" and drive.beams == () and drive.tones[0].detuning_hz == 4.5
