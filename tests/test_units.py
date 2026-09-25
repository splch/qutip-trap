"""CODATA constants, the cm^-1 conversion and the Lande factor (PLAN.md Section 13)."""

from __future__ import annotations

import pytest

from qutip_trap import units
from qutip_trap.species.zeeman import MU_B_OVER_H_HZ_PER_G
from qutip_trap.units import lande_g_j


def test_codata_2022_values_are_pinned() -> None:
    """mu_B, mu_B/h, m_e and g_S (positive, Steck's sign) are the CODATA 2022 values to 1e-12 (Section 5.6)."""
    assert units.MU_B_J_PER_T == pytest.approx(9.2740100657e-24, rel=1e-12)
    assert units.MU_B_OVER_H_HZ_PER_T == pytest.approx(13996244917.1, rel=1e-12)
    assert units.ELECTRON_MASS_U == pytest.approx(0.0005485799090441, rel=1e-12)
    assert units.G_S == pytest.approx(2.00231930436092, rel=1e-13), "Steck's sign convention: g_S positive"
    assert MU_B_OVER_H_HZ_PER_G * 1e-6 == pytest.approx(1.399624, abs=5e-7), "the MHz/G of Section 4.5.1"


def test_lande_factors_with_the_measured_g_s() -> None:
    """g_J of S1/2, P1/2, P3/2, D3/2, D5/2 is 2, 2/3, 4/3, 4/5, 6/5 moved by the electron's g_S - 2, to 1e-10."""
    half = 0.5
    assert lande_g_j(0, half, half) == pytest.approx(units.G_S, rel=1e-15)
    assert lande_g_j(1, half, half) == pytest.approx(0.6658935652, abs=1e-10)
    assert lande_g_j(1, half, 1.5) == pytest.approx(1.3341064348, abs=1e-10)
    assert lande_g_j(2, half, 1.5) == pytest.approx(0.7995361391, abs=1e-10)
    assert lande_g_j(2, half, 2.5) == pytest.approx(1.2004638609, abs=1e-10)


def test_wavenumber_to_wavelength_reproduces_the_729_nm_line() -> None:
    """NIST Ca II D5/2 at 13710.88 cm^-1 is the 729.347 nm vacuum line of Section 4.5.7."""
    assert units.C_M_PER_S / units.hz_from_wavenumber_cm(13710.88) == pytest.approx(729.347e-9, rel=2e-6)
