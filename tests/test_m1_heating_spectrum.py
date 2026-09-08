"""M1 regressions on the ``NoiseSpectrum`` -> S_E adapter of the heating layer (PLAN.md Section 4.1.5; Section 13 rows
"Electric-field noise density" and "Heating-rate meaning"; audit item E.1).

Two failures the shipped spectra never exposed, because every one of them is either flat (invariant under both bugs) or
tabulated on a non-negative grid:

- a legitimate SYMMETRIC two-sided tabulation (-omega_max .. +omega_max, which ``NoiseSpectrum.__post_init__`` accepts,
  requiring only a strictly increasing grid) has a NON-monotonic ``|omega_rad_s|``, on which the old
  ``np.interp(|omega|, |grid|, S)`` returned a constant: the whole spectral shape was lost, a factor 10 low at DC on a
  Lorentzian-like table;
- above the tabulated band ``np.interp`` clamps to S[-1], where ``NoiseSpectrum.tabulated`` is 0 by construction
  ("zero outside the band") because the white level is the density there. The heating layer therefore reported
  2 S[-1] + white instead of white - exactly where ``micromotion_sideband_heating_rate`` evaluates S_E, at tens of MHz.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from qutip_trap.noise.spectra import NoiseSpectrum, power_law_spectrum
from qutip_trap.trap.heating import heating_rate_quanta_per_s, single_sided_from_spectrum
from qutip_trap.units import ATOMIC_MASS_KG
from tests.m4_fixtures import two_ion_device


def test_symmetric_two_sided_tabulation_keeps_its_shape_at_dc_mid_band_and_the_edge() -> None:
    """S_E = 2 x NoiseSpectrum.tabulated at DC, mid-band and the band edge for a symmetric NON-flat table."""
    spectrum = NoiseSpectrum(
        np.array([-3e7, -1e7, 0.0, 1e7, 3e7]),
        np.array([1e-14, 5e-14, 1e-13, 5e-14, 1e-14]),
        "(V/m)^2/(rad/s)",
    )
    s_e = single_sided_from_spectrum(spectrum)
    for omega, expected in ((0.0, 2e-13), (1e7, 1e-13), (2e7, 6e-14), (3e7, 2e-14)):
        assert s_e(omega) == pytest.approx(expected, rel=1e-12)
        assert s_e(omega) == pytest.approx(2.0 * float(spectrum.tabulated(omega)), rel=1e-12)
        assert s_e(-omega) == pytest.approx(s_e(omega), rel=1e-12), "S(-omega) = S(omega)"
    # the shape is not flat: the old |omega|-on-a-non-monotonic-grid interpolation collapsed all four to 2 S[0] = 2e-14
    assert s_e(0.0) / s_e(3e7) == pytest.approx(10.0, rel=1e-12)


def test_above_the_tabulated_band_the_density_is_the_white_level_alone() -> None:
    """A mode above the band heats at the white level, not at 2 S[-1] + white (``np.interp`` clamped at the right edge)."""
    white_two_sided = 3e-15
    spectrum = power_law_spectrum(
        1e-13,
        2.0 * np.pi * 1e6,
        1.0,
        "(V/m)^2/(rad/s)",
        omega_min_rad_s=2.0 * np.pi * 1e3,
        omega_max_rad_s=2.0 * np.pi * 0.5e6,
        n=201,
    )
    spectrum = dataclasses.replace(spectrum, white_level=white_two_sided)
    s_e = single_sided_from_spectrum(spectrum)
    top = spectrum.omega_max_rad_s
    inside = s_e(0.5 * top)
    assert inside > 2.0 * white_two_sided, "the tabulated band dominates inside it"
    for omega in (1.5 * top, 10.0 * top, 100.0 * top):
        assert s_e(omega) == pytest.approx(2.0 * white_two_sided, rel=1e-12)
    assert s_e(top) == pytest.approx(2.0 * (float(spectrum.tabulated(top)) + white_two_sided), rel=1e-12)

    # end to end: a mode above the band gets the white level's rate, and the white level is counted exactly once
    dev = two_ion_device()
    dev = dataclasses.replace(
        dev, noise=dataclasses.replace(dev.noise, S_E=spectrum, correlation_length_m=0.0)
    )
    mass = dev.crystal.species[0].mass_u * ATOMIC_MASS_KG
    rates = dev.noise.heating_rates_quanta_per_s(dev)
    assert all(m.omega_rad_s > top for m in dev.crystal.modes), "every fixture mode is above the band"
    for k, mode in enumerate(dev.crystal.modes):
        assert rates[k] == pytest.approx(
            heating_rate_quanta_per_s(2.0 * white_two_sided, mass, mode.omega_rad_s), rel=1e-9
        )
