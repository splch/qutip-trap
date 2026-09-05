"""The M2 experiments (PLAN.md Sections 7.5, 7.9): Rabi flopping with the thermal Debye-Waller fit, Ramsey fringes, the
Ramsey-frequency experiment and sideband spectroscopy on the JOINT_EXACT engine."""

from __future__ import annotations

import numpy as np
import pytest

from qutip_trap.api import rabi_scan, ramsey, ramsey_frequency, sideband_spectroscopy
from qutip_trap.experiments import thermal_rabi_model
from qutip_trap.light.raman import derive_raman_drive
from tests.m2_fixtures import microwave_device, single_ion_raman_device


@pytest.fixture(scope="module")
def raman():  # type: ignore[no-untyped-def]
    dev = single_ion_raman_device()
    dd = derive_raman_drive(dev, 0, (0, 1), scattering=False)
    return dev, dd


def test_rabi_scan_fits_the_rabi_frequency_and_the_thermal_occupation(raman) -> None:  # type: ignore[no-untyped-def]
    """Section 7.9: Rabi flopping fitted with the thermal Debye-Waller envelope returns the carrier Rabi frequency and nbar."""
    dev, dd = raman
    f = dd.carrier_rabi_hz
    ts = np.linspace(0.0, 1.2 / f, 13)
    res = rabi_scan(dev, 0, ts, nbar={1: 0.6}, include_stark=False)
    assert res.data.shape == (13, 2) and res.model == "thermal_debye_waller_rabi"
    assert res.fitted["f_rabi_hz"][0] == pytest.approx(f, rel=1e-3)
    assert res.fitted["nbar"][0] == pytest.approx(0.6, abs=0.05)
    assert res.fitted["contrast"][0] == pytest.approx(1.0, abs=1e-3)
    model = thermal_rabi_model(np.array([f, 0.6, 1.0]), ts, dd.etas[1])
    assert np.max(np.abs(model - res.data[:, 1])) < 5e-3


def test_ramsey_fringe_and_ramsey_frequency_recover_the_detuning(raman) -> None:  # type: ignore[no-untyped-def]
    dev, _ = raman
    delays = np.linspace(0.0, 2e-3, 9)
    res = ramsey(dev, 0, delays, detuning_hz=1000.0, include_stark=False)
    assert res.fitted["delta_hz"][0] == pytest.approx(1000.0, abs=1e-3)
    assert res.fitted["contrast"][0] == pytest.approx(0.5, abs=1e-3)
    freq = ramsey_frequency(dev, 0, delays, include_stark=False, probe_hz=1000.0, qubit_shifts_hz={0: 37.0})
    assert freq.fitted["qubit_offset_hz"][0] == pytest.approx(37.0, abs=0.05)
    assert freq.model == "ramsey_two_probe"


def test_microwave_ramsey_frequency() -> None:
    dev = microwave_device()
    freq = ramsey_frequency(
        dev,
        0,
        np.linspace(0.0, 2e-3, 9),
        rabi_hz=2e4,
        probe_hz=1000.0,
        qubit_shifts_hz={0: -12.5},
        frame_hz=1.0e9,
    )
    assert freq.fitted["qubit_offset_hz"][0] == pytest.approx(-12.5, abs=0.05)
    assert freq.fitted["qubit_freq_hz"][0] == pytest.approx(1.0e9 - 12.5, abs=0.05)
    with pytest.raises(ValueError, match="rabi_hz"):
        rabi_scan(dev, 0, [0.0, 1e-6, 2e-6, 3e-6])


def test_sideband_spectroscopy_finds_the_blue_sideband_and_the_dark_red_one(raman) -> None:  # type: ignore[no-untyped-def]
    """Section 4.2.7: from n = 0 the red sideband is dark (sideband-asymmetry thermometry) while the blue sideband flops."""
    dev, dd = raman
    f, eta = dd.carrier_rabi_hz, dd.etas[1]
    mus = np.concatenate(
        [np.linspace(-3.02e6, -2.98e6, 5), np.linspace(-1e4, 1e4, 3), np.linspace(2.98e6, 3.02e6, 5)]
    )
    res = sideband_spectroscopy(dev, 0, mus, duration_s=0.5 / (f * eta), include_stark=False)
    assert res.fitted["blue_sideband_hz"][0] == pytest.approx(3.0e6, abs=1e4)
    assert res.fitted["carrier_hz"][0] == pytest.approx(0.0, abs=1e4)
    blue = res.data[np.argmin(np.abs(res.data[:, 0] - 3.0e6)), 1]
    red = res.data[np.argmin(np.abs(res.data[:, 0] + 3.0e6)), 1]
    assert blue > 0.9 and red < 1e-3
