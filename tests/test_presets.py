"""The example devices of ``device/presets.py`` (PLAN.md Section 3.2; M10): the 171Yb+ chain preset IS the validated M6 fixture,
hash for hash, and carries the drive maps the scheduler needs."""

from __future__ import annotations

import pytest

from qutip_trap.control.schedule import default_gate_drives
from qutip_trap.device.model import BeamRoles
from qutip_trap.device.presets import DevicePreset, yb171_chain
from tests.m6_fixtures import circuit_fixture


def test_yb171_chain_is_the_m6_fixture_device() -> None:
    for n, waist in ((2, 2.5e-6), (3, 2.0e-6)):
        preset = yb171_chain(n, address_waist_m=waist)
        fixture = circuit_fixture(n, address_waist_m=waist)
        assert isinstance(preset, DevicePreset) and preset.n_ions == n
        assert preset.device.hash() == fixture.device.hash(), (
            "the preset and the validated fixture are one device"
        )
        assert (
            preset.gate_drives == fixture.gate_drives
            and preset.entangling_drives == fixture.entangling_drives
        )
        assert preset.detection_beam == fixture.detection_beam == len(preset.device.beams) - 1
        # 0.2.0: the device carries the drive maps as its roles, so the scheduler's default reads them without ambiguity
        assert preset.device.roles == BeamRoles(
            gate=preset.gate_drives, entangling=preset.entangling_drives, detection=preset.detection_beam
        )
        assert default_gate_drives(preset.device) == preset.gate_drives
        assert not hasattr(preset, "run_kwargs"), "deprecated in 0.2.0, removed in 0.4.0"
        assert preset.device.preparation is not None and preset.device.noise.is_quiet(preset.device)


def test_preset_overrides_and_refusals() -> None:
    a = yb171_chain(2)
    b = yb171_chain(2, phase_continuous=True)
    assert a.device.hash() != b.device.hash() and b.device.hardware.phase_continuous
    assert not a.device.hardware.phase_continuous
    with pytest.raises(ValueError):
        yb171_chain(0)
    single = yb171_chain(1)
    assert single.n_ions == 1 and single.gate_drives[0].beams == (2, 3)
