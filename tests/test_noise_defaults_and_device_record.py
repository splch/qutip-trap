"""``NoiseModel()`` is the quiet model and ``summary`` lists the channels that follow from what was set; ``Device.to_dict`` and
``from_dict`` round-trip both presets exactly, and ``Device.specs`` renders the derived quantities with their provenance
ids."""

from __future__ import annotations

import dataclasses
import json
import math

import pytest

from qutip_trap.device.model import Device
from qutip_trap.device.presets import ca40_optical, yb171_chain
from qutip_trap.device.serial import parse
from qutip_trap.hashing import canonical_digest
from qutip_trap.noise.model import NoiseModel, quiet_drift
from qutip_trap.noise.spectra import Collisions, power_law_spectrum, white_spectrum


def test_the_default_noise_model_is_quiet() -> None:
    quiet = NoiseModel()
    assert quiet.is_quiet() and quiet.S_E.is_zero() and quiet.correlation_length_m == 0.0
    assert all(d.quiet for d in quiet.drifts.values()) and quiet.collisions is None and quiet.mains is None
    assert quiet.summary(yb171_chain(1).device) == {} and quiet.apparatus() == ()
    assert canonical_digest(yb171_chain(2).device.noise) == canonical_digest(NoiseModel())


def test_summary_lists_heating_when_the_field_spectrum_is_set_and_nothing_when_it_is_not() -> None:
    device = yb171_chain(2).device
    s_e = power_law_spectrum(
        level_at_ref=1e-12,
        omega_ref_rad_s=2 * math.pi * 1e6,
        alpha=1.0,
        unit="(V/m)^2/(rad/s)",
        omega_min_rad_s=2 * math.pi * 1e3,
        omega_max_rad_s=2 * math.pi * 1e7,
    )
    noisy = NoiseModel(S_E=s_e, correlation_length_m=0.0)
    summary = noisy.summary(device)
    heating = {k: v for k, v in summary.items() if k.startswith("heating_rate_per_s[")}
    assert heating and all(unit == "quanta/s" and value > 0.0 for value, unit in heating.values())
    assert heating.keys() == {f"heating_rate_per_s[{m}]" for m in noisy.heating_rates_quanta_per_s(device)}
    assert not any(k.startswith(("qubit_dephasing", "collision", "intensity")) for k in summary)
    # a drift and a collision model add their rows with the units the module documents
    drifting = dataclasses.replace(
        noisy,
        rabi_drift=dataclasses.replace(quiet_drift(), rms=0.01),
        field_drift=dataclasses.replace(quiet_drift(), rms=1e-9),
        collisions=Collisions(
            pressure_pa=1e-9,
            gas={"H2": 1.0},
            outcome_probabilities={"heating_kick": 0.9, "reorder": 0.05, "loss": 0.04, "dark_ion": 0.01},
        ),
        laser_intensity=white_spectrum(1e-12, "1/(rad/s)"),
    )
    more = drifting.summary(device)
    assert more["rabi_drift_rms"] == (0.01, "1") and more["field_drift_rms"] == (1e-9, "T")
    assert more["intensity_noise_density"] == (1e-12, "1/(rad/s)")
    assert {k for k in more if k.startswith("collision_rate_per_ion[")} == {
        "collision_rate_per_ion[0]",
        "collision_rate_per_ion[1]",
    }
    assert all(more[f"collision_rate_per_ion[{i}]"][1] == "1/s" for i in range(2))


@pytest.mark.parametrize(
    "make", [lambda: yb171_chain(2), lambda: ca40_optical(1)], ids=["yb171_chain(2)", "ca40_optical(1)"]
)
def test_the_device_round_trip_is_exact(make) -> None:  # type: ignore[no-untyped-def]
    device = make().device
    record = device.to_dict()
    assert record["schema_version"] == 1 and record["device_hash"] == device.hash() and "device" in record
    text = json.dumps(record)  # standard JSON: no NaN or Infinity tokens
    back = Device.from_dict(json.loads(text))
    assert back.hash() == device.hash() and json.dumps(back.to_dict()) == text
    assert (
        back.hardware.amplifier_bandwidth_hz == device.hardware.amplifier_bandwidth_hz
    )  # inf survives as "inf"
    assert back.roles == device.roles and back.beams[0].polarization == device.beams[0].polarization


def test_the_record_refuses_what_it_cannot_carry() -> None:
    device = yb171_chain(1).device
    with pytest.raises(ValueError, match="schema version 1"):
        Device.from_dict({**device.to_dict(), "schema_version": 2})
    bad = device.to_dict()
    bad["device"]["detector"]["colour"] = "blue"
    with pytest.raises(ValueError, match="unknown fields \\['colour'\\]"):
        Device.from_dict(bad)
    with pytest.raises(ValueError, match="no JSON form for the annotation"):
        parse("set[int]")
    assert (
        parse("tuple[float, ...]").variadic and parse("dict[tuple[int, int], float]").args[0].kind == "tuple"
    )


def test_specs_renders_the_derived_quantities_with_their_provenance_ids() -> None:
    preset = yb171_chain(2)
    text = preset.device.specs()
    derived = preset.device.derived()
    assert text.startswith(f"device {preset.device.hash()[:12]}") and "crystal: 2 ion(s) of 171Yb+" in text
    for key, value in list(derived.values.items())[:5]:
        assert f"{key} = {value:.6g}  [{derived.provenance[key]}]" in text
    assert all(f"[{pid}]" in text for pid in set(derived.provenance.values()))
    assert "noise channels: none (the noise model is quiet)" in text
    machine_text = preset.machine().specs()
    assert (
        machine_text.startswith(text) and "gate drives = " in machine_text and "level = auto" in machine_text
    )
