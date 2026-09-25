"""docs/api_implementation_plan.md 2.4: table edits are proposals. ``CalibrationTable.with_params`` sets entries by field and
refuses unknown names, ``updated_with`` maps a typed ``ExperimentResult`` onto exactly the entries it fitted with the stamps of
Section 7.5, ``CalEntry.kind`` partitions setpoints from characterisation with a default derived from the field, and the new
field stays out of every digest while unset."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from qutip_trap.control.table import ENTRY_KINDS, CalEntry, CalibrationTable
from qutip_trap.experiments.result import (
    CrystalImage,
    ExperimentResult,
    HeatingRateFit,
    ParityScan,
    RabiScan,
    RamseyFringe,
)
from qutip_trap.experiments.single_ion import rabi_scan
from qutip_trap.hashing import canonical_digest
from qutip_trap.machine import Machine
from tests.m6_fixtures import CircuitFixture, circuit_fixture

WINDOWS = tuple(float(x) for x in np.linspace(10e-6, 40e-6, 7))


@pytest.fixture(scope="module")
def fx() -> CircuitFixture:
    return circuit_fixture(2)


@pytest.fixture(scope="module")
def machine(fx: CircuitFixture) -> Machine:
    return Machine(fx.device).calibrated(pairs=[(0, 1)], detection_records=300, detection_windows_s=WINDOWS)


def test_with_params_merges_mapping_fields_and_refuses_unknown_names(machine: Machine) -> None:
    table = machine.table
    assert table is not None
    entry = table.rabi[(0, 2)]
    wrong = table.with_params(
        rabi={(0, 2): dataclasses.replace(entry, value=1.02 * entry.value, status="calibrated")}
    )
    assert (
        wrong.rabi[(0, 2)].value == pytest.approx(1.02 * entry.value)
        and wrong.rabi[(0, 2)].status == "calibrated"
    )
    assert (
        wrong.rabi[(1, 4)] is table.rabi[(1, 4)] and wrong.device_hash == table.device_hash
    )  # merged, not replaced
    assert (
        wrong.modes == table.modes and wrong is not table and table.rabi[(0, 2)] is entry
    )  # the original is untouched
    with pytest.raises(TypeError, match="unknown or unsettable field"):
        table.with_params(rabbi={})
    with pytest.raises(TypeError, match="device_hash"):
        table.with_params(device_hash="0" * 64)
    with pytest.raises(TypeError, match="takes a mapping"):
        table.with_params(rabi=3.0)
    # a waveform under the other key order replaces the stored pair rather than adding a second one
    wf = table.waveform_for((0, 1))
    assert wf is not None
    marked = dataclasses.replace(wf, phi_s=dataclasses.replace(wf.phi_s, value=wf.phi_s.value + 0.1))
    swapped = table.with_params(ms={(1, 0): marked})
    assert set(swapped.ms) == set(table.ms) and swapped.waveform_for((0, 1)) is marked
    # a scalar field is replaced
    assert (
        table.with_params(fitted_at_s=7.5).fitted_at_s == 7.5
        and table.with_params(surrogate=False).surrogate is False
    )


def test_updated_with_a_rabi_scan_changes_exactly_the_rabi_entry_with_the_stamps(machine: Machine) -> None:
    table = machine.table
    assert table is not None
    scan = rabi_scan(machine, 0, np.linspace(0.0, 20e-6, 9), shots=200, seed=1)
    assert isinstance(scan, RabiScan) and scan.converged and scan.subject == {"ion": 0, "beam": 2}
    proposal = table.updated_with(scan, fitted_at_s=12.5, sample_id=3)
    changed = {k for k in table.entries() if table.entries()[k] != proposal.entries().get(k)}
    assert changed == {"rabi[(0, 2)]"}
    new = proposal.rabi[(0, 2)]
    assert new.value == scan.f_rabi_hz and new.uncertainty == scan.uncertainty("f_rabi_hz")
    assert (
        new.status == "calibrated"
        and new.experiment == "rabi_scan"
        and new.provenance_id == scan.provenance_id
    )
    assert (
        new.fitted_at_s == 12.5 and new.sample_id == 3 and proposal.fitted_at_s == 12.5 and new.kind is None
    )
    assert proposal.kind_of("rabi[(0, 2)]") == "setpoint" and proposal.ms == table.ms
    # the default time is the table's own; a failed fit proposes an uncalibrated entry
    assert table.updated_with(scan).rabi[(0, 2)].fitted_at_s == table.fitted_at_s
    failed = dataclasses.replace(scan, converged=False)
    assert table.updated_with(failed).rabi[(0, 2)].status == "uncalibrated" and failed.quality == "failed"
    # the results that set no entry refuse rather than returning the table unchanged
    for bare in (
        ParityScan(data=np.zeros((0, 2)), fitted={}, model="parity_oscillation", provenance_id="p"),
        CrystalImage(data=np.zeros((0, 2)), fitted={}, model="crystal_image", provenance_id="p"),
        RamseyFringe(
            data=np.zeros((0, 2)), fitted={"delta_hz": (1.0, 0.1)}, model="ramsey_fringe", provenance_id="p"
        ),
        ExperimentResult(data=np.zeros((0, 2)), fitted={}, model="x", provenance_id="p"),
    ):
        with pytest.raises(ValueError, match="sets no|only ramsey_frequency"):
            table.updated_with(bare)
    heat = HeatingRateFit(
        data=np.zeros((0, 2)),
        fitted={"ndot_per_s": (12.0, 1.0)},
        model="heating_rate_sideband_asymmetry",
        provenance_id="anchor.trap.heating_dynamics",
        subject={"mode": 3},
    )
    assert table.updated_with(heat).heating[3].value == 12.0 and heat.experiment == "heating_rate"


def test_entry_kinds_partition_the_table(machine: Machine) -> None:
    table = machine.table
    assert table is not None
    setpoints = table.entries(kind="setpoint")
    characterisation = table.entries(kind="characterisation")
    assert set(setpoints) | set(characterisation) == set(table.entries()) and not (
        set(setpoints) & set(characterisation)
    )
    assert "rabi[(0, 2)]" in setpoints and "field" in characterisation and "ms[(0, 1)].phi_s" in setpoints
    assert all(
        table.kind_of(k) == "characterisation" for k in table.entries() if k.startswith(("nbar[", "heating["))
    )
    assert set(ENTRY_KINDS) == {f.name for f in dataclasses.fields(table)} - {
        "device_hash",
        "seed",
        "surrogate",
        "fitted_at_s",
    }
    tagged = dataclasses.replace(table.rabi[(0, 2)], kind="characterisation")
    assert (
        table.with_params(rabi={(0, 2): tagged}).kind_of("rabi[(0, 2)]") == "characterisation"
    )  # the entry's own kind wins


def test_the_kind_field_is_digest_neutral_while_unset_and_round_trips(machine: Machine) -> None:
    table = machine.table
    assert table is not None
    entry = table.rabi[(0, 2)]
    assert entry.kind is None and "kind" not in entry.to_dict()
    assert canonical_digest(entry) == canonical_digest(dataclasses.replace(entry))
    tagged = dataclasses.replace(entry, kind="setpoint")
    assert canonical_digest(tagged) != canonical_digest(entry) and tagged.to_dict()["kind"] == "setpoint"
    assert CalEntry.from_dict(tagged.to_dict()) == tagged and CalEntry.from_dict(entry.to_dict()) == entry
    assert CalibrationTable.from_dict(table.to_dict()).rabi[(0, 2)] == entry
