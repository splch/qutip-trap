"""Section 9.11 row "Record round trip": export then import reproduces bitstrings, traces, seeds and hashes bitwise."""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest

from qutip_trap_app.codec import to_document
from qutip_trap_app.record import RECORD_FORMAT, LiveRun, Record, RecordError
from qutip_trap_app.storage import export_bytes, export_record, import_bytes, import_record


def _arrays_equal(a: Record, b: Record) -> None:
    doc_a, arr_a = to_document(a)
    doc_b, arr_b = to_document(b)
    assert doc_a == doc_b, "the JSON documents differ"
    assert set(arr_a) == set(arr_b)
    for key in arr_a:
        x, y = arr_a[key], arr_b[key]
        assert x.dtype == y.dtype and x.shape == y.shape, key
        assert np.array_equal(x, y, equal_nan=True), key
        assert x.tobytes() == y.tobytes(), key


def test_export_import_is_bitwise(bell: tuple[Record, LiveRun], tmp_path) -> None:  # type: ignore[no-untyped-def]
    record, _live = bell
    path = tmp_path / "bell.qtrec.zip"
    record_id = export_record(record, path)
    back = import_record(path)
    assert back.digest() == record.digest() == record_id
    _arrays_equal(record, back)
    assert np.array_equal(back.results.bitstrings, record.results.bitstrings)
    assert back.results.bitstrings.dtype == np.uint8
    assert back.job.seed == record.job.seed == 7 and back.diagnostics.root_seed == 7
    assert back.device_hash == record.device_hash == record.job.device.hash
    assert back.table.device_hash == record.device_hash
    assert len(back.traces) == len(record.traces) == record.n_samples * record.n_branches
    for tr_a, tr_b in zip(record.traces, back.traces):
        assert np.array_equal(tr_a.times_s, tr_b.times_s)
        assert np.array_equal(tr_a.reduced_internal, tr_b.reduced_internal)
        assert tr_a.final_joint is not None and tr_b.final_joint is not None
        assert np.array_equal(tr_a.final_joint, tr_b.final_joint)
    # exporting the imported record gives the same file bytes: the export is a pure function of the data
    again, again_id = export_bytes(back)
    first, first_id = export_bytes(record)
    assert again == first and again_id == first_id


def test_manifest_and_layout(bell: tuple[Record, LiveRun]) -> None:
    record, _ = bell
    data, record_id = export_bytes(record)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = zf.namelist()
        manifest = json.loads(zf.read("manifest.json"))
    assert manifest["format"] == RECORD_FORMAT and manifest["record_id"] == record_id
    assert "record.json" in names and all(n.endswith(".npy") for n in names if n.startswith("arrays/"))
    assert len(manifest["arrays"]) == sum(1 for n in names if n.startswith("arrays/"))
    assert manifest["versions"]["qutip_trap"] == record.versions["qutip_trap"]


def test_tampering_is_detected(bell: tuple[Record, LiveRun]) -> None:
    record, _ = bell
    data, _ = export_bytes(record)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        entries = {n: zf.read(n) for n in zf.namelist()}
    doc = entries["record.json"]
    tampered = doc.replace(b'"shots":200', b'"shots":201')
    assert tampered != doc, "the probe string must exist in the document"
    entries["record.json"] = tampered
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for n, b in entries.items():
            zf.writestr(n, b)
    with pytest.raises(RecordError, match="digest mismatch"):
        import_bytes(buf.getvalue())


def test_record_holds_the_ladder(bell: tuple[Record, LiveRun]) -> None:
    """Section 14.3: job -> compiled circuit -> schedule -> traces -> readout -> results, all present and consistent."""
    record, live = bell
    assert record.job.circuit.n_qubits == 2 and record.results.bitstrings.shape == (200, 2)
    assert record.compiled.native.ops and all(
        op.name in ("gpi", "gpi2", "ms", "zz") for op in record.compiled.native.ops
    )
    assert record.schedule.pulses and record.schedule.steps and record.schedule.targets
    assert {s.kind for s in record.schedule.steps} == {"gate", "idle"}
    assert record.diagnostics.level == "JOINT_EXACT" and record.space.dimension == live.space.dimension
    assert record.readout.levels.shape == (200, 2) and record.readout.mode == "fast"
    assert set(record.results.target_probabilities) == {"00", "11"}
    assert record.results.probabilities["00"] + record.results.probabilities["11"] > 0.98
    assert record.results.register_fidelity is not None and record.results.register_fidelity > 0.99
    assert record.core_gaps and record.device_card.n_ions == 2
    assert record.table.entries and not record.table.uncalibrated
    assert record.preparation.nbar and all(v >= 0.0 for v in record.preparation.nbar.values())
