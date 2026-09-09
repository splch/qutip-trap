"""Export and bitwise re-import of run records (PLAN.md Section 14.3; Section 9.11 row "Record round trip").

A record file is one zip archive: ``manifest.json`` (format, the record's digest, the array table), ``record.json`` (the
canonical document, :mod:`qutip_trap_app.codec`) and ``arrays/NNNN.npy`` (every array as NumPy's ``.npy``, no pickling).
Entry timestamps are fixed, so exporting the same record twice yields the same bytes, and importing verifies the digest
of what was read against the manifest before returning a record: an export that re-imports is bitwise the record that
was exported, which is what makes a published screenshot reproducible (Section 14.3).
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np

from qutip_trap_app.codec import ArrayStore, digest, from_document, loads, to_document
from qutip_trap_app.record import RECORD_FORMAT, Record, RecordError

_EPOCH = (1980, 1, 1, 0, 0, 0)
"""The fixed zip timestamp (the earliest a zip entry can carry): the export is a pure function of the record."""


def _entry(name: str, data: bytes, *, compress: bool) -> tuple[zipfile.ZipInfo, bytes]:
    info = zipfile.ZipInfo(name, date_time=_EPOCH)
    info.compress_type = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    return info, data


def _npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(arr), allow_pickle=False)
    return buf.getvalue()


def export_bytes(record: Record) -> tuple[bytes, str]:
    """The record file's bytes and the record's digest."""
    document, arrays = to_document(record)
    record_id = digest(document, arrays)
    names = {key: f"arrays/{k:04d}.npy" for k, key in enumerate(sorted(arrays))}
    manifest = {
        "format": RECORD_FORMAT,
        "record_id": record_id,
        "created_utc": record.created_utc,
        "versions": record.versions,
        "arrays": names,
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            *_entry("manifest.json", json.dumps(manifest, sort_keys=True, indent=1).encode(), compress=True)
        )
        zf.writestr(*_entry("record.json", document, compress=True))
        for key in sorted(arrays):
            zf.writestr(*_entry(names[key], _npy_bytes(arrays[key]), compress=False))
    return buf.getvalue(), record_id


def export_record(record: Record, path: str | Path) -> str:
    """Write the record file; returns the record's digest (its ``record_id``)."""
    data, record_id = export_bytes(record)
    Path(path).write_bytes(data)
    return record_id


def import_bytes(data: bytes) -> Record:
    with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
        names = set(zf.namelist())
        if "manifest.json" not in names or "record.json" not in names:
            raise RecordError("not a qutip-trap-app record: manifest.json or record.json missing")
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        if manifest.get("format") != RECORD_FORMAT:
            raise RecordError(
                f"unsupported record format {manifest.get('format')!r}; this reader knows {RECORD_FORMAT}"
            )
        document = zf.read("record.json")
        arrays: ArrayStore = {}
        for key, name in manifest["arrays"].items():
            if name not in names:
                raise RecordError(f"array {name} named by the manifest is missing")
            arrays[key] = np.load(io.BytesIO(zf.read(name)), allow_pickle=False)
    found = digest(document, arrays)
    if found != manifest["record_id"]:
        raise RecordError(
            f"record digest mismatch: manifest says {manifest['record_id'][:12]}, the contents give {found[:12]}"
        )
    loads(document)  # a syntax check with a clear error before decoding
    return from_document(Record, document, arrays)


def import_record(path: str | Path) -> Record:
    """Read a record file and verify it bitwise against its manifest digest."""
    return import_bytes(Path(path).read_bytes())


__all__ = ["export_bytes", "export_record", "import_bytes", "import_record"]
