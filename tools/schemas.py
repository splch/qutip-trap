"""The JSON schemas of the ``Result`` envelope (docs/api_implementation_plan.md 1.7; 0.2.0) and of the ``Device`` record (2.5;
0.3.0): ``docs/schemas/result.schema.json`` is written from ``RESULT_SCHEMA`` below, which mirrors ``Result.to_dict``, and
``docs/schemas/device.schema.json`` from ``qutip_trap.device.serial.device_schema``, generated from the device tree's field
annotations; ``--check`` exits 1 when the committed file is
stale (CI, like the ledger tables), and ``validate`` checks an instance against the subset of JSON Schema the file uses
(``type``, ``properties``, ``required``, ``additionalProperties``, ``items``, ``enum``, ``minimum``, ``const``), so a test
can validate a real result without a validator dependency. ``tests/test_m6_results_export.py`` asserts that the schema's
properties are exactly the keys a result writes.

    uv run python tools/schemas.py            # rewrite docs/schemas/result.schema.json
    uv run python tools/schemas.py --check    # exit 1 if the file is stale
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from qutip_trap.provenance import repository_root

SCHEMA_PATH = Path("docs") / "schemas" / "result.schema.json"
DEVICE_SCHEMA_PATH = Path("docs") / "schemas" / "device.schema.json"


def _numbers(keys: str = "string") -> dict[str, Any]:
    return {"type": "object", "additionalProperties": {"type": "number"}}


def _integers() -> dict[str, Any]:
    return {"type": "object", "additionalProperties": {"type": "integer"}}


def _pair(kind: str = "number") -> dict[str, Any]:
    return {"type": "array", "items": {"type": kind}, "minItems": 2, "maxItems": 2}


CAL_ENTRY: dict[str, Any] = {
    "type": "object",
    "required": ["value", "uncertainty", "status", "experiment", "provenance_id", "fitted_at_s", "sample_id"],
    "additionalProperties": False,
    "properties": {
        "value": {"type": "number"},
        "uncertainty": {"type": "number"},
        "status": {"type": "string", "enum": ["seed", "calibrated", "uncalibrated"]},
        "experiment": {"type": "string"},
        "provenance_id": {"type": "string"},
        "fitted_at_s": {"type": "number"},
        "sample_id": {"type": "integer"},
    },
}

CALIBRATION: dict[str, Any] = {
    "type": "object",
    "required": ["device_hash", "seed", "surrogate", "fitted_at_s", "entries", "waveforms"],
    "additionalProperties": False,
    "properties": {
        "device_hash": {"type": "string"},
        "seed": {"type": "integer"},
        "surrogate": {"type": "boolean"},
        "fitted_at_s": {"type": "number"},
        "entries": {"type": "object", "additionalProperties": CAL_ENTRY},
        "waveforms": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "required": ["duration_s", "kind", "chi_m", "alpha_m", "phi_s", "phi_m"],
                "additionalProperties": False,
                "properties": {
                    "duration_s": {"type": "number"},
                    "kind": {"type": "string"},
                    "chi_m": _numbers(),
                    "alpha_m": {"type": "object", "additionalProperties": _pair()},
                    "phi_s": CAL_ENTRY,
                    "phi_m": CAL_ENTRY,
                },
            },
        },
    },
}

SPACE: dict[str, Any] = {
    "type": "object",
    "required": ["ion_dims", "resolved", "enr_group", "frozen", "ions", "dropped"],
    "additionalProperties": False,
    "properties": {
        "ion_dims": {"type": "array", "items": {"type": "integer"}},
        "resolved": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["mode", "d", "expected_n_range", "eta_max", "element_tol"],
                "additionalProperties": False,
                "properties": {
                    "mode": {"type": "integer"},
                    "d": {"type": "integer", "minimum": 2},
                    "expected_n_range": _pair("integer"),
                    "eta_max": {"type": "number"},
                    "element_tol": {"type": ["number", "null"]},
                },
            },
        },
        "enr_group": {"type": ["array", "null"]},
        "frozen": {"type": "array", "items": {"type": "integer"}},
        "ions": {"type": "array", "items": {"type": "integer"}},
        "dropped": {"type": "array", "items": {"type": "integer"}},
    },
}

RUN_STATE: dict[str, Any] = {
    "type": "object",
    "required": ["order", "dark", "lost", "events"],
    "additionalProperties": False,
    "properties": {
        "order": {"type": "array", "items": {"type": "integer"}},
        "dark": {"type": "array", "items": {"type": "integer"}},
        "lost": {"type": "array", "items": {"type": "integer"}},
        "events": {"type": "array", "items": {"type": "array", "minItems": 2, "maxItems": 2}},
    },
}

DIAGNOSTICS: dict[str, Any] = {
    "type": "object",
    "required": [
        "level",
        "level_reason",
        "space",
        "mode_class",
        "run_state",
        "wall_clock_span_s",
        "boundary_population",
        "margin_levels",
        "dropped_modes",
        "frozen_contribution",
        "integrator",
        "tolerances",
        "samples",
        "trajectories",
        "shots_per_sample",
        "effective_sample_size",
        "root_seed",
        "calibration",
        "approximations",
        "intrinsic_budget",
        "dropped_branch_weight",
        "frozen_excitation_bound",
        "dropped_contribution",
        "margin_reached",
        "populated_n_max",
        "cap_growth",
        "gate_local",
        "kernel",
        "workers",
        "propagator_cache_hits",
        "branches",
        "convergence",
        "shots_per_sample_realized",
    ],
    "additionalProperties": False,
    "properties": {
        "level": {"type": "string", "enum": ["JOINT_EXACT", "GATE_LOCAL"]},
        "level_reason": {"type": "string"},
        "space": SPACE,
        "mode_class": {
            "type": "object",
            "additionalProperties": {"type": "string", "enum": ["resolved", "frozen", "dropped", "enr"]},
        },
        "run_state": RUN_STATE,
        "wall_clock_span_s": {"type": "number"},
        "boundary_population": _numbers(),
        "margin_levels": _integers(),
        "dropped_modes": {"type": "array", "items": {"type": "integer"}},
        "frozen_contribution": {"type": "object", "additionalProperties": _pair()},
        "integrator": {"type": "string"},
        "tolerances": _pair(),
        "samples": {"type": "integer"},
        "trajectories": {"type": "integer"},
        "shots_per_sample": {"type": "integer"},
        "effective_sample_size": {"type": "number"},
        "root_seed": {"type": "integer"},
        "calibration": CALIBRATION,
        "approximations": {"type": "array", "items": {"type": "string"}},
        "intrinsic_budget": _numbers(),
        "dropped_branch_weight": {"type": "number"},
        "frozen_excitation_bound": _numbers(),
        "dropped_contribution": _pair(),
        "margin_reached": _integers(),
        "populated_n_max": _integers(),
        "cap_growth": _integers(),
        "gate_local": {"type": "boolean"},
        "kernel": {"type": "string"},
        "workers": {"type": "integer"},
        "propagator_cache_hits": {"type": "integer"},
        "branches": {"type": "integer"},
        "convergence": {"type": ["string", "null"]},
        "shots_per_sample_realized": {"type": "array", "items": {"type": "integer"}},
    },
}

RESULT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://github.com/splch/qutip-trap/blob/main/docs/schemas/result.schema.json",
    "title": "qutip-trap Result",
    "description": (
        "The envelope Result.to_dict() writes and Result.from_dict() reads (schema version 1): the identity of the run, "
        "the histogram with its error bars, the SPAM errors, the herald tallies, the run state and the diagnostics summary; "
        "per_shot carries the arrays when asked for. Bitstring keys put qubit 0 rightmost (docs/conventions.md)."
    ),
    "type": "object",
    "required": [
        "schema_version",
        "qutip_trap_version",
        "device_hash",
        "machine_hash",
        "created_at",
        "duration_s",
        "shots",
        "n_qubits",
        "qubits",
        "bit_order",
        "root_seed",
        "counts",
        "probabilities",
        "error_bars",
        "spam",
        "discarded_shots",
        "registers",
        "heralds",
        "run_state",
        "diagnostics",
    ],
    "additionalProperties": False,
    "properties": {
        "schema_version": {"const": 1},
        "qutip_trap_version": {"type": "string"},
        "device_hash": {"type": "string"},
        "machine_hash": {"type": ["string", "null"]},
        "created_at": {"type": "string"},
        "duration_s": {"type": "number", "minimum": 0},
        "shots": {"type": "integer", "minimum": 0},
        "n_qubits": {"type": "integer", "minimum": 1},
        "qubits": {"type": "array", "items": {"type": "integer", "minimum": 0}},
        "bit_order": {"type": "string", "enum": ["qubit0_lsb", "qubit0_msb"]},
        "root_seed": {"type": "integer"},
        "counts": _integers(),
        "probabilities": _numbers(),
        "error_bars": _numbers(),
        "spam": {"type": "object", "additionalProperties": _pair()},
        "discarded_shots": {"type": "integer", "minimum": 0},
        "registers": {
            "type": ["object", "null"],
            "additionalProperties": {"type": "array", "items": {"type": "integer", "minimum": 0}},
        },
        "heralds": {
            "type": "object",
            "required": ["collision", "dark_or_lost", "count_anomaly"],
            "additionalProperties": False,
            "properties": {
                "collision": {"type": "integer", "minimum": 0},
                "dark_or_lost": {"type": "integer", "minimum": 0},
                "count_anomaly": {"type": "integer", "minimum": 0},
            },
        },
        "run_state": RUN_STATE,
        "diagnostics": DIAGNOSTICS,
        "per_shot": {
            "type": "object",
            "required": ["bitstrings", "heralds", "photon_records", "posteriors"],
            "additionalProperties": False,
            "properties": {
                "bitstrings": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "integer", "enum": [0, 1]}},
                },
                "heralds": {"type": "array", "items": {"type": "integer", "minimum": 0}},
                "photon_records": {"type": ["array", "null"]},
                "posteriors": {"type": ["array", "null"]},
            },
        },
    },
}


def _type_ok(value: object, kind: str) -> bool:
    if kind == "object":
        return isinstance(value, Mapping)
    if kind == "array":
        return isinstance(value, Sequence) and not isinstance(value, str | bytes)
    if kind == "string":
        return isinstance(value, str)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "null":
        return value is None
    raise ValueError(f"unknown JSON Schema type {kind!r}")


def validate(
    instance: object, schema: Mapping[str, Any], path: str = "$", root: Mapping[str, Any] | None = None
) -> list[str]:
    """The violations of ``instance`` against ``schema`` (the subset in the module docstring, plus ``anyOf`` and the local
    ``$ref`` into ``$defs`` the device schema uses), empty when it validates."""
    root = schema if root is None else root
    if "$ref" in schema:
        ref = str(schema["$ref"])
        if not ref.startswith("#/$defs/"):
            raise ValueError(f"only local $defs references are supported, got {ref!r}")
        return validate(instance, root["$defs"][ref[len("#/$defs/") :]], path, root)
    if "anyOf" in schema:
        attempts = [validate(instance, alt, path, root) for alt in schema["anyOf"]]
        if any(not a for a in attempts):
            return []
        return [f"{path}: no alternative of anyOf fits ({'; '.join(e for a in attempts for e in a[:1])})"]
    errors: list[str] = []
    if "const" in schema and instance != schema["const"]:
        errors.append(f"{path}: expected the constant {schema['const']!r}, got {instance!r}")
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: {instance!r} is not one of {schema['enum']}")
    kinds = schema.get("type")
    if kinds is not None:
        allowed = [kinds] if isinstance(kinds, str) else list(kinds)
        if not any(_type_ok(instance, k) for k in allowed):
            errors.append(f"{path}: expected {allowed}, got {type(instance).__name__}")
            return errors
    if isinstance(instance, int | float) and not isinstance(instance, bool):
        minimum = schema.get("minimum")
        if minimum is not None and (math.isnan(instance) or instance < minimum):
            errors.append(f"{path}: {instance!r} is below the minimum {minimum!r}")
    if isinstance(instance, Mapping):
        properties = schema.get("properties", {})
        for key in schema.get("required", ()):
            if key not in instance:
                errors.append(f"{path}: the required key {key!r} is missing")
        extra = schema.get("additionalProperties", True)
        for key, value in instance.items():
            if key in properties:
                errors.extend(validate(value, properties[key], f"{path}.{key}", root))
            elif extra is False:
                errors.append(f"{path}: the key {key!r} is not allowed")
            elif isinstance(extra, Mapping):
                errors.extend(validate(value, extra, f"{path}.{key}", root))
    elif isinstance(instance, Sequence) and not isinstance(instance, str | bytes):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errors.append(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            errors.append(f"{path}: more than {schema['maxItems']} items")
        items = schema.get("items")
        if items is not None:
            for k, value in enumerate(instance):
                errors.extend(validate(value, items, f"{path}[{k}]", root))
    return errors


def render() -> str:
    return json.dumps(RESULT_SCHEMA, indent=2, ensure_ascii=False) + "\n"


def render_device() -> str:
    """The device schema, generated from the field annotations of the device tree (``qutip_trap.device.serial``)."""
    from qutip_trap.device.serial import device_schema

    return json.dumps(device_schema(), indent=2, ensure_ascii=False) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="exit 1 when a committed schema is stale")
    args = parser.parse_args(argv)
    root = repository_root()
    stale = 0
    for path, text in ((root / SCHEMA_PATH, render()), (root / DEVICE_SCHEMA_PATH, render_device())):
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                print(f"{path} is stale: run `uv run python tools/schemas.py`")
                stale = 1
            else:
                print(f"{path} is current")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            print(f"wrote {path}")
    return stale


if __name__ == "__main__":
    sys.exit(main())
