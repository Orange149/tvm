"""Stable identities and failure vocabulary for C3 VTA tuning candidates."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping


CANDIDATE_ID_SCHEMA = "c3_candidate_id_v1"
FAILURE_SCHEMA = "c3_candidate_failure_v1"
RESIDENCE_MODES = (
    "original",
    "input_stationary",
    "weight_stationary",
    "paper_inspired_hybrid",
    "weight_resident_barrier",
    "input_weight_resident_barrier",
)
FAILURE_CATEGORIES = (
    "lower",
    "compile",
    "wrong_answer",
    "timeout",
    "rpc",
    "environment",
)

_DEBUG_CONFIG_KEYS = frozenset(("index", "config_index", "debug_index"))


def _json_value(value, path="$"):
    """Return a strict JSON value while rejecting ambiguous/non-finite inputs."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float at {}".format(path))
        return value
    if isinstance(value, Mapping):
        normalized = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("non-string mapping key at {}: {!r}".format(path, key))
            normalized[key] = _json_value(item, "{}.{}".format(path, key))
        return normalized
    if isinstance(value, (list, tuple)):
        return [_json_value(item, "{}[{}]".format(path, index)) for index, item in enumerate(value)]
    raise TypeError("unsupported JSON value at {}: {}".format(path, type(value).__name__))


def canonical_json_bytes(value):
    """Encode a JSON-compatible object deterministically for hashing."""

    normalized = _json_value(value)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def normalize_config_entity(config_entity):
    """Keep the complete semantic ConfigEntity and remove only debug indices.

    AutoTVM commonly serializes a ConfigEntity as ``index``, ``code_hash`` and
    ``entity``. The config-space index is deliberately excluded because adding
    a new knob can renumber an otherwise identical entity.
    """

    normalized = _json_value(config_entity, "$.complete_config_entity")
    if isinstance(normalized, dict):
        normalized = {
            key: value for key, value in normalized.items() if key not in _DEBUG_CONFIG_KEYS
        }
    return normalized


def candidate_identity_payload(
    hardware_fingerprint,
    template_name,
    schedule_version,
    workload,
    residence_mode,
    complete_config_entity,
):
    """Build the exact semantic payload covered by ``candidate_id``."""

    if not isinstance(template_name, str) or not template_name:
        raise ValueError("template_name must be a non-empty string")
    if not isinstance(schedule_version, str) or not schedule_version:
        raise ValueError("schedule_version must be a non-empty string")
    if residence_mode not in RESIDENCE_MODES:
        raise ValueError("unsupported residence_mode: {}".format(residence_mode))
    return {
        "schema": CANDIDATE_ID_SCHEMA,
        "hardware_fingerprint": _json_value(hardware_fingerprint, "$.hardware_fingerprint"),
        "template_name": template_name,
        "schedule_version": schedule_version,
        "workload": _json_value(workload, "$.workload"),
        "residence_mode": residence_mode,
        "complete_config_entity": normalize_config_entity(complete_config_entity),
    }


def candidate_id(
    hardware_fingerprint,
    template_name,
    schedule_version,
    workload,
    residence_mode,
    complete_config_entity,
):
    """Return a lowercase SHA-256 identifier for one semantic candidate."""

    payload = candidate_identity_payload(
        hardware_fingerprint,
        template_name,
        schedule_version,
        workload,
        residence_mode,
        complete_config_entity,
    )
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def candidate_identity_record(
    hardware_fingerprint,
    template_name,
    schedule_version,
    workload,
    residence_mode,
    complete_config_entity,
    config_index=None,
):
    """Return identity payload plus an explicitly non-semantic debug index."""

    payload = candidate_identity_payload(
        hardware_fingerprint,
        template_name,
        schedule_version,
        workload,
        residence_mode,
        complete_config_entity,
    )
    record = {
        "candidate_id": hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
        "identity": payload,
    }
    if config_index is not None:
        record["debug"] = {"config_index": int(config_index)}
    return record


def failure_schema_record():
    """Return the frozen machine-readable failure category schema."""

    return {
        "schema": FAILURE_SCHEMA,
        "categories": list(FAILURE_CATEGORIES),
        "category_meanings": {
            "lower": "schedule construction or TIR lowering failed",
            "compile": "lowered module could not be built or cross-compiled",
            "wrong_answer": "execution completed but correctness oracle failed",
            "timeout": "bounded operation exceeded its declared timeout",
            "rpc": "RPC transport/server/session failed",
            "environment": "hardware, bitstream, runtime, dependency, or fingerprint mismatch",
        },
    }


def make_failure(category, message, phase=None, retryable=False):
    """Create a validated failure record without guessing missing diagnostics."""

    if category not in FAILURE_CATEGORIES:
        raise ValueError("unsupported failure category: {}".format(category))
    if not isinstance(message, str) or not message:
        raise ValueError("failure message must be a non-empty string")
    record = {
        "schema": FAILURE_SCHEMA,
        "category": category,
        "message": message,
        "retryable": bool(retryable),
    }
    if phase is not None:
        record["phase"] = str(phase)
    return record
