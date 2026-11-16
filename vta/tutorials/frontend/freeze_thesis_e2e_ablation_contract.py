#!/usr/bin/env python3
"""Freeze the Stage-06 six-condition identities and measurement schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from thesis_e2e_ablation_protocol import CONDITIONS, frozen_schedule


FORBIDDEN = ("latency", "throughput", "fps", "measured", "top1", "top5", "accuracy")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_results(value, path="root"):
    if isinstance(value, dict):
        for key, nested in value.items():
            if any(token in key.lower() for token in FORBIDDEN):
                raise ValueError("result field is forbidden before freeze: {}.{}".format(path, key))
            _reject_results(nested, "{}.{}".format(path, key))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_results(nested, "{}[{}]".format(path, index))


def freeze(payload):
    if payload.get("schema") != "thesis_e2e_ablation_freeze_input_v1":
        raise ValueError("unexpected ablation-freeze schema")
    if payload.get("result_labels_read") is not False:
        raise ValueError("result labels must remain unread")
    _reject_results({key: value for key, value in payload.items() if key != "result_labels_read"})
    common = payload.get("common_artifacts", {})
    required_common = ("model_hash", "qir_hash", "input_manifest_hash", "label_manifest_hash")
    if any(not common.get(field) for field in required_common):
        raise ValueError("common model/qIR/input/label hashes are required")
    rows = payload.get("conditions", [])
    if [row.get("condition") for row in rows] != list(CONDITIONS):
        raise ValueError("conditions must be frozen in A/B/C/Ca/D/E order")
    for row in rows:
        for field in ("package_hash", "actual_policy_hash", "plan_hash", "capacity_certificate_hash"):
            if not row.get(field):
                raise ValueError("{} lacks {}".format(row.get("condition"), field))
    return {
        "schema": "thesis_e2e_ablation_contract_v1",
        "status": "frozen_before_results",
        "result_labels_read": False,
        "common_artifacts": common,
        "conditions": rows,
        "schedule": frozen_schedule(),
        "measurement_contract": {
            "boots": 5,
            "distinct_boot_ids_required": True,
            "blocks_per_boot": 2,
            "warmup_calls_per_condition_block": 5,
            "serial_scored_calls_per_condition_block": 15,
            "windows_per_condition_per_boot": 2,
            "completion_rows_per_window": 302,
            "discard_rows_per_window": 2,
            "scored_gaps_per_window": 299,
            "window_timeout_seconds": 300,
            "accuracy_images_per_condition": 1000,
            "numeric_real_images_per_condition": 64,
            "numeric_diagnostic_inputs_per_condition": 3,
        },
        "planned_counts": {
            "windows": 60,
            "completion_rows": 18120,
            "scored_ii_gaps": 17940,
            "serial_scored_calls": 900,
            "warmup_calls": 300,
            "accuracy_inferences": 6000,
            "numeric_qualification_calls": 402,
        },
        "claim_boundary": "schedule/identity contract only; no correctness, accuracy, performance, resource, deployment, or completion result",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    input_path = Path(args.input).resolve()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    contract = freeze(json.loads(input_path.read_text(encoding="utf-8")))
    contract["input"] = {"path": str(input_path), "sha256": sha256_file(input_path)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": contract["status"], **contract["planned_counts"]}, indent=2))


if __name__ == "__main__":
    main()

