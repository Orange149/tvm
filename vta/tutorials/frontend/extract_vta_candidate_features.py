"""Adapt archived static VTA DMA evidence to the C3 candidate feature schema.

This local-only adapter does not lower, compile, execute, or contact a board.
It intentionally reports command footprints as ``not_measured``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from c3_candidate_identity import (
    candidate_identity_record,
    canonical_json_bytes,
    failure_schema_record,
)


FEATURE_SCHEMA = "c3_candidate_feature_v1"
SUMMARY_SCHEMA = "c3_candidate_feature_summary_v1"


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def _workload_key(workload):
    return canonical_json_bytes(workload).decode("utf-8")


def build_config_lookup(selected_tasks):
    """Index archived full ConfigEntity records by workload and debug index."""

    lookup = {}
    for task in selected_tasks:
        workload = task["workload"]
        workload_key = _workload_key(workload)
        for candidate in task.get("candidates", []):
            debug_index = int(candidate["config_index"])
            key = (workload_key, debug_index)
            if key in lookup:
                raise ValueError("duplicate candidate mapping for index {}".format(debug_index))
            if "config" not in candidate:
                raise ValueError("candidate {} has no full config record".format(debug_index))
            lookup[key] = candidate["config"]
    return lookup


def _descriptor_aggregates(descriptors):
    exact_histogram = {"load": defaultdict(int), "store": defaultdict(int)}
    by_memory = defaultdict(lambda: {"calls": 0, "bytes": 0, "padded_calls": 0})
    padded_calls = {"load": 0, "store": 0}
    descriptor_rows = 0
    expanded_calls = 0
    expanded_bytes = 0

    for descriptor in descriptors:
        direction = descriptor["direction"]
        if direction not in exact_histogram:
            raise ValueError("unknown descriptor direction: {}".format(direction))
        multiplicity = int(descriptor.get("multiplicity", 1))
        request_bytes = int(descriptor["bytes_per_request"])
        memory = str(descriptor.get("memory_name", "unknown"))
        if multiplicity < 0 or request_bytes < 0:
            raise ValueError("negative descriptor multiplicity or byte count")

        descriptor_rows += 1
        expanded_calls += multiplicity
        expanded_bytes += multiplicity * request_bytes
        exact_histogram[direction][str(request_bytes)] += multiplicity
        by_memory[memory]["calls"] += multiplicity
        by_memory[memory]["bytes"] += multiplicity * request_bytes
        if bool(descriptor.get("padded", False)):
            padded_calls[direction] += multiplicity
            by_memory[memory]["padded_calls"] += multiplicity

    return {
        "descriptor_rows": descriptor_rows,
        "expanded_calls": expanded_calls,
        "expanded_bytes": expanded_bytes,
        "padded_calls": padded_calls,
        "exact_request_bytes_histogram": {
            direction: dict(sorted(histogram.items(), key=lambda item: int(item[0])))
            for direction, histogram in exact_histogram.items()
        },
        "by_memory": dict(sorted(by_memory.items())),
    }


def adapt_row(
    row,
    config_lookup,
    hardware_fingerprint,
    template_name,
    schedule_version,
    residence_mode,
    small_request_threshold_bytes,
    source_sha256,
):
    workload = row["workload"]
    config_index = int(row["config_index"])
    lookup_key = (_workload_key(workload), config_index)
    if lookup_key not in config_lookup:
        raise ValueError(
            "missing full ConfigEntity for workload/config_index {}".format(config_index)
        )
    full_config = config_lookup[lookup_key]
    identity = candidate_identity_record(
        hardware_fingerprint,
        template_name,
        schedule_version,
        workload,
        residence_mode,
        full_config,
        config_index=config_index,
    )

    static_dma = row["static_dma"]
    descriptor_aggregates = _descriptor_aggregates(static_dma.get("request_descriptors", []))
    record = {
        "schema": FEATURE_SCHEMA,
        **identity,
        "status": "ok",
        "failure": None,
        "source": {
            "kind": "archived_static_workload_dma",
            "sha256": source_sha256,
            "lower_status": "historical_static_extraction_succeeded",
            "build_status": "not_reexecuted_by_adapter",
        },
        "legality": {
            "status": "archived_success",
            "sram_working_set": "not_available_in_archived_source",
        },
        "transfer_signature": {
            "small_request_threshold_bytes": int(small_request_threshold_bytes),
            "totals": static_dma["totals"],
            "max_request_bytes_by_memory": static_dma.get("max_request_bytes_by_memory", {}),
            "unique_tensor_bytes": row.get("unique_tensor_bytes", {}),
            "reload_ratio": row.get("redundancy", {}),
            "descriptor_aggregates": descriptor_aggregates,
        },
        "command_signature": {
            "status": "not_measured",
            "instruction_bytes": None,
            "serialized_uop_bytes": None,
            "finish_replay_requirement": None,
            "submit_count": None,
        },
    }
    if "runtime_comparison" in row:
        record["archived_validation"] = row["runtime_comparison"]
    return record


def adapt_archive(
    static_archive,
    selected_tasks,
    hardware_fingerprint,
    template_name,
    schedule_version,
    residence_mode,
    source_sha256,
):
    if static_archive.get("status") != "completed":
        raise ValueError("static DMA archive is not completed")
    rows = static_archive.get("rows", [])
    expected_count = int(static_archive.get("workload_count", len(rows)))
    if expected_count != len(rows):
        raise ValueError("archive workload_count does not match rows")
    threshold = int(static_archive["small_request_threshold_bytes"])
    lookup = build_config_lookup(selected_tasks)
    records = [
        adapt_row(
            row,
            lookup,
            hardware_fingerprint,
            template_name,
            schedule_version,
            residence_mode,
            threshold,
            source_sha256,
        )
        for row in rows
    ]
    candidate_ids = [record["candidate_id"] for record in records]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("adapted archive contains duplicate candidate IDs")
    return records


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-workload-dma", required=True)
    parser.add_argument("--selected-tasks", required=True)
    parser.add_argument("--hardware-fingerprint-json", required=True)
    parser.add_argument("--template-name", default="conv2d_packed.vta")
    parser.add_argument("--schedule-version", required=True)
    parser.add_argument("--residence-mode", default="original")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    static_path = Path(args.static_workload_dma)
    selected_path = Path(args.selected_tasks)
    output_path = Path(args.output)
    summary_path = Path(args.summary)
    hardware_fingerprint = json.loads(args.hardware_fingerprint_json)

    source_sha256 = _sha256_file(static_path)
    selected_sha256 = _sha256_file(selected_path)
    records = adapt_archive(
        _load_json(static_path),
        _load_json(selected_path),
        hardware_fingerprint,
        args.template_name,
        args.schedule_version,
        args.residence_mode,
        source_sha256,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")

    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed",
        "mode": "local_archive_adapter_only",
        "record_count": len(records),
        "unique_candidate_ids": len({record["candidate_id"] for record in records}),
        "command_signature_statuses": sorted(
            {record["command_signature"]["status"] for record in records}
        ),
        "source": {"path": str(static_path), "sha256": source_sha256},
        "selected_tasks": {"path": str(selected_path), "sha256": selected_sha256},
        "output": {"path": str(output_path), "sha256": _sha256_file(output_path)},
        "failure_schema": failure_schema_record(),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps(summary, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
