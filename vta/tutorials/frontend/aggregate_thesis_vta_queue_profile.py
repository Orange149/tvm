#!/usr/bin/env python3
"""Aggregate all native-runner worker [VTA_QUEUE] records into a Q1 peak source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PREFIX = "[VTA_QUEUE] "
DEFAULT_MAX_XFER = 1 << 25


def load_records(path):
    records = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.startswith(PREFIX):
            continue
        try:
            record = json.loads(line[len(PREFIX) :])
        except json.JSONDecodeError as error:
            raise ValueError("invalid VTA_QUEUE JSON at line {}: {}".format(line_number, error))
        record["source_line"] = line_number
        records.append(record)
    if not records:
        raise ValueError("native stderr contains no VTA_QUEUE records")
    return records


def aggregate(records, environment, max_xfer=DEFAULT_MAX_XFER):
    if environment.get("VTA_REPLAY_POLICY") != "disabled":
        raise ValueError("default profile must explicitly disable replay")
    if environment.get("VTA_INSN_SUBMIT_THRESHOLD_BYTES") not in (None, ""):
        raise ValueError("default profile must leave early submit threshold unset")
    if int(environment.get("VTA_INSN_BUFFER_BYTES", -1)) != max_xfer or int(
        environment.get("VTA_UOP_BUFFER_BYTES", -1)
    ) != max_xfer:
        raise ValueError("default profile must use VTA_MAX_XFER for both queues")
    by_queue = {}
    global_insn_peak = 0
    global_uop_peak = 0
    for record in records:
        queue = record.get("queue_id")
        if not isinstance(queue, str) or not queue:
            raise ValueError("queue record has no queue_id")
        if int(record.get("timeout", -1)) != 0:
            raise ValueError("queue record contains a device timeout")
        if int(record.get("insn_capacity", -1)) != max_xfer or int(
            record.get("uop_capacity", -1)
        ) != max_xfer:
            raise ValueError("queue record capacity differs from default")
        if record.get("submit_threshold_configured") is not False:
            raise ValueError("queue record used early submit threshold")
        state = by_queue.setdefault(queue, {"submissions": 0, "insn_peak": 0, "uop_peak": 0})
        expected_submit = state["submissions"] + 1
        if int(record.get("submit", -1)) != expected_submit:
            raise ValueError("non-contiguous submit sequence for queue {}".format(queue))
        insn_bytes = int(record.get("insn_bytes", 0))
        uop_bytes = int(record.get("uop_bytes", 0))
        if min(insn_bytes, uop_bytes) <= 0:
            raise ValueError("queue record has a non-positive serialized command size")
        state["submissions"] = expected_submit
        state["insn_peak"] = max(state["insn_peak"], insn_bytes)
        state["uop_peak"] = max(state["uop_peak"], uop_bytes)
        if int(record.get("insn_peak", -1)) != state["insn_peak"] or int(
            record.get("uop_peak", -1)
        ) != state["uop_peak"]:
            raise ValueError("runtime running peak is inconsistent for queue {}".format(queue))
        global_insn_peak = max(global_insn_peak, insn_bytes)
        global_uop_peak = max(global_uop_peak, uop_bytes)
    return {
        "schema": "vta_queue_capacity_status_v1",
        "aggregation_schema": "thesis_vta_worker_queue_aggregation_v1",
        "insn_capacity_bytes": max_xfer,
        "uop_capacity_bytes": max_xfer,
        "insn_peak_bytes": global_insn_peak,
        "uop_peak_bytes": global_uop_peak,
        "submissions": len(records),
        "queue_instances": len(by_queue),
        "per_queue": by_queue,
        "insn_submit_threshold_bytes": max_xfer,
        "insn_submit_threshold_configured": False,
        "command_manifest_id": environment.get("VTA_COMMAND_MANIFEST_ID", ""),
        "replay_policy": "disabled",
        "finish_included": True,
        "finish_evidence": "every VTA_QUEUE record is emitted after Synchronize appends FINISH",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stderr-log", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vta-max-xfer-bytes", type=int, default=DEFAULT_MAX_XFER)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    environment = json.loads(Path(args.environment).read_text(encoding="utf-8"))
    result = aggregate(load_records(args.stderr_log), environment, args.vta_max_xfer_bytes)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
