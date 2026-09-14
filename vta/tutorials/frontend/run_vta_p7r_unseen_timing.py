#!/usr/bin/env python3
"""Run prospective paired timing for one passed P7R unseen-geometry contract."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import tvm
from tvm import rpc
import vta

from qualify_vta_residency_fsim import reference_data_from_workload
from run_vta_p7r_joint_correctness import (
    build_module,
    load_contract,
    sha256_file,
    ssh_preflight,
    write_json,
)


def correctness_gate(path, expected_sha256):
    if sha256_file(path) != expected_sha256:
        raise ValueError("correctness ledger hash mismatch")
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if len(rows) != 5 or not all(row.get("correct") for row in rows):
        raise ValueError("unseen timing requires five passed correctness records")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--correctness", required=True)
    parser.add_argument("--correctness-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--blocks", type=int, default=7)
    parser.add_argument("--number", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--order-seed", type=int, default=20260911)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    output.mkdir(parents=True)
    contract = load_contract(args.contract)
    correctness_gate(args.correctness, args.correctness_sha256)
    env = vta.get_env()
    before = ssh_preflight(contract, args.host)

    role_labels = {
        "request_shape_original_before": "shape_original",
        "request_shape_residency": "shape_residency",
        "contiguous_original": "control_original",
        "contiguous_residency": "control_residency",
    }
    entries = []
    for entry in contract["execution_order"]:
        if entry["role"] in role_labels:
            entries.append({**entry, "label": role_labels[entry["role"]]})
    if {entry["label"] for entry in entries} != set(role_labels.values()):
        raise ValueError("contract does not contain the four timing roles")
    labels = sorted(role_labels.values())
    rng = random.Random(args.order_seed)
    orders = []
    for _ in range(args.blocks):
        order = list(labels)
        rng.shuffle(order)
        orders.append(order)
    write_json(
        output / "preregistered.json",
        {
            "schema": "c3_p7r_unseen_timing_protocol_v1",
            "frozen_before_timing": True,
            "workload_id": contract["workload_id"],
            "correctness_sha256": args.correctness_sha256,
            "blocks": args.blocks,
            "number": args.number,
            "warmup": args.warmup,
            "order_seed": args.order_seed,
            "orders": orders,
            "primary": "paired median speedup(shape) > paired median speedup(contiguous control)",
        },
    )

    remote = rpc.connect(args.host, args.port, session_timeout=120)
    device = remote.ext_dev(0)
    records = []
    start = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="c3_p7r_unseen_timing_") as temporary:
            modules, functions = {}, {}
            for entry in entries:
                binary, _, _ = build_module(
                    contract["workload"], entry["mode"], entry["config_index"], env,
                    Path(temporary), entry.get("tir_sha256")
                )
                remote.upload(str(binary))
                modules[entry["label"]] = remote.load_module(binary.name)
                functions[entry["label"]] = modules[entry["label"]]["main"]
            data, weight, expected = reference_data_from_workload(contract["workload"], 0)
            buffers = [tvm.nd.array(data, device), tvm.nd.array(weight, device),
                       tvm.nd.empty(expected.shape, "int8", device)]
            for label in labels:
                functions[label](*buffers)
                if int(np.count_nonzero(buffers[-1].numpy() != expected)):
                    raise RuntimeError("pre-timing correctness failed for " + label)
                for _ in range(args.warmup):
                    functions[label](*buffers)
            with (output / "timing.jsonl").open("x") as stream:
                for block, order in enumerate(orders):
                    for position, label in enumerate(order):
                        evaluator = modules[label].time_evaluator(
                            "main", device, number=args.number, repeat=1
                        )
                        value = float(evaluator(*buffers).mean * 1000.0)
                        row = {"block": block, "position": position, "label": label,
                               "latency_ms": value, "number": args.number}
                        records.append(row)
                        stream.write(json.dumps(row, sort_keys=True) + "\n")
                        stream.flush()
                    print("block {}/{} complete".format(block + 1, args.blocks), flush=True)
            for label in labels:
                functions[label](*buffers)
                if int(np.count_nonzero(buffers[-1].numpy() != expected)):
                    raise RuntimeError("post-timing correctness failed for " + label)
    except Exception as error:
        write_json(output / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise

    samples = {
        label: [row["latency_ms"] for row in records if row["label"] == label]
        for label in labels
    }
    comparisons = {}
    for name in ("shape", "control"):
        original = {row["block"]: row["latency_ms"] for row in records
                    if row["label"] == name + "_original"}
        residency = {row["block"]: row["latency_ms"] for row in records
                     if row["label"] == name + "_residency"}
        paired = [(original[b] - residency[b]) / original[b] for b in sorted(original)]
        comparisons[name] = {
            "original_median_ms": statistics.median(original.values()),
            "residency_median_ms": statistics.median(residency.values()),
            "paired_speedups": paired,
            "paired_median_speedup_fraction": statistics.median(paired),
            "wins": sum(value > 0 for value in paired),
        }
    summary = {
        "schema": "c3_p7r_unseen_timing_v1",
        "status": "completed",
        "workload_id": contract["workload_id"],
        "scope": "prospective single-boot unseen-geometry operator timing; not stage/FPS",
        "comparisons": comparisons,
        "prediction_correct": comparisons["shape"]["paired_median_speedup_fraction"]
        > comparisons["control"]["paired_median_speedup_fraction"],
        "elapsed_seconds": time.monotonic() - start,
        "board_before": before,
        "board_after": ssh_preflight(contract, args.host),
    }
    write_json(output / "summary.json", summary)
    (output / "STATUS.md").write_text(
        "# P7R {} unseen timing\n\n- Shape paired median: {:+.2%} ({}/{} wins)\n"
        "- Control paired median: {:+.2%} ({}/{} wins)\n- Frozen ranking prediction correct: `{}`\n"
        "- Scope: operator-only, single boot; no stage/FPS claim\n".format(
            contract["workload_id"], comparisons["shape"]["paired_median_speedup_fraction"],
            comparisons["shape"]["wins"], args.blocks,
            comparisons["control"]["paired_median_speedup_fraction"],
            comparisons["control"]["wins"], args.blocks, summary["prediction_correct"]
        )
    )
    hashes = {path.name: sha256_file(path) for path in output.iterdir()
              if path.is_file() and path.name != "artifact_hashes.json"}
    write_json(output / "artifact_hashes.json", {"artifacts": hashes})
    print(json.dumps(summary["comparisons"], indent=2), flush=True)


if __name__ == "__main__":
    main()
