#!/usr/bin/env python3
"""Time the frozen P7R W04 pairs after real-FPGA correctness qualification."""

from __future__ import annotations

import argparse
import hashlib
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


def load_correctness(path, expected_sha256):
    if sha256_file(path) != expected_sha256:
        raise ValueError("correctness ledger hash mismatch")
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if len(rows) != 6 or not all(row.get("correct") for row in rows):
        raise ValueError("timing gate requires six passed correctness records")
    return rows


def verify_output(function, buffers, expected):
    function(*buffers)
    actual = buffers[-1].numpy()
    mismatch = int(np.count_nonzero(actual != expected))
    if mismatch:
        raise RuntimeError("timing correctness sentinel failed with {} mismatches".format(mismatch))


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
    correctness = load_correctness(args.correctness, args.correctness_sha256)
    env = vta.get_env()
    if env.TARGET != "axu5evb":
        raise RuntimeError("VTA TARGET must be axu5evb")
    before = ssh_preflight(contract, args.host)

    entries = [
        {"label": "incumbent_463", "mode": "original", "config_index": 463},
        {"label": "original_455", "mode": "original", "config_index": 455},
        {
            "label": "input_stationary_455",
            "mode": "input_stationary",
            "config_index": 455,
            "tir_sha256": "cbc27aad1be773574fff2d5bfbeaba31ef06e17eaa9da0d9623d678d93766141",
        },
        {"label": "original_461", "mode": "original", "config_index": 461},
        {
            "label": "input_stationary_461",
            "mode": "input_stationary",
            "config_index": 461,
            "tir_sha256": "2725d118840080d050b4e97014bcc618bb6022068ac7c0901a9c938ece4cf51d",
        },
    ]
    middle_labels = [entry["label"] for entry in entries if entry["label"] != "incumbent_463"]
    rng = random.Random(args.order_seed)
    block_orders = []
    for _ in range(args.blocks):
        middle = list(middle_labels)
        rng.shuffle(middle)
        block_orders.append(["incumbent_463"] + middle + ["incumbent_463"])

    preregistered = {
        "schema": "c3_p7r_w04_timing_protocol_v1",
        "frozen_before_timing": True,
        "correctness_sha256": args.correctness_sha256,
        "correctness_records": len(correctness),
        "blocks": args.blocks,
        "number_per_sample": args.number,
        "warmup_per_module": args.warmup,
        "order_seed": args.order_seed,
        "block_orders": block_orders,
        "primary_comparisons": [
            ["input_stationary_455", "original_455"],
            ["input_stationary_461", "original_461"],
        ],
        "protected_comparison": "every candidate median versus incumbent_463 median",
    }
    write_json(output / "preregistered.json", preregistered)

    remote = rpc.connect(args.host, args.port, session_timeout=120)
    device = remote.ext_dev(0)
    records = []
    start = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="c3_p7r_timing_") as temporary:
            directory = Path(temporary)
            functions = {}
            modules = {}
            binaries = {}
            for entry in entries:
                binary, tir_hash, _ = build_module(
                    contract["workload"],
                    entry["mode"],
                    entry["config_index"],
                    env,
                    directory,
                    entry.get("tir_sha256"),
                )
                remote.upload(str(binary))
                loaded = remote.load_module(binary.name)
                modules[entry["label"]] = loaded
                functions[entry["label"]] = loaded["main"]
                binaries[entry["label"]] = {
                    "tir_sha256": tir_hash,
                    "binary_sha256": sha256_file(binary),
                }

            data, weight, expected = reference_data_from_workload(contract["workload"], 0)
            buffers = [
                tvm.nd.array(data, device),
                tvm.nd.array(weight, device),
                tvm.nd.empty(expected.shape, "int8", device),
            ]
            for entry in entries:
                function = functions[entry["label"]]
                verify_output(function, buffers, expected)
                for _ in range(args.warmup):
                    function(*buffers)

            by_label = {entry["label"]: entry for entry in entries}
            with (output / "timing.jsonl").open("x", encoding="utf-8") as stream:
                for block, order in enumerate(block_orders):
                    for position, label in enumerate(order):
                        entry = by_label[label]
                        evaluator = modules[label].time_evaluator(
                            "main", device, number=args.number, repeat=1
                        )
                        result = evaluator(*buffers)
                        record = {
                            "block": block,
                            "position": position,
                            "label": label,
                            "mode": entry["mode"],
                            "config_index": entry["config_index"],
                            "number": args.number,
                            "latency_ms": float(result.mean * 1000.0),
                        }
                        records.append(record)
                        stream.write(json.dumps(record, sort_keys=True) + "\n")
                        stream.flush()
                    print("block {}/{} complete".format(block + 1, args.blocks), flush=True)
            for entry in entries:
                verify_output(functions[entry["label"]], buffers, expected)
    except Exception as error:
        write_json(output / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise

    after = ssh_preflight(contract, args.host)
    samples = {}
    for entry in entries:
        label = entry["label"]
        values = [row["latency_ms"] for row in records if row["label"] == label]
        samples[label] = {
            "count": len(values),
            "median_ms": statistics.median(values),
            "min_ms": min(values),
            "max_ms": max(values),
            "values_ms": values,
        }
    comparisons = {}
    for index in (455, 461):
        original = samples["original_{}".format(index)]["median_ms"]
        residency = samples["input_stationary_{}".format(index)]["median_ms"]
        incumbent = samples["incumbent_463"]["median_ms"]
        comparisons[str(index)] = {
            "same_tile_original_median_ms": original,
            "input_stationary_median_ms": residency,
            "same_tile_speedup_fraction": (original - residency) / original,
            "incumbent_median_ms": incumbent,
            "candidate_vs_incumbent_fraction": (incumbent - residency) / incumbent,
        }
    incumbent_values = samples["incumbent_463"]["values_ms"]
    summary = {
        "schema": "c3_p7r_w04_joint_timing_v1",
        "status": "completed",
        "scope": "single-boot paired development timing; not unseen confirmation or FPS",
        "samples": samples,
        "comparisons": comparisons,
        "incumbent_sentinel_drift_fraction":
            (max(incumbent_values) - min(incumbent_values)) / statistics.median(incumbent_values),
        "elapsed_seconds": time.monotonic() - start,
        "board_before": before,
        "board_after": after,
        "correctness_gate_sha256": args.correctness_sha256,
    }
    write_json(output / "summary.json", summary)
    (output / "command.txt").write_text(" ".join([str(Path(__file__).resolve())] + __import__("sys").argv) + "\n")
    status = [
        "# P7R W04 joint timing",
        "",
        "- Status: `completed`",
        "- Scope: single-boot paired development timing; not unseen confirmation or FPS",
        "- Incumbent median: {:.6f} ms".format(samples["incumbent_463"]["median_ms"]),
    ]
    for index in (455, 461):
        item = comparisons[str(index)]
        status.append(
            "- Config{}: original {:.6f} ms, input-stationary {:.6f} ms, same-tile speedup {:+.2%}, versus incumbent {:+.2%}".format(
                index,
                item["same_tile_original_median_ms"],
                item["input_stationary_median_ms"],
                item["same_tile_speedup_fraction"],
                item["candidate_vs_incumbent_fraction"],
            )
        )
    (output / "STATUS.md").write_text("\n".join(status) + "\n")
    hashes = {
        path.name: sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    write_json(output / "artifact_hashes.json", {"artifacts": hashes})
    print(json.dumps(summary["comparisons"], indent=2), flush=True)


if __name__ == "__main__":
    main()
