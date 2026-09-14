#!/usr/bin/env python3
"""Time one hardware-qualified P7R support-transfer candidate on real VTA.

The protected TopHub incumbent brackets every randomized complete block.  The
runner refuses to time unless the exact original/residency/original correctness
ledger passed all three frozen seeds.
"""

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


EXPECTED_TOPHUB = {
    "W05": {
        "config_index": 575,
        "entity": [
            ["tile_b", "sp", [-1, 1]],
            ["tile_h", "sp", [-1, 14]],
            ["tile_w", "sp", [-1, 14]],
            ["tile_ci", "sp", [-1, 1]],
            ["tile_co", "sp", [-1, 4]],
            ["oc_nthread", "ot", 2],
            ["h_nthread", "ot", 1],
        ],
    }
}


def load_correctness(path, expected_sha256, contract):
    if sha256_file(path) != expected_sha256:
        raise ValueError("correctness ledger hash mismatch")
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    expected_order = [
        ("original", int(contract["execution_order"][0]["config_index"])),
        ("input_stationary", int(contract["execution_order"][1]["config_index"])),
        ("original", int(contract["execution_order"][2]["config_index"])),
    ]
    actual_order = [(row.get("mode"), int(row.get("config_index", -1))) for row in rows]
    if len(rows) != 3 or actual_order != expected_order:
        raise ValueError("timing requires the exact three-record frozen correctness order")
    if not all(row.get("correct") for row in rows):
        raise ValueError("timing requires every FPGA correctness record to pass")
    expected_seeds = list(contract["seeds"])
    for row in rows:
        if [item.get("seed") for item in row.get("seeds", [])] != expected_seeds:
            raise ValueError("correctness ledger seed set does not match the frozen contract")
        if not all(item.get("correct") for item in row["seeds"]):
            raise ValueError("timing requires all nine exact seed checks to pass")
    return rows


def verify_output(function, buffers, expected, label):
    function(*buffers)
    mismatch = int(np.count_nonzero(buffers[-1].numpy() != expected))
    if mismatch:
        raise RuntimeError(
            "{} correctness sentinel failed with {} mismatches".format(label, mismatch)
        )


def normalized_entity(config):
    value = config["entity"]
    return [[item[0], item[1], item[2]] for item in value]


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
    if contract["workload_id"] not in EXPECTED_TOPHUB:
        raise ValueError("no frozen TopHub identity for " + contract["workload_id"])
    correctness = load_correctness(args.correctness, args.correctness_sha256, contract)
    env = vta.get_env()
    if env.TARGET != "axu5evb":
        raise RuntimeError("VTA TARGET must be axu5evb")
    before = ssh_preflight(contract, args.host)

    candidate_index = int(contract["execution_order"][1]["config_index"])
    incumbent_index = int(EXPECTED_TOPHUB[contract["workload_id"]]["config_index"])
    entries = [
        {"label": "tophub_incumbent", "mode": "original", "config_index": incumbent_index},
        {"label": "transferred_original", "mode": "original", "config_index": candidate_index},
        {
            "label": "transferred_residency",
            "mode": "input_stationary",
            "config_index": candidate_index,
            "tir_sha256": contract["execution_order"][1]["tir_sha256"],
        },
    ]
    middle_labels = ["transferred_original", "transferred_residency"]
    rng = random.Random(args.order_seed)
    block_orders = []
    for _ in range(args.blocks):
        middle = list(middle_labels)
        rng.shuffle(middle)
        block_orders.append(["tophub_incumbent"] + middle + ["tophub_incumbent"])

    write_json(
        output / "preregistered.json",
        {
            "schema": "c3_p7r_support_transfer_timing_protocol_v1",
            "frozen_before_timing": True,
            "workload_id": contract["workload_id"],
            "candidate_config_index": candidate_index,
            "tophub_config_index": incumbent_index,
            "tophub_complete_config_entity": EXPECTED_TOPHUB[contract["workload_id"]]["entity"],
            "correctness_sha256": args.correctness_sha256,
            "correctness_records": len(correctness),
            "exact_seed_checks": sum(len(row["seeds"]) for row in correctness),
            "blocks": args.blocks,
            "number_per_sample": args.number,
            "warmup_per_module": args.warmup,
            "order_seed": args.order_seed,
            "block_orders": block_orders,
            "primary": "median paired same-tile speedup is positive",
            "protected": "residency median must beat TopHub or dispatch falls back to TopHub",
            "scope": "prospective support-transfer operator timing; no stage or FPS claim",
            "runner_sha256": sha256_file(__file__),
        },
    )

    remote = rpc.connect(args.host, args.port, session_timeout=120)
    device = remote.ext_dev(0)
    records = []
    start = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="c3_p7r_support_timing_") as temporary:
            directory = Path(temporary)
            modules, functions, build_identities = {}, {}, {}
            for entry in entries:
                binary, tir_hash, config = build_module(
                    contract["workload"], entry["mode"], entry["config_index"], env,
                    directory, entry.get("tir_sha256")
                )
                if entry["label"] == "tophub_incumbent":
                    if normalized_entity(config) != EXPECTED_TOPHUB[contract["workload_id"]]["entity"]:
                        raise RuntimeError("TopHub complete config entity changed")
                remote.upload(str(binary))
                loaded = remote.load_module(binary.name)
                modules[entry["label"]] = loaded
                functions[entry["label"]] = loaded["main"]
                build_identities[entry["label"]] = {
                    "tir_sha256": tir_hash,
                    "binary_sha256": sha256_file(binary),
                    "complete_config_entity": config,
                }

            data, weight, expected = reference_data_from_workload(contract["workload"], 0)
            buffers = [
                tvm.nd.array(data, device),
                tvm.nd.array(weight, device),
                tvm.nd.empty(expected.shape, "int8", device),
            ]
            for entry in entries:
                label = entry["label"]
                verify_output(functions[label], buffers, expected, "pre-timing " + label)
                for _ in range(args.warmup):
                    functions[label](*buffers)

            by_label = {entry["label"]: entry for entry in entries}
            with (output / "timing.jsonl").open("x", encoding="utf-8") as stream:
                for block, order in enumerate(block_orders):
                    for position, label in enumerate(order):
                        evaluator = modules[label].time_evaluator(
                            "main", device, number=args.number, repeat=1
                        )
                        latency_ms = float(evaluator(*buffers).mean * 1000.0)
                        row = {
                            "block": block,
                            "position": position,
                            "label": label,
                            "mode": by_label[label]["mode"],
                            "config_index": by_label[label]["config_index"],
                            "number": args.number,
                            "latency_ms": latency_ms,
                        }
                        records.append(row)
                        stream.write(json.dumps(row, sort_keys=True) + "\n")
                        stream.flush()
                    print("block {}/{} complete".format(block + 1, args.blocks), flush=True)
            for entry in entries:
                label = entry["label"]
                verify_output(functions[label], buffers, expected, "post-timing " + label)
    except Exception as error:
        write_json(output / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise

    samples = {}
    for label in ("tophub_incumbent", "transferred_original", "transferred_residency"):
        values = [row["latency_ms"] for row in records if row["label"] == label]
        samples[label] = {
            "count": len(values),
            "median_ms": statistics.median(values),
            "min_ms": min(values),
            "max_ms": max(values),
            "values_ms": values,
        }
    original_by_block = {
        row["block"]: row["latency_ms"] for row in records
        if row["label"] == "transferred_original"
    }
    residency_by_block = {
        row["block"]: row["latency_ms"] for row in records
        if row["label"] == "transferred_residency"
    }
    paired = [
        (original_by_block[block] - residency_by_block[block]) / original_by_block[block]
        for block in sorted(original_by_block)
    ]
    original = samples["transferred_original"]["median_ms"]
    residency = samples["transferred_residency"]["median_ms"]
    incumbent = samples["tophub_incumbent"]["median_ms"]
    incumbent_values = samples["tophub_incumbent"]["values_ms"]
    comparison = {
        "same_tile_original_median_ms": original,
        "input_stationary_median_ms": residency,
        "paired_speedup_fractions": paired,
        "paired_median_speedup_fraction": statistics.median(paired),
        "paired_wins": sum(value > 0 for value in paired),
        "tophub_incumbent_median_ms": incumbent,
        "residency_vs_tophub_fraction": (incumbent - residency) / incumbent,
        "dispatch_choice": "transferred_residency" if residency < incumbent else "tophub_incumbent",
    }
    summary = {
        "schema": "c3_p7r_support_transfer_timing_v1",
        "status": "completed",
        "scope": "prospective support-transfer operator timing; no stage or FPS claim",
        "workload_id": contract["workload_id"],
        "samples": samples,
        "comparison": comparison,
        "incumbent_sentinel_drift_fraction":
            (max(incumbent_values) - min(incumbent_values)) / statistics.median(incumbent_values),
        "build_identities": build_identities,
        "correctness_gate_sha256": args.correctness_sha256,
        "elapsed_seconds": time.monotonic() - start,
        "board_before": before,
        "board_after": ssh_preflight(contract, args.host),
    }
    write_json(output / "summary.json", summary)
    (output / "command.txt").write_text(
        " ".join([str(Path(__file__).resolve())] + __import__("sys").argv) + "\n",
        encoding="utf-8",
    )
    (output / "STATUS.md").write_text(
        "# P7R {} support-transfer timing\n\n"
        "- Status: `completed`\n"
        "- Same-tile paired median speedup: {:+.2%} ({}/{} wins)\n"
        "- Residency versus protected TopHub: {:+.2%}\n"
        "- Safe dispatch choice: `{}`\n"
        "- Scope: prospective operator timing; no stage/FPS claim\n".format(
            contract["workload_id"], comparison["paired_median_speedup_fraction"],
            comparison["paired_wins"], args.blocks,
            comparison["residency_vs_tophub_fraction"], comparison["dispatch_choice"]
        ),
        encoding="utf-8",
    )
    hashes = {
        path.name: sha256_file(path) for path in sorted(output.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    write_json(output / "artifact_hashes.json", {"artifacts": hashes})
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    main()
