#!/usr/bin/env python3
"""Paired timing for FPGA-qualified W05 bounded-hybrid t2 candidates."""

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


def load_correctness(path, expected_sha256, contract):
    if sha256_file(path) != expected_sha256:
        raise ValueError("correctness ledger hash mismatch")
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    expected = [(item["mode"], int(item["config_index"]))
                for item in contract["execution_order"]]
    actual = [(item["mode"], int(item["config_index"])) for item in rows]
    if actual != expected or not all(item.get("correct") for item in rows):
        raise ValueError("timing requires the exact fully passed correctness contract")
    if sum(len(item.get("seeds", [])) for item in rows) != 18:
        raise ValueError("timing requires all 18 frozen seed checks")
    if not all(seed.get("correct") for item in rows for seed in item["seeds"]):
        raise ValueError("one or more exact seed checks failed")
    return rows


def verify_output(function, buffers, expected, label):
    function(*buffers)
    mismatch = int(np.count_nonzero(buffers[-1].numpy() != expected))
    if mismatch:
        raise RuntimeError("{} sentinel has {} mismatches".format(label, mismatch))


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
    correctness = load_correctness(
        args.correctness, args.correctness_sha256, contract
    )
    env = vta.get_env()
    if env.TARGET != "axu5evb":
        raise RuntimeError("VTA TARGET must be axu5evb")
    before = ssh_preflight(contract, args.host)

    entries = []
    seen = set()
    for item in contract["execution_order"]:
        key = (item["mode"], int(item["config_index"]))
        if key in seen:
            continue
        seen.add(key)
        if int(item["config_index"]) == 575:
            label = "tophub_incumbent"
        else:
            label = "{}_{}".format(item["mode"], item["config_index"])
        entries.append({**item, "label": label})
    expected_labels = {
        "tophub_incumbent", "original_574", "paper_inspired_hybrid_574",
        "original_494", "paper_inspired_hybrid_494",
    }
    if {item["label"] for item in entries} != expected_labels:
        raise ValueError("unexpected hybrid timing identities")
    middle = sorted(expected_labels - {"tophub_incumbent"})
    rng = random.Random(args.order_seed)
    orders = []
    for _ in range(args.blocks):
        shuffled = list(middle)
        rng.shuffle(shuffled)
        orders.append(["tophub_incumbent"] + shuffled + ["tophub_incumbent"])
    write_json(output / "preregistered.json", {
        "schema": "c3_p7r_w05_hybrid_timing_protocol_v1",
        "status": "frozen_before_latency",
        "correctness_sha256": args.correctness_sha256,
        "correctness_records": len(correctness),
        "exact_seed_checks": 18,
        "blocks": args.blocks,
        "number_per_sample": args.number,
        "warmup_per_module": args.warmup,
        "order_seed": args.order_seed,
        "block_orders": orders,
        "primary": "config574 has lowest final latency among new candidates",
        "same_tile": "each hybrid is compared with its exact original config",
        "protected": "only a hybrid faster than TopHub config575 may replace it",
        "runner_sha256": sha256_file(__file__),
    })

    remote = rpc.connect(args.host, args.port, session_timeout=120)
    device = remote.ext_dev(0)
    records = []
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="c3_p7r_hybrid_timing_") as temporary:
            modules, functions, identities = {}, {}, {}
            for item in entries:
                binary, tir_hash, config = build_module(
                    contract["workload"], item["mode"], item["config_index"], env,
                    Path(temporary), item.get("tir_sha256")
                )
                remote.upload(str(binary))
                module = remote.load_module(binary.name)
                modules[item["label"]] = module
                functions[item["label"]] = module["main"]
                identities[item["label"]] = {
                    "tir_sha256": tir_hash,
                    "binary_sha256": sha256_file(binary),
                    "complete_config_entity": config,
                }
            data, weight, expected = reference_data_from_workload(contract["workload"], 0)
            buffers = [tvm.nd.array(data, device), tvm.nd.array(weight, device),
                       tvm.nd.empty(expected.shape, "int8", device)]
            for label in sorted(expected_labels):
                verify_output(functions[label], buffers, expected, "pre-timing " + label)
                for _ in range(args.warmup):
                    functions[label](*buffers)
            with (output / "timing.jsonl").open("x", encoding="utf-8") as stream:
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
            for label in sorted(expected_labels):
                verify_output(functions[label], buffers, expected, "post-timing " + label)
    except Exception as error:
        write_json(output / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise

    samples = {}
    for label in sorted(expected_labels):
        values = [row["latency_ms"] for row in records if row["label"] == label]
        samples[label] = {
            "count": len(values), "median_ms": statistics.median(values),
            "min_ms": min(values), "max_ms": max(values), "values_ms": values,
        }
    comparisons = {}
    incumbent = samples["tophub_incumbent"]["median_ms"]
    for index in (574, 494):
        original_label = "original_{}".format(index)
        hybrid_label = "paper_inspired_hybrid_{}".format(index)
        original_by_block = {row["block"]: row["latency_ms"] for row in records
                             if row["label"] == original_label}
        hybrid_by_block = {row["block"]: row["latency_ms"] for row in records
                           if row["label"] == hybrid_label}
        paired = [(original_by_block[b] - hybrid_by_block[b]) / original_by_block[b]
                  for b in sorted(original_by_block)]
        original = samples[original_label]["median_ms"]
        hybrid = samples[hybrid_label]["median_ms"]
        comparisons[str(index)] = {
            "original_median_ms": original,
            "hybrid_median_ms": hybrid,
            "paired_speedup_fractions": paired,
            "paired_median_speedup_fraction": statistics.median(paired),
            "paired_wins": sum(value > 0 for value in paired),
            "tophub_median_ms": incumbent,
            "hybrid_vs_tophub_fraction": (incumbent - hybrid) / incumbent,
        }
    winning_index = min((574, 494), key=lambda i: samples[
        "paper_inspired_hybrid_{}".format(i)]["median_ms"])
    winning_hybrid = samples["paper_inspired_hybrid_{}".format(winning_index)]["median_ms"]
    dispatch = (
        "paper_inspired_hybrid_{}".format(winning_index)
        if winning_hybrid < incumbent else "tophub_incumbent"
    )
    incumbent_values = samples["tophub_incumbent"]["values_ms"]
    summary = {
        "schema": "c3_p7r_w05_hybrid_timing_v1",
        "status": "completed",
        "scope": "prospective W05 operator timing; no stage or FPS claim",
        "samples": samples,
        "comparisons": comparisons,
        "prediction_config574_best_new_candidate": winning_index == 574,
        "best_new_candidate": "paper_inspired_hybrid_{}".format(winning_index),
        "safe_dispatch_choice": dispatch,
        "incumbent_sentinel_drift_fraction":
            (max(incumbent_values) - min(incumbent_values)) / statistics.median(incumbent_values),
        "build_identities": identities,
        "elapsed_seconds": time.monotonic() - started,
        "board_before": before,
        "board_after": ssh_preflight(contract, args.host),
    }
    write_json(output / "summary.json", summary)
    (output / "STATUS.md").write_text(
        "# W05 bounded-hybrid t2 timing\n\n"
        "- Config574 same-tile: {:+.2%} ({}/{} wins); versus TopHub: {:+.2%}\n"
        "- Config494 same-tile: {:+.2%} ({}/{} wins); versus TopHub: {:+.2%}\n"
        "- Frozen nearest-TopHub prediction correct: `{}`\n"
        "- Safe dispatch: `{}`\n- Scope: operator only; no stage/FPS claim\n".format(
            comparisons["574"]["paired_median_speedup_fraction"],
            comparisons["574"]["paired_wins"], args.blocks,
            comparisons["574"]["hybrid_vs_tophub_fraction"],
            comparisons["494"]["paired_median_speedup_fraction"],
            comparisons["494"]["paired_wins"], args.blocks,
            comparisons["494"]["hybrid_vs_tophub_fraction"],
            summary["prediction_config574_best_new_candidate"], dispatch,
        ), encoding="utf-8"
    )
    (output / "command.txt").write_text(
        " ".join([str(Path(__file__).resolve())] + __import__("sys").argv) + "\n",
        encoding="utf-8",
    )
    hashes = {path.name: sha256_file(path) for path in output.iterdir()
              if path.is_file() and path.name != "artifact_hashes.json"}
    write_json(output / "artifact_hashes.json", {"artifacts": hashes})
    print(json.dumps({"comparisons": comparisons, "safe_dispatch": dispatch}, indent=2))


if __name__ == "__main__":
    main()
