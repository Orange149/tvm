#!/usr/bin/env python3
"""Gated first-boot YOLO-224 graph comparison for node34's token bridge."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
import traceback

import numpy as np
from tvm.contrib import graph_executor

import c3_yolo224_incremental as protocol
from run_vta_c3_yolo608_graph_boot import random_input, output_summary
from run_vta_c3_node34_board_qualify import (
    GRAPH, PREPARED, Qualification, SEEDS, equal, read, sha, write,
)
from run_vta_c3_node34_board_operator import QUALIFY

GRAPH["stock"] = PREPARED / "stock"
VERSIONS = ("G2", "W1", "BRIDGE")
ORDERS = (("G2", "W1", "BRIDGE"), ("BRIDGE", "W1", "G2"))
SEQUENCE = tuple(protocol.GRAPH_SEEDS)


def evaluate(rows):
    medians = {}
    blocks = {}
    for name in VERSIONS:
        values = [row["latency_ms"] for row in rows
                  if row["phase"] == "timing" and row["variant"] == name]
        if len(values) != 10 or not all(math.isfinite(x) and x > 0 for x in values):
            raise ValueError("missing or invalid graph timing sample")
        medians[name] = median(values)
        blocks[name] = [median(row["latency_ms"] for row in rows
                               if row["phase"] == "timing" and row["variant"] == name
                               and row["block"] == block)
                        for block in (0, 1)]
    gain = 100 * (1 - medians["BRIDGE"] / medians["G2"])
    decision = {"bridge_vs_g2_pct": gain,
                "both_blocks_bridge_faster_than_g2": all(
                    blocks["BRIDGE"][block] < blocks["G2"][block]
                    for block in (0, 1)),
                "request_second_linux_boot": gain >= 1 and all(
                    blocks["BRIDGE"][block] < blocks["G2"][block]
                    for block in (0, 1))}
    return {"median_ms": medians,
            "single_graph_equivalent_fps": {name: 1000 / value for name, value in medians.items()},
            "block_median_ms": blocks, "decision": decision}


def make_executor(remote, lib, name):
    root = GRAPH[name]
    executor = graph_executor.create((root / "graph.json").read_text(), lib,
                                     [remote.ext_dev(0), remote.cpu(0)])
    executor.load_params((root / "params.bin").read_bytes())
    return executor


def call(runner, executor, functions, name, seed, phase, reference, block=None, index=None, timer=None):
    executor.set_input("data", random_input(seed, resolution=224))
    functions["vta.runtime.profiler_clear"]()
    latency = float(timer(executor.module)) if phase == "timing" else None
    if phase != "timing":
        executor.run()
    outputs = [np.array(executor.get_output(i).numpy(), copy=True)
               for i in range(executor.get_num_outputs())]
    profile = json.loads(str(functions["vta.runtime.profiler_status"]()))
    good = reference is None or equal(outputs, reference)
    runner.record(phase=phase, variant=name, seed=seed, block=block, sample=index,
                  latency_ms=latency, correct=good,
                  outputs=output_summary(outputs), runtime_profile=profile)
    if not good or profile.get("driver_timeout_calls", 0) != 0 \
            or profile.get("driver_run_calls", 0) <= 0 \
            or (phase == "timing" and (not math.isfinite(latency) or latency <= 0)):
        np.savez_compressed(runner.out / ("failed_%s_%s.npz" % (name, phase)), *outputs)
        raise ValueError("graph correctness/completion failed; whole performance session invalid")
    return outputs


def run(args):
    op = read(args.operator / "summary.json")
    analysis = read(args.operator / "analysis.json")
    cert = read(QUALIFY / "summary.json")
    if op["status"] != "OPERATOR_FIRST_BOOT_COMPLETE" or not op["all_correct"] \
            or not op["graph_eligible"] or not analysis["decision"]["graph_eligible"]:
        raise ValueError("operator gate did not authorize graph comparison")
    if op["boot_id"] != cert["boot_id"]:
        raise ValueError("operator/qualification boot mismatch")
    runner = Qualification(args)
    if runner.boot != op["boot_id"]:
        raise ValueError("first-boot graph requires same Linux boot")
    for name in VERSIONS:
        if sha(GRAPH[name] / "graphlib.so") != cert["graphlib_sha256"][name]:
            raise ValueError("graph binary drift after qualification")
    write(runner.out / "contract.json", {
        "stage": "graph_first_boot", "boot_id": runner.boot,
        "operator_summary": str(args.operator / "summary.json"),
        "operator_summary_sha256": sha(args.operator / "summary.json"),
        "stock_graphlib_sha256": sha(GRAPH["stock"] / "graphlib.so"),
        "graphlib_sha256": {name: cert["graphlib_sha256"][name] for name in VERSIONS},
        "order": ORDERS, "input_seeds": SEQUENCE,
        "completion_mode": "run_only", "timed_calls_per_graph": 10,
        "warmup_per_five_call_block": 1,
    })
    refs = {}
    with runner.session("stock") as (remote, lib, functions):
        executor = make_executor(remote, lib, "stock")
        for seed in SEEDS:
            refs[seed] = call(runner, executor, functions, "stock", seed,
                              "reference", None)
            np.savez_compressed(runner.out / ("stock_%d.npz" % seed), *refs[seed])
    for name in VERSIONS:
        with runner.session(name) as (remote, lib, functions):
            executor = make_executor(remote, lib, name)
            for seed in SEEDS:
                call(runner, executor, functions, name, seed, "qualification", refs[seed])
    for block, order in enumerate(ORDERS):
        seeds = SEQUENCE[block * 5:block * 5 + 5]
        if len(seeds) != 5 or set(seeds) != set(SEEDS):
            raise ValueError("frozen five-call seed block changed")
        for name in order:
            with runner.session(name) as (remote, lib, functions):
                executor = make_executor(remote, lib, name)
                call(runner, executor, functions, name, seeds[0], "warmup", refs[seeds[0]], block)
                timer = remote.get_function("c3.v5.graph_run_only_timed_once")
                for index, seed in enumerate(seeds):
                    call(runner, executor, functions, name, seed, "timing", refs[seed],
                         block, index, timer)
                counts = json.loads(str(remote.get_function("c3.v5.counts")()))
                if counts["graph_calls"] != 5 or counts["explicit_sync_calls"] != 0:
                    raise ValueError("hidden graph call or extra explicit sync")
    counts = {phase: sum(row["phase"] == phase for row in runner.rows)
              for phase in ("reference", "qualification", "warmup", "timing")}
    if counts != {"reference": 3, "qualification": 9, "warmup": 6, "timing": 30}:
        raise ValueError("graph call ledger drift")
    result = evaluate(runner.rows)
    write(runner.out / "analysis.json", result)
    write(runner.out / "summary.json", {
        "status": "GRAPH_FIRST_BOOT_COMPLETE", "boot_id": runner.boot,
        "all_correct": True, "phase_counts": counts,
        "request_second_linux_boot": result["decision"]["request_second_linux_boot"],
        "operator_gate_sha256": sha(args.operator / "summary.json"),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--known-hosts", default=str(Path.home() / ".ssh/known_hosts"))
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--execute-board", action="store_true")
    args = parser.parse_args()
    if not args.execute_board:
        raise SystemExit("explicit --execute-board required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        run(args)
    except BaseException as error:
        write(args.output_dir / "invalid_attempt.json", {
            "status": "INVALID_NON_SPLICEABLE", "error": str(error),
            "traceback": traceback.format_exc(), "all_timing_invalid": True,
        })
        raise


if __name__ == "__main__":
    main()
