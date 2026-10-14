#!/usr/bin/env python3
"""Six-round fixed-Tile, single-call FPGA timing after node34 qualification."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
import traceback

import numpy as np
import tvm

from run_vta_c3_node34_board_qualify import (
    GROUPED, GRAPH, Qualification, SEEDS, SYMBOL, equal, read, sentinel, sha, write,
)
from run_vta_p7r120_yolo_rpc_board import load_ephemeral

QUALIFY = GROUPED / "20261008_c3_node34_board_qualify_run02"
TIMER = GROUPED / "20261008_c3_node34_timer_build_run01/c3_node34_fused_timed_once.so"
VERSIONS = ("G2", "O1", "I1", "W1", "BRIDGE")
ORDERS = (
    ("G2", "O1", "I1", "W1", "BRIDGE"),
    ("O1", "I1", "W1", "BRIDGE", "G2"),
    ("I1", "W1", "BRIDGE", "G2", "O1"),
    ("W1", "BRIDGE", "G2", "O1", "I1"),
    ("BRIDGE", "G2", "O1", "I1", "W1"),
    ("G2", "W1", "I1", "O1", "BRIDGE"),
)


def evaluate(rows):
    by_round = {}
    for round_no in range(6):
        by_round[round_no] = {}
        for name in VERSIONS:
            values = [row["latency_ms"] for row in rows if row["phase"] == "timing"
                      and row["round"] == round_no and row["variant"] == name]
            if len(values) != 3 or not all(math.isfinite(x) and x > 0 for x in values):
                raise ValueError("missing or invalid three-seed timing round")
            by_round[round_no][name] = median(values)
    overall = {name: median(by_round[r][name] for r in range(6)) for name in VERSIONS}
    gains = {name: [100 * (1 - by_round[r]["BRIDGE"] / by_round[r][name])
                    for r in range(6)] for name in VERSIONS if name != "BRIDGE"}
    decision = {
        "mode4_median_improvement_pct": median(gains["W1"]),
        "mode4_round_wins": sum(g > 0 for g in gains["W1"]),
        "beats_all_original_input_controls": all(overall["BRIDGE"] < overall[name]
                                                  for name in ("G2", "O1", "I1")),
    }
    decision["graph_eligible"] = (
        decision["mode4_median_improvement_pct"] >= 2
        and decision["mode4_round_wins"] >= 5
        and decision["beats_all_original_input_controls"]
    )
    return {"round_medians_ms": by_round, "overall_medians_ms": overall,
            "paired_improvement_pct": gains, "decision": decision}


def run(args):
    certificate = read(QUALIFY / "summary.json")
    if certificate["status"] != "BOARD_FUSION_AND_GRAPH_QUALIFIED_NO_TIMING" \
            or not certificate["all_correct"]:
        raise ValueError("board fused/graph correctness prerequisite missing")
    for name, root in GRAPH.items():
        if sha(root / "graphlib.so") != certificate["graphlib_sha256"][name]:
            raise ValueError("candidate binary drift after FPGA qualification")
    if not TIMER.is_file():
        raise ValueError("four-argument timer binary missing")
    runner = Qualification(args)
    if runner.boot != certificate["boot_id"]:
        raise ValueError("operator timing requires same verified first Linux boot")
    write(runner.out / "contract.json", {
        "stage": "operator_first_boot", "boot_id": runner.boot,
        "qualification": str(QUALIFY), "timer_sha256": sha(TIMER),
        "graphlib_sha256": certificate["graphlib_sha256"],
        "round_orders": ORDERS, "seed_order": SEEDS,
        "warmup_per_candidate_session": 3, "timed_per_candidate_session": 3,
        "latency_name": "board single full-fusion function completion time, post explicit warmup",
        "no_graph_performance_claim": True,
    })
    x = {seed: np.load(QUALIFY / ("X_%d.npy" % seed)) for seed in SEEDS}
    y = {seed: np.load(QUALIFY / ("Y34_%d.npy" % seed)) for seed in SEEDS}
    weight = np.load(QUALIFY / "W_A.npy")
    add = np.load(QUALIFY / "ADD.npy")
    for round_no, order in enumerate(ORDERS):
        # Rotate the within-group seed order without altering the seed set.
        seeds = SEEDS[round_no % 3:] + SEEDS[:round_no % 3]
        for name in order:
            with runner.session(name) as (remote, lib, functions):
                timer_ref = load_ephemeral(remote, TIMER)
                reset = remote.get_function("c3.node34.reset_counts")
                counts = remote.get_function("c3.node34.counts")
                timed_once = remote.get_function("c3.node34.fused_timed_once")
                reset()
                dev = remote.ext_dev(0)
                xbuf = tvm.nd.array(x[seeds[0]], dev)
                wbuf = tvm.nd.array(weight, dev)
                abuf = tvm.nd.array(add, dev)
                ybuf = tvm.nd.array(sentinel(y[seeds[0]]), dev)
                for phase in ("warmup", "timing"):
                    for position, seed in enumerate(seeds):
                        xbuf.copyfrom(x[seed])
                        wbuf.copyfrom(weight)
                        abuf.copyfrom(add)
                        ybuf.copyfrom(sentinel(y[seed]))
                        functions["vta.runtime.profiler_clear"]()
                        latency = float(timed_once(lib, SYMBOL, xbuf, wbuf, abuf, ybuf)) \
                            if phase == "timing" else None
                        if phase == "warmup":
                            lib[SYMBOL](xbuf, wbuf, abuf, ybuf)
                        output = np.array(ybuf.numpy(), copy=True)
                        profile = json.loads(str(functions["vta.runtime.profiler_status"]()))
                        good = equal([output], [y[seed]]) and equal(
                            [xbuf.numpy(), wbuf.numpy(), abuf.numpy()],
                            [x[seed], weight, add],
                        )
                        runner.record(phase=phase, variant=name, round=round_no,
                                      position=position, seed=seed, latency_ms=latency,
                                      correct=good, timer_calls=int(counts()),
                                      runtime_profile=profile)
                        if not good or profile.get("driver_timeout_calls", 0) != 0 \
                                or profile.get("driver_run_calls", 0) <= 0 \
                                or (phase == "timing" and (not math.isfinite(latency) or latency <= 0)):
                            np.save(runner.out / ("failure_%s_r%d_s%d.npy" % (name, round_no, seed)), output)
                            raise ValueError("invalid fused timing call; entire session excluded")
                if int(counts()) != 3:
                    raise ValueError("hidden or missing timed target invocation")
                del timer_ref
    if len([row for row in runner.rows if row["phase"] == "timing"]) != 90 or \
            len([row for row in runner.rows if row["phase"] == "warmup"]) != 90:
        raise ValueError("six-round call ledger mismatch")
    result = evaluate(runner.rows)
    write(runner.out / "analysis.json", result)
    write(runner.out / "summary.json", {
        "status": "OPERATOR_FIRST_BOOT_COMPLETE", "boot_id": runner.boot,
        "all_correct": True, "timed_calls": 90, "warmup_calls": 90,
        "graph_eligible": result["decision"]["graph_eligible"],
        "runtime_contract_sha256": sha(QUALIFY / "summary.json"),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
