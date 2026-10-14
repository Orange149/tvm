#!/usr/bin/env python3
"""Frozen six-round first-boot fused-operator timing for node24."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
import traceback

import numpy as np
import tvm

from run_vta_c3_node24_partial_board import (
    Gate, G2, GROUPED, GRAPH_SYMBOL, SEEDS, SYMBOL, VARIANTS,
    read, same, sentinel, sha, write,
)
from run_vta_p7r120_yolo_rpc_board import load_ephemeral


ROOT = GROUPED / "20261009_c3_node24_partial_run01"
TIMER = GROUPED / "20261008_c3_node34_timer_build_run01/c3_node34_fused_timed_once.so"
QUALIFICATION = {
    "input_fixed": ROOT / "board_input_qualify01/summary.json",
    "input_alt": ROOT / "board_alt_qualify01/summary.json",
    "a12": ROOT / "board_qualify03/summary.json",
    "f12": ROOT / "board_qualify03/summary.json",
    "a13": ROOT / "board_qualify03/summary.json",
    "f13": ROOT / "board_qualify03/summary.json",
}
VERSIONS = ("G2", "input_fixed", "a12", "f12", "a13", "f13", "input_alt")
# Six-label even Williams square plus the separately deployed alternative
# Input reference.  Position balancing for the six primary programs is exact;
# input_alt alternates before/after the square to avoid being one-sided.
SQUARE = (
    ("G2", "input_fixed", "f13", "a12", "a13", "f12"),
    ("input_fixed", "a12", "G2", "f12", "f13", "a13"),
    ("a12", "f12", "input_fixed", "a13", "G2", "f13"),
    ("f12", "a13", "a12", "f13", "input_fixed", "G2"),
    ("a13", "f13", "f12", "G2", "a12", "input_fixed"),
    ("f13", "G2", "a13", "input_fixed", "f12", "a12"),
)
ORDERS = tuple(("input_alt",) + row if i % 2 == 0 else row + ("input_alt",)
               for i, row in enumerate(SQUARE))


def analyze(rows):
    by_round = {}
    for round_no in range(6):
        by_round[round_no] = {}
        for name in VERSIONS:
            samples = [row["latency_ms"] for row in rows
                       if row["phase"] == "timing" and row["round"] == round_no
                       and row["variant"] == name]
            if len(samples) != 3 or not all(math.isfinite(x) and x > 0 for x in samples):
                raise ValueError("missing three-seed node24 timing round: " + name)
            by_round[round_no][name] = median(samples)
    overall = {name: median(by_round[r][name] for r in range(6)) for name in VERSIONS}
    comparisons = {}
    for name in ("a12", "f12", "a13", "f13"):
        comparisons[name] = {
            "vs_input_fixed_pct": [100 * (1 - by_round[r][name] /
                                         by_round[r]["input_fixed"]) for r in range(6)],
            "vs_g2_pct": [100 * (1 - by_round[r][name] / by_round[r]["G2"])
                          for r in range(6)],
        }
    for resident, control in (("f12", "a12"), ("f13", "a13")):
        comparisons[resident]["vs_reload_pct"] = [
            100 * (1 - by_round[r][resident] / by_round[r][control]) for r in range(6)
        ]
    decisions = {}
    for name in ("f12", "f13"):
        gains = comparisons[name]["vs_input_fixed_pct"]
        decisions[name] = {
            "paired_median_improvement_pct": median(gains),
            "paired_round_wins": sum(g > 0 for g in gains),
            "beats_g2_and_alternative_input": all(
                overall[name] < overall[baseline] for baseline in ("G2", "input_alt")
            ),
        }
        decisions[name]["graph_eligible"] = (
            decisions[name]["paired_median_improvement_pct"] >= 2
            and decisions[name]["paired_round_wins"] >= 5
            and decisions[name]["beats_g2_and_alternative_input"]
        )
    eligible = [name for name in ("f12", "f13") if decisions[name]["graph_eligible"]]
    winner = sorted(eligible,
                    key=lambda name: (-decisions[name]["paired_median_improvement_pct"], name))[0] \
        if eligible else None
    return {
        "round_medians_ms": by_round, "overall_medians_ms": overall,
        "comparisons": comparisons, "decisions": decisions, "frozen_graph_winner": winner,
    }


def run(args):
    if not TIMER.is_file():
        raise ValueError("frozen four-argument timer helper absent")
    for name, path in QUALIFICATION.items():
        certificate = read(path)
        if certificate["status"] != "BOARD_CORRECT_NO_TIMING" or \
                not certificate["all_correct"] or \
                certificate["candidate_graphlib_sha256"][name] != \
                sha(args.binaries / name / "graphlib.so"):
            raise ValueError("board correctness prerequisite missing: " + name)
    runner = Gate(args)
    boots = {read(path)["boot_id"] for path in QUALIFICATION.values()}
    if boots != {runner.boot}:
        raise ValueError("node24 timing must use qualified first Linux boot")
    write(runner.out / "contract.json", {
        "status": "PRE_LATENCY_FROZEN", "boot_id": runner.boot,
        "orders": ORDERS, "seed_order": SEEDS, "timer_sha256": sha(TIMER),
        "binaries": {k: sha(v) for k, v in runner.libraries.items()},
        "g2_sha256": sha(G2 / "graphlib.so"),
        "per_session_warmup_calls": 3, "per_session_timed_calls": 3,
        "same_tile_residency_gate_pct": 2,
        "same_tile_residency_round_wins": 5,
        "required_practical_controls": ("G2", "input_alt"),
        "mode4_included": False,
        "mode4_exclusion": "same_tile_full_fusion_fsim_uop_destination_hazard",
    })
    x = {seed: np.load(ROOT / "board_qualify03" / ("X_%d.npy" % seed)) for seed in SEEDS}
    y = {seed: np.load(ROOT / "board_qualify03" / ("Y24_%d.npy" % seed)) for seed in SEEDS}
    weight = np.load(ROOT / "board_qualify03/W_A.npy")
    add = np.load(ROOT / "board_qualify03/ADD.npy")
    for round_no, order in enumerate(ORDERS):
        seeds = SEEDS[round_no % 3:] + SEEDS[:round_no % 3]
        for name in order:
            binary = G2 / "graphlib.so" if name == "G2" else runner.libraries[name]
            symbol = GRAPH_SYMBOL if name == "G2" else SYMBOL
            with runner.session(name, binary) as (remote, lib, functions):
                timer_ref = load_ephemeral(remote, TIMER)
                reset = remote.get_function("c3.node34.reset_counts")
                counts = remote.get_function("c3.node34.counts")
                timed_once = remote.get_function("c3.node34.fused_timed_once")
                reset()
                dev = remote.ext_dev(0)
                buffers = (
                    tvm.nd.array(x[seeds[0]], dev), tvm.nd.array(weight, dev),
                    tvm.nd.array(add, dev), tvm.nd.array(sentinel(y[seeds[0]]), dev),
                )
                xbuf, wbuf, abuf, ybuf = buffers
                for phase in ("warmup", "timing"):
                    for position, seed in enumerate(seeds):
                        xbuf.copyfrom(x[seed])
                        wbuf.copyfrom(weight)
                        abuf.copyfrom(add)
                        ybuf.copyfrom(sentinel(y[seed]))
                        functions["vta.runtime.profiler_clear"]()
                        latency = float(timed_once(lib, symbol, *buffers)) \
                            if phase == "timing" else None
                        if phase == "warmup":
                            lib[symbol](*buffers)
                        actual = np.array(ybuf.numpy(), copy=True)
                        profile = json.loads(str(functions["vta.runtime.profiler_status"]()))
                        unchanged = all(same(current, original) for current, original in zip(
                            (xbuf.numpy(), wbuf.numpy(), abuf.numpy()),
                            (x[seed], weight, add),
                        ))
                        correct = same(actual, y[seed]) and unchanged
                        runner.record(
                            phase=phase, round=round_no, variant=name,
                            position=position, seed=seed, latency_ms=latency,
                            correct=correct, timer_calls=int(counts()),
                            runtime_profile=profile,
                        )
                        if not correct or profile.get("driver_timeout_calls", 0) or \
                                profile.get("driver_run_calls", 0) <= 0 or \
                                (phase == "timing" and
                                 (not math.isfinite(latency) or latency <= 0)):
                            np.save(runner.out / ("failure_%s_r%d_s%d.npy" %
                                                      (name, round_no, seed)), actual)
                            raise ValueError("invalid node24 timed call; entire session excluded")
                if int(counts()) != 3:
                    raise ValueError("timer executed an extra/missing fused call")
                del timer_ref
    expected = 6 * len(VERSIONS) * 3
    if sum(row["phase"] == "timing" for row in runner.rows) != expected or \
            sum(row["phase"] == "warmup" for row in runner.rows) != expected:
        raise ValueError("node24 timing call ledger mismatch")
    result = analyze(runner.rows)
    write(runner.out / "analysis.json", result)
    write(runner.out / "winner.json", {
        "status": "WINNER_SEALED_FIRST_BOOT",
        "candidate": result["frozen_graph_winner"],
        "boot_id": runner.boot,
        "contract_sha256": sha(runner.out / "contract.json"),
        "binary_sha256": sha(runner.libraries[result["frozen_graph_winner"]])
        if result["frozen_graph_winner"] else None,
    })
    write(runner.out / "summary.json", {
        "status": "OPERATOR_FIRST_BOOT_COMPLETE", "boot_id": runner.boot,
        "all_correct": True, "timed_calls": expected, "warmup_calls": expected,
        "graph_winner": result["frozen_graph_winner"],
        "mode4_excluded": True,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--binaries", type=Path, required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--known-hosts", default=str(Path.home() / ".ssh/known_hosts"))
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--execute-board", action="store_true")
    args = parser.parse_args()
    if not args.execute_board:
        raise SystemExit("explicit --execute-board required")
    args.variants = VARIANTS
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        run(args)
    except BaseException as error:
        write(args.output_dir / "invalid_attempt.json", {
            "status": "INVALID_NON_SPLICEABLE", "error": str(error),
            "traceback": traceback.format_exc(), "timing_valid": False,
        })
        raise


if __name__ == "__main__":
    main()
