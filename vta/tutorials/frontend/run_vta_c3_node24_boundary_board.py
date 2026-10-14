#!/usr/bin/env python3
"""Controlled 14/15-group shape boundary; never participates in node24 winner selection."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
import traceback

import numpy as np
import tvm

import run_vta_c3_reuse_board as reuse
from c3_thread_sync_board import ExperimentBoard
from run_vta_c3_node24_partial_board import (
    G2, GROUPED, RUNTIME, SEEDS, SYMBOL, Gate, read, same, sentinel, sha, write,
)
from run_vta_c3_node24_partial_local import _reference
from run_vta_p7r120_yolo_rpc_board import load_ephemeral


ROOT = GROUPED / "20261009_c3_node24_partial_run01"
TIMER = GROUPED / "20261008_c3_node34_timer_build_run01/c3_node34_fused_timed_once.so"
NAMES = ("input14", "f13_14", "full14", "input15", "f13_15")
PAIRS = ((14, "input14", "f13_14"), (15, "input15", "f13_15"),
         (16, "input_fixed", "f13"))


class BoundaryGate(Gate):
    def __init__(self, args):
        self.args, self.out = args, args.output_dir
        self.rows, self.sessions = [], []
        self.variants = NAMES
        self.runtime = read(RUNTIME)
        for name, path in self.runtime["runtime_paths"].items():
            if sha(path) != self.runtime["runtime_sha256"][name]:
                raise ValueError("runtime identity drift: " + name)
        reuse.RUNTIME = self.runtime["runtime_sha256"]
        self.libraries = {name: args.binaries / (name + "_retry01") / "graphlib.so"
                          for name in NAMES}
        for name, binary in self.libraries.items():
            built = read(binary.parent / "result.json")
            local = read(ROOT / "boundary/local" /
                         ("input14_retry02" if name == "input14" else name) /
                         "result.json")
            if built["group_count"] != (14 if name.endswith("14") else 15) or \
                    built["address_status"] != "passed" or \
                    built["target"] != "axu5evb" or \
                    built["graphlib_sha256"] != sha(binary) or \
                    local["status"] != "LOCAL_FSIM_PASSED_BOARD_NOT_QUALIFIED" or \
                    local["request_totals"] != built["request_totals"]:
                raise ValueError("boundary local/board certificate mismatch: " + name)
            def final_digest(row):
                return next(item["sha256"] for item in row["snapshots"]
                            if item["pass"].endswith("CPUAccessRewrite"))
            if final_digest(local) != final_digest(built):
                raise ValueError("boundary final TIR mismatch: " + name)
        board = ExperimentBoard(args, "/var/volatile/" + self.out.name + "_preflight", args.port)
        self.initial = board.state()
        reuse.validate_state(self.initial, exclusive=True)
        self.boot = self.initial["boot_id"]
        write(self.out / "contract.json", {
            "status": "BOUNDARY_PRE_LATENCY_FROZEN", "boot_id": self.boot,
            "source_node": 24, "groups": [14, 15, 16],
            "group_16_uses_prior_operator_boot1": str(ROOT / "operator_boot1_01"),
            "binary_sha256": {n: sha(p) for n, p in self.libraries.items()},
            "timer_sha256": sha(TIMER),
            "order": NAMES, "seeds": SEEDS,
            "interpretation": "changed output channels also change computation; within-shape trends only",
        })

    def run_boundary(self):
        source = ROOT / "board_qualify03"
        inputs = {seed: np.load(source / ("X_%d.npy" % seed)) for seed in SEEDS}
        all_weight = np.load(source / "W_A.npy")
        all_add = np.load(source / "ADD.npy")
        # Real node24 data are sliced in group order; no synthetic random
        # replacement and no reselection based on the resulting latencies.
        for name in NAMES:
            groups = 14 if name.endswith("14") else 15
            weight = all_weight[:groups].copy()
            add = all_add[:groups].copy()
            expected = {seed: _reference(inputs[seed], weight, add) for seed in SEEDS}
            second = np.roll(weight, 1, axis=0).copy()
            expected_b = _reference(inputs[0], second, add)
            if same(expected[0], expected_b):
                raise ValueError("boundary A/B reference has no distinguishing output")
            with self.session(name, self.libraries[name]) as (remote, lib, functions):
                dev = remote.ext_dev(0)
                buffers = (
                    tvm.nd.array(inputs[0], dev), tvm.nd.array(weight, dev),
                    tvm.nd.array(add, dev), tvm.nd.array(sentinel(expected[0]), dev),
                )
                for seed in SEEDS:
                    self.fused(lib, functions, name, buffers, inputs[seed], weight,
                               add, expected[seed], "A_seed_" + str(seed))
                for label, changed, answer in (("WA_repeat", weight, expected[0]),
                                               ("WB", second, expected_b),
                                               ("WA_return", weight, expected[0])):
                    self.fused(lib, functions, name, buffers, inputs[0], changed,
                               add, answer, label)
        # Run timing only after *all* boundary candidates have passed FPGA
        # qualification.  Sessions are independent but all use the same board.
        timing = {}
        for name in NAMES:
            groups = 14 if name.endswith("14") else 15
            weight = all_weight[:groups].copy()
            add = all_add[:groups].copy()
            expected = {seed: _reference(inputs[seed], weight, add) for seed in SEEDS}
            with self.session(name + "_timing", self.libraries[name]) as (remote, lib, functions):
                timer_ref = load_ephemeral(remote, TIMER)
                timer = remote.get_function("c3.node34.fused_timed_once")
                count = remote.get_function("c3.node34.counts")
                remote.get_function("c3.node34.reset_counts")()
                dev = remote.ext_dev(0)
                buffers = (
                    tvm.nd.array(inputs[0], dev), tvm.nd.array(weight, dev),
                    tvm.nd.array(add, dev), tvm.nd.array(sentinel(expected[0]), dev),
                )
                xbuf, wbuf, abuf, ybuf = buffers
                for seed in SEEDS:
                    xbuf.copyfrom(inputs[seed]); wbuf.copyfrom(weight); abuf.copyfrom(add)
                    ybuf.copyfrom(sentinel(expected[seed]))
                    lib[SYMBOL](*buffers)
                    if not same(ybuf.numpy(), expected[seed]):
                        raise ValueError("boundary warmup output mismatch")
                    self.record(phase="warmup", variant=name, seed=seed, correct=True)
                for repeat in range(3):
                    for seed in SEEDS:
                        xbuf.copyfrom(inputs[seed]); wbuf.copyfrom(weight); abuf.copyfrom(add)
                        ybuf.copyfrom(sentinel(expected[seed]))
                        latency = float(timer(lib, SYMBOL, *buffers))
                        correct = same(ybuf.numpy(), expected[seed]) and \
                            same(xbuf.numpy(), inputs[seed]) and \
                            same(wbuf.numpy(), weight) and same(abuf.numpy(), add)
                        self.record(phase="timing", variant=name, seed=seed,
                                    repeat=repeat, correct=correct, latency_ms=latency)
                        if not correct or not math.isfinite(latency) or latency <= 0:
                            raise ValueError("boundary timed call invalid")
                if int(count()) != 9:
                    raise ValueError("boundary timer hidden/extra target calls")
                timing[name] = median(row["latency_ms"] for row in self.rows
                                      if row["phase"] == "timing" and row["variant"] == name)
                del timer_ref
        original = read(ROOT / "operator_boot1_01/analysis.json")["overall_medians_ms"]
        summary = {
            "status": "BOUNDARY_SINGLE_BOOT_COMPLETE", "boot_id": self.boot,
            "all_correct": True, "medians_ms": {**timing,
                "input16_prior_protocol": original["input_fixed"],
                "f13_16_prior_protocol": original["f13"]},
            "cross_shape_direct_comparison_valid": False,
            "group_16_uses_different_prior_protocol": True,
            "shape_relative_gain_pct": {
                str(g): 100 * (1 - timing[f] / timing[i]) if g < 16 else
                100 * (1 - original["f13"] / original["input_fixed"])
                for g, i, f in PAIRS},
            "full14_vs_input14_pct": 100 * (1 - timing["full14"] / timing["input14"]),
            "timed_calls": sum(r["phase"] == "timing" for r in self.rows),
        }
        write(self.out / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binaries", type=Path, default=ROOT / "boundary/binaries")
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
        BoundaryGate(args).run_boundary()
    except BaseException as error:
        write(args.output_dir / "invalid_attempt.json", {
            "status": "INVALID_NON_SPLICEABLE", "error": str(error),
            "traceback": traceback.format_exc(), "timing_valid": False,
        })
        raise


if __name__ == "__main__":
    main()
