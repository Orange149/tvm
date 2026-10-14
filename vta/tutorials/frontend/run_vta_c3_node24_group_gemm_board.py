#!/usr/bin/env python3
"""One-boot, three-arm node24 group-GEMM experiment; no graph deployment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
import traceback

import numpy as np
import tvm

from c3_thread_sync_board import ExperimentBoard
import run_vta_c3_reuse_board as reuse
from run_vta_c3_node24_partial_board import (
    Gate, GROUPED, RUNTIME, SEEDS, SYMBOL, read, same, sentinel, sha, write,
)
from run_vta_p7r120_yolo_rpc_board import load_ephemeral


OLD = GROUPED / "20261009_c3_node24_partial_run01"
DMA = GROUPED / "20261009_c3_node24_group_dma_run01"
GEMM = GROUPED / "20261009_c3_node24_group_gemm_run01"
TIMER = GROUPED / "20261008_c3_node34_timer_build_run01/c3_node34_fused_timed_once.so"
VERSIONS = ("f13_b", "f13_g", "input_alt")
ORDERS = (
    ("f13_b", "f13_g", "input_alt"),
    ("f13_g", "input_alt", "f13_b"),
    ("input_alt", "f13_b", "f13_g"),
    ("input_alt", "f13_g", "f13_b"),
    ("f13_g", "f13_b", "input_alt"),
    ("f13_b", "input_alt", "f13_g"),
)


class GroupGEMMGate(Gate):
    def __init__(self, args):
        self.args, self.out = args, args.output_dir
        self.rows, self.sessions = [], []
        self.runtime = read(RUNTIME)
        for name, path in self.runtime["runtime_paths"].items():
            if sha(path) != self.runtime["runtime_sha256"][name]:
                raise ValueError("frozen runtime changed: " + name)
        reuse.RUNTIME = self.runtime["runtime_sha256"]
        self.libraries = {
            "f13_b": DMA / "binaries/f13_b/graphlib.so",
            "f13_g": GEMM / "binaries/f13_g/graphlib.so",
            "input_alt": OLD / "binaries/input_alt/graphlib.so",
        }
        local = {
            "f13_b": DMA / "local/f13_b/result.json",
            "f13_g": GEMM / "local/f13_g02/result.json",
            "input_alt": OLD / "local/input_alt03/result.json",
        }
        for name, binary in self.libraries.items():
            row, certificate = read(binary.parent / "result.json"), read(local[name])
            if (sha(binary) != row.get("graphlib_sha256") or
                    row.get("address_status") != "passed" or
                    row.get("target") != "axu5evb" or
                    certificate.get("status") != "LOCAL_FSIM_PASSED_BOARD_NOT_QUALIFIED" or
                    certificate.get("address_status") != "passed" or
                    row.get("request_totals") != certificate.get("request_totals") or
                    len(certificate.get("fsim", [])) != 4 or
                    not all(x["correct"] and x["arguments_unchanged"]
                            for x in certificate["fsim"])):
                raise ValueError("candidate identity/local certificate failed: " + name)
            def final_digest(record):
                return next(x["sha256"] for x in record["snapshots"]
                            if x["pass"].endswith("CPUAccessRewrite"))
            if final_digest(row) != final_digest(certificate):
                raise ValueError("sim/board final TIR mismatch: " + name)
            if bool(row.get("group_gemm", False)) != (name == "f13_g"):
                raise ValueError("merged GEMM candidate identity mismatch: " + name)
        old_request = read(DMA / "binaries/f13_b/result.json")["request_totals"]
        new_request = read(GEMM / "binaries/f13_g/result.json")["request_totals"]
        if old_request != new_request or new_request["load_buffer_2d_wgt_calls"] != 34 or \
                new_request["load_buffer_2d_wgt_bytes"] != 626688:
            raise ValueError("GEMM merge changed the frozen data-request contract")
        state = ExperimentBoard(args, "/var/volatile/" + self.out.name + "_preflight",
                                args.port).state()
        reuse.validate_state(state, exclusive=True)
        self.initial, self.boot = state, state["boot_id"]
        write(self.out / "preflight.json", state)


def analyze(rows):
    rounds = {}
    for round_id in range(6):
        rounds[round_id] = {}
        for name in VERSIONS:
            samples = [row["latency_ms"] for row in rows if row["phase"] == "timing"
                       and row["round"] == round_id and row["variant"] == name]
            if len(samples) != 3 or not all(math.isfinite(x) and x > 0 for x in samples):
                raise ValueError("incomplete timed round")
            rounds[round_id][name] = median(samples)
    comparisons = {}
    for baseline in ("f13_b", "input_alt"):
        gains = [100 * (1 - rounds[r]["f13_g"] / rounds[r][baseline])
                 for r in range(6)]
        comparisons[baseline + "_to_f13_g"] = {
            "round_improvement_pct": gains,
            "median_improvement_pct": median(gains),
            "wins": sum(x > 0 for x in gains),
        }
    primary = comparisons["f13_b_to_f13_g"]
    practical = comparisons["input_alt_to_f13_g"]
    return {
        "round_medians_ms": rounds,
        "version_medians_ms": {n: median(rounds[r][n] for r in range(6)) for n in VERSIONS},
        "comparisons": comparisons,
        "primary_engineering_gate_passed": primary["median_improvement_pct"] >= 2
        and primary["wins"] >= 5,
        "single_boot_deployment_candidate": practical["median_improvement_pct"] >= 1
        and practical["wins"] >= 5,
    }


def run(args):
    runner = GroupGEMMGate(args)
    x = {s: np.load(OLD / "board_qualify03" / ("X_%d.npy" % s)) for s in SEEDS}
    y = {s: np.load(OLD / "board_qualify03" / ("Y24_%d.npy" % s)) for s in SEEDS}
    weight = np.load(OLD / "board_qualify03/W_A.npy")
    weight_b = np.load(OLD / "board_qualify03/W_B.npy")
    y_b = np.load(OLD / "board_qualify03/Y_B.npy")
    add = np.load(OLD / "board_qualify03/ADD.npy")
    if same(y[0], y_b):
        raise ValueError("A/B reference outputs are not distinct")
    for name in VERSIONS:
        with runner.session(name, runner.libraries[name]) as (remote, lib, functions):
            dev = remote.ext_dev(0)
            buffers = (tvm.nd.array(x[0], dev), tvm.nd.array(weight, dev),
                       tvm.nd.array(add, dev), tvm.nd.array(sentinel(y[0]), dev))
            for seed in SEEDS:
                runner.fused(lib, functions, name, buffers, x[seed], weight, add,
                             y[seed], "seed_" + str(seed))
            if name == "f13_g":
                for label, w, expected in (("WA", weight, y[0]),
                                           ("WB", weight_b, y_b),
                                           ("WA_return", weight, y[0])):
                    runner.fused(lib, functions, name, buffers, x[0], w, add,
                                 expected, label)
    write(runner.out / "qualification.json", {
        "status": "THREE_ARMS_FPGA_CORRECT", "boot_id": runner.boot,
        "calls": len(runner.rows), "wa_wb_reference_distinct": True,
    })
    write(runner.out / "contract.json", {
        "status": "PRE_LATENCY_FROZEN", "boot_id": runner.boot,
        "orders": ORDERS, "seed_rotation": list(SEEDS),
        "binaries_sha256": {k: sha(v) for k, v in runner.libraries.items()},
        "timer_sha256": sha(TIMER), "runtime_sha256": runner.runtime["runtime_sha256"],
        "warmup_per_cell": 3, "timed_per_cell": 3,
        "primary": "f13_b_to_f13_g", "primary_min_gain_pct": 2,
        "practical": "input_alt_to_f13_g", "practical_min_gain_pct": 1,
        "min_wins": 5, "no_graph_or_second_boot": True,
    })
    for round_id, order in enumerate(ORDERS):
        seeds = SEEDS[round_id % 3:] + SEEDS[:round_id % 3]
        for name in order:
            with runner.session(name, runner.libraries[name]) as (remote, lib, functions):
                timer_ref = load_ephemeral(remote, TIMER)
                timed_once = remote.get_function("c3.node34.fused_timed_once")
                reset = remote.get_function("c3.node34.reset_counts")
                counts = remote.get_function("c3.node34.counts")
                reset()
                dev = remote.ext_dev(0)
                buffers = (tvm.nd.array(x[seeds[0]], dev), tvm.nd.array(weight, dev),
                           tvm.nd.array(add, dev), tvm.nd.array(sentinel(y[seeds[0]]), dev))
                xbuf, wbuf, abuf, ybuf = buffers
                for phase in ("warmup", "timing"):
                    for position, seed in enumerate(seeds):
                        xbuf.copyfrom(x[seed])
                        wbuf.copyfrom(weight)
                        abuf.copyfrom(add)
                        ybuf.copyfrom(sentinel(y[seed]))
                        functions["vta.runtime.profiler_clear"]()
                        latency = float(timed_once(lib, SYMBOL, *buffers)) if phase == "timing" else None
                        if phase == "warmup":
                            lib[SYMBOL](*buffers)
                        actual = np.array(ybuf.numpy(), copy=True)
                        profile = json.loads(str(functions["vta.runtime.profiler_status"]()))
                        unchanged = all(same(got, want) for got, want in
                                        zip((xbuf.numpy(), wbuf.numpy(), abuf.numpy()),
                                            (x[seed], weight, add)))
                        correct = same(actual, y[seed]) and unchanged
                        runner.record(phase=phase, round=round_id, variant=name,
                                      position=position, seed=seed, correct=correct,
                                      inputs_unchanged=unchanged, latency_ms=latency,
                                      timer_calls=int(counts()), runtime_profile=profile)
                        expected_gemm = {"f13_b": 1008, "f13_g": 224,
                                         "input_alt": 12}[name]
                        if not correct or profile.get("driver_timeout_calls", 0) or \
                                profile.get("driver_run_calls", 0) <= 0 or \
                                profile.get("push_gemm_op_calls") != expected_gemm or \
                                (phase == "timing" and
                                 (not math.isfinite(latency) or latency <= 0)):
                            np.save(runner.out / ("failure_%s_r%d_s%d.npy" %
                                                      (name, round_id, seed)), actual)
                            raise ValueError("invalid call; no timing samples may be spliced")
                if int(counts()) != 3:
                    raise ValueError("unexpected extra/missing timed target call")
                del timer_ref
    timing = [x for x in runner.rows if x["phase"] == "timing"]
    warmup = [x for x in runner.rows if x["phase"] == "warmup"]
    if len(timing) != 54 or len(warmup) != 54:
        raise ValueError("call ledger not 54+54")
    write(runner.out / "analysis.json", analyze(runner.rows))
    write(runner.out / "summary.json", {
        "status": "SINGLE_BOOT_COMPLETE", "boot_id": runner.boot,
        "qualification_calls": 12, "warmup_calls": 54, "timed_calls": 54,
        "all_correct": True, "no_graph_or_second_boot": True,
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
            "traceback": traceback.format_exc(), "timing_valid": False,
        })
        raise


if __name__ == "__main__":
    main()
