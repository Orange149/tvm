#!/usr/bin/env python3
"""Single-boot YOLO-224 node24 G2/InputAlt/F13-G graph confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import traceback

import numpy as np
from tvm.contrib import graph_executor

from c3_thread_sync_board import ExperimentBoard
import run_vta_c3_reuse_board as reuse
from run_vta_c3_node24_partial_board import Gate, GROUPED, RUNTIME, SEEDS, read, sha, write
from run_vta_c3_yolo608_graph_boot import random_input
from run_vta_p7r120_yolo_rpc_board import load_ephemeral


PILOT = GROUPED / "20261009_c3_node24_group_gemm_run01"
HELPER = GROUPED / "20260926_c3_evaluation_v5_helper_build_run02/c3_evaluation_v5_helper.so"
POLICY = GROUPED / "20260926_c3_yolo512_evaluation_v5_completion_audit_run01/completion_policy.json"
ARMS = ("G0", "InputAlt", "F13G")
GRAPH_SEEDS = (0, 20250901, 20260910, 0, 20250901,
               20260910, 0, 20250901, 20260910, 0)
ORDERS = (("G0", "InputAlt", "F13G"), ("F13G", "InputAlt", "G0"))


def equal(left, right):
    return len(left) == len(right) and all(
        x.shape == y.shape and x.dtype == y.dtype and np.array_equal(x, y)
        for x, y in zip(left, right)
    )


class GraphGate(Gate):
    def __init__(self, args):
        self.args, self.out = args, args.output_dir
        self.rows, self.sessions = [], []
        self.runtime = read(RUNTIME)
        for name, path in self.runtime["runtime_paths"].items():
            if sha(path) != self.runtime["runtime_sha256"][name]:
                raise ValueError("frozen runtime changed: " + name)
        reuse.RUNTIME = self.runtime["runtime_sha256"]
        policy = read(POLICY)
        if policy.get("status") != "COMPLETION_POLICY_FROZEN" or \
                policy.get("mode") != "run_only" or \
                sha(HELPER) != policy.get("helper_sha256"):
            raise ValueError("run-only graph completion helper identity changed")
        contract = read(args.prepared / "contract.json")
        cross = read(PILOT / "boot2_operator_run02/cross_boot_analysis.json")
        if contract["status"] != "GRAPH_LOCAL_BUILT_BOARD_NOT_QUALIFIED" or \
                not contract["unchanged_non_target_tir"] or \
                not contract["target_body_matches_isolated_operator"] or \
                cross["status"] != "TWO_BOOT_OPERATOR_CONFIRMED" or \
                sha(PILOT / "boot2_operator_run02/cross_boot_analysis.json") != \
                contract["operator_cross_boot_sha256"]:
            raise ValueError("graph/operator identity or correctness gate missing")
        self.libraries = {name: args.prepared / name / "graphlib.so"
                          for name in ("stock", *ARMS)}
        graph_hash = contract["base_graph_sha256"]
        for name, binary in self.libraries.items():
            build = read(binary.parent / "build.json")
            if sha(binary) != build["graphlib_sha256"] or \
                    sha(binary.parent / "graph.json") != build["graph_sha256"] or \
                    build["graph_sha256"] != graph_hash or \
                    not build["memory_contract"]["fits"]:
                raise ValueError("graph build/hash/memory contract mismatch: " + name)
        self.initial = ExperimentBoard(args, "/var/volatile/" + self.out.name + "_preflight",
                                       args.port).state()
        reuse.validate_state(self.initial, exclusive=True)
        self.boot = self.initial["boot_id"]
        if self.boot != cross["boot_ids"][1]:
            raise ValueError("graph must follow confirmed operator on this Linux boot")
        write(self.out / "preflight.json", self.initial)


def calls(gate, name, seeds, phase, references, block=None, prewarm=False):
    with gate.session(name, gate.libraries[name]) as (remote, lib, functions):
        helper_ref = load_ephemeral(remote, HELPER)
        observed_symbols = json.loads(str(remote.get_function("c3.v5.runtime_symbols")()))
        frozen_symbols = read(POLICY)["runtime_symbols"]
        for symbol, observed in observed_symbols.items():
            if symbol not in frozen_symbols or \
                    Path(observed["dso"]).name != Path(frozen_symbols[symbol]["dso"]).name:
                raise ValueError("graph helper resolves a different VTA runtime")
        graph = (gate.args.prepared / name / "graph.json").read_text()
        executor = graph_executor.create(graph, lib, [remote.ext_dev(0), remote.cpu(0)])
        executor.load_params((gate.args.prepared / name / "params.bin").read_bytes())
        timer = remote.get_function("c3.v5.graph_run_only_timed_once")
        reset = remote.get_function("c3.v5.reset_counts")
        counts = remote.get_function("c3.v5.counts")
        reset()
        results = []

        def invoke(seed, call_phase, position):
            executor.set_input("data", random_input(seed, resolution=224))
            functions["vta.runtime.profiler_clear"]()
            latency = float(timer(executor.module)) if call_phase == "timing" else None
            if call_phase != "timing":
                executor.run()
            profile = json.loads(str(functions["vta.runtime.profiler_status"]()))
            outputs = [np.array(executor.get_output(i).numpy(), copy=True)
                       for i in range(executor.get_num_outputs())]
            correct = references is None or equal(outputs, references[seed])
            gate.record(phase=call_phase, variant=name, block=block, position=position,
                        seed=seed, latency_ms=latency, correct=correct,
                        output_hashes=[hashlib.sha256(x.tobytes()).hexdigest()
                                       for x in outputs], runtime_profile=profile)
            if not correct or profile.get("driver_timeout_calls", 0) or \
                    profile.get("driver_run_calls", 0) <= 0 or \
                    (call_phase == "timing" and (not math.isfinite(latency) or latency <= 0)):
                raise ValueError("graph output/completion/timing failed: " + name)
            return outputs

        if prewarm:
            if phase != "timing" or not seeds:
                raise ValueError("prewarm is only valid for a nonempty timing block")
            invoke(seeds[0], "warmup", 0)
        for position, seed in enumerate(seeds):
            results.append(invoke(seed, phase, position))
        if phase == "timing":
            actual = json.loads(str(counts()))["graph_calls"]
            if actual != len(seeds):
                raise ValueError("hidden or missing timed GraphExecutor.run")
        del helper_ref
        return results


def run(args):
    gate = GraphGate(args)
    write(gate.out / "contract.json", {
        "status": "PRE_GRAPH_LATENCY_FROZEN", "boot_id": gate.boot,
        "order": ORDERS, "seed_sequence": GRAPH_SEEDS,
        "binary_sha256": {name: sha(path) for name, path in gate.libraries.items()},
        "helper_sha256": sha(HELPER), "runtime_sha256": gate.runtime["runtime_sha256"],
        "primary": "F13G_vs_G0", "primary_min_graph_gain_pct": 1,
        "block_direction": "F13G faster than G0 in both five-call blocks",
        "incremental": "F13G_vs_InputAlt, descriptive; no graph-wide attribution",
        "single_boot_only": True,
    })
    stock = calls(gate, "stock", SEEDS, "reference", None)
    references = dict(zip(SEEDS, stock))
    for name in ARMS:
        calls(gate, name, SEEDS, "qualification", references)
    write(gate.out / "qualification.json", {
        "status": "THREE_GRAPHS_CORRECT_VS_STOCK", "boot_id": gate.boot,
        "stock_reference_calls": 3, "candidate_qualification_calls": 9,
    })
    for block, order in enumerate(ORDERS):
        seeds = GRAPH_SEEDS[:5] if block == 0 else GRAPH_SEEDS[5:]
        for name in order:
            calls(gate, name, seeds, "timing", references, block=block, prewarm=True)
    timed = [row for row in gate.rows if row["phase"] == "timing"]
    warm = [row for row in gate.rows if row["phase"] == "warmup"]
    if len(timed) != 30 or len(warm) != 6 or len(gate.rows) != 48:
        raise ValueError("graph call ledger is not 3+9+6+30")
    per_block = {block: {name: median(row["latency_ms"] for row in timed
                                      if row["block"] == block and row["variant"] == name)
                         for name in ARMS} for block in (0, 1)}
    per_arm = {name: median(row["latency_ms"] for row in timed if row["variant"] == name)
               for name in ARMS}
    primary_gain = 100 * (1 - per_arm["F13G"] / per_arm["G0"])
    input_gain = 100 * (1 - per_arm["F13G"] / per_arm["InputAlt"])
    primary_blocks = all(per_block[block]["F13G"] < per_block[block]["G0"]
                         for block in (0, 1))
    input_blocks = all(per_block[block]["F13G"] < per_block[block]["InputAlt"]
                       for block in (0, 1))
    write(gate.out / "analysis.json", {
        "arm_median_ms": per_arm, "arm_equivalent_fps": {name: 1000 / value
                                                for name, value in per_arm.items()},
        "block_median_ms": per_block,
        "F13G_vs_G0_gain_pct": primary_gain,
        "F13G_vs_InputAlt_gain_pct": input_gain,
        "F13G_vs_G0_block_direction": primary_blocks,
        "F13G_vs_InputAlt_block_direction": input_blocks,
        "single_boot_graph_gate_passed": primary_gain >= 1 and primary_blocks,
        "not_pipeline_throughput": True,
    })
    final = ExperimentBoard(args, "/var/volatile/" + gate.out.name + "_postflight",
                            args.port).state()
    reuse.validate_state(final, exclusive=True)
    if final["boot_id"] != gate.boot:
        raise ValueError("Linux boot changed during graph experiment")
    write(gate.out / "postflight.json", final)
    write(gate.out / "sessions.json", gate.sessions)
    write(gate.out / "summary.json", {
        "status": "SINGLE_BOOT_GRAPH_COMPLETE", "boot_id": gate.boot,
        "all_correct": True, "reference_calls": 3, "qualification_calls": 9,
        "warmup_calls": 6, "timed_calls": 30,
        "no_second_graph_boot": True,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
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
