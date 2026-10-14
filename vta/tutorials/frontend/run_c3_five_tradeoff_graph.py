#!/usr/bin/env python3
"""Single-boot graph test of exactly the pre-reference sealed node24 winner."""

import argparse
import json
import math
from pathlib import Path
from statistics import median
import traceback

import numpy as np
import tvm
from tvm.contrib import graph_executor

import c3_residency_tradeoff as p
from c3_five_tradeoff_board import Runner, execution_sources
from c3_five_tradeoff_policy import NEW
from c3_residency_tradeoff_board import append, load_ephemeral
from run_vta_c3_node34_board_qualify import HELPER, POLICY
from run_vta_c3_yolo608_graph_boot import random_input, output_summary
from run_vta_c3_mode11_graph_context_diagnostic import local_params
from c3_yolo224_incremental import GRAPH_SEEDS

VERSIONS = ("current", "Input", "rule")
ORDERS = (VERSIONS, tuple(reversed(VERSIONS)))


def same_outputs(left, right):
    return len(left) == len(right) and all(x.shape == y.shape and x.dtype == y.dtype and np.array_equal(x, y)
                                         for x, y in zip(left, right))


def evaluate(rows):
    times, blocks = {}, {}
    for name in VERSIONS:
        part = [r for r in rows if r["program"] == name and r["call_kind"] == "graph_timing"]
        if len(part) != 10 or any(not r["correct"] or r["status"] != "PASS" or
                                  not math.isfinite(r["latency_ms"]) or r["latency_ms"] <= 0 for r in part):
            raise ValueError("incomplete/incorrect graph cannot enter performance analysis")
        times[name] = median(r["latency_ms"] for r in part)
        blocks[name] = [median(r["latency_ms"] for r in part if r["block"] == i) for i in range(2)]
    gain = {name: 100 * (1 - times["rule"] / times[name]) for name in ("current", "Input")}
    direction = {name: all(blocks["rule"][i] < blocks[name][i] for i in range(2)) for name in gain}
    return {"median_ms": times, "single_graph_equivalent_fps": {n: 1000 / t for n, t in times.items()},
            "block_median_ms": blocks, "rule_improvement_pct": gain, "both_blocks_faster": direction,
            "single_boot_practical_increment_vs_Input": gain["Input"] >= 1 and direction["Input"],
            "single_boot_improvement_vs_current": gain["current"] >= 1 and direction["current"],
            "not_cross_boot_stable_or_application_pipeline_throughput": True}


def run(args):
    root = args.output_dir.resolve()
    deployment = p.read(root / "graph_prepare/deployment_contract.json")
    results, sealed = p.read(root / "analysis/results.json"), p.read(root / "winners.json")
    if not results["node24_new_graph_eligible"] or sealed["winners"]["rule"] != NEW or \
            deployment["winner_sha256"] != p.digest(root / "winners.json") or \
            deployment["operator_result_sha256"] != p.digest(root / "analysis/results.json"):
        raise ValueError("exact pre-reference winner has not passed independent gate")
    directories = {name: Path(path) for name, path in deployment["deployments"].items()}
    graph = p.digest(directories["current"] / "graph.json")
    parameters = local_params(directories["current"])
    if p.digest(directories["stock"] / "graphlib.so") != deployment["binary_sha256"]["stock"]:
        raise ValueError("stock reference binary identity changed")
    stock_params = local_params(directories["stock"])
    if set(stock_params) != set(parameters) or any(not np.array_equal(stock_params[k], parameters[k]) for k in stock_params):
        raise ValueError("stock reference does not share frozen model parameters")
    old_functions = p.read(directories["current"] / "functions.json")
    for name in VERSIONS:
        directory = directories[name]
        if p.digest(directory / "graphlib.so") != deployment["binary_sha256"][name] or \
                p.digest(directory / "graph.json") != graph:
            raise ValueError("graph identity changed")
        other = local_params(directory)
        if set(other) != set(parameters) or any(not np.array_equal(other[k], parameters[k]) for k in other):
            raise ValueError("actual parameter tensors differ")
        inventory = p.read(directory / "functions.json")
        if set(inventory) != set(old_functions):
            raise ValueError("actual implementation inventory differs")
        for symbol in inventory:
            if symbol != deployment["target_symbol"]:
                old = tvm.ir.load_json((directories["current"] / (symbol + ".tir.json")).read_text())
                new = tvm.ir.load_json((directory / (symbol + ".tir.json")).read_text())
                if not tvm.ir.structural_equal(old, new, map_free_vars=True):
                    raise ValueError("non-target function changed: " + symbol)
    helper, completion = p.read(HELPER), p.read(POLICY)
    if completion["status"] != "COMPLETION_POLICY_FROZEN" or completion["mode"] != "run_only" or \
            p.digest(helper["binary"]) != helper["binary_sha256"] or \
            completion["helper_sha256"] != helper["binary_sha256"]:
        raise ValueError("graph completion helper identity unresolved")
    runner = Runner(args)
    if runner.boot != results["boot_id"]:
        raise ValueError("graph cannot change the single qualified Linux boot")
    runner.start_phase("graph_node24")
    runner.contract(graph_deployment_sha256=p.digest(root / "graph_prepare/deployment_contract.json"),
                    graph_helpers={"binary": helper["binary"], "sha256": helper["binary_sha256"]},
                    completion_policy_sha256=p.digest(POLICY), graph_orders=ORDERS, graph_seeds=GRAPH_SEEDS,
                    graph_source_sha256=p.digest(__file__), qualification_calls=9, reference_calls=3,
                    explicit_warmup_calls=6, timed_calls=30)

    def executor(remote, lib, name):
        obj = graph_executor.create((directories[name] / "graph.json").read_text(), lib,
                                    [remote.ext_dev(0), remote.cpu(0)])
        obj.load_params((directories[name] / "params.bin").read_bytes())
        return obj

    def call(obj, funcs, name, seed, kind, expected=None, timer=None, **extra):
        obj.set_input("data", random_input(seed, resolution=224))
        funcs["vta.runtime.profiler_clear"]()
        iid = runner.out.name + ":%06d" % len(runner.rows)
        append(runner.out / "dispatch.jsonl", {"invocation_id": iid, "event": "begin", "program": name, "call_kind": kind})
        t = float(timer(obj.module)) if timer is not None else None
        if timer is None:
            obj.run()
        profile = json.loads(str(funcs["vta.runtime.profiler_status"]()))
        outputs = [np.array(obj.get_output(i).numpy(), copy=True) for i in range(obj.get_num_outputs())]
        correct = expected is None or same_outputs(outputs, expected)
        runner.record({"program": name, "seed": seed, "call_kind": kind, "correct": correct,
                       "arguments_unchanged": True, "argument_check_scope": "full graph outputs; graph input may legally reuse storage",
                       "latency_ms": t, "runtime_profile": profile, "profile_before_output_reads": True,
                       "complete_graph_calls": 1, "hidden_warmup_calls": 0, "outputs": output_summary(outputs), **extra})
        append(runner.out / "dispatch.jsonl", {"invocation_id": iid, "event": "end"})
        if not correct or profile.get("driver_timeout_calls", 0) or profile.get("driver_run_calls", 0) <= 0 \
                or t is not None and (not math.isfinite(t) or t <= 0):
            np.savez_compressed(runner.out / ("failure_" + iid.replace(":", "_") + ".npz"), *outputs)
            raise ValueError("graph output/completion/timer failure; stop subsequent board work")
        return outputs

    refs = {}
    with runner.session("stock", directories["stock"] / "graphlib.so") as (remote, lib, funcs, retained, sid):
        obj = executor(remote, lib, "stock")
        for seed in (0, 20250901, 20260910):
            refs[seed] = call(obj, funcs, "stock", seed, "graph_reference", session_id=sid)
            np.savez_compressed(runner.out / ("stock_reference_%d.npz" % seed), *refs[seed])
        del obj
    for name in VERSIONS:
        with runner.session(name, directories[name] / "graphlib.so") as (remote, lib, funcs, retained, sid):
            obj = executor(remote, lib, name)
            for seed in (0, 20250901, 20260910):
                call(obj, funcs, name, seed, "graph_correctness", refs[seed], session_id=sid)
            del obj
    for block, order in enumerate(ORDERS):
        seeds = GRAPH_SEEDS[block * 5:block * 5 + 5]
        for name in order:
            with runner.session(name, directories[name] / "graphlib.so") as (remote, lib, funcs, retained, sid):
                retained.append(load_ephemeral(remote, Path(helper["binary"])))
                symbols = json.loads(str(remote.get_function("c3.v5.runtime_symbols")()))
                runtime_directory = "/var/volatile/c3_tradeoff_%s_graph_node24_%03d" % (
                    runner.root.name, int(sid.split(":")[-1]))
                for symbol, info in symbols.items():
                    if Path(info["dso"]).name != Path(completion["runtime_symbols"][symbol]["dso"]).name or \
                            str(Path(info["dso"]).parent) != runtime_directory:
                        raise ValueError("helper resolved a different VTA runtime DSO")
                p.write_new(runner.out / ("runtime_symbols_%s_b%d.json" % (name, block)), symbols)
                remote.get_function("c3.v5.reset_counts")()
                obj = executor(remote, lib, name)
                call(obj, funcs, name, seeds[0], "graph_warmup", refs[seeds[0]], session_id=sid, block=block)
                timer = remote.get_function("c3.v5.graph_run_only_timed_once")
                for i, seed in enumerate(seeds):
                    call(obj, funcs, name, seed, "graph_timing", refs[seed], timer=timer,
                         session_id=sid, block=block, sample=i)
                    count = json.loads(str(remote.get_function("c3.v5.counts")()))
                    if count["graph_calls"] != i + 1 or count["explicit_sync_calls"] != 0:
                        raise ValueError("extra graph invocation/synchronization")
                    append(runner.out / "helper_counts.jsonl", {"invocation_id": runner.rows[-1]["invocation_id"],
                                                               "expected_graph_calls": i + 1, "counts": count})
                del obj
            print("Graph block complete: %d %s" % (block + 1, name), flush=True)
    counts = {kind: sum(r["call_kind"] == kind for r in runner.rows) for kind in
              ("graph_reference", "graph_correctness", "graph_warmup", "graph_timing")}
    if counts != {"graph_reference": 3, "graph_correctness": 9, "graph_warmup": 6, "graph_timing": 30}:
        raise ValueError("graph stage call ledger differs")
    result = evaluate(runner.rows)
    p.write_new(runner.out / "results.json", result)
    p.write_new(runner.out / "summary.json", {"status": "SEALED_NODE24_GRAPH_SINGLE_BOOT_COMPLETE",
                "boot_id": runner.boot, "all_outputs_equal_stock": True, "call_counts": counts,
                "winner_sha256": p.digest(root / "winners.json"), "final_state": runner.final_state(),
                "no_second_boot_or_reselection": True})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--known-hosts", default=str(Path.home() / ".ssh/known_hosts"))
    parser.add_argument("--execute-board", action="store_true")
    args = parser.parse_args()
    if not args.execute_board:
        raise SystemExit("explicit --execute-board required")
    try:
        run(args)
    except BaseException as error:
        directory = args.output_dir / "graph_node24"
        directory.mkdir(exist_ok=True)
        p.write_new(directory / "invalid_attempt.json", {"status": "STOP_SUBSEQUENT_BOARD_WORK",
                    "error": str(error), "traceback": traceback.format_exc(), "no_retry": True})
        raise
