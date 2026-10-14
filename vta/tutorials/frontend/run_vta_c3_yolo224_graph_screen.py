#!/usr/bin/env python3
"""Run a clean-deployment one-boot YOLO-224 full-graph screen."""

from __future__ import annotations

import argparse
import gc
import json
import math
import shlex
import statistics
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
from tvm import rpc

import run_vta_c3_reuse_board as reuse_board
from c3_evaluation_v5 import (
    GRAPH_SEED_SEQUENCE,
    SEEDS,
    append_jsonl,
    read_json,
    seal_flat,
    sha256_file,
    verify_flat,
    write_json,
)
from c3_thread_sync_board import ExperimentBoard
from run_vta_c3_yolo608_graph_boot import (
    load_executor,
    mismatch_counts,
    output_summary,
    random_input,
)
from run_vta_p7r120_yolo_rpc_board import load_ephemeral, runtime_inventory


VARIANTS = ("stock", "incumbent", "input_control", "selector_v4")
BLOCK_ORDERS = {
    "A": ("stock", "incumbent", "input_control", "selector_v4"),
    "B": ("selector_v4", "input_control", "incumbent", "stock"),
}
BLOCK_SEEDS = {"A": GRAPH_SEED_SEQUENCE[:5], "B": GRAPH_SEED_SEQUENCE[5:]}


def connect(host, port, timeout):
    for _ in range(30):
        try:
            return rpc.connect(host, port, session_timeout=timeout)
        except Exception:
            time.sleep(0.25)
    raise RuntimeError("RPC unavailable for YOLO-224 graph session")


def exact(outputs, references):
    return all(value == 0 for value in mismatch_counts(references, outputs))


def run(args):
    build_dir = Path(args.build_dir).resolve()
    helper_dir = Path(args.helper_build_dir).resolve()
    completion_dir = Path(args.completion_dir).resolve()
    state_diagnostic_dir = Path(args.state_diagnostic_dir).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    manifests = {
        "build": verify_flat(build_dir),
        "helper": verify_flat(helper_dir),
        "completion": verify_flat(completion_dir),
        "state_diagnostic": verify_flat(state_diagnostic_dir),
    }
    build = read_json(build_dir / "summary.json")
    helper_contract = read_json(helper_dir / "contract.json")
    policy = read_json(completion_dir / "completion_policy.json")
    state_diagnostic = read_json(state_diagnostic_dir / "summary.json")
    if (build.get("status") != "C3_FOUR_YOLO224_DEPLOYMENTS_BUILT"
            or build.get("input_shape") != [1, 3, 224, 224]):
        raise ValueError("YOLO-224 graph build is incomplete")
    if (policy.get("status") != "COMPLETION_POLICY_FROZEN"
            or policy.get("mode") != "run_only"):
        raise ValueError("the frozen runtime requires the audited run-only completion boundary")
    if state_diagnostic.get("classification") != "GENERAL_CROSS_RPC_FPGA_STATE_CONTAMINATION":
        raise ValueError("clean-start requirement is not backed by the frozen diagnostic")
    helper = Path(helper_contract["binary"])
    if (sha256_file(helper) != helper_contract["binary_sha256"]
            or policy["helper_sha256"] != helper_contract["binary_sha256"]):
        raise ValueError("graph helper identity drift")
    runtime = read_json(args.runtime_contract)
    for name, path in runtime["runtime_paths"].items():
        if sha256_file(path) != runtime["runtime_sha256"][name]:
            raise ValueError("runtime changed: " + name)

    output.mkdir(parents=True)
    (output / "calls.jsonl").write_text("", encoding="utf-8")
    contract = {
        "schema": "c3_yolo224_graph_screen_v1",
        "status": "FROZEN_BEFORE_FIRST_YOLO224_GRAPH_LATENCY",
        "input_resolution": 224,
        "variants": list(VARIANTS),
        "qualification_order": list(VARIANTS),
        "block_orders": {name: list(order) for name, order in BLOCK_ORDERS.items()},
        "formal_seed_sequence": list(GRAPH_SEED_SEQUENCE),
        "completion_mode": "run_only",
        "fresh_rpc_per_qualification_variant": True,
        "fresh_rpc_per_five_call_block": True,
        "clean_bitstream_before_every_rpc": True,
        "one_executor_per_rpc": True,
        "qualification_calls": 12,
        "explicit_warmup_calls": 8,
        "timing_calls": 40,
        "formal_calls": 60,
        "bitstream_reloads": 12,
        "timing_boundary": "board steady_clock around exactly one graph run",
        "timing_excludes": [
            "bitstream reload", "RPC startup", "upload", "executor creation",
            "parameter load", "set_input", "get_output", "output comparison",
        ],
        "same_boot_first_screen": True,
        "cross_boot_stability_claim": False,
        "real_image_accuracy_claim": False,
        "manifests": manifests,
    }
    write_json(output / "contract.json", contract)
    write_json(output / "pre_observation_hashes.json", {
        "contract_sha256": sha256_file(output / "contract.json")
    })

    remote_runtime = "/var/volatile/" + output.name
    board = ExperimentBoard(args, remote_runtime, args.port)
    remote = None
    initial = None
    rows = []
    references = {}
    sessions = []
    session_index = 0
    started = time.monotonic()
    try:
        initial = board.state()
        reuse_board.validate_state(initial, exclusive=True)
        environment = board.environment()
        reuse_board.RUNTIME = runtime["runtime_sha256"]
        write_json(output / "preflight.json", {"state": initial, "environment": environment})
        write_json(output / "runtime_attestation.json", board.deploy(runtime["runtime_paths"]))

        def one_session(variant, purpose, block=None):
            nonlocal remote, session_index
            state = board.state()
            reuse_board.validate_state(state, exclusive=True)
            if state["boot_id"] != initial["boot_id"]:
                raise RuntimeError("boot changed before graph session")
            reload_attestation = board.reload()
            board.port = args.port + 2 * session_index
            board.pid = None
            board.start()
            remote = connect(args.host, board.port, args.session_timeout)
            _, funcs = runtime_inventory(remote)
            retained = [load_ephemeral(remote, helper)]
            symbols = json.loads(str(remote.get_function("c3.v5.runtime_symbols")()))
            for name, observed in symbols.items():
                expected = policy["runtime_symbols"][name]
                if observed["dso"].split("/")[-1] != expected["dso"].split("/")[-1]:
                    raise RuntimeError("runtime symbol DSO drift")
            timer = remote.get_function("c3.v5.graph_run_only_timed_once")
            reset_counts = remote.get_function("c3.v5.reset_counts")
            helper_counts = remote.get_function("c3.v5.counts")
            clear = funcs["vta.runtime.profiler_clear"]
            profiler = funcs["vta.runtime.profiler_status"]
            reset_counts()
            with tempfile.TemporaryDirectory(prefix="c3_yolo224_graph_") as temporary:
                executor = load_executor(
                    remote, build_dir, variant,
                    [remote.ext_dev(0), remote.cpu(0)], temporary,
                )
                current = board.environment(require_affinity=True)
                if current["governor"] != environment["governor"]:
                    raise RuntimeError("CPU governor changed during graph screen")

                def untimed(seed, phase):
                    executor.set_input("data", random_input(seed, resolution=224))
                    clear()
                    executor.run()
                    outputs = [
                        executor.get_output(index).numpy()
                        for index in range(executor.get_num_outputs())
                    ]
                    row = {
                        "phase": phase, "variant": variant, "seed": seed,
                        "session": session_index, "block": block,
                        "latency_ms": None, "outputs": output_summary(outputs),
                        "runtime_profile_complete": json.loads(profiler()),
                        "invocation_index": len(rows),
                    }
                    return outputs, row

                def timed(seed, repeat):
                    executor.set_input("data", random_input(seed, resolution=224))
                    clear()
                    latency = float(timer(executor.module))
                    if not math.isfinite(latency) or latency <= 0:
                        raise RuntimeError("invalid YOLO-224 graph latency")
                    outputs = [
                        executor.get_output(index).numpy()
                        for index in range(executor.get_num_outputs())
                    ]
                    profile = json.loads(profiler())
                    if (int(profile.get("driver_timeout_calls", 0)) != 0
                            or int(profile.get("driver_run_calls", 0)) <= 0):
                        raise RuntimeError("invalid graph completion profile")
                    row = {
                        "phase": "timing", "variant": variant, "seed": seed,
                        "session": session_index, "block": block, "repeat": repeat,
                        "latency_ms": latency, "outputs": output_summary(outputs),
                        "runtime_profile_complete": profile,
                        "invocation_index": len(rows),
                    }
                    return outputs, row

                if purpose == "qualification":
                    for seed in SEEDS:
                        outputs, row = untimed(seed, "qualification")
                        if variant == "stock":
                            references[seed] = [
                                np.array(value, copy=True) for value in outputs
                            ]
                            np.savez_compressed(
                                output / "stock_reference_{}.npz".format(seed), *outputs
                            )
                        elif seed not in references or not exact(outputs, references[seed]):
                            raise RuntimeError(
                                "{} qualification output differs from stock".format(variant)
                            )
                        rows.append(row)
                        append_jsonl(output / "calls.jsonl", row)
                else:
                    seeds = BLOCK_SEEDS[block]
                    outputs, warmup = untimed(seeds[0], "explicit_warmup")
                    if not exact(outputs, references[seeds[0]]):
                        raise RuntimeError("{} warmup output differs from stock".format(variant))
                    rows.append(warmup)
                    append_jsonl(output / "calls.jsonl", warmup)
                    offset = 0 if block == "A" else 5
                    for local_index, seed in enumerate(seeds):
                        outputs, row = timed(seed, offset + local_index)
                        if not exact(outputs, references[seed]):
                            raise RuntimeError(
                                "{} timed output differs from stock".format(variant)
                            )
                        rows.append(row)
                        append_jsonl(output / "calls.jsonl", row)
                counts = json.loads(str(helper_counts()))
                if purpose == "timing" and counts["graph_calls"] != 5:
                    raise RuntimeError("five-call block did not execute exactly five graphs")
                sessions.append({
                    "session": session_index, "variant": variant,
                    "purpose": purpose, "block": block,
                    "helper_counts": counts, "runtime_symbols": symbols,
                    "clean_bitstream_reload": reload_attestation,
                })
                write_json(
                    output / "session_{:02d}.json".format(session_index),
                    sessions[-1],
                )
                del executor
            retained.clear()
            remote = None
            gc.collect()
            board.stop()
            board.pid = None
            session_index += 1

        for variant in VARIANTS:
            one_session(variant, "qualification")
            print("qualification {} complete".format(variant), flush=True)
        write_json(output / "stock_reference_inventory.json", {
            str(seed): output_summary(references[seed]) for seed in SEEDS
        })
        for block in ("A", "B"):
            for variant in BLOCK_ORDERS[block]:
                one_session(variant, "timing", block)
                print("graph block {} {} complete".format(block, variant), flush=True)

        if session_index != 12 or len(rows) != 60:
            raise RuntimeError("YOLO-224 graph session/call count drift")
        timing = [row for row in rows if row["phase"] == "timing"]
        warmups = [row for row in rows if row["phase"] == "explicit_warmup"]
        qualification = [row for row in rows if row["phase"] == "qualification"]
        if (len(timing), len(warmups), len(qualification)) != (40, 8, 12):
            raise RuntimeError("YOLO-224 graph phase count drift")
        stats = {}
        for variant in VARIANTS:
            values = [
                float(row["latency_ms"]) for row in timing
                if row["variant"] == variant
            ]
            blocks = {
                block: [
                    float(row["latency_ms"]) for row in timing
                    if row["variant"] == variant and row["block"] == block
                ] for block in ("A", "B")
            }
            if len(values) != 10 or any(len(value) != 5 for value in blocks.values()):
                raise RuntimeError("incomplete graph timing coverage")
            median_ms = float(statistics.median(values))
            block_medians = {
                name: float(statistics.median(value)) for name, value in blocks.items()
            }
            stats[variant] = {
                "values_ms": values,
                "median_ms": median_ms,
                "throughput_fps": 1000.0 / median_ms,
                "block_median_ms": block_medians,
            }

        final = board.state()
        reuse_board.validate_state(final, exclusive=True)
        final_environment = board.environment()
        if (final["boot_id"] != initial["boot_id"]
                or final_environment["governor"] != environment["governor"]):
            raise RuntimeError("board identity changed during graph screen")
        selector = stats["selector_v4"]["median_ms"]
        result_summary = {
            "schema": contract["schema"],
            "status": "YOLO224_SINGLE_BOOT_GRAPH_SCREEN_COMPLETE",
            "boot_id": initial["boot_id"],
            "formal_calls": len(rows),
            "qualification_calls": len(qualification),
            "warmup_calls": len(warmups),
            "timing_calls": len(timing),
            "sessions": session_index,
            "bitstream_reloads": len(sessions),
            "all_outputs_equal_to_stock": True,
            "statistics": stats,
            "selector_improvement_pct": {
                baseline: 100.0 * (1.0 - selector / stats[baseline]["median_ms"])
                for baseline in ("stock", "incumbent", "input_control")
            },
            "selector_faster_in_both_blocks_vs_input_control": all(
                stats["selector_v4"]["block_median_ms"][block]
                < stats["input_control"]["block_median_ms"][block]
                for block in ("A", "B")
            ),
            "cross_boot_stability_claim": False,
            "wall_seconds": time.monotonic() - started,
        }
        write_json(output / "sessions.json", sessions)
        write_json(output / "postflight.json", {
            "state": final, "environment": final_environment,
        })
        write_json(output / "summary.json", result_summary)
        print(json.dumps(result_summary, indent=2, sort_keys=True), flush=True)
    except BaseException as error:
        write_json(output / "invalid_session.json", {
            "status": "INVALID_NON_SPLICEABLE_YOLO224_GRAPH_SCREEN",
            "message": str(error), "type": type(error).__name__,
            "traceback": traceback.format_exc()[-16000:],
            "calls_completed": len(rows), "all_timing_invalid": True,
        })
        raise
    finally:
        remote = None
        gc.collect()
        try:
            board.stop()
        except Exception:
            pass
        try:
            if board.runtime != remote_runtime:
                raise RuntimeError("refusing unexpected graph cleanup target")
            target = shlex.quote(remote_runtime)
            board.ssh("rm -rf " + target + " && test ! -e " + target)
        except Exception:
            pass
        seal_flat(output, (Path(__file__).resolve(),))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--helper-build-dir", type=Path, required=True)
    parser.add_argument("--completion-dir", type=Path, required=True)
    parser.add_argument("--state-diagnostic-dir", type=Path, required=True)
    parser.add_argument("--runtime-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--session-timeout", type=int, default=1800)
    parser.add_argument("--known-hosts", default="/home/orange/.ssh/known_hosts")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
