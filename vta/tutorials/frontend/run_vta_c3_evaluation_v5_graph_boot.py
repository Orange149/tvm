#!/usr/bin/env python3
"""Run one clean-deployment-isolated C3 v5 YOLO-512 graph boot."""
from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
from tvm import rpc

import run_vta_c3_reuse_board as reuse_board
from c3_evaluation_v5 import (GRAPH_SEED_SEQUENCE, SEEDS, append_jsonl, read_json,
                              seal_flat, sha256_file, verify_flat, write_json)
from c3_thread_sync_board import ExperimentBoard
from run_vta_c3_yolo608_graph_boot import load_executor, mismatch_counts, output_summary, random_input
from run_vta_p7r120_yolo_rpc_board import load_ephemeral, runtime_inventory

HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_BUILD = GROUPED / "20260926_c3_yolo512_evaluation_v5_graph_build_run01"
DEFAULT_FREEZE = GROUPED / "20260926_c3_yolo512_evaluation_v5_freeze_run01"
DEFAULT_COMPLETION = GROUPED / "20260926_c3_yolo512_evaluation_v5_completion_audit_run01"
DEFAULT_STATE_DIAGNOSTIC = GROUPED / "20260926_c3_mode11_cross_program_state_run06"
VARIANTS = ("stock", "incumbent", "selector_v4")
BLOCK_ORDERS = {
    1: {"A": ("stock", "incumbent", "selector_v4"),
        "B": ("selector_v4", "incumbent", "stock")},
    2: {"A": ("incumbent", "selector_v4", "stock"),
        "B": ("stock", "selector_v4", "incumbent")},
}
BLOCK_SEEDS = {"A": GRAPH_SEED_SEQUENCE[:5], "B": GRAPH_SEED_SEQUENCE[5:]}


def connect(host, port, timeout):
    for _ in range(30):
        try:
            return rpc.connect(host, port, session_timeout=timeout)
        except Exception:
            time.sleep(0.25)
    raise RuntimeError("RPC unavailable for graph session")


def exact(outputs, references):
    return all(value == 0 for value in mismatch_counts(references, outputs))


def run(args):
    build_dir = Path(args.build_dir).resolve()
    freeze_dir = Path(args.freeze_dir).resolve()
    completion_dir = Path(args.completion_dir).resolve()
    state_diagnostic_dir = Path(args.state_diagnostic_dir).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    build_manifest = verify_flat(build_dir)
    freeze_manifest = verify_flat(freeze_dir)
    completion_manifest = verify_flat(completion_dir)
    state_diagnostic_manifest = verify_flat(state_diagnostic_dir)
    build = read_json(build_dir / "summary.json")
    freeze = read_json(freeze_dir / "contract.json")
    policy = read_json(completion_dir / "completion_policy.json")
    state_diagnostic = read_json(state_diagnostic_dir / "summary.json")
    if state_diagnostic.get("classification") != "GENERAL_CROSS_RPC_FPGA_STATE_CONTAMINATION":
        raise ValueError("cross-RPC state contamination evidence is not frozen")
    if build.get("status") != "C3_EVALUATION_V5_THREE_YOLO512_DEPLOYMENTS_BUILT":
        raise ValueError("graph build incomplete")
    if policy.get("status") != "COMPLETION_POLICY_FROZEN" or policy["mode"] not in ("run_only", "run_plus_sync"):
        raise ValueError("completion policy is not frozen")
    helper = Path(freeze["helper_binary"])
    if sha256_file(helper) != freeze["helper_sha256"] or policy["helper_sha256"] != freeze["helper_sha256"]:
        raise ValueError("helper identity drift")
    runtime = read_json(freeze["runtime_contract"])
    for name, path in runtime["runtime_paths"].items():
        if sha256_file(path) != runtime["runtime_sha256"][name]:
            raise ValueError("runtime changed: " + name)
    previous = None
    if args.boot_index == 2:
        if args.previous_boot_dir is None:
            raise ValueError("graph boot 2 requires --previous-boot-dir")
        previous_dir = Path(args.previous_boot_dir).resolve()
        verify_flat(previous_dir)
        previous = read_json(previous_dir / "summary.json")
        if previous.get("status") != "C3_EVALUATION_V5_GRAPH_BOOT1_COMPLETE":
            raise ValueError("previous graph boot is incomplete")
        if (previous["build_manifest_sha256"] != build_manifest
                or previous["completion_manifest_sha256"] != completion_manifest
                or previous["state_diagnostic_manifest_sha256"] != state_diagnostic_manifest):
            raise ValueError("graph build/completion/state-diagnostic identity changed between boots")
    elif args.previous_boot_dir is not None:
        raise ValueError("graph boot 1 cannot import a previous boot")

    output.mkdir(parents=True)
    (output / "calls.jsonl").write_text("", encoding="utf-8")
    contract = {
        "schema": "c3_evaluation_v5_graph_boot_v2_clean_deployment",
        "status": "FROZEN_BEFORE_GRAPH_TIMING",
        "boot_index": args.boot_index,
        "build_manifest_sha256": build_manifest,
        "freeze_manifest_sha256": freeze_manifest,
        "completion_manifest_sha256": completion_manifest,
        "state_diagnostic_manifest_sha256": state_diagnostic_manifest,
        "completion_mode": policy["mode"],
        "variants": list(VARIANTS),
        "qualification_order": list(VARIANTS),
        "block_orders": {key: list(value) for key, value in BLOCK_ORDERS[args.boot_index].items()},
        "formal_seed_sequence": list(GRAPH_SEED_SEQUENCE),
        "fresh_rpc_per_qualification_variant": True,
        "fresh_rpc_per_five_call_block": True,
        "qualification_calls": 9, "explicit_warmup_calls": 6, "timing_calls": 30,
        "formal_calls": 45,
        "one_executor_per_rpc": True, "bitstream_reloads_this_graph_boot": 9,
        "clean_bitstream_before_every_rpc": True,
        "bitstream_reload_excluded_from_timing": True,
        "steady_state_scope": "five timed calls in one executor after one explicit warmup",
        "timing_boundary": "board steady_clock around one graph run under frozen completion mode",
        "excludes": ["RPC startup", "upload", "executor creation", "params load", "set_input",
                     "get_output", "output comparison"],
        "wrong_output_or_identity_drift_invalidates_whole_boot": True,
    }
    write_json(output / "contract.json", contract)
    write_json(output / "pre_observation_hashes.json", {"contract_sha256": sha256_file(output / "contract.json")})
    board = ExperimentBoard(args, "/var/volatile/" + output.name, args.port)
    remote = None
    initial = None
    rows = []
    references = {}
    session_index = 0
    bitstream_reloads = []
    started = time.monotonic()
    try:
        initial = board.state()
        reuse_board.validate_state(initial, exclusive=True)
        environment = board.environment()
        if previous and initial["boot_id"] == previous["boot_id"]:
            raise RuntimeError("graph boot 2 requires a different kernel boot")
        reuse_board.RUNTIME = runtime["runtime_sha256"]
        write_json(output / "runtime_attestation.json", board.deploy(runtime["runtime_paths"]))
        write_json(output / "preflight.json", {"state": initial, "environment": environment})

        def one_session(variant, purpose, block=None):
            nonlocal remote, session_index
            state = board.state()
            reuse_board.validate_state(state, exclusive=True)
            if state["boot_id"] != initial["boot_id"]:
                raise RuntimeError("boot changed before graph session")
            reload_attestation = board.reload()
            bitstream_reloads.append({
                "session": session_index, "variant": variant, "purpose": purpose,
                "block": block, "attestation": reload_attestation,
            })
            board.port = args.port + 2 * session_index
            board.pid = None
            board.start()
            remote = connect(args.host, board.port, args.session_timeout)
            _, funcs = runtime_inventory(remote)
            retained = [load_ephemeral(remote, helper)]
            symbols = json.loads(str(remote.get_function("c3.v5.runtime_symbols")()))
            if symbols != policy["runtime_symbols"]:
                # ASLR changes addresses, so only DSO identity is invariant.
                for name in symbols:
                    if symbols[name]["dso"].split("/")[-1] != policy["runtime_symbols"][name]["dso"].split("/")[-1]:
                        raise RuntimeError("runtime symbol DSO drift")
            timer_name = ("c3.v5.graph_run_only_timed_once" if policy["mode"] == "run_only"
                          else "c3.v5.graph_run_sync_timed_once")
            timer = remote.get_function(timer_name)
            explicit_sync = remote.get_function("c3.v5.empty_sync")
            reset_counts = remote.get_function("c3.v5.reset_counts")
            helper_counts = remote.get_function("c3.v5.counts")
            clear = funcs["vta.runtime.profiler_clear"]
            profiler = funcs["vta.runtime.profiler_status"]
            reset_counts()
            with tempfile.TemporaryDirectory(prefix="c3_v5_graph_session_") as temporary:
                executor = load_executor(remote, build_dir, variant,
                                         [remote.ext_dev(0), remote.cpu(0)], temporary)
                current = board.environment(require_affinity=True)
                if current["governor"] != environment["governor"]:
                    raise RuntimeError("governor changed")

                def untimed(seed, phase):
                    executor.set_input("data", random_input(seed, resolution=512))
                    clear()
                    executor.run()
                    if policy["mode"] == "run_plus_sync":
                        explicit_sync()
                    outputs = [executor.get_output(i).numpy() for i in range(executor.get_num_outputs())]
                    profile = json.loads(profiler())
                    row = {"phase": phase, "variant": variant, "seed": seed,
                           "session": session_index, "block": block,
                           "latency_ms": None, "outputs": output_summary(outputs),
                           "runtime_profile_complete": profile,
                           "invocation_index": len(rows)}
                    return outputs, row

                def timed(seed, repeat):
                    executor.set_input("data", random_input(seed, resolution=512))
                    clear()
                    latency = float(timer(executor.module))
                    if not math.isfinite(latency) or latency <= 0:
                        raise RuntimeError("invalid graph latency")
                    outputs = [executor.get_output(i).numpy() for i in range(executor.get_num_outputs())]
                    profile = json.loads(profiler())
                    if int(profile.get("driver_timeout_calls", 0)) != 0 or int(profile.get("driver_run_calls", 0)) <= 0:
                        raise RuntimeError("invalid graph completion profile")
                    row = {"phase": "timing", "variant": variant, "seed": seed,
                           "session": session_index, "block": block, "repeat": repeat,
                           "latency_ms": latency, "outputs": output_summary(outputs),
                           "runtime_profile_complete": profile,
                           "invocation_index": len(rows)}
                    return outputs, row

                if purpose == "qualification":
                    for seed in SEEDS:
                        outputs, row = untimed(seed, "qualification")
                        if variant == "stock":
                            references[seed] = [np.array(value, copy=True) for value in outputs]
                            np.savez_compressed(output / ("stock_reference_{}.npz".format(seed)), *outputs)
                        elif seed not in references or not exact(outputs, references[seed]):
                            raise RuntimeError("qualification output differs from stock")
                        rows.append(row); append_jsonl(output / "calls.jsonl", row)
                else:
                    seeds = BLOCK_SEEDS[block]
                    _, warm = untimed(seeds[0], "explicit_warmup")
                    if not exact([np.array(value) for value in [executor.get_output(i).numpy()
                                 for i in range(executor.get_num_outputs())]], references[seeds[0]]):
                        raise RuntimeError("warmup output differs from stock")
                    rows.append(warm); append_jsonl(output / "calls.jsonl", warm)
                    offset = 0 if block == "A" else 5
                    for local_index, seed in enumerate(seeds):
                        outputs, row = timed(seed, offset + local_index)
                        if not exact(outputs, references[seed]):
                            raise RuntimeError("timed graph output differs from stock")
                        rows.append(row); append_jsonl(output / "calls.jsonl", row)
                counts = json.loads(str(helper_counts()))
                if purpose == "timing" and counts["graph_calls"] != 5:
                    raise RuntimeError("five-call block did not execute exactly five timed graphs")
                write_json(output / "session_{:02d}.json".format(session_index), {
                    "variant": variant, "purpose": purpose, "block": block,
                    "helper_counts": counts, "runtime_symbols": symbols,
                    "clean_bitstream_reload": reload_attestation,
                })
                del executor
            retained.clear()
            remote = None
            gc.collect()
            board.stop()
            board.pid = None
            session_index += 1

        # References are generated first and sealed before incumbent/selector qualification.
        for variant in VARIANTS:
            one_session(variant, "qualification")
        write_json(output / "stock_reference_inventory.json", {
            str(seed): output_summary(references[seed]) for seed in SEEDS
        })
        for block in ("A", "B"):
            for variant in BLOCK_ORDERS[args.boot_index][block]:
                one_session(variant, "timing", block)
                print("graph boot {} block {} {} complete".format(args.boot_index, block, variant), flush=True)

        write_json(output / "bitstream_reloads.json", {
            "count": len(bitstream_reloads),
            "before_every_rpc": True,
            "reloads": bitstream_reloads,
        })
        if len(bitstream_reloads) != 9 or session_index != 9:
            raise RuntimeError("clean deployment isolation count drift")
        if len(rows) != 45:
            raise RuntimeError("graph formal call count drift")
        timing = [row for row in rows if row["phase"] == "timing"]
        warmups = [row for row in rows if row["phase"] == "explicit_warmup"]
        qualification = [row for row in rows if row["phase"] == "qualification"]
        if (len(timing), len(warmups), len(qualification)) != (30, 6, 9):
            raise RuntimeError("graph phase call counts drift")
        stats = {}
        for variant in VARIANTS:
            values = [float(row["latency_ms"]) for row in timing if row["variant"] == variant]
            blocks = {block: [float(row["latency_ms"]) for row in timing
                              if row["variant"] == variant and row["block"] == block]
                      for block in ("A", "B")}
            if len(values) != 10 or any(len(value) != 5 for value in blocks.values()):
                raise RuntimeError("incomplete graph timing coverage")
            median_ms = float(statistics.median(values))
            block_median_ms = {k: float(statistics.median(v)) for k, v in blocks.items()}
            stats[variant] = {
                "values_ms": values,
                "median_ms": median_ms,
                "throughput_fps": 1000.0 / median_ms,
                "block_median_ms": block_median_ms,
                "block_throughput_fps": {k: 1000.0 / v for k, v in block_median_ms.items()},
            }
        running = board.state()
        reuse_board.validate_state(running, exclusive=True)
        final_environment = board.environment()
        if running["boot_id"] != initial["boot_id"] or final_environment["governor"] != environment["governor"]:
            raise RuntimeError("board identity changed during graph boot")
        summary = {
            "schema": contract["schema"],
            "status": "C3_EVALUATION_V5_GRAPH_BOOT{}_COMPLETE".format(args.boot_index),
            "boot_id": initial["boot_id"], "build_manifest_sha256": build_manifest,
            "completion_manifest_sha256": completion_manifest,
            "state_diagnostic_manifest_sha256": state_diagnostic_manifest,
            "completion_mode": policy["mode"], "formal_calls": len(rows),
            "qualification_calls": len(qualification), "warmup_calls": len(warmups),
            "timing_calls": len(timing), "sessions": session_index,
            "bitstream_reloads": len(bitstream_reloads),
            "clean_bitstream_before_every_rpc": True,
            "all_outputs_equal_to_stock": True, "statistics": stats,
            "selector_vs_incumbent_latency_improvement_pct": 100.0 * (
                1.0 - stats["selector_v4"]["median_ms"] / stats["incumbent"]["median_ms"]),
            "selector_vs_incumbent_throughput_improvement_pct": 100.0 * (
                stats["selector_v4"]["throughput_fps"]
                / stats["incumbent"]["throughput_fps"] - 1.0),
            "wall_seconds": time.monotonic() - started,
        }
        write_json(output / "postflight.json", {"state": running, "environment": final_environment})
        write_json(output / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    except BaseException as error:
        write_json(output / "invalid_session.json", {
            "status": "INVALID_NON_SPLICEABLE_GRAPH_BOOT", "message": str(error),
            "type": type(error).__name__, "traceback": traceback.format_exc()[-16000:],
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
        seal_flat(output, (Path(__file__).resolve(),))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--boot-index", type=int, choices=(1, 2), required=True)
    p.add_argument("--previous-boot-dir", type=Path)
    p.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD)
    p.add_argument("--freeze-dir", type=Path, default=DEFAULT_FREEZE)
    p.add_argument("--completion-dir", type=Path, default=DEFAULT_COMPLETION)
    p.add_argument("--state-diagnostic-dir", type=Path, default=DEFAULT_STATE_DIAGNOSTIC)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--host", default="192.168.1.247")
    p.add_argument("--port", type=int, default=9090)
    p.add_argument("--session-timeout", type=int, default=1800)
    p.add_argument("--known-hosts", default="/home/orange/.ssh/known_hosts")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
