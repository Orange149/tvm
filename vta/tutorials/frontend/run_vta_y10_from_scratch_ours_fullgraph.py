#!/usr/bin/env python3
"""Reexecute the complete Y10 proposed-method path and charge T0-to-T1 wall time.

This is deliberately a cost reexecution, not a new holdout: candidate generation
must reproduce the already frozen P7R428 identities byte-for-byte, and target
performance artifacts are never passed to any subprocess.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_vta_p7r132_y00_search_confirmation import CleanStartBoard
from run_vta_resnet50_fused_program_pareto_board import assert_board_state


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_phase(name, command, output, phases, t0):
    started = time.perf_counter()
    log = output / (name + ".log")
    with log.open("x", encoding="utf-8") as stream:
        stream.write("COMMAND " + " ".join(str(value) for value in command) + "\n")
        stream.flush()
        result = subprocess.run(
            [str(value) for value in command],
            cwd=REPO,
            env=os.environ.copy(),
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
    row = {
        "phase": name,
        "seconds": time.perf_counter() - started,
        "finished_t0_seconds": time.perf_counter() - t0,
        "returncode": result.returncode,
        "log": log.name,
    }
    phases.append(row)
    write(output / "phase_timeline.json", phases)
    if result.returncode:
        raise RuntimeError("{} failed with return code {}; see {}".format(
            name, result.returncode, log
        ))
    return row


def clean_start(args):
    board = CleanStartBoard(args.host, args.ssh_known_hosts)
    before = board.state()
    assert_board_state(before)
    rpc = board.rpc_state()
    stopped = None
    if rpc.get("PID") and rpc.get("CWD"):
        board.stop_rpc(rpc)
        stopped = rpc
    bitstream = board.reload_frozen_bitstream()
    fresh = board.start_rpc(args.default_runtime, "p7r453_ours_t0.log")
    after = board.state()
    assert_board_state(after)
    return {"before": before, "stopped_rpc": stopped, "bitstream": bitstream,
            "fresh_rpc": fresh, "after": after}


def sum_board_resources(candidate_results):
    totals = {
        "logical_load_bytes": 0,
        "logical_store_bytes": 0,
        "logical_dma_calls": 0,
        "fpga_kernel_invocations": 0,
    }
    row_count = 0
    for result in candidate_results:
        for group in ("correctness", "timings"):
            for row in result.get(group, []):
                profile = row.get("runtime_profile_complete")
                if not profile:
                    continue
                totals["logical_load_bytes"] += int(profile.get("load_buffer_2d_bytes", 0))
                totals["logical_store_bytes"] += int(profile.get("store_buffer_2d_bytes", 0))
                totals["logical_dma_calls"] += int(profile.get("load_buffer_2d_calls", 0))
                totals["logical_dma_calls"] += int(profile.get("store_buffer_2d_calls", 0))
                totals["fpga_kernel_invocations"] += int(profile.get("driver_run_calls", 0))
                row_count += 1
    totals["profiled_invocation_rows"] = row_count
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    c3 = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune"
    grouped = c3 / "07_grouped_holdout"
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--session-timeout", type=int, default=600)
    parser.add_argument("--ssh-known-hosts", required=True)
    parser.add_argument("--default-runtime", default="/var/volatile/vta_c3_ram/runtime")
    parser.add_argument("--budget-seconds", type=float, default=600.0)
    parser.add_argument(
        "--target-contract", type=Path,
        default=grouped / "20260913_p7r428_y10_yolov3_tiny_320_conv18_candidate_contract_run01",
    )
    parser.add_argument(
        "--policy-contract", type=Path,
        default=grouped / "20260913_p7r429_y10_invalid_dominator_peeling_policy_run01",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    preflight = clean_start(args)
    write(output / "t0_clean_start.json", preflight)
    t0 = time.perf_counter()
    phases = []
    python = sys.executable
    replay = output / "candidate_generation_cost_replay"
    local = output / "local_qualification"
    front = output / "front"
    online = output / "online"

    try:
        run_phase("candidate_generation", [
            python, HERE / "prepare_vta_model_conv_adaptive_holdout.py",
            "--workload-id", "Y10", "--model", "yolov3_tiny_320",
            "--layer", "conv18",
            "--model-source", HERE / "yolov3-tiny-320.cfg",
            "--darknet-weights", "/tmp/tvm_test_data/darknet/yolov3-tiny.weights",
            "--darknet-lib", "/home/orange/.tvm_test_data/darknet/libdarknet2.0.so",
            "--layer-derivation",
            "Darknet cfg/weights import followed by Relay InferType; conv18 parameter identity",
            "--ci", "256", "--co", "128", "--height", "10", "--width", "10",
            "--kernel", "1", "--stride", "1", "--padding", "0", "--families", "8",
            "--selection-seed", "c3-p7r428-y10-yolov3-tiny-320-conv18-v1",
            "--cost-replay-of", args.target_contract.resolve(), "--output-dir", replay,
        ], output, phases, t0)
        reference_candidates = args.target_contract.resolve() / "candidates.jsonl"
        if sha256(replay / "candidates.jsonl") != sha256(reference_candidates):
            raise RuntimeError("candidate generation replay differs from frozen target")

        run_phase("static_and_fsim_qualification", [
            python, HERE / "run_vta_p7r127_y00_v2_local_pool.py",
            "--candidates", reference_candidates, "--workload-id", "Y10",
            "--output-dir", local, "--fsim-hw-path", "/tmp/vta-hw-fsim",
            "--fsim-timeout", "300",
        ], output, phases, t0)
        run_phase("proxy_front", [
            python, HERE / "freeze_vta_invalid_dominator_peeling_front.py",
            "--target-contract", args.target_contract.resolve(),
            "--policy-contract", args.policy_contract.resolve(),
            "--local-qualification", local, "--output-dir", front,
        ], output, phases, t0)
        run_phase("online_fullgraph_search", [
            python, HERE / "run_vta_invalid_dominator_peeling_online.py",
            "--target-contract", args.target_contract.resolve(),
            "--policy-contract", args.policy_contract.resolve(),
            "--local-qualification", local, "--front-contract", front,
            "--output-dir", online, "--budget-seconds", str(args.budget_seconds),
            "--strategy", "peeling", "--host", args.host, "--port", str(args.port),
            "--session-timeout", str(args.session_timeout),
            "--ssh-known-hosts", str(Path(args.ssh_known_hosts).resolve()),
            "--default-runtime", args.default_runtime,
        ], output, phases, t0)

        local_summary = read(local / "summary.json")
        online_summary = read(online / "summary.json")
        passed = [row for row in online_summary["candidate_results"] if row["status"] == "passed"]
        if not passed:
            raise RuntimeError("no FPGA-correct candidate reached T1")
        selected = min(
            passed,
            key=lambda row: row["median_latency_ms"][row["candidate_id"]],
        )
        selected_id = selected["candidate_id"]
        candidate_ms = selected["median_latency_ms"][selected_id]
        stock_ms = selected["median_latency_ms"]["stock_reference"]
        before_online = next(
            row["finished_t0_seconds"] - row["seconds"]
            for row in phases if row["phase"] == "online_fullgraph_search"
        )
        first_correct_seconds = before_online + min(
            row["outer_elapsed_seconds"] for row in passed
        )
        final_best_plus_2_seconds = before_online + min(
            row["outer_elapsed_seconds"] for row in passed
            if row["median_latency_ms"][row["candidate_id"]] <= 1.02 * candidate_ms
        )
        t1_seconds = time.perf_counter() - t0
        summary = {
            "schema": "c3_y10_from_scratch_ours_fullgraph_v1",
            "status": "completed_t0_to_t1",
            "workload_id": "Y10",
            "target_layer": "YOLOv3-tiny-320 conv18",
            "method": "capacity/hash family generation + exact lowering + three-seed FSim + invalid-dominator peeling + full-graph FPGA gate",
            "t0_definition": "after clean bitstream reload and fresh healthy default RPC",
            "t1_definition": "after selected candidate passed three-input/all-output correctness and seven paired full-graph rounds",
            "t0_to_t1_seconds": t1_seconds,
            "phase_timeline": phases,
            "config_space_original_tiles": 1280,
            "generated_mode_identities": 32,
            "qualified_mode_identities": local_summary["gross_identities"],
            "static_ok": local_summary["static_ok"],
            "fsim_passed": local_summary["fsim_passed"],
            "fpga_dispatch_count": online_summary["dispatch_count"],
            "fpga_correctness_invocations": online_summary["correctness_invocations"],
            "fpga_timing_invocations": online_summary["timing_invocations"],
            "first_fpga_correct_fullgraph_t0_seconds": first_correct_seconds,
            "first_final_best_plus_2_percent_t0_seconds": final_best_plus_2_seconds,
            "selected_candidate_id": selected_id,
            "selected_public_mode": selected["public_mode"],
            "selected_family_id": selected["family_id"],
            "selected_fullgraph_median_ms": candidate_ms,
            "paired_stock_fullgraph_median_ms": stock_ms,
            "selected_over_stock_ratio": selected["median_paired_latency_ratio"],
            "selected_speedup_percent": (1.0 - selected["median_paired_latency_ratio"]) * 100.0,
            "selected_paired_wins": selected["candidate_paired_wins"],
            "selected_correctness_inputs": len(selected["correctness"]) // 2,
            "selected_all_outputs_equal": all(
                row.get("paired_equal") for row in selected["correctness"]
            ),
            "online_search_board_resources_including_paired_stock": sum_board_resources(
                online_summary["candidate_results"]
            ),
            "candidate_generation_replay_proof": read(replay / "contract.json")["cost_replay_proof"],
            "old_target_labels_used_as_search_input": False,
            "claim_boundary": (
                "One deterministic retrospective cost reexecution on exposed Y10. Identity-exact "
                "candidate generation, qualification and board execution were rerun; this is not "
                "a new prospective holdout or a multi-workload generalization result."
            ),
        }
        write(output / "summary.json", summary)
        (output / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
        artifacts = {
            str(path.relative_to(output)): sha256(path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "artifact_hashes.json"
        }
        write(output / "artifact_hashes.json", {
            "artifacts": artifacts,
            "source_sha256": sha256(Path(__file__).resolve()),
        })
        print(json.dumps(summary, indent=2, sort_keys=True))
    except Exception as error:
        write(output / "failure.json", {
            "status": "failed_closed", "type": type(error).__name__,
            "message": str(error), "t0_elapsed_seconds": time.perf_counter() - t0,
            "phases": phases,
        })
        raise


if __name__ == "__main__":
    main()
