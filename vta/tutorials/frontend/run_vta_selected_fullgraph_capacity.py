#!/usr/bin/env python3
"""Validate exact selected full-graph command capacity and fail-closed rejection.

The selected graph is fixed by a completed online run plus a completed-pool
analysis.  A diagnostic pass first observes the maximum instruction and UOP
bytes of every real submission.  The script then page-aligns those peaks,
replays three deterministic inputs at the reduced capacity, and removes one
page from the limiting queue to require rejection before the violating device
submission.  No latency is measured.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import traceback

import tvm
from tvm import rpc

from analyze_vta_fused_program_service_proxy_norefit import verify_artifacts_compatible
from run_vta_p7r132_y00_search_confirmation import (
    CleanStartBoard,
    EXPECTED_DEFAULT_RUNTIME_HASHES,
)
from run_vta_resnet50_fused_program_pareto_board import (
    load_executor,
    model_input_for_seed,
)
from run_vta_resnet50_relay_residency_dispatch_pair_board import sha256, write_json
from run_vta_yolov3_tiny_relay_residency_dispatch_pair_board import output_summary


PAGE_BYTES = 4096
SEEDS = (0, 20250901, 20260910)
QUEUE_SCHEMA = "vta_queue_capacity_status_v1"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical_sha256(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def align_up(value, alignment=PAGE_BYTES):
    return (int(value) + alignment - 1) // alignment * alignment


def finish(output):
    write_json(output / "artifact_hashes.json", {"artifacts": {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }})


def environment(capacity, manifest_id):
    result = {
        "VTA_QUEUE_DIAGNOSTICS": 1,
        "VTA_REPLAY_POLICY": "disabled",
        "VTA_COMMAND_MANIFEST_ID": manifest_id,
    }
    if capacity is not None:
        result.update({
            "VTA_INSN_BUFFER_BYTES": int(capacity["insn_bytes"]),
            "VTA_UOP_BUFFER_BYTES": int(capacity["uop_bytes"]),
        })
    return result


def verify_selected(selected_run, analysis_run, candidate_id=None):
    verified = {
        "selected_run_artifacts": len(verify_artifacts_compatible(selected_run)),
        "analysis_run_artifacts": len(verify_artifacts_compatible(analysis_run)),
    }
    selected = read_json(selected_run / "summary.json")
    contract = read_json(selected_run / "contract.json")
    analysis = read_json(analysis_run / "analysis.json")
    oracle = analysis["oracle"]
    candidate_id = candidate_id or oracle["candidate_id"]
    if candidate_id != oracle["candidate_id"]:
        raise ValueError("requested candidate is not the completed-pool oracle")
    if candidate_id not in selected["passed_candidate_ids"]:
        raise ValueError("selected identity did not pass the bound online run")
    program = next(
        (row for row in selected["built_programs"] if row["candidate_id"] == candidate_id),
        None,
    )
    if program is None:
        raise ValueError("oracle program was not built in the bound online run")
    local = selected_run / program["relative_dir"]
    for name, key in (
        ("graph.json", "graph_sha256"),
        ("graphlib.so", "graphlib_sha256"),
        ("params.bin", "params_sha256"),
    ):
        if sha256(local / name) != program[key]:
            raise RuntimeError("selected graph artifact hash mismatch: " + name)
    input_spec = contract.get("input_specification")
    if not input_spec or not input_spec.get("all_outputs"):
        raise ValueError("this experiment requires a bound all-output model input specification")
    if analysis.get("status") not in {
        "complete_fpga_correct_pool_analyzed",
        "prospective_online_peeling_exact_oracle_complete",
    }:
        raise ValueError("completed-pool analysis is not final")
    return verified, selected, contract, analysis, program, local, input_spec


def run_graph(args, local, input_spec, expected=None, layout_padding_bytes=0):
    remote = rpc.connect(args.host, args.port, session_timeout=args.session_timeout)
    device = remote.ext_dev(0)
    padding = None
    if layout_padding_bytes:
        padding = tvm.nd.empty((int(layout_padding_bytes),), "uint8", device)
    rows = []
    with tempfile.TemporaryDirectory(prefix="c3_selected_capacity_") as temporary:
        executor = load_executor(
            remote, local, "selected_capacity_graph.so", [device, remote.cpu(0)], temporary
        )
        for seed in SEEDS:
            executor.set_input("data", model_input_for_seed(seed, input_spec))
            remote.get_function("vta.runtime.profiler_clear")()
            executor.run()
            outputs = output_summary([
                executor.get_output(index).numpy()
                for index in range(executor.get_num_outputs())
            ])
            profile = json.loads(remote.get_function("vta.runtime.profiler_status")())
            row = {
                "seed": seed,
                "outputs": outputs,
                "all_outputs_nonzero": all(item["nonzero"] > 0 for item in outputs),
                "runtime_profile": profile,
            }
            if expected is not None:
                row["exact_baseline_match"] = outputs == expected[str(seed)]
            rows.append(row)
        queue = json.loads(remote.get_function("vta.runtime.queue_capacity_status")())
    del executor, padding, device, remote
    gc.collect()
    time.sleep(0.25)
    if queue.get("schema") != QUEUE_SCHEMA:
        raise RuntimeError("unexpected queue status schema")
    return {"rows": rows, "queue_status": queue}


def run_negative(args, local, input_spec, capacity, manifest_id, short_queue):
    remote = rpc.connect(args.host, args.port, session_timeout=args.session_timeout)
    device = remote.ext_dev(0)
    # Reinsert the removed page immediately after queue allocation so later
    # graph/tensor allocation keeps the positive phase's physical layout.
    padding = tvm.nd.empty((PAGE_BYTES,), "uint8", device)
    error = None
    with tempfile.TemporaryDirectory(prefix="c3_selected_capacity_negative_") as temporary:
        executor = load_executor(
            remote, local, "selected_capacity_negative.so", [device, remote.cpu(0)], temporary
        )
        executor.set_input("data", model_input_for_seed(SEEDS[0], input_spec))
        remote.get_function("vta.runtime.profiler_clear")()
        try:
            executor.run()
        except Exception as caught:  # The deliberate rejection crosses RPC.
            error = {
                "exception_type": type(caught).__name__,
                "message": str(caught),
                "traceback": traceback.format_exc()[-10000:],
            }
        profile = json.loads(remote.get_function("vta.runtime.profiler_status")())
        queue = json.loads(remote.get_function("vta.runtime.queue_capacity_status")())
    del executor, padding, device, remote
    gc.collect()
    time.sleep(0.25)
    if error is None or "queue backing capacity exceeded before submission" not in error["message"]:
        raise RuntimeError("one-page-short capacity was not rejected")
    if queue.get("command_manifest_id") != manifest_id:
        raise RuntimeError("negative manifest attestation mismatch")
    if int(profile.get("driver_run_calls", -1)) != int(queue.get("submissions", -2)):
        raise RuntimeError("violating command reached the device or counters disagree")
    if int(queue.get(short_queue + "_capacity_bytes", -1)) != int(capacity[short_queue + "_bytes"]):
        raise RuntimeError("negative queue capacity mismatch")
    return {
        "capacity": capacity,
        "short_queue": short_queue,
        "layout_padding_bytes": PAGE_BYTES,
        "error": error,
        "runtime_profile": profile,
        "queue_status": queue,
        "rejected_before_violating_submission": True,
    }


def run(args):
    selected_run = Path(args.selected_run).resolve()
    analysis_run = Path(args.analysis_run).resolve()
    diagnostic_build = Path(args.diagnostic_build).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output " + str(output))
    output.mkdir(parents=True)
    caught = None
    active = None
    board = CleanStartBoard(args.host, args.ssh_known_hosts)
    original_rpc = None
    before = None
    try:
        verified, selected, source_contract, analysis, program, local, input_spec = (
            verify_selected(selected_run, analysis_run, args.candidate_id)
        )
        diagnostic_manifest = read_json(diagnostic_build / "build_manifest.json")
        driver_hash = diagnostic_manifest["artifacts"]["libvta.so"]
        if sha256(diagnostic_build / "libvta.so") != driver_hash:
            raise RuntimeError("local diagnostic driver hash mismatch")
        before = board.state()
        if before.get("FPGA") != "operating" or before.get("UDMABUF") != "201326592":
            raise RuntimeError("FPGA/u-dma-buf preflight failed")
        if before.get("STORAGE_ERRORS"):
            raise RuntimeError("storage errors present before experiment")
        if board.runtime_hashes(args.default_runtime) != EXPECTED_DEFAULT_RUNTIME_HASHES:
            raise RuntimeError("default runtime hash mismatch")
        remote_driver_hash = board.run(
            "sha256sum " + args.diagnostic_runtime.rstrip("/") + "/libvta.so"
        ).stdout.split()[0]
        if remote_driver_hash != driver_hash:
            raise RuntimeError("board diagnostic driver hash mismatch")
        original_rpc = board.rpc_state()
        if original_rpc.get("CWD") != args.default_runtime:
            raise RuntimeError("default RPC is not active")

        identity = {
            "candidate_id": program["candidate_id"],
            "public_mode": program["public_mode"],
            "graph_sha256": program["graph_sha256"],
            "graphlib_sha256": program["graphlib_sha256"],
            "params_sha256": program["params_sha256"],
            "input_specification": input_spec,
            "selected_run_manifest_sha256": sha256(selected_run / "artifact_hashes.json"),
            "analysis_run_manifest_sha256": sha256(analysis_run / "artifact_hashes.json"),
        }
        manifest_id = canonical_sha256(identity)
        initial_contract = {
            "schema": "c3_selected_fullgraph_peak_contract_v1",
            "status": "frozen_before_diagnostic_peak_execution",
            "boot_id": before["BOOT"],
            "identity": identity,
            "completed_pool_oracle": analysis["oracle"],
            "verified_artifact_counts": verified,
            "diagnostic_driver_sha256": driver_hash,
            "manifest_id": manifest_id,
            "seeds": list(SEEDS),
            "latency_measurement": "not_collected",
            "claim_boundary": (
                "Exact selected static-shape graph; logical command queue/backing, not physical "
                "AXI traffic, dynamic-shape coverage, or model accuracy."
            ),
        }
        write_json(output / "peak_contract.json", initial_contract)
        write_json(output / "pre_peak_hashes.json", {
            "artifacts": {"peak_contract.json": sha256(output / "peak_contract.json")}
        })

        board.stop_rpc(original_rpc)
        board.reload_frozen_bitstream()
        active = board.start_rpc(
            args.diagnostic_runtime, "c3_selected_capacity_peak.log",
            environment(None, manifest_id),
        )
        peak = run_graph(args, local, input_spec)
        queue = peak["queue_status"]
        expected_submissions = sum(
            int(row["runtime_profile"]["driver_run_calls"]) for row in peak["rows"]
        )
        if (not all(row["all_outputs_nonzero"] for row in peak["rows"])
                or int(queue["submissions"]) != expected_submissions
                or queue.get("command_manifest_id") != manifest_id):
            raise RuntimeError("diagnostic peak phase failed its correctness/profile contract")
        write_json(output / "peak_raw.json", peak)
        board.stop_rpc(active)
        active = None

        positive_capacity = {
            "insn_bytes": align_up(queue["insn_peak_bytes"]),
            "uop_bytes": align_up(queue["uop_peak_bytes"]),
        }
        ratios = {
            name: int(queue[name + "_peak_bytes"]) / positive_capacity[name + "_bytes"]
            for name in ("insn", "uop")
        }
        short_queue = max(ratios, key=ratios.get)
        negative_capacity = dict(positive_capacity)
        negative_capacity[short_queue + "_bytes"] -= PAGE_BYTES
        if negative_capacity[short_queue + "_bytes"] < 0:
            raise RuntimeError("cannot construct nonnegative one-page-short control")
        measured_peak = int(queue[short_queue + "_peak_bytes"])
        if not (
            negative_capacity[short_queue + "_bytes"] < measured_peak
            <= positive_capacity[short_queue + "_bytes"]
        ):
            raise RuntimeError("one-page-short capacity does not straddle measured peak")
        capacity_contract = {
            "schema": "c3_selected_fullgraph_capacity_contract_v1",
            "status": "frozen_after_peak_before_reduced_capacity_execution",
            "peak_contract_sha256": sha256(output / "peak_contract.json"),
            "peak_raw_sha256": sha256(output / "peak_raw.json"),
            "manifest_id": manifest_id,
            "measured_peak": {
                "insn_bytes": int(queue["insn_peak_bytes"]),
                "uop_bytes": int(queue["uop_peak_bytes"]),
                "submissions_for_three_inputs": int(queue["submissions"]),
            },
            "derivation": "4 KiB page-align each observed maximum per-submission queue use",
            "positive_capacity": positive_capacity,
            "negative_capacity": negative_capacity,
            "negative_short_queue": short_queue,
            "negative_layout_padding_bytes": PAGE_BYTES,
            "failure_policy": "always restore frozen default RPC; reject before violating submit",
            "latency_measurement": "not_collected",
        }
        write_json(output / "capacity_contract.json", capacity_contract)
        write_json(output / "pre_capacity_hashes.json", {"artifacts": {
            "peak_contract.json": sha256(output / "peak_contract.json"),
            "peak_raw.json": sha256(output / "peak_raw.json"),
            "capacity_contract.json": sha256(output / "capacity_contract.json"),
        }})

        baseline = {str(row["seed"]): row["outputs"] for row in peak["rows"]}
        board.reload_frozen_bitstream()
        active = board.start_rpc(
            args.diagnostic_runtime, "c3_selected_capacity_positive.log",
            environment(positive_capacity, manifest_id),
        )
        positive = run_graph(args, local, input_spec, expected=baseline)
        positive_queue = positive["queue_status"]
        if (not all(row.get("exact_baseline_match") for row in positive["rows"])
                or int(positive_queue["insn_capacity_bytes"]) != positive_capacity["insn_bytes"]
                or int(positive_queue["uop_capacity_bytes"]) != positive_capacity["uop_bytes"]
                or int(positive_queue["insn_peak_bytes"]) != int(queue["insn_peak_bytes"])
                or int(positive_queue["uop_peak_bytes"]) != int(queue["uop_peak_bytes"])
                or int(positive_queue["submissions"]) != expected_submissions):
            raise RuntimeError("positive reduced-capacity replay violated its contract")
        write_json(output / "positive_raw.json", positive)
        board.stop_rpc(active)
        active = None

        board.reload_frozen_bitstream()
        active = board.start_rpc(
            args.diagnostic_runtime, "c3_selected_capacity_negative.log",
            environment(negative_capacity, manifest_id),
        )
        negative = run_negative(
            args, local, input_spec, negative_capacity, manifest_id, short_queue
        )
        write_json(output / "negative_raw.json", negative)
        board.stop_rpc(active)
        active = None

        board.reload_frozen_bitstream()
        board.start_rpc(args.default_runtime, "c3_selected_capacity_default_restored.log")
        health = run_graph_under_default(args, local, input_spec, baseline)
        write_json(output / "default_restore_health.json", health)
        after = board.state()
        if after.get("BOOT") != before.get("BOOT") or after.get("STORAGE_ERRORS"):
            raise RuntimeError("board/storage state changed during experiment")
        default_bytes = 2 * 33554432
        reduced_bytes = sum(positive_capacity.values())
        summary = {
            "schema": "c3_selected_fullgraph_capacity_summary_v1",
            "status": "peak_positive_and_one_page_short_fail_closed_passed",
            "boot_id": before["BOOT"],
            "candidate_id": program["candidate_id"],
            "public_mode": program["public_mode"],
            "measured_peak": capacity_contract["measured_peak"],
            "positive_capacity": positive_capacity,
            "positive_correctness": "3/3 deterministic inputs, all graph outputs exact",
            "negative_capacity": negative_capacity,
            "negative_short_queue": short_queue,
            "negative_prior_device_submissions": int(negative["queue_status"]["submissions"]),
            "negative_rejected_before_violating_submission": True,
            "default_command_backing_bytes": default_bytes,
            "selected_command_backing_bytes": reduced_bytes,
            "command_backing_reduction_percent": (1.0 - reduced_bytes / default_bytes) * 100.0,
            "default_rpc_restored": after.get("RPC_CWD") == args.default_runtime,
            "post_restore_exact_health": True,
            "board_after": after,
            "latency_measurement": "not_collected",
            "claim_boundary": initial_contract["claim_boundary"],
        }
        write_json(output / "summary.json", summary)
        (output / "STATUS.md").write_text(
            "# Selected full-graph command capacity\n\n"
            "- Status: `passed`\n"
            "- Candidate: `{}` (`{}`)\n"
            "- Observed peak: instruction {} B, UOP {} B\n"
            "- Page-aligned backing: instruction {} B, UOP {} B\n"
            "- Positive replay: three inputs and all outputs match the peak run exactly\n"
            "- Negative replay: one page removed from `{}` and rejected before the violating "
            "device submission\n"
            "- Default RPC restored; one exact post-restore health inference passed\n".format(
                program["candidate_id"], program["public_mode"],
                queue["insn_peak_bytes"], queue["uop_peak_bytes"],
                positive_capacity["insn_bytes"], positive_capacity["uop_bytes"], short_queue,
            ),
            encoding="utf-8",
        )
    except Exception as error:
        caught = error
        write_json(output / "failure.json", {
            "status": "failed_closed",
            "exception_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc()[-16000:],
        })
    finally:
        try:
            current = board.rpc_state()
            if current.get("CWD") != args.default_runtime:
                if current.get("PID") and current.get("CWD"):
                    board.stop_rpc(current)
                board.reload_frozen_bitstream()
                board.start_rpc(args.default_runtime, "c3_selected_capacity_emergency_restore.log")
        except Exception as restore_error:
            write_json(output / "restoration_failure.json", {
                "exception_type": type(restore_error).__name__,
                "message": str(restore_error),
            })
            if caught is None:
                caught = restore_error
        (output / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
        finish(output)
    if caught is not None:
        raise caught
    print(json.dumps(read_json(output / "summary.json"), indent=2, sort_keys=True))


def run_graph_under_default(args, local, input_spec, baseline):
    remote = rpc.connect(args.host, args.port, session_timeout=args.session_timeout)
    device = remote.ext_dev(0)
    with tempfile.TemporaryDirectory(prefix="c3_selected_capacity_health_") as temporary:
        executor = load_executor(
            remote, local, "selected_capacity_health.so", [device, remote.cpu(0)], temporary
        )
        executor.set_input("data", model_input_for_seed(SEEDS[0], input_spec))
        executor.run()
        outputs = output_summary([
            executor.get_output(index).numpy() for index in range(executor.get_num_outputs())
        ])
    del executor, device, remote
    gc.collect()
    if outputs != baseline[str(SEEDS[0])]:
        raise RuntimeError("post-restore default-runtime inference differs from baseline")
    return {"seed": SEEDS[0], "outputs": outputs, "exact_baseline_match": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-run", required=True)
    parser.add_argument("--analysis-run", required=True)
    parser.add_argument("--candidate-id")
    parser.add_argument("--diagnostic-build", required=True)
    parser.add_argument("--diagnostic-runtime", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--session-timeout", type=int, default=600)
    parser.add_argument("--ssh-known-hosts", required=True)
    parser.add_argument("--default-runtime", default="/var/volatile/vta_c3_ram/runtime")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
