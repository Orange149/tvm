#!/usr/bin/env python3
"""Qualify selected YOLO graph command capacity and one-page-short rejection."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import tempfile
import time
import traceback

from tvm import rpc

from run_vta_yolov3_tiny_relay_residency_dispatch_pair_board import (
    CleanStartBoard,
    image_input,
    load_executor,
    output_summary,
    random_input,
    sha256,
    write_json,
)


VARIANT = "y00_input_only"
CASES = ("person_image", "random_20250901", "random_20260910")


def allocation_snapshot(remote):
    return json.loads(remote.get_function("vta.runtime.udmabuf_allocation_snapshot")())


def expected_outputs(board_dir):
    rows = json.loads((Path(board_dir) / "correctness.json").read_text(encoding="utf-8"))
    result = {}
    for row in rows:
        if row.get("deployment_variant") == VARIANT:
            result[row["case"]] = row["outputs"]
    if tuple(result) != CASES:
        raise ValueError("archived selected-graph correctness cases are incomplete")
    return result


def environment(capacity, manifest_id):
    return {
        "VTA_INSN_BUFFER_BYTES": capacity["insn_bytes"],
        "VTA_UOP_BUFFER_BYTES": capacity["uop_bytes"],
        "VTA_QUEUE_DIAGNOSTICS": 1,
        "VTA_REPLAY_POLICY": "disabled",
        "VTA_COMMAND_MANIFEST_ID": manifest_id,
    }


def run_positive(args, build_dir, inputs, expected, capacity, manifest_id):
    remote = rpc.connect(args.host, args.port, session_timeout=args.session_timeout)
    device = remote.ext_dev(0)
    rows = []
    snapshots = [{"phase": "rpc_fresh", **allocation_snapshot(remote)}]
    with tempfile.TemporaryDirectory(prefix="c3_yolo_graph_capacity_positive_") as temporary:
        executor = load_executor(
            remote, build_dir, VARIANT, [device, remote.cpu(0)], temporary
        )
        snapshots.append({"phase": "graph_and_params_loaded", **allocation_snapshot(remote)})
        for case in CASES:
            executor.set_input("data", inputs[case])
            remote.get_function("vta.runtime.profiler_clear")()
            executor.run()
            actual = output_summary([
                executor.get_output(index).numpy()
                for index in range(executor.get_num_outputs())
            ])
            row = {
                "case": case,
                "outputs": actual,
                "exact_archived_match": actual == expected[case],
                "runtime_profile": json.loads(
                    remote.get_function("vta.runtime.profiler_status")()
                ),
            }
            rows.append(row)
            snapshots.append({"phase": "after_" + case, **allocation_snapshot(remote)})
        queue = json.loads(remote.get_function("vta.runtime.queue_capacity_status")())
    if not all(row["exact_archived_match"] for row in rows):
        raise RuntimeError("reduced-capacity selected graph differs from archived outputs")
    if (
        queue["insn_capacity_bytes"] != capacity["insn_bytes"]
        or queue["uop_capacity_bytes"] != capacity["uop_bytes"]
        or queue["command_manifest_id"] != manifest_id
        or queue["replay_policy"] != "disabled"
        or queue["insn_peak_bytes"] > capacity["insn_bytes"]
        or queue["uop_peak_bytes"] > capacity["uop_bytes"]
        or queue["submissions"] != 36
    ):
        raise RuntimeError("positive queue status violates the graph capacity contract")
    del remote
    gc.collect()
    time.sleep(0.25)
    return {"capacity": capacity, "rows": rows, "snapshots": snapshots, "queue_status": queue}


def run_negative(args, build_dir, data, capacity, manifest_id, positive_peak):
    remote = rpc.connect(args.host, args.port, session_timeout=args.session_timeout)
    device = remote.ext_dev(0)
    snapshots = [{"phase": "rpc_fresh", **allocation_snapshot(remote)}]
    error = None
    with tempfile.TemporaryDirectory(prefix="c3_yolo_graph_capacity_negative_") as temporary:
        executor = load_executor(
            remote, build_dir, VARIANT, [device, remote.cpu(0)], temporary
        )
        snapshots.append({"phase": "graph_and_params_loaded", **allocation_snapshot(remote)})
        executor.set_input("data", data)
        remote.get_function("vta.runtime.profiler_clear")()
        try:
            executor.run()
        except Exception as caught:  # RPC transports the deliberate runtime rejection.
            error = {
                "exception_type": type(caught).__name__,
                "message": str(caught),
                "traceback": traceback.format_exc()[-8000:],
            }
        profile = json.loads(remote.get_function("vta.runtime.profiler_status")())
        queue = json.loads(remote.get_function("vta.runtime.queue_capacity_status")())
        snapshots.append({"phase": "after_rejection", **allocation_snapshot(remote)})
    if error is None or "queue backing capacity exceeded before submission" not in error["message"]:
        raise RuntimeError("one-page-short instruction capacity was not rejected")
    if not (capacity["insn_bytes"] < positive_peak <= capacity["insn_bytes"] + 4096):
        raise RuntimeError("negative capacity does not straddle the measured peak by one page")
    if (
        queue["insn_capacity_bytes"] != capacity["insn_bytes"]
        or queue["uop_capacity_bytes"] != capacity["uop_bytes"]
        or queue["command_manifest_id"] != manifest_id
        or queue["replay_policy"] != "disabled"
        or queue["insn_peak_bytes"] > capacity["insn_bytes"]
        or profile["driver_run_calls"] != queue["submissions"]
    ):
        raise RuntimeError("negative queue status is not fail-closed at the violating submission")
    del remote
    gc.collect()
    time.sleep(0.25)
    return {
        "capacity": capacity,
        "error": error,
        "runtime_profile": profile,
        "queue_status": queue,
        "snapshots": snapshots,
        "rejected_before_violating_submission": True,
    }


def run(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    output.mkdir(parents=True)
    build_dir = Path(args.build_dir)
    manifest_path = Path(args.graph_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["selected_deployment_variant"] != VARIANT:
        raise ValueError("graph manifest does not select Y00 input residency")
    for name, record in manifest["graph_artifacts"].items():
        expected_name = "params.bin" if name == "params.bin" else name
        if sha256(build_dir / VARIANT / expected_name) != record["sha256"]:
            raise RuntimeError("selected graph artifact hash mismatch: " + name)

    peaks = json.loads((Path(args.peak_run) / "summary.json").read_text(encoding="utf-8"))
    peak = peaks["derived"]
    selected_insn_peak = int(peak["selected_y00_insn_peak_bytes"])
    selected_uop_peak = int(peak["selected_y00_uop_peak_bytes"])
    positive = {
        "insn_bytes": (selected_insn_peak + 4095) // 4096 * 4096,
        "uop_bytes": (selected_uop_peak + 4095) // 4096 * 4096,
    }
    negative = {"insn_bytes": positive["insn_bytes"] - 4096, "uop_bytes": positive["uop_bytes"]}
    manifest_id = manifest["manifest_id"]

    board = CleanStartBoard(args.host, args.ssh_known_hosts)
    before = board.state()
    original_rpc = board.rpc_state()
    if original_rpc.get("CWD") != args.default_runtime:
        raise RuntimeError("default RPC is not active")
    if before.get("STORAGE_ERRORS") or before.get("FPGA") != "operating":
        raise RuntimeError("board preflight failed")
    driver_hash = json.loads(
        (Path(args.diagnostic_build) / "build_manifest.json").read_text(encoding="utf-8")
    )["artifacts"]["libvta.so"]
    remote_driver_hash = board.run(
        "sha256sum " + args.diagnostic_runtime + "/libvta.so"
    ).stdout.split()[0]
    if remote_driver_hash != driver_hash:
        raise RuntimeError("diagnostic runtime driver hash mismatch")

    archived = Path(args.archived_board_run)
    expected = expected_outputs(archived)
    inputs = {
        "person_image": image_input(args.image),
        "random_20250901": random_input(20250901),
        "random_20260910": random_input(20260910),
    }
    contract = {
        "schema": "c3_yolov3_tiny_selected_graph_capacity_contract_v1",
        "status": "frozen_before_capacity_execution",
        "boot_id": before["BOOT"],
        "graph_manifest_id": manifest_id,
        "selected_variant": VARIANT,
        "selected_graph_binary_sha256": manifest["graph_artifacts"]["graphlib.so"]["sha256"],
        "measured_peak": {"insn_bytes": selected_insn_peak, "uop_bytes": selected_uop_peak},
        "positive_capacity": positive,
        "negative_one_insn_page_capacity": negative,
        "expected_cases": CASES,
        "archived_board_evidence_sha256": sha256(archived / "correctness.json"),
        "peak_evidence_sha256": sha256(Path(args.peak_run) / "summary.json"),
        "diagnostic_driver_sha256": driver_hash,
        "failure_policy": "persist raw result and always restore default RPC",
        "performance_measurement": "not_collected",
    }
    write_json(output / "contract.json", contract)

    active = None
    positive_result = None
    negative_result = None
    caught = None
    try:
        board.stop_rpc(original_rpc)
        board.reload_frozen_bitstream()
        active = board.start_rpc(
            args.diagnostic_runtime,
            "p7r261_selected_graph_capacity_positive.log",
            environment(positive, manifest_id),
        )
        positive_result = run_positive(
            args, build_dir, inputs, expected, positive, manifest_id
        )
        write_json(output / "positive_raw.json", positive_result)
        board.stop_rpc(active)
        active = board.start_rpc(
            args.diagnostic_runtime,
            "p7r261_selected_graph_capacity_negative.log",
            environment(negative, manifest_id),
        )
        negative_result = run_negative(
            args, build_dir, inputs["person_image"], negative, manifest_id, selected_insn_peak
        )
        write_json(output / "negative_raw.json", negative_result)
        board.stop_rpc(active)
        active = None
    except Exception as error:
        caught = error
        write_json(output / "failure.json", {
            "exception_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc()[-12000:],
        })
    finally:
        current = board.rpc_state()
        if current.get("CWD") != args.default_runtime:
            if current.get("PID") and current.get("CWD"):
                board.stop_rpc(current)
            board.reload_frozen_bitstream()
            board.start_rpc(args.default_runtime, "p7r261_default_rpc_restored.log")
    if caught is not None:
        raise caught

    positive_after = positive_result["snapshots"][-1]
    default_command_bytes = 2 * 33554432
    selected_command_bytes = positive["insn_bytes"] + positive["uop_bytes"]
    summary = {
        "schema": "c3_yolov3_tiny_selected_graph_capacity_summary_v1",
        "status": "selected_graph_capacity_positive_and_negative_passed",
        "boot_id": before["BOOT"],
        "graph_manifest_id": manifest_id,
        "positive_capacity": positive,
        "positive_correctness": "3/3 exact archived output sets",
        "positive_queue_status": positive_result["queue_status"],
        "positive_udmabuf_high_water_bytes": positive_after["high_water_bytes"],
        "negative_capacity": negative,
        "negative_error": negative_result["error"]["message"].splitlines()[-1],
        "negative_prior_device_submissions": negative_result["queue_status"]["submissions"],
        "negative_rejected_before_violating_submission": True,
        "command_backing_default_bytes": default_command_bytes,
        "command_backing_selected_bytes": selected_command_bytes,
        "command_backing_reduction_percent": (
            1.0 - selected_command_bytes / default_command_bytes
        ) * 100.0,
        "default_rpc_restored": board.rpc_state().get("CWD") == args.default_runtime,
        "board_after": board.state(),
        "claim_boundary": (
            "Exact selected static-shape graph on one boot; logical queue/backing allocation, "
            "not physical AXI or dynamic-shape guarantee"
        ),
    }
    write_json(output / "summary.json", summary)
    (output / "command.txt").write_text(" ".join(__import__("sys").argv) + "\n", encoding="utf-8")
    write_json(output / "artifact_hashes.json", {"artifacts": {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }})
    print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--graph-manifest", required=True)
    parser.add_argument("--peak-run", required=True)
    parser.add_argument("--archived-board-run", required=True)
    parser.add_argument("--diagnostic-build", required=True)
    parser.add_argument("--diagnostic-runtime", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--session-timeout", type=int, default=600)
    parser.add_argument("--ssh-known-hosts", required=True)
    parser.add_argument("--default-runtime", default="/var/volatile/vta_c3_ram/runtime")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
