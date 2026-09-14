#!/usr/bin/env python3
"""Measure YOLO Executor and first-run u-dma-buf allocation without provoking known OOM."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import tempfile
import time

from tvm import rpc

from run_vta_yolov3_tiny_relay_residency_dispatch_pair_board import (
    CleanStartBoard,
    image_input,
    load_executor,
    output_summary,
    sha256,
    write_json,
)


VARIANTS = ["stock_all", "y00_input_only", "y02_weight_only", "y00_input_y02_weight"]
SNAPSHOT_SCHEMA = "vta_udmabuf_allocation_snapshot_v1"


def snapshot(remote, phase):
    row = json.loads(remote.get_function("vta.runtime.udmabuf_allocation_snapshot")())
    if row.get("schema") != SNAPSHOT_SCHEMA:
        raise RuntimeError("unexpected allocation snapshot schema")
    row["phase"] = phase
    row["remaining_bytes"] = row["capacity_bytes"] - row["high_water_bytes"]
    return row


def run_scenario(args, build_dir, variants, run_variant, data):
    remote = rpc.connect(args.host, args.port, session_timeout=args.session_timeout)
    device = remote.ext_dev(0)
    contexts = [device, remote.cpu(0)]
    rows = [snapshot(remote, "rpc_fresh")]
    if rows[0]["initialized"] != 0 or rows[0]["high_water_bytes"] != 0:
        raise RuntimeError("RPC child did not begin with an uninitialized allocation pool")
    executors = {}
    result = None
    with tempfile.TemporaryDirectory(prefix="c3_yolo_allocation_") as temporary:
        for index, variant in enumerate(variants, start=1):
            executors[variant] = load_executor(
                remote, build_dir, variant, contexts, temporary
            )
            rows.append(snapshot(remote, "after_load_{}_{}".format(index, variant)))
        if run_variant is not None:
            executor = executors[run_variant]
            executor.set_input("data", data)
            remote.get_function("vta.runtime.profiler_clear")()
            started = time.perf_counter()
            executor.run()
            host_wall_ms = (time.perf_counter() - started) * 1000.0
            outputs = [
                executor.get_output(index).numpy()
                for index in range(executor.get_num_outputs())
            ]
            rows.append(snapshot(remote, "after_first_run_" + run_variant))
            result = {
                "run_variant": run_variant,
                "host_wall_ms": host_wall_ms,
                "outputs": output_summary(outputs),
                "all_outputs_nonzero": all(item["nonzero"] > 0 for item in output_summary(outputs)),
                "queue_capacity_status": json.loads(
                    remote.get_function("vta.runtime.queue_capacity_status")()
                ),
                "runtime_profile": json.loads(
                    remote.get_function("vta.runtime.profiler_status")()
                ),
            }
    record = {"loaded_variants": list(variants), "snapshots": rows, "run": result}
    executors.clear()
    del remote
    gc.collect()
    time.sleep(0.25)
    return record


def run(args):
    build_dir = Path(args.build_dir)
    build = json.loads((build_dir / "summary.json").read_text(encoding="utf-8"))
    if build.get("status") != "yolov3_tiny_two_route_factorial_cross_build_verified":
        raise ValueError("input is not the verified YOLO 2x2 factorial build")
    if [row["deployment_variant"] for row in build["variants"]] != VARIANTS:
        raise ValueError("unexpected factorial variant order")

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    output.mkdir(parents=True)
    diagnostic_build = Path(args.diagnostic_build)
    diagnostic_manifest = json.loads(
        (diagnostic_build / "build_manifest.json").read_text(encoding="utf-8")
    )
    expected_driver = diagnostic_manifest["artifacts"]["libvta.so"]
    if sha256(diagnostic_build / "libvta.so") != expected_driver:
        raise RuntimeError("diagnostic driver build hash mismatch")

    board = CleanStartBoard(args.host, args.ssh_known_hosts)
    before = board.state()
    if before.get("FPGA") != "operating" or before.get("UDMABUF") != "201326592":
        raise RuntimeError("FPGA/u-dma-buf preflight failed")
    if before.get("STORAGE_ERRORS"):
        raise RuntimeError("storage errors present before run")
    rpc_before = board.rpc_state()
    if rpc_before.get("CWD") != args.diagnostic_runtime:
        raise RuntimeError("diagnostic RPC is not active: " + repr(rpc_before))
    remote_driver_hash = board.run(
        "sha256sum " + args.diagnostic_runtime + "/libvta.so"
    ).stdout.split()[0]
    if remote_driver_hash != expected_driver:
        raise RuntimeError("board diagnostic driver hash mismatch")

    contract = {
        "schema": "c3_yolov3_tiny_udmabuf_allocation_diagnostic_contract_v1",
        "purpose": "localize the 3-Executor pass / 4-Executor first-run OOM",
        "boot_id": before["BOOT"],
        "udmabuf_capacity_bytes": int(before["UDMABUF"]),
        "variants": VARIANTS,
        "scenarios": [
            {"load_count": 1, "run": "stock_all"},
            {"load_count": 3, "run": "stock_all"},
            {"load_count": 4, "run": None, "reason": "do not repeat known fatal OOM"},
            {"load_count": 1, "run": "y00_input_only", "role": "selected deployment"},
            {"load_count": 1, "run": "y02_weight_only", "role": "factorial allowlist"},
            {"load_count": 1, "run": "y00_input_y02_weight", "role": "factorial allowlist"},
        ],
        "diagnostic_driver_sha256": expected_driver,
        "diagnostic_runtime": args.diagnostic_runtime,
        "default_runtime_to_restore": args.default_runtime,
        "factorial_build_summary_sha256": sha256(build_dir / "summary.json"),
        "script_sha256_before_run": sha256(__file__),
        "claim_boundary": (
            "Read-only logical allocator high-water and runtime queue status; "
            "not physical AXI traffic and not a latency comparison"
        ),
    }
    write_json(output / "contract.json", contract)

    data = image_input(args.image)
    scenarios = []
    restored = None
    try:
        scenarios.append(run_scenario(args, build_dir, VARIANTS[:1], "stock_all", data))
        scenarios.append(run_scenario(args, build_dir, VARIANTS[:3], "stock_all", data))
        scenarios.append(run_scenario(args, build_dir, VARIANTS, None, data))
        scenarios.append(run_scenario(args, build_dir, VARIANTS[1:2], "y00_input_only", data))
        scenarios.append(run_scenario(args, build_dir, VARIANTS[2:3], "y02_weight_only", data))
        scenarios.append(run_scenario(
            args, build_dir, VARIANTS[3:4], "y00_input_y02_weight", data
        ))
    finally:
        current = board.rpc_state()
        if current.get("CWD") == args.diagnostic_runtime:
            board.stop_rpc(current)
            bitstream = board.reload_frozen_bitstream()
            restored = {
                "bitstream_reload": bitstream,
                "rpc": board.start_rpc(args.default_runtime, "p7r255_default_rpc_restored.log"),
            }

    single = scenarios[0]
    triple = scenarios[1]
    four = scenarios[2]
    selected = scenarios[3]
    single_variant_scenarios = [scenarios[index] for index in (0, 3, 4, 5)]
    write_json(output / "raw_scenarios.json", {"scenarios": scenarios})
    single_load = single["snapshots"][-2]
    single_run = single["snapshots"][-1]
    triple_load = triple["snapshots"][-2]
    triple_run = triple["snapshots"][-1]
    four_load = four["snapshots"][-1]
    queue = single["run"]["queue_capacity_status"]
    profile = single["run"]["runtime_profile"]
    whole_graph_peak_available = (
        queue["submissions"] > 0
        and queue["submissions"] == profile["synchronize_calls"]
    )
    first_run_allocation = single_run["high_water_bytes"] - single_load["high_water_bytes"]
    explicit_queue_reservation = queue["uop_capacity_bytes"] + queue["insn_capacity_bytes"]
    other_first_run_allocation = first_run_allocation - explicit_queue_reservation
    remaining_after_uop_queue = four_load["remaining_bytes"] - queue["uop_capacity_bytes"]
    predicted_insn_allocation_deficit = max(
        0, queue["insn_capacity_bytes"] - remaining_after_uop_queue
    )
    if first_run_allocation < explicit_queue_reservation:
        raise RuntimeError("first-run allocation is smaller than the two command queues")
    if triple_run["high_water_bytes"] > triple_run["capacity_bytes"]:
        raise RuntimeError("triple scenario exceeded capacity despite a successful run")
    if predicted_insn_allocation_deficit <= 0:
        raise RuntimeError("four-Executor dry run does not explain the archived second queue OOM")
    if four_load["high_water_bytes"] + queue["uop_capacity_bytes"] != 170758144:
        raise RuntimeError("dry-run allocation reconstruction differs from archived P7R246 used bytes")
    if single["run"]["outputs"] != triple["run"]["outputs"]:
        raise RuntimeError("stock output changed between one- and three-Executor contexts")
    if single["run"]["outputs"] != selected["run"]["outputs"]:
        raise RuntimeError("selected Y00 graph output differs from stock")
    if any(row["run"]["outputs"] != single["run"]["outputs"]
           for row in single_variant_scenarios[1:]):
        raise RuntimeError("a factorial variant output differs from stock")
    selected_queue = selected["run"]["queue_capacity_status"]
    if selected_queue["submissions"] != selected["run"]["runtime_profile"]["synchronize_calls"]:
        raise RuntimeError("selected graph queue diagnostics missed a synchronization")
    variant_queues = {
        row["run"]["run_variant"]: row["run"]["queue_capacity_status"]
        for row in single_variant_scenarios
    }
    max_variant_insn = max(row["insn_peak_bytes"] for row in variant_queues.values())
    max_variant_uop = max(row["uop_peak_bytes"] for row in variant_queues.values())

    after = board.state()
    if after.get("STORAGE_ERRORS"):
        raise RuntimeError("storage errors appeared during diagnostic")
    summary = {
        "schema": "c3_yolov3_tiny_udmabuf_allocation_diagnostic_v1",
        "status": "three_executor_pass_four_executor_second_queue_oom_explained",
        "boot_id": after["BOOT"],
        "scenarios": scenarios,
        "derived": {
            "single_executor_graph_bytes": single_load["high_water_bytes"],
            "three_executor_graph_bytes": triple_load["high_water_bytes"],
            "four_executor_graph_bytes": four_load["high_water_bytes"],
            "single_to_three_graph_increment_bytes": (
                triple_load["high_water_bytes"] - single_load["high_water_bytes"]
            ),
            "first_run_total_allocation_bytes": first_run_allocation,
            "explicit_two_queue_reservation_bytes": explicit_queue_reservation,
            "other_first_run_allocation_bytes": other_first_run_allocation,
            "uop_queue_reservation_bytes": queue["uop_capacity_bytes"],
            "insn_queue_reservation_bytes": queue["insn_capacity_bytes"],
            "three_executor_post_run_remaining_bytes": triple_run["remaining_bytes"],
            "four_executor_pre_run_remaining_bytes": four_load["remaining_bytes"],
            "four_executor_after_uop_queue_high_water_bytes": (
                four_load["high_water_bytes"] + queue["uop_capacity_bytes"]
            ),
            "four_executor_remaining_after_uop_queue_bytes": remaining_after_uop_queue,
            "four_executor_second_insn_allocation_deficit_bytes": (
                predicted_insn_allocation_deficit
            ),
            "queue_status_insn_peak_bytes": queue["insn_peak_bytes"],
            "queue_status_uop_peak_bytes": queue["uop_peak_bytes"],
            "queue_status_submissions": queue["submissions"],
            "whole_graph_peak_available": whole_graph_peak_available,
            "whole_graph_peak_gap": None if whole_graph_peak_available else (
                "VTA_QUEUE_DIAGNOSTICS was not enabled or the queried queue did not "
                "observe every graph synchronization."
            ),
            "selected_y00_insn_peak_bytes": selected["run"]["queue_capacity_status"][
                "insn_peak_bytes"
            ],
            "selected_y00_uop_peak_bytes": selected["run"]["queue_capacity_status"][
                "uop_peak_bytes"
            ],
            "selected_y00_submissions": selected["run"]["queue_capacity_status"][
                "submissions"
            ],
            "factorial_variant_queue_peaks": {
                variant: {
                    "insn_peak_bytes": row["insn_peak_bytes"],
                    "uop_peak_bytes": row["uop_peak_bytes"],
                    "submissions": row["submissions"],
                }
                for variant, row in variant_queues.items()
            },
            "factorial_allowlist_max_insn_peak_bytes": max_variant_insn,
            "factorial_allowlist_max_uop_peak_bytes": max_variant_uop,
            "factorial_allowlist_insn_capacity_4k_bytes": (
                (max_variant_insn + 4095) // 4096 * 4096
            ),
            "factorial_allowlist_uop_capacity_4k_bytes": (
                (max_variant_uop + 4095) // 4096 * 4096
            ),
        },
        "archived_failure_link": {
            "run": "P7R246",
            "observed_request_bytes": 33554432,
            "observed_used_bytes": 170758144,
            "observed_capacity_bytes": 201326592,
            "interpretation": (
                "uop allocation raised high-water to the archived used value; "
                "the second instruction-queue allocation then failed"
            ),
        },
        "default_runtime_restored": restored,
        "board_after": after,
        "claim_boundary": contract["claim_boundary"],
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
    parser.add_argument("--diagnostic-build", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--session-timeout", type=int, default=600)
    parser.add_argument("--ssh-known-hosts", required=True)
    parser.add_argument("--diagnostic-runtime", required=True)
    parser.add_argument("--default-runtime", default="/var/volatile/vta_c3_ram/runtime")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
