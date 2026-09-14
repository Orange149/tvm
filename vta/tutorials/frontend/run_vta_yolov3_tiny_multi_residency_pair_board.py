#!/usr/bin/env python3
"""Clean-start board A/B of one versus two exact YOLO residency routes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import tempfile

from tvm import rpc

from run_vta_yolov3_tiny_relay_residency_dispatch_pair_board import (
    CleanStartBoard,
    DEFAULT_RUNTIME,
    ROUNDS,
    image_input,
    invoke,
    load_executor,
    mismatch_count,
    random_input,
    semantic_params_hash,
    sha256,
    write_json,
)


VARIANTS = ["y00_input_only", "y00_input_y02_weight"]
PROFILE_KEYS = (
    "load_buffer_2d_bytes",
    "load_buffer_2d_calls",
    "load_buffer_2d_inp_bytes",
    "load_buffer_2d_inp_calls",
    "load_buffer_2d_wgt_bytes",
    "load_buffer_2d_wgt_calls",
    "store_buffer_2d_bytes",
    "store_buffer_2d_calls",
    "driver_run_calls",
    "driver_run_insns",
    "synchronize_calls",
)


def median_profiles(rows, variant):
    selected = [row["runtime_profile_complete"] for row in rows if row["deployment_variant"] == variant]
    return {key: statistics.median(float(row[key]) for row in selected) for key in PROFILE_KEYS}


def run(args):
    build_dir = Path(args.build_dir)
    build = json.loads((build_dir / "summary.json").read_text(encoding="utf-8"))
    if build.get("status") != "yolov3_tiny_one_vs_two_residency_routes_cross_build_verified":
        raise ValueError("input build is not the verified multi-route pair")
    variants = [row["deployment_variant"] for row in build["variants"]]
    if variants != VARIANTS:
        raise ValueError("unexpected deployment variants: " + repr(variants))
    semantic = [semantic_params_hash(build_dir / variant / "params.bin") for variant in variants]
    if semantic[0] != semantic[1]:
        raise RuntimeError("A/B parameter content differs")
    image = image_input(args.image)

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    output.mkdir(parents=True)
    write_json(output / "parameter_equivalence.json", {
        "status": "identical",
        "semantic_sha256": semantic[0][0],
        "parameter_count": len(semantic[0][1]),
        "inventory": semantic[0][1],
    })

    board = CleanStartBoard(args.host, args.ssh_known_hosts)
    before = board.state()
    if before.get("FPGA") != "operating" or before.get("UDMABUF") != "201326592":
        raise RuntimeError("FPGA/u-dma-buf preflight failed")
    if before.get("STORAGE_ERRORS"):
        raise RuntimeError("storage errors present before run")
    prior_rpc = board.rpc_state()
    if prior_rpc.get("CWD") != args.default_runtime:
        raise RuntimeError("unexpected RPC directory: " + repr(prior_rpc))
    board.stop_rpc(prior_rpc)
    bitstream = board.reload_frozen_bitstream()
    fresh_rpc = board.start_rpc(args.default_runtime, "p7r241_yolov3_multi_residency.log")
    write_json(output / "clean_board_start.json", {
        "before": before,
        "stopped_rpc": prior_rpc,
        "bitstream_reload": bitstream,
        "fresh_rpc": fresh_rpc,
    })

    remote = rpc.connect(args.host, args.port, session_timeout=args.session_timeout)
    device = remote.ext_dev(0)
    contexts = [device, remote.cpu(0)]
    functions = {
        "clear": remote.get_function("vta.runtime.profiler_clear"),
        "status": remote.get_function("vta.runtime.profiler_status"),
    }
    correctness = []
    timings = []
    with tempfile.TemporaryDirectory(prefix="c3_yolo_multi_residency_") as temporary:
        executors = {
            variant: load_executor(remote, build_dir, variant, contexts, temporary)
            for variant in variants
        }
        cases = [("person_image", image)] + [
            ("random_{}".format(seed), random_input(seed)) for seed in (20250901, 20260910)
        ]
        for case, data in cases:
            outputs = {}
            for variant in variants:
                actual, row = invoke(executors[variant], data, functions, device, False)
                outputs[variant] = actual
                row.update(case=case, deployment_variant=variant)
                correctness.append(row)
            mismatches = mismatch_count(outputs[variants[0]], outputs[variants[1]])
            for row in correctness[-2:]:
                row["paired_mismatch_count_by_output"] = mismatches
                row["paired_equal"] = all(value == 0 for value in mismatches)
            if any(mismatches):
                write_json(output / "correctness.json", correctness)
                raise RuntimeError("whole-model YOLO multi-route A/B output mismatch")

        for round_index in range(ROUNDS):
            order = variants if round_index % 2 == 0 else list(reversed(variants))
            round_outputs = {}
            for position, variant in enumerate(order):
                actual, row = invoke(executors[variant], image, functions, device, True)
                round_outputs[variant] = actual
                row.update(round=round_index, position=position, deployment_variant=variant)
                timings.append(row)
            mismatches = mismatch_count(round_outputs[variants[0]], round_outputs[variants[1]])
            for row in timings[-2:]:
                row["paired_mismatch_count_by_output"] = mismatches
                row["paired_equal"] = all(value == 0 for value in mismatches)
            if any(mismatches):
                raise RuntimeError("timed whole-model YOLO multi-route A/B output mismatch")
            print("timing round {}/{}".format(round_index + 1, ROUNDS), flush=True)

    write_json(output / "correctness.json", correctness)
    (output / "timing.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in timings), encoding="utf-8"
    )
    medians = {
        variant: statistics.median(
            row["latency_ms"] for row in timings if row["deployment_variant"] == variant
        )
        for variant in variants
    }
    profiles = {variant: median_profiles(timings, variant) for variant in variants}
    deltas = {
        key: profiles[variants[1]][key] - profiles[variants[0]][key] for key in PROFILE_KEYS
    }
    wins = sum(
        next(
            row["latency_ms"] for row in timings
            if row["round"] == index and row["deployment_variant"] == variants[1]
        )
        < next(
            row["latency_ms"] for row in timings
            if row["round"] == index and row["deployment_variant"] == variants[0]
        )
        for index in range(ROUNDS)
    )
    after = board.state()
    if after.get("STORAGE_ERRORS"):
        raise RuntimeError("storage errors appeared during run")
    summary = {
        "schema": "c3_vta_yolov3_tiny_multi_residency_board_v1",
        "status": "yolov3_tiny_multi_residency_board_correctness_and_timing_complete",
        "boot_id": after.get("BOOT"),
        "model": "yolov3-tiny",
        "parameter_semantic_sha256": semantic[0][0],
        "image_sha256": sha256(args.image),
        "variants": build["variants"],
        "incremental_route": build["incremental_route"],
        "correctness_calls": len(correctness),
        "output_count": len(correctness[0]["outputs"]),
        "all_paired_outputs_equal": all(row["paired_equal"] for row in correctness + timings),
        "all_correctness_outputs_nonzero": all(
            item["nonzero"] > 0 for row in correctness for item in row["outputs"]
        ),
        "timing_calls": len(timings),
        "median_latency_ms": medians,
        "incremental_speedup_percent": (medians[variants[0]] / medians[variants[1]] - 1.0) * 100.0,
        "combined_paired_wins": wins,
        "paired_rounds": ROUNDS,
        "median_runtime_profiles": profiles,
        "incremental_profile_delta": deltas,
        "profile_counter_note": (
            "The runtime time_evaluator path records two graph executions per timed call; "
            "divide counter values and deltas by two for per-inference quantities"
        ),
        "board_after": after,
        "claim_boundary": (
            "Full YOLO paired-output equivalence and incremental Y02 route effect; "
            "no mAP or physical AXI evaluation"
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
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--session-timeout", type=int, default=600)
    parser.add_argument("--ssh-known-hosts", required=True)
    parser.add_argument("--default-runtime", default=DEFAULT_RUNTIME)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
