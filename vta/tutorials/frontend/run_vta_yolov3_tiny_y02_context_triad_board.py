#!/usr/bin/env python3
"""Clean-start FPGA triad for Y02 stock tile, same-tile original and residency."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import statistics
import tempfile

from tvm import rpc

from run_vta_yolov3_tiny_multi_residency_pair_board import PROFILE_KEYS, median_profiles
from run_vta_yolov3_tiny_relay_residency_dispatch_pair_board import (
    CleanStartBoard,
    DEFAULT_RUNTIME,
    image_input,
    invoke,
    load_executor,
    mismatch_count,
    random_input,
    semantic_params_hash,
    sha256,
    write_json,
)


VARIANTS = [
    "y00_input_y02_stock",
    "y00_input_y02_same_tile_original",
    "y00_input_y02_same_tile_weight",
]
ORDERS = [list(order) for order in itertools.permutations(VARIANTS)]


def row_latency(rows, round_index, variant):
    return next(
        row["latency_ms"] for row in rows
        if row["round"] == round_index and row["deployment_variant"] == variant
    )


def comparison(rows, lhs, rhs):
    lhs_values = [row["latency_ms"] for row in rows if row["deployment_variant"] == lhs]
    rhs_values = [row["latency_ms"] for row in rows if row["deployment_variant"] == rhs]
    lhs_median = statistics.median(lhs_values)
    rhs_median = statistics.median(rhs_values)
    return {
        "lhs": lhs,
        "rhs": rhs,
        "lhs_median_ms": lhs_median,
        "rhs_median_ms": rhs_median,
        "rhs_speedup_percent": (lhs_median / rhs_median - 1.0) * 100.0,
        "rhs_paired_wins": sum(
            row_latency(rows, index, rhs) < row_latency(rows, index, lhs)
            for index in range(len(ORDERS))
        ),
        "paired_rounds": len(ORDERS),
    }


def profile_delta(profiles, lhs, rhs):
    return {key: profiles[rhs][key] - profiles[lhs][key] for key in PROFILE_KEYS}


def run(args):
    build_dir = Path(args.build_dir)
    build = json.loads((build_dir / "summary.json").read_text(encoding="utf-8"))
    if build.get("status") != "yolov3_tiny_y02_context_triad_cross_build_verified":
        raise ValueError("input build is not the verified Y02 context triad")
    variants = [row["deployment_variant"] for row in build["variants"]]
    if variants != VARIANTS:
        raise ValueError("unexpected deployment variants: " + repr(variants))
    semantic = [semantic_params_hash(build_dir / variant / "params.bin") for variant in variants]
    if any(value != semantic[0] for value in semantic[1:]):
        raise RuntimeError("triad parameter content differs")
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
    fresh_rpc = board.start_rpc(args.default_runtime, "p7r243_yolov3_y02_context_triad.log")
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
    with tempfile.TemporaryDirectory(prefix="c3_yolo_y02_context_") as temporary:
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
            for variant in variants[1:]:
                mismatches = mismatch_count(outputs[variants[0]], outputs[variant])
                if any(mismatches):
                    write_json(output / "correctness.json", correctness)
                    raise RuntimeError("whole-model YOLO triad output mismatch")
            for row in correctness[-3:]:
                row["all_variants_equal"] = True

        for round_index, order in enumerate(ORDERS):
            round_outputs = {}
            for position, variant in enumerate(order):
                actual, row = invoke(executors[variant], image, functions, device, True)
                round_outputs[variant] = actual
                row.update(round=round_index, position=position, deployment_variant=variant)
                timings.append(row)
            for variant in variants[1:]:
                if any(mismatch_count(round_outputs[variants[0]], round_outputs[variant])):
                    raise RuntimeError("timed whole-model YOLO triad output mismatch")
            for row in timings[-3:]:
                row["all_variants_equal"] = True
            print("timing order {}/{}".format(round_index + 1, len(ORDERS)), flush=True)

    write_json(output / "correctness.json", correctness)
    (output / "timing.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in timings), encoding="utf-8"
    )
    profiles = {variant: median_profiles(timings, variant) for variant in variants}
    comparisons = {
        "same_tile_original_to_weight": comparison(timings, variants[1], variants[2]),
        "stock_to_same_tile_original": comparison(timings, variants[0], variants[1]),
        "stock_to_same_tile_weight": comparison(timings, variants[0], variants[2]),
    }
    after = board.state()
    if after.get("STORAGE_ERRORS"):
        raise RuntimeError("storage errors appeared during run")
    summary = {
        "schema": "c3_vta_yolov3_tiny_y02_context_triad_board_v1",
        "status": "yolov3_tiny_y02_context_triad_board_complete",
        "boot_id": after.get("BOOT"),
        "model": "yolov3-tiny",
        "parameter_semantic_sha256": semantic[0][0],
        "image_sha256": sha256(args.image),
        "controlled_factors": build["controlled_factors"],
        "correctness_calls": len(correctness),
        "output_count": len(correctness[0]["outputs"]),
        "all_variant_outputs_equal": all(row["all_variants_equal"] for row in correctness + timings),
        "all_correctness_outputs_nonzero": all(
            item["nonzero"] > 0 for row in correctness for item in row["outputs"]
        ),
        "timing_calls": len(timings),
        "balanced_orders": ORDERS,
        "comparisons": comparisons,
        "median_runtime_profiles": profiles,
        "profile_deltas": {
            "same_tile_original_to_weight": profile_delta(profiles, variants[1], variants[2]),
            "stock_to_same_tile_original": profile_delta(profiles, variants[0], variants[1]),
            "stock_to_same_tile_weight": profile_delta(profiles, variants[0], variants[2]),
        },
        "profile_counter_note": (
            "The runtime time_evaluator path records two graph executions per timed call; "
            "divide counter values and deltas by two for per-inference quantities"
        ),
        "board_after": after,
        "claim_boundary": (
            "Full YOLO controlled Y02 context triad; same-tile mechanism effect is "
            "separate from deployability against TopHub; no mAP or physical AXI evaluation"
        ),
    }
    write_json(output / "summary.json", summary)
    (output / "command.txt").write_text(" ".join(__import__("sys").argv) + "\n", encoding="utf-8")
    write_json(output / "artifact_hashes.json", {"artifacts": {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }})
    print(json.dumps({
        "status": summary["status"],
        "boot_id": summary["boot_id"],
        "correctness_calls": summary["correctness_calls"],
        "timing_calls": summary["timing_calls"],
        "comparisons": comparisons,
        "profile_deltas": summary["profile_deltas"],
        "board_after": after,
    }, indent=2, sort_keys=True))


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
