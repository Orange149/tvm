#!/usr/bin/env python3
"""Diagnose a frozen selected-ResNet18 full-graph correctness failure.

This runner never collects performance labels.  It reuses the exact binaries from
the build contract, records output hashes and pairwise mismatch counts for forward
and reverse execution orders, and treats a board interruption as non-spliceable.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import tempfile
import time
import traceback
from pathlib import Path

from tvm import rpc

from analyze_vta_fused_program_service_proxy_norefit import verify_artifacts_compatible
from run_vta_c3_resnet18_fullgraph_programs import SEEDS, append_jsonl
from run_vta_p7r132_y00_search_confirmation import (
    CleanStartBoard,
    EXPECTED_DEFAULT_RUNTIME_HASHES,
)
from run_vta_resnet50_fused_program_pareto_board import (
    assert_board_state,
    invoke_model,
    load_executor,
    model_input_for_seed,
)
from run_vta_resnet50_relay_residency_dispatch_pair_board import sha256, write_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def output_hashes(outputs):
    return [hashlib.sha256(value.tobytes(order="C")).hexdigest() for value in outputs]


def mismatch_by_output(left, right):
    if len(left) != len(right):
        return {"output_count_mismatch": [len(left), len(right)]}
    return [int((a != b).sum()) for a, b in zip(left, right)]


def finish(output):
    write_json(
        output / "artifact_hashes.json",
        {
            "artifacts": {
                str(path.relative_to(output)): sha256(path)
                for path in sorted(output.rglob("*"))
                if path.is_file() and path.name != "artifact_hashes.json"
            },
            "source_sha256": sha256(Path(__file__).resolve()),
        },
    )


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    build_dir = Path(args.build_dir).resolve()
    verify_artifacts_compatible(build_dir)
    built = read_json(build_dir / "summary.json")
    if built.get("status") != "selected_fullgraphs_built_once_before_board":
        raise RuntimeError("selected full-graph builds are incomplete")
    program_ids = ["stock_reference"] + [row["program_id"] for row in built["programs"]]
    local_dirs = {"stock_reference": build_dir / "stock_reference"}
    local_dirs.update(
        {row["program_id"]: build_dir / row["relative_dir"] for row in built["programs"]}
    )
    orders = [program_ids, list(reversed(program_ids))]
    output.mkdir(parents=True)
    w0_wall = time.time()
    w0_mono = time.monotonic()
    write_json(
        output / "contract.json",
        {
            "schema": "c3_resnet18_selected_fullgraph_mismatch_diagnostic_v1",
            "status": "frozen_before_diagnostic_board_observation",
            "build_manifest_sha256": sha256(build_dir / "artifact_hashes.json"),
            "program_ids": program_ids,
            "seeds": list(SEEDS),
            "orders": orders,
            "performance_timing_enabled": False,
            "purpose": "identify the program and scale of the fail-closed P7R516 mismatch",
            "session_policy": "any RPC, device or power interruption invalidates the entire diagnostic session",
        },
    )
    write_json(output / "pre_observation_hashes.json", {"contract.json": sha256(output / "contract.json")})
    results_path = output / "results.jsonl"
    timeline_path = output / "timeline.jsonl"
    results_path.write_text("", encoding="utf-8")
    timeline_path.write_text("", encoding="utf-8")
    append_jsonl(timeline_path, {"phase": "contract_frozen", "W0_elapsed_seconds": time.monotonic() - w0_mono})
    remote = None
    rows = []
    try:
        board = CleanStartBoard(args.host, args.ssh_known_hosts)
        before = board.state()
        assert_board_state(before)
        if board.runtime_hashes(args.default_runtime) != EXPECTED_DEFAULT_RUNTIME_HASHES:
            raise RuntimeError("default runtime hash mismatch")
        prior = board.rpc_state()
        if not prior.get("PID") or prior.get("CWD") != args.default_runtime:
            raise RuntimeError("expected healthy default RPC before clean start")
        board.stop_rpc(prior)
        bitstream = board.reload_frozen_bitstream()
        fresh = board.start_rpc(args.default_runtime, "p7r_resnet18_fullgraph_mismatch.log")
        t0_wall = time.time()
        t0_mono = time.monotonic()
        write_json(
            output / "clean_start.json",
            {
                "before": before,
                "stopped_rpc": prior,
                "bitstream": bitstream,
                "fresh_rpc": fresh,
                "W0_wall_unix": w0_wall,
                "T0_wall_unix": t0_wall,
                "W0_to_T0_seconds": t0_mono - w0_mono,
            },
        )
        append_jsonl(timeline_path, {"phase": "T0_clean_bitstream_and_rpc_ready", "W0_elapsed_seconds": t0_mono - w0_mono})
        remote = rpc.connect(args.host, args.port, session_timeout=args.session_timeout)
        device = remote.ext_dev(0)
        contexts = [device, remote.cpu(0)]
        functions = {
            "clear": remote.get_function("vta.runtime.profiler_clear"),
            "status": remote.get_function("vta.runtime.profiler_status"),
        }
        input_spec = {"shape": [1, 3, 224, 224], "uniform_range": [0.0, 1.0]}
        with tempfile.TemporaryDirectory(prefix="c3_r18_mismatch_") as temporary:
            executors = {
                program_id: load_executor(
                    remote,
                    local_dirs[program_id],
                    "r18_diag_{}_{:02d}.so".format(program_id[:18], index),
                    contexts,
                    temporary,
                )
                for index, program_id in enumerate(program_ids)
            }
            references = {}
            for seed in SEEDS:
                data = model_input_for_seed(seed, input_spec)
                for order_index, order in enumerate(orders):
                    observed = {}
                    for position, program_id in enumerate(order):
                        actual, invocation = invoke_model(
                            executors[program_id], data, functions, device, False, True
                        )
                        observed[program_id] = actual
                        if program_id == "stock_reference":
                            references.setdefault(int(seed), actual)
                        reference = references.get(int(seed))
                        mismatch = None if reference is None else mismatch_by_output(reference, actual)
                        row = {
                            "record_type": "diagnostic_invocation",
                            "seed": int(seed),
                            "order_index": order_index,
                            "position": position,
                            "program_id": program_id,
                            "output_sha256": output_hashes(actual),
                            "stock_mismatch_count_by_output": mismatch,
                            "stock_equal": mismatch is not None and sum(mismatch) == 0,
                            **invocation,
                        }
                        rows.append(row)
                        append_jsonl(results_path, row)
                    reference = references[int(seed)]
                    for left_index, left_id in enumerate(program_ids):
                        for right_id in program_ids[left_index + 1 :]:
                            mismatch = mismatch_by_output(observed[left_id], observed[right_id])
                            append_jsonl(
                                results_path,
                                {
                                    "record_type": "pairwise_comparison",
                                    "seed": int(seed),
                                    "order_index": order_index,
                                    "left_program_id": left_id,
                                    "right_program_id": right_id,
                                    "mismatch_count_by_output": mismatch,
                                    "equal": sum(mismatch) == 0,
                                },
                            )
        per_program = {}
        for program_id in program_ids:
            selected = [row for row in rows if row["program_id"] == program_id]
            per_program[program_id] = {
                "observations": len(selected),
                "stock_equal_observations": sum(bool(row["stock_equal"]) for row in selected),
                "mismatch_counts": [sum(row["stock_mismatch_count_by_output"]) for row in selected],
                "stable_hash_by_seed": all(
                    len({tuple(row["output_sha256"]) for row in selected if row["seed"] == seed}) == 1
                    for seed in SEEDS
                ),
            }
        t1_wall = time.time()
        t1_mono = time.monotonic()
        summary = {
            "schema": "c3_resnet18_selected_fullgraph_mismatch_diagnostic_v1",
            "status": "complete_non_spliced_correctness_diagnostic",
            "boot_id": before["BOOT"],
            "programs": per_program,
            "all_selected_equal_stock": all(
                row["stock_equal"] for row in rows if row["program_id"] != "stock_reference"
            ),
            "W0_to_T1_seconds": t1_mono - w0_mono,
            "T0_to_T1_seconds": t1_mono - t0_mono,
            "W0_wall_unix": w0_wall,
            "T0_wall_unix": t0_wall,
            "T1_wall_unix": t1_wall,
            "session_spliced": False,
            "performance_labels_collected": False,
        }
        write_json(output / "summary.json", summary)
        append_jsonl(timeline_path, {"phase": "T1_diagnostic_complete", "W0_elapsed_seconds": t1_mono - w0_mono})
        (output / "command.txt").write_text(
            " ".join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]) + "\n",
            encoding="utf-8",
        )
        finish(output)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    except Exception as error:
        invalid = {
            "schema": "c3_resnet18_selected_fullgraph_mismatch_diagnostic_v1",
            "status": "invalid_entire_diagnostic_session_fail_closed",
            "exception_type": type(error).__name__,
            "message": (str(error) or type(error).__name__)[:4000],
            "traceback": traceback.format_exc()[-12000:],
            "W0_elapsed_seconds": time.monotonic() - w0_mono,
            "may_splice_with_other_session": False,
        }
        write_json(output / "invalid_session.json", invalid)
        write_json(output / "summary.json", invalid)
        append_jsonl(timeline_path, {"phase": "session_invalidated", **invalid})
        (output / "command.txt").write_text(
            " ".join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]) + "\n",
            encoding="utf-8",
        )
        finish(output)
        raise
    finally:
        if remote is not None:
            del remote
            gc.collect()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--session-timeout", type=int, default=600)
    parser.add_argument("--ssh-known-hosts", required=True)
    parser.add_argument("--default-runtime", default="/var/volatile/vta_c3_ram/runtime")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
