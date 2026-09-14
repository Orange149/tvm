#!/usr/bin/env python3
"""Correctness-qualify P4b lowering-success residency candidates in local FSim.

Each candidate is built exactly once inside a bounded worker process, then run
against three independently generated NumPy references.  No execution timing is
collected or used as a performance signal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import tvm
from tvm import autotvm, rpc
from tvm.contrib import utils
import vta

import vta.top.vta_conv2d_residency  # pylint: disable=unused-import
from c3_candidate_identity import canonical_json_bytes, make_failure, normalize_config_entity


RESULT_SCHEMA = "c3_residency_fsim_qualification_v1"
SUMMARY_SCHEMA = "c3_residency_fsim_summary_v1"
TEMPLATE_NAME = "conv2d_packed_residency.vta"
SEEDS = (0, 20250901, 20260910)
MODE_NUMBERS = {
    "original": 0,
    "input_stationary": 1,
    "weight_stationary": 2,
    "paper_inspired_hybrid": 3,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def select_candidates(rows, roles=("residency_experiment",)):
    roles = frozenset(roles)
    selected = [
        row
        for row in rows
        if row.get("candidate_role") in roles and row.get("status") == "ok"
    ]
    selected.sort(key=lambda row: (row["workload_id"], row["residence_mode"], row["candidate_id"]))
    ids = [row["candidate_id"] for row in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("P4b lowering-success selection has duplicate candidate IDs")
    unsupported = set(row["residence_mode"] for row in selected) - set(MODE_NUMBERS)
    if unsupported:
        raise ValueError("unsupported residency modes: {}".format(sorted(unsupported)))
    return selected


def create_task(candidate, env):
    workload = candidate["identity"]["workload"]
    mode = candidate["residence_mode"]
    return autotvm.task.create(
        TEMPLATE_NAME,
        args=tuple(workload[1:]) + (MODE_NUMBERS[mode],),
        target=env.target,
        target_host=env.target_host,
    )


def semantic_config_key(config):
    return canonical_json_bytes(normalize_config_entity(config)).decode("utf-8")


def reference_data_from_workload(workload, seed):
    """Independent packed-int8 NumPy reference; does not call the TE schedule."""

    data_arg, weight_arg, strides, padding, dilation, _, out_dtype = workload[1:]
    if tuple(dilation) != (1, 1) or out_dtype != "int32":
        raise ValueError("reference supports dilation=1 and int32 accumulation only")
    data_shape, weight_shape = tuple(data_arg[1]), tuple(weight_arg[1])
    if data_arg[2] != "int8" or weight_arg[2] != "int8":
        raise ValueError("reference requires int8 data and weights")
    batch_outer, ci_outer, height, width, batch_inner, block_in = data_shape
    co_outer, kernel_ci, kernel_h, kernel_w, block_out, kernel_block_in = weight_shape
    if ci_outer != kernel_ci or block_in != kernel_block_in:
        raise ValueError("packed channel mismatch")
    rng = np.random.default_rng(int(seed))
    data = rng.integers(-32, 33, data_shape, dtype=np.int8)
    weight = rng.integers(-8, 9, weight_shape, dtype=np.int8)
    nchw = data.transpose(0, 4, 1, 5, 2, 3).reshape(
        batch_outer * batch_inner, ci_outer * block_in, height, width
    )
    kernel = weight.transpose(0, 4, 1, 5, 2, 3).reshape(
        co_outer * block_out, ci_outer * block_in, kernel_h, kernel_w
    )
    if len(padding) == 2:
        pad_top, pad_left = padding
        pad_bottom, pad_right = pad_top, pad_left
    else:
        pad_top, pad_left, pad_bottom, pad_right = padding
    padded = np.pad(
        nchw.astype("int32"),
        ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
    )
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, (kernel_h, kernel_w), axis=(2, 3)
    )
    windows = windows[:, :, :: strides[0], :: strides[1], :, :]
    accum = np.einsum(
        "ncyxij,ocij->noyx", windows, kernel.astype("int32"), optimize=True
    )
    result = np.clip(accum >> 8, 0, 127).astype("int8")
    output_h, output_w = result.shape[2:]
    packed = result.reshape(
        batch_outer, batch_inner, co_outer, block_out, output_h, output_w
    ).transpose(0, 2, 4, 5, 1, 3)
    return data, weight, np.ascontiguousarray(packed)


def array_sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def base_result(candidate):
    return {
        "schema": RESULT_SCHEMA,
        "candidate_id": candidate["candidate_id"],
        "workload_id": candidate["workload_id"],
        "residence_mode": candidate["residence_mode"],
        "p4b_tir_sha256": candidate["tir_sha256"],
        "p4b_tir_hash_encoding": candidate.get("tir_hash_encoding"),
        "build": {"status": "not_attempted", "count": 0, "failure": None},
        "seeds": [],
        "overall_status": "not_attempted",
        "failure": None,
        "performance_measurement": "not_collected",
    }


def worker_result(candidate):
    result = base_result(candidate)
    env = vta.get_env()
    if env.TARGET != "sim":
        result["failure"] = make_failure(
            "environment", "VTA TARGET must be sim for local FSim qualification", phase="preflight"
        )
        result["overall_status"] = "failed"
        return result
    try:
        from vta.testing import simulator

        if not simulator.enabled():
            raise RuntimeError("VTA simulator runtime is not enabled")
    except Exception as error:
        result["failure"] = make_failure(
            "environment", str(error) or type(error).__name__, phase="preflight"
        )
        result["overall_status"] = "failed"
        return result

    task = create_task(candidate, env)
    config_index = int(candidate["debug"]["config_index"])
    config = task.config_space.get(config_index)
    expected_config = candidate["identity"]["complete_config_entity"]
    if semantic_config_key(config.to_json_dict()) != semantic_config_key(expected_config):
        result["failure"] = make_failure(
            "environment", "ConfigEntity no longer matches P4b identity", phase="identity_check"
        )
        result["overall_status"] = "failed"
        return result

    result["build"]["count"] = 1
    try:
        with task.target:
            schedule, tensors = task.instantiate(config)
        with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
            lowered = tvm.lower(schedule, tensors, name="main")
        relowered_hash = hashlib.sha256(tvm.ir.save_json(lowered).encode("utf-8")).hexdigest()
        result["relowered_tir_sha256"] = relowered_hash
        result["tir_hash_matches_p4b"] = relowered_hash == candidate["tir_sha256"]
        if not result["tir_hash_matches_p4b"]:
            raise RuntimeError("re-lowered TIR hash differs from P4b")
        with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
            module = vta.build(
                schedule,
                tensors,
                target=tvm.target.Target(env.target, host=env.target_host),
                name="main",
            )
        result["build"]["status"] = "ok"
    except Exception as error:
        failure = make_failure(
            "compile", (str(error) or type(error).__name__)[:4000], phase="fsim_build"
        )
        failure["exception_type"] = type(error).__name__
        result["build"].update(status="failed", failure=failure)
        result["failure"] = failure
        result["overall_status"] = "failed"
        return result

    try:
        temporary = utils.tempdir()
        module_path = temporary.relpath("{}.o".format(candidate["candidate_id"]))
        module.save(module_path)
        remote = rpc.LocalSession()
        remote.upload(module_path)
        function = remote.load_module(Path(module_path).name)["main"]
        device = remote.ext_dev(0)
    except Exception as error:
        failure = make_failure(
            "environment", (str(error) or type(error).__name__)[:4000], phase="fsim_load"
        )
        failure["exception_type"] = type(error).__name__
        result["failure"] = failure
        result["overall_status"] = "failed"
        return result

    workload = candidate["identity"]["workload"]
    for seed in SEEDS:
        seed_record = {"seed": seed, "status": "not_attempted", "correct": None, "failure": None}
        try:
            data, weight, expected = reference_data_from_workload(workload, seed)
            output = tvm.nd.empty(expected.shape, "int8", device=device)
            function(tvm.nd.array(data, device), tvm.nd.array(weight, device), output)
            actual = output.numpy()
            mismatches = int(np.count_nonzero(actual != expected))
            seed_record.update(
                {
                    "status": "passed" if mismatches == 0 else "failed",
                    "correct": mismatches == 0,
                    "mismatch_count": mismatches,
                    "max_absolute_error": int(
                        np.max(np.abs(actual.astype("int32") - expected.astype("int32")))
                    ),
                    "expected_sha256": array_sha256(expected),
                    "actual_sha256": array_sha256(actual),
                }
            )
            if mismatches:
                seed_record["failure"] = make_failure(
                    "wrong_answer",
                    "{} output elements differ from NumPy reference".format(mismatches),
                    phase="fsim_execute",
                )
        except Exception as error:
            failure = make_failure(
                "environment", (str(error) or type(error).__name__)[:4000], phase="fsim_execute"
            )
            failure["exception_type"] = type(error).__name__
            seed_record.update(status="failed", correct=False, failure=failure)
        result["seeds"].append(seed_record)

    result["overall_status"] = (
        "passed" if len(result["seeds"]) == len(SEEDS) and all(x["correct"] for x in result["seeds"])
        else "failed"
    )
    if result["overall_status"] == "failed":
        result["failure"] = next(
            (item["failure"] for item in result["seeds"] if item.get("failure")),
            make_failure("wrong_answer", "one or more seeds failed", phase="fsim_execute"),
        )
    return result


def summarize(records, selected_count):
    modes = {}
    for mode in MODE_NUMBERS:
        rows = [row for row in records if row["residence_mode"] == mode]
        categories = Counter(
            row["failure"]["category"] for row in rows if row.get("failure") is not None
        )
        modes[mode] = {
            "selected": len(rows),
            "passed_all_seeds": sum(row["overall_status"] == "passed" for row in rows),
            "failed": sum(row["overall_status"] == "failed" for row in rows),
            "timed_out": sum(
                row.get("failure", {}).get("category") == "timeout"
                for row in rows
                if row.get("failure")
            ),
            "failure_categories": dict(sorted(categories.items())),
        }
    workloads = {}
    for workload_id in sorted({row["workload_id"] for row in records}):
        rows = [row for row in records if row["workload_id"] == workload_id]
        workloads[workload_id] = {
            mode: {
                "selected": sum(row["residence_mode"] == mode for row in rows),
                "passed_all_seeds": sum(
                    row["residence_mode"] == mode and row["overall_status"] == "passed"
                    for row in rows
                ),
            }
            for mode in MODE_NUMBERS
        }
    return {
        "schema": SUMMARY_SCHEMA,
        "status": "completed" if len(records) == selected_count else "partial",
        "scope": "local FSim correctness only; no performance timing collected",
        "frozen_seeds": list(SEEDS),
        "selected_candidates": selected_count,
        "result_records": len(records),
        "passed_all_seeds": sum(row["overall_status"] == "passed" for row in records),
        "failed": sum(row["overall_status"] == "failed" for row in records),
        "not_attempted": sum(row["overall_status"] == "not_attempted" for row in records),
        "modes": modes,
        "workloads": workloads,
    }


def timeout_result(candidate, message, phase):
    result = base_result(candidate)
    failure = make_failure("timeout", message, phase=phase, retryable=True)
    result["failure"] = failure
    result["overall_status"] = "failed"
    return result


def find_candidate(path, candidate_id, roles=("residency_experiment",)):
    selected = select_candidates(load_jsonl(path), roles=roles)
    matches = [row for row in selected if row["candidate_id"] == candidate_id]
    if len(matches) != 1:
        raise ValueError("candidate ID lookup returned {} rows".format(len(matches)))
    return matches[0]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def run_worker(args):
    candidate = find_candidate(args.p4b_results, args.worker_candidate_id, roles=args.roles)
    try:
        result = worker_result(candidate)
    except Exception as error:
        result = base_result(candidate)
        failure = make_failure(
            "environment", (str(error) or type(error).__name__)[:4000], phase="worker_uncaught"
        )
        failure["exception_type"] = type(error).__name__
        result.update(overall_status="failed", failure=failure)
    write_json(args.worker_output, result)


def run_main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(args.p4b_results)
    candidates = select_candidates(rows, roles=args.roles)
    if args.limit is not None:
        candidates = candidates[: args.limit]
    results_path = output_dir / "results.jsonl"
    results_path.write_text("")
    stdout_lines, stderr_lines, records = [], [], []
    start = time.monotonic()

    for position, candidate in enumerate(candidates):
        remaining = args.overall_timeout_seconds - (time.monotonic() - start)
        if remaining <= 0:
            result = timeout_result(
                candidate, "overall qualification deadline reached before candidate", "global_deadline"
            )
        else:
            with tempfile.TemporaryDirectory(prefix="c3_p4c_worker_") as directory:
                worker_output = Path(directory) / "result.json"
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--p4b-results",
                    str(Path(args.p4b_results).resolve()),
                    "--worker-candidate-id",
                    candidate["candidate_id"],
                    "--worker-output",
                    str(worker_output),
                    "--roles",
                    ",".join(args.roles),
                ]
                try:
                    completed = subprocess.run(
                        command,
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=min(args.candidate_timeout_seconds, max(1, int(remaining))),
                        env=os.environ.copy(),
                    )
                    if completed.stderr.strip():
                        stderr_lines.append(
                            "candidate={} stderr={}\n".format(
                                candidate["candidate_id"], completed.stderr[-4000:]
                            )
                        )
                    if completed.returncode != 0 or not worker_output.is_file():
                        result = base_result(candidate)
                        failure = make_failure(
                            "environment",
                            "worker exit={} output_present={}".format(
                                completed.returncode, worker_output.is_file()
                            ),
                            phase="worker_process",
                        )
                        result.update(overall_status="failed", failure=failure)
                    else:
                        result = json.loads(worker_output.read_text())
                except subprocess.TimeoutExpired:
                    result = timeout_result(
                        candidate,
                        "candidate exceeded {} second correctness timeout".format(
                            args.candidate_timeout_seconds
                        ),
                        "candidate_worker",
                    )
        records.append(result)
        with open(results_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, sort_keys=True) + "\n")
            stream.flush()
        stdout_lines.append(
            "{}/{} {} {} {}".format(
                position + 1,
                len(candidates),
                candidate["workload_id"],
                candidate["residence_mode"],
                result["overall_status"],
            )
        )

    summary = summarize(records, len(candidates))
    write_json(output_dir / "summary.json", summary)
    (output_dir / "stdout.log").write_text("\n".join(stdout_lines) + "\n")
    (output_dir / "stderr.log").write_text("\n".join(stderr_lines))
    (output_dir / "command.txt").write_text(" ".join([sys.executable] + sys.argv) + "\n")
    write_json(
        output_dir / "preregistered.json",
        {
            "schema": "c3_p4c_preregistered_v1",
            "selection": "P4b records with status=ok and candidate_role in {}".format(
                list(args.roles)
            ),
            "frozen_seeds": list(SEEDS),
            "builds_per_candidate": 1,
            "oracle": "independent NumPy packed int8 convolution, shift-right 8, clip [0,127]",
            "correctness": "exact elementwise equality for all three seeds",
            "candidate_timeout_seconds": args.candidate_timeout_seconds,
            "overall_timeout_seconds": args.overall_timeout_seconds,
            "performance_rule": "FSim time is never collected or used",
        },
    )
    env = vta.get_env()
    write_json(
        output_dir / "manifest.json",
        {
            "schema": "c3_p4c_local_manifest_v1",
            "date": "2026-09-10",
            "environment": {
                "python": sys.executable,
                "python_version": platform.python_version(),
                "tvm_version": tvm.__version__,
                "vta_target": env.TARGET,
                "vta_hw_path": os.environ.get("VTA_HW_PATH"),
                "ssh_used": False,
                "board_contacted": False,
                "performance_timing_collected": False,
            },
            "input": {
                "p4b_results": args.p4b_results,
                "sha256": sha256_file(args.p4b_results),
                "selected_lowering_success": len(candidates),
                "candidate_roles": list(args.roles),
            },
            "source": {
                "qualifier": str(Path(__file__)),
                "sha256": sha256_file(Path(__file__)),
            },
            "summary": {
                "records": len(records),
                "passed_all_seeds": summary["passed_all_seeds"],
                "failed": summary["failed"],
            },
        },
    )
    status = [
        "# C3-P4c local FSim qualification run 01",
        "",
        "- Status: `{}`".format(summary["status"]),
        "- Selected P4b lowering-success candidates: {}".format(len(candidates)),
        "- Passed all three seeds: {}".format(summary["passed_all_seeds"]),
        "- Failed: {}".format(summary["failed"]),
        "- Frozen seeds: `{}`".format(list(SEEDS)),
        "- Build policy: exactly once per candidate",
        "- Oracle: independent NumPy exact elementwise comparison",
        "- SSH/board: not used",
        "- FSim performance timing: not collected",
        "",
    ]
    for mode, data in summary["modes"].items():
        status.append(
            "- {}: {}/{} passed all seeds; failures {}".format(
                mode, data["passed_all_seeds"], data["selected"], data["failure_categories"]
            )
        )
    (output_dir / "STATUS.md").write_text("\n".join(status) + "\n")
    (output_dir / "HANDOFF.md").write_text(
        "# HANDOFF\n\n"
        "`results.jsonl` is the candidate-level correctness ledger joined by candidate ID and "
        "P4b TIR hash. Each successful worker built once and tested seeds 0, 20250901, and "
        "20260910 against an independent NumPy reference. `summary.json` gives mode/workload "
        "counts. These are correctness qualifications only; no FSim latency is present or valid "
        "for performance ranking.\n"
    )
    artifacts = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "artifact_hashes.json":
            artifacts[path.name] = sha256_file(path)
    test_source = Path(__file__).with_name("test_qualify_vta_residency_fsim.py")
    source_hashes = {str(Path(__file__)): sha256_file(Path(__file__))}
    if test_source.is_file():
        source_hashes[str(test_source)] = sha256_file(test_source)
    write_json(
        output_dir / "artifact_hashes.json",
        {"output_sha256": artifacts, "source_sha256": source_hashes},
    )
    print(
        "selected={} passed={} failed={}".format(
            len(candidates), summary["passed_all_seeds"], summary["failed"]
        )
    )


def parse_args():
    base = Path(__file__).resolve().parent / "report_out" / "stage_tile_cotuning"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--p4b-results",
        default=str(
            base
            / "c3_dma_residency_autotune"
            / "04_dma_command_signatures"
            / "20260910_p4b_local_residency_pool_run01"
            / "results.jsonl"
        ),
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--candidate-timeout-seconds", type=int, default=120)
    parser.add_argument("--overall-timeout-seconds", type=int, default=3600)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--roles",
        type=lambda value: tuple(item for item in value.split(",") if item),
        default=("residency_experiment",),
        help="Comma-separated P4b candidate_role values; defaults to residency_experiment",
    )
    parser.add_argument("--worker-candidate-id")
    parser.add_argument("--worker-output")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.worker_candidate_id:
        if not args.worker_output:
            raise ValueError("--worker-output is required in worker mode")
        run_worker(args)
    else:
        if not args.output_dir:
            raise ValueError("--output-dir is required")
        run_main(args)


if __name__ == "__main__":
    main()
