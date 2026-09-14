#!/usr/bin/env python3
"""Cross-compile every P4e-qualified VTA candidate for AXU5EVB.

Workers are process-isolated and bounded by a per-candidate timeout.  Exported
shared objects live only in a temporary directory; the ledger retains their
size and SHA-256, not the binaries.  This tool never contacts RPC or a board.
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

import tvm
from tvm import autotvm
from tvm.contrib import cc
import vta

import vta.top.vta_conv2d_residency  # pylint: disable=unused-import
from c3_candidate_identity import canonical_json_bytes, make_failure, normalize_config_entity


RESULT_SCHEMA = "c3_axu_cross_compile_qualification_v1"
SUMMARY_SCHEMA = "c3_axu_cross_compile_summary_v1"
TEMPLATE_NAME = "conv2d_packed_residency.vta"
MODE_NUMBERS = {
    "original": 0,
    "input_stationary": 1,
    "weight_stationary": 2,
    "paper_inspired_hybrid": 3,
}
EXPECTED_PASSED = 165
EXPECTED_INCUMBENTS = 10


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def unique_by(rows, key, source):
    answer = {}
    for row in rows:
        value = row[key]
        if value in answer:
            raise ValueError("duplicate {} in {}: {}".format(key, source, value))
        answer[value] = row
    return answer


def select_candidates(p4b_rows, p4e_rows):
    """Join the exact P4e passing set to complete P4b candidate definitions."""

    p4b = unique_by(p4b_rows, "candidate_id", "P4b")
    p4e = unique_by(p4e_rows, "candidate_id", "P4e")
    selected = []
    for candidate_id, certificate in p4e.items():
        if certificate.get("local_status") != "fsim_passed":
            continue
        if candidate_id not in p4b:
            raise ValueError("P4e candidate missing from P4b: {}".format(candidate_id))
        candidate = p4b[candidate_id]
        if candidate.get("status") != "ok" or candidate.get("tir_sha256") is None:
            raise ValueError("P4e passing candidate is not P4b lower-ok: {}".format(candidate_id))
        if certificate.get("tir_sha256") != candidate.get("tir_sha256"):
            raise ValueError("P4b/P4e TIR hash mismatch: {}".format(candidate_id))
        if certificate.get("candidate_role") != candidate.get("candidate_role"):
            raise ValueError("P4b/P4e role mismatch: {}".format(candidate_id))
        if certificate.get("residence_mode") != candidate.get("residence_mode"):
            raise ValueError("P4b/P4e mode mismatch: {}".format(candidate_id))
        selected.append(candidate)
    selected.sort(key=lambda row: (row["workload_id"], row["candidate_role"], row["candidate_id"]))
    if len(selected) != EXPECTED_PASSED:
        raise ValueError("expected {} P4e passing candidates, found {}".format(EXPECTED_PASSED, len(selected)))
    incumbents = [row for row in selected if row["candidate_role"] == "protected_original_incumbent"]
    if len(incumbents) != EXPECTED_INCUMBENTS:
        raise ValueError("expected {} protected incumbents, found {}".format(EXPECTED_INCUMBENTS, len(incumbents)))
    unsupported = {row["residence_mode"] for row in selected} - set(MODE_NUMBERS)
    if unsupported:
        raise ValueError("unsupported residence modes: {}".format(sorted(unsupported)))
    return selected


def semantic_config_key(value):
    return canonical_json_bytes(normalize_config_entity(value)).decode("utf-8")


def create_task(candidate, env):
    workload = candidate["identity"]["workload"]
    return autotvm.task.create(
        TEMPLATE_NAME,
        args=tuple(workload[1:]) + (MODE_NUMBERS[candidate["residence_mode"]],),
        target=env.target,
        target_host=env.target_host,
    )


def base_result(candidate):
    return {
        "schema": RESULT_SCHEMA,
        "candidate_id": candidate["candidate_id"],
        "workload_id": candidate["workload_id"],
        "candidate_role": candidate["candidate_role"],
        "residence_mode": candidate["residence_mode"],
        "config_index_debug_only": int(candidate["debug"]["config_index"]),
        "tir_sha256": candidate["tir_sha256"],
        "qualification_source": None,
        "identity_check": "not_attempted",
        "tir_hash_check": "not_attempted",
        "build": {"status": "not_attempted", "elapsed_seconds": None},
        "export": {
            "status": "not_attempted",
            "elapsed_seconds": None,
            "binary_size_bytes": None,
            "binary_sha256": None,
            "binary_retained": False,
        },
        "overall_status": "not_attempted",
        "failure": None,
        "board_access": False,
        "performance_measurement": "not_collected",
    }


def fail_result(result, category, message, phase, subcategory, retryable=False, error=None):
    failure = make_failure(category, message[:4000], phase=phase, retryable=retryable)
    failure["subcategory"] = subcategory
    if error is not None:
        failure["exception_type"] = type(error).__name__
    result["failure"] = failure
    result["overall_status"] = "failed"
    return result


def cross_options():
    sysroot = os.environ.get("SDKTARGETSYSROOT")
    if not sysroot:
        raise RuntimeError("SDKTARGETSYSROOT is unset; source the aarch64 SDK environment")
    return [
        "--sysroot=" + sysroot,
        "-Wl,-rpath-link," + sysroot + "/lib",
        "-Wl,-rpath-link," + sysroot + "/usr/lib",
        "-L" + sysroot + "/lib",
        "-L" + sysroot + "/usr/lib",
    ]


def worker_result(candidate):
    result = base_result(candidate)
    result["qualification_source"] = "fresh_isolated_cross_compile"
    try:
        env = vta.get_env()
        if env.TARGET != "axu5evb":
            raise RuntimeError("VTA TARGET must be axu5evb, got {}".format(env.TARGET))
        if str(env.target_host) != "llvm -keys=arm_cpu,cpu -mtriple=aarch64-linux-gnu":
            raise RuntimeError("unexpected AXU5EVB target_host: {}".format(env.target_host))
        options = cross_options()
    except Exception as error:
        return fail_result(
            result,
            "environment",
            str(error) or type(error).__name__,
            "preflight",
            "cross_toolchain_environment",
            error=error,
        )

    try:
        task = create_task(candidate, env)
        config = task.config_space.get(int(candidate["debug"]["config_index"]))
        observed_key = semantic_config_key(config.to_json_dict())
        expected_key = semantic_config_key(candidate["identity"]["complete_config_entity"])
        if observed_key != expected_key:
            raise RuntimeError("ConfigEntity no longer matches P4b identity")
        result["identity_check"] = "passed"
    except Exception as error:
        return fail_result(
            result,
            "environment",
            str(error) or type(error).__name__,
            "identity_check",
            "candidate_identity_mismatch",
            error=error,
        )

    build_start = time.monotonic()
    try:
        with task.target:
            schedule, tensors = task.instantiate(config)
        with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
            lowered = tvm.lower(schedule, tensors, name="main")
        observed_tir_hash = hashlib.sha256(tvm.ir.save_json(lowered).encode("utf-8")).hexdigest()
        result["relowered_tir_sha256"] = observed_tir_hash
        if observed_tir_hash != candidate["tir_sha256"]:
            raise RuntimeError("re-lowered TIR hash differs from P4b/P4e certificate")
        result["tir_hash_check"] = "passed"
        with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
            module = vta.build(
                schedule,
                tensors,
                target=tvm.target.Target(env.target, host=env.target_host),
                name="main",
            )
        result["build"] = {
            "status": "passed",
            "elapsed_seconds": time.monotonic() - build_start,
        }
    except Exception as error:
        result["build"] = {
            "status": "failed",
            "elapsed_seconds": time.monotonic() - build_start,
        }
        return fail_result(
            result,
            "compile",
            str(error) or type(error).__name__,
            "vta_build",
            "vta_build_or_tir_mismatch",
            error=error,
        )

    export_start = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="c3_p4f_binary_") as temporary:
            binary = Path(temporary) / (candidate["candidate_id"] + ".so")
            module.export_library(
                str(binary),
                fcompile=cc.cross_compiler("aarch64-xilinx-linux-g++"),
                options=options,
            )
            result["export"] = {
                "status": "passed",
                "elapsed_seconds": time.monotonic() - export_start,
                "binary_size_bytes": binary.stat().st_size,
                "binary_sha256": sha256_file(binary),
                "binary_retained": False,
            }
        result["overall_status"] = "passed"
        return result
    except Exception as error:
        result["export"] = {
            "status": "failed",
            "elapsed_seconds": time.monotonic() - export_start,
            "binary_size_bytes": None,
            "binary_sha256": None,
            "binary_retained": False,
        }
        return fail_result(
            result,
            "compile",
            str(error) or type(error).__name__,
            "cross_export",
            "aarch64_shared_library_export",
            error=error,
        )


def referenced_incumbent_result(candidate, p2_record):
    result = base_result(candidate)
    if int(candidate["debug"]["config_index"]) != int(p2_record["config_index"]):
        raise ValueError("P2 incumbent config mismatch for {}".format(candidate["workload_id"]))
    result.update(
        qualification_source="referenced_p2_run02_cross_compile",
        identity_check="passed_by_workload_config_join",
        tir_hash_check="passed_by_p4e_p2_tir_certificate",
        build={"status": "referenced_passed", "elapsed_seconds": None},
        export={
            "status": "referenced_passed",
            "elapsed_seconds": None,
            "binary_size_bytes": int(p2_record["binary_size_bytes"]),
            "binary_sha256": p2_record["binary_sha256"],
            "binary_retained": False,
        },
        overall_status="passed",
    )
    return result


def timeout_result(candidate, seconds, phase):
    result = base_result(candidate)
    result["qualification_source"] = "fresh_isolated_cross_compile"
    return fail_result(
        result,
        "timeout",
        "candidate exceeded {} second cross-compile timeout".format(seconds),
        phase,
        "candidate_worker_timeout",
        retryable=True,
    )


def summarize(records, expected):
    categories = Counter(
        row["failure"]["category"] for row in records if row.get("failure") is not None
    )
    subcategories = Counter(
        row["failure"].get("subcategory", "unspecified")
        for row in records
        if row.get("failure") is not None
    )
    modes = {}
    for mode in MODE_NUMBERS:
        rows = [row for row in records if row["residence_mode"] == mode]
        modes[mode] = {
            "records": len(rows),
            "passed": sum(row["overall_status"] == "passed" for row in rows),
            "failed": sum(row["overall_status"] == "failed" for row in rows),
        }
    return {
        "schema": SUMMARY_SCHEMA,
        "status": "completed" if len(records) == expected else "partial",
        "scope": "local AXU5EVB build/export qualification; no board or performance",
        "selected_p4e_fsim_passed": expected,
        "result_records": len(records),
        "passed": sum(row["overall_status"] == "passed" for row in records),
        "failed": sum(row["overall_status"] == "failed" for row in records),
        "fresh_cross_compile_records": sum(
            row["qualification_source"] == "fresh_isolated_cross_compile" for row in records
        ),
        "referenced_p2_incumbents": sum(
            row["qualification_source"] == "referenced_p2_run02_cross_compile" for row in records
        ),
        "failure_categories": dict(sorted(categories.items())),
        "failure_subcategories": dict(sorted(subcategories.items())),
        "modes": modes,
        "binary_retained_count": sum(row["export"]["binary_retained"] for row in records),
        "board_access": False,
        "performance_measurement": "not_collected",
    }


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def find_candidate(p4b_path, p4e_path, candidate_id):
    selected = select_candidates(load_jsonl(p4b_path), load_jsonl(p4e_path))
    matches = [row for row in selected if row["candidate_id"] == candidate_id]
    if len(matches) != 1:
        raise ValueError("worker candidate lookup returned {} rows".format(len(matches)))
    if matches[0]["candidate_role"] == "protected_original_incumbent":
        raise ValueError("protected incumbents must cite P2 rather than enter a worker")
    return matches[0]


def worker_main(args):
    candidate = find_candidate(args.p4b_results, args.p4e_results, args.worker_candidate_id)
    try:
        result = worker_result(candidate)
    except Exception as error:  # Last-resort worker boundary.
        result = base_result(candidate)
        result["qualification_source"] = "fresh_isolated_cross_compile"
        fail_result(
            result,
            "environment",
            str(error) or type(error).__name__,
            "worker_uncaught",
            "uncaught_worker_exception",
            error=error,
        )
    write_json(args.worker_output, result)


def guard_hashes(paths):
    return {str(path): sha256_file(path) for path in map(Path, paths)}


def main_run(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    candidates = select_candidates(load_jsonl(args.p4b_results), load_jsonl(args.p4e_results))
    p2 = unique_by(load_json(args.p2_cross_results), "workload_id", "P2 cross results")
    guards_before = guard_hashes(args.guard_files)
    results_path = output / "results.jsonl"
    records, stdout_lines, stderr_lines = [], [], []
    started = time.monotonic()
    with open(results_path, "w", encoding="utf-8") as result_stream:
        for position, candidate in enumerate(candidates, 1):
            if candidate["candidate_role"] == "protected_original_incumbent":
                result = referenced_incumbent_result(candidate, p2[candidate["workload_id"]])
            else:
                remaining = args.overall_timeout_seconds - (time.monotonic() - started)
                if remaining <= 0:
                    result = timeout_result(candidate, args.overall_timeout_seconds, "global_deadline")
                else:
                    with tempfile.TemporaryDirectory(prefix="c3_p4f_worker_") as temporary:
                        worker_output = Path(temporary) / "result.json"
                        command = [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "--p4b-results",
                            str(Path(args.p4b_results).resolve()),
                            "--p4e-results",
                            str(Path(args.p4e_results).resolve()),
                            "--worker-candidate-id",
                            candidate["candidate_id"],
                            "--worker-output",
                            str(worker_output),
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
                                result["qualification_source"] = "fresh_isolated_cross_compile"
                                fail_result(
                                    result,
                                    "environment",
                                    "worker exit={} output_present={}".format(
                                        completed.returncode, worker_output.is_file()
                                    ),
                                    "worker_process",
                                    "worker_process_failure",
                                )
                            else:
                                result = load_json(worker_output)
                        except subprocess.TimeoutExpired:
                            result = timeout_result(
                                candidate, args.candidate_timeout_seconds, "candidate_worker"
                            )
            records.append(result)
            result_stream.write(json.dumps(result, sort_keys=True) + "\n")
            result_stream.flush()
            stdout_lines.append(
                "{}/{} {} {} {} {}".format(
                    position,
                    len(candidates),
                    candidate["workload_id"],
                    candidate["residence_mode"],
                    result["qualification_source"],
                    result["overall_status"],
                )
            )

    guards_after = guard_hashes(args.guard_files)
    if guards_before != guards_after:
        raise RuntimeError("production source guard changed during qualification")
    summary = summarize(records, len(candidates))
    summary["wall_clock_seconds"] = time.monotonic() - started
    write_json(output / "summary.json", summary)
    (output / "stdout.log").write_text("\n".join(stdout_lines) + "\n", encoding="utf-8")
    (output / "stderr.log").write_text("".join(stderr_lines), encoding="utf-8")
    (output / "command.txt").write_text(" ".join([sys.executable] + sys.argv) + "\n", encoding="utf-8")
    compiler_version = subprocess.run(
        ["aarch64-xilinx-linux-g++", "--version"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.splitlines()[0]
    manifest = {
        "schema": "c3_p4f_axu_cross_compile_manifest_v1",
        "date": "2026-09-11",
        "environment": {
            "python": sys.executable,
            "python_version": platform.python_version(),
            "tvm_version": tvm.__version__,
            "vta_target": vta.get_env().TARGET,
            "target_host": str(vta.get_env().target_host),
            "cross_compiler": compiler_version,
            "sysroot": os.environ.get("SDKTARGETSYSROOT"),
            "ssh_used": False,
            "rpc_used": False,
            "board_contacted": False,
        },
        "inputs": {
            "p4b_results": {"path": args.p4b_results, "sha256": sha256_file(args.p4b_results)},
            "p4e_results": {"path": args.p4e_results, "sha256": sha256_file(args.p4e_results)},
            "p2_cross_results": {
                "path": args.p2_cross_results,
                "sha256": sha256_file(args.p2_cross_results),
            },
        },
        "source_guards_before": guards_before,
        "source_guards_after": guards_after,
        "source": {"path": str(Path(__file__)), "sha256": sha256_file(__file__)},
    }
    write_json(output / "manifest.json", manifest)
    write_json(
        output / "preregistered.json",
        {
            "schema": "c3_p4f_axu_cross_compile_protocol_v1",
            "selection": "exact P4e local_status=fsim_passed joined to P4b by candidate_id/TIR hash",
            "expected_candidates": EXPECTED_PASSED,
            "protected_incumbents": "cite exact P2 run02 workload/config cross-compile evidence",
            "fresh_candidates": EXPECTED_PASSED - EXPECTED_INCUMBENTS,
            "worker_isolation": "one subprocess and temporary binary directory per fresh candidate",
            "candidate_timeout_seconds": args.candidate_timeout_seconds,
            "overall_timeout_seconds": args.overall_timeout_seconds,
            "failure_categories": ["lower", "compile", "wrong_answer", "timeout", "rpc", "environment"],
            "binary_retention": "size/hash only; no candidate .so retained",
            "claim": "deployability qualification only; performance use forbidden",
        },
    )
    status = (
        "# C3-P4f AXU5EVB cross-compile qualification\n\n"
        "- Status: `{}`\n"
        "- P4e FSim-passed candidates: {}\n"
        "- Passed build/export qualification: {}\n"
        "- Failed: {}\n"
        "- Fresh isolated cross-compiles: {}\n"
        "- P2 run02 incumbent citations: {}\n"
        "- Candidate binaries retained: {}\n"
        "- SSH/RPC/board: not used\n"
        "- Performance: not measured; this proves build/export deployability only\n"
    ).format(
        summary["status"],
        summary["selected_p4e_fsim_passed"],
        summary["passed"],
        summary["failed"],
        summary["fresh_cross_compile_records"],
        summary["referenced_p2_incumbents"],
        summary["binary_retained_count"],
    )
    (output / "STATUS.md").write_text(status, encoding="utf-8")
    (output / "HANDOFF.md").write_text(
        "# HANDOFF\n\n"
        "`results.jsonl` is the candidate-level AXU5EVB build/export ledger. Fresh candidates "
        "were process-isolated, re-lowered with an exact TIR-hash check, built, and exported "
        "with the aarch64 Xilinx toolchain. Protected incumbents cite P2 run02. Shared objects "
        "were deleted with their temporary directories; only size/hash remain. Passing this "
        "stage does not establish board correctness, loadability on the current boot, latency, "
        "or FPS.\n",
        encoding="utf-8",
    )
    artifacts = {
        path.name: sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    test_source = Path(__file__).with_name("test_qualify_vta_axu_cross_compile.py")
    sources = {str(Path(__file__)): sha256_file(__file__)}
    if test_source.is_file():
        sources[str(test_source)] = sha256_file(test_source)
    write_json(output / "artifact_hashes.json", {"output_sha256": artifacts, "source_sha256": sources})
    print(json.dumps(summary, sort_keys=True))


def parse_args():
    base = Path(__file__).resolve().parent / "report_out" / "stage_tile_cotuning"
    p4 = base / "c3_dma_residency_autotune" / "04_dma_command_signatures"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--p4b-results",
        default=str(p4 / "20260910_p4b_local_residency_pool_run01" / "results.jsonl"),
    )
    parser.add_argument(
        "--p4e-results",
        default=str(p4 / "20260911_p4e_unified_local_qualification_run01" / "results.jsonl"),
    )
    parser.add_argument(
        "--p2-cross-results",
        default=str(
            base
            / "c3_dma_residency_autotune"
            / "02_baselines"
            / "20260910_p2_local_baseline_run02"
            / "cross_compile_results.json"
        ),
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--candidate-timeout-seconds", type=int, default=180)
    parser.add_argument("--overall-timeout-seconds", type=int, default=7200)
    parser.add_argument(
        "--guard-files",
        nargs="+",
        default=[
            str(Path(__file__).parents[2] / "python" / "vta" / "top" / "vta_conv2d.py"),
            str(
                Path(__file__).parents[2]
                / "python"
                / "vta"
                / "top"
                / "vta_conv2d_residency.py"
            ),
            str(Path(__file__).parents[2] / "python" / "vta" / "transform.py"),
            str(Path(__file__).parents[2] / "runtime" / "runtime.cc"),
            str(Path(__file__).parents[3] / "src" / "tir" / "transforms" / "coproc_sync.cc"),
            str(Path(__file__).parents[3] / "build" / "libtvm.so"),
        ],
    )
    parser.add_argument("--worker-candidate-id")
    parser.add_argument("--worker-output")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.worker_candidate_id:
        if not args.worker_output:
            raise ValueError("--worker-output required in worker mode")
        worker_main(args)
    else:
        if not args.output_dir:
            raise ValueError("--output-dir required in main mode")
        main_run(args)


if __name__ == "__main__":
    main()
