#!/usr/bin/env python3
"""Cross-compile the P4g-qualified weight-barrier pool for AXU5EVB.

Every candidate is re-identified, re-lowered, built, and exported in a bounded
subprocess.  Candidate shared objects exist only in temporary directories.  This
tool never uses SSH, RPC, a board, or execution timing as a performance signal.
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


RESULT_SCHEMA = "c3_weight_barrier_axu_cross_compile_v1"
SUMMARY_SCHEMA = "c3_weight_barrier_axu_cross_compile_summary_v1"
EXPECTED_CANDIDATES = 32
PUBLIC_MODE = "weight_stationary_barrier"
IMPLEMENTATION_MODE = 4
TEMPLATE_NAME = "conv2d_packed_residency.vta"


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


def semantic_config_key(value):
    return canonical_json_bytes(normalize_config_entity(value)).decode("utf-8")


def expected_source_hashes(repo_root):
    paths = {
        "vta_conv2d.py": repo_root / "vta/python/vta/top/vta_conv2d.py",
        "vta_conv2d_residency.py": repo_root / "vta/python/vta/top/vta_conv2d_residency.py",
        "transform.py": repo_root / "vta/python/vta/transform.py",
        "coproc_sync.cc": repo_root / "src/tir/transforms/coproc_sync.cc",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def validate_p4g_artifacts(results_path):
    results_path = Path(results_path).resolve()
    run_dir = results_path.parent
    ledger_path = run_dir / "artifact_hashes.json"
    ledger = load_json(ledger_path)
    expected = ledger.get("output_sha256", {})
    if expected.get("results.jsonl") != sha256_file(results_path):
        raise ValueError("P4g results hash mismatch")
    verified = {}
    for name, wanted in expected.items():
        path = run_dir / name
        if not path.is_file():
            raise ValueError("P4g artifact is missing: {}".format(name))
        observed = sha256_file(path)
        if observed != wanted:
            raise ValueError("P4g artifact hash mismatch: {}".format(name))
        verified[name] = observed
    return {
        "run_dir": str(run_dir),
        "artifact_hashes_path": str(ledger_path),
        "artifact_hashes_sha256": sha256_file(ledger_path),
        "verified_output_sha256": verified,
    }


def select_candidates(rows, source_hashes):
    selected = []
    seen = set()
    for row in rows:
        qualification = row.get("fsim_qualification") or {}
        audit = row.get("dependency_audit") or {}
        if not (
            row.get("status") == "ok"
            and row.get("lower_status") == "ok"
            and audit.get("valid") is True
            and qualification.get("overall_status") == "passed"
        ):
            continue
        candidate_id = row.get("candidate_id")
        if candidate_id in seen:
            raise ValueError("duplicate P4g candidate ID: {}".format(candidate_id))
        seen.add(candidate_id)
        if row.get("residence_mode") != PUBLIC_MODE or row.get("implementation_mode") != 4:
            raise ValueError("unexpected P4g mode: {}".format(candidate_id))
        identity = row.get("identity")
        recomputed = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
        if recomputed != candidate_id:
            raise ValueError("stable ID mismatch: {}".format(candidate_id))
        if identity.get("complete_config_entity") != normalize_config_entity(
            identity.get("complete_config_entity")
        ):
            raise ValueError("identity ConfigEntity contains debug-only fields")
        if identity.get("schedule_source_sha256") != source_hashes:
            raise ValueError("schedule/compiler source hash mismatch: {}".format(candidate_id))
        if qualification.get("tir_hash_matches_static") is not True:
            raise ValueError("P4g FSim TIR certificate is false: {}".format(candidate_id))
        if qualification.get("relowered_tir_sha256") != row.get("tir_sha256"):
            raise ValueError("P4g static/FSim TIR hash mismatch: {}".format(candidate_id))
        selected.append(row)
    selected.sort(key=lambda row: (row["workload_id"], row["candidate_id"]))
    if len(selected) != EXPECTED_CANDIDATES:
        raise ValueError(
            "expected {} fully qualified P4g candidates, found {}".format(
                EXPECTED_CANDIDATES, len(selected)
            )
        )
    return selected


def create_task(candidate, env):
    workload = candidate["identity"]["workload"]
    return autotvm.task.create(
        TEMPLATE_NAME,
        args=tuple(workload[1:]) + (IMPLEMENTATION_MODE,),
        target=env.target,
        target_host=env.target_host,
    )


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


def base_result(candidate):
    return {
        "schema": RESULT_SCHEMA,
        "candidate_id": candidate["candidate_id"],
        "workload_id": candidate["workload_id"],
        "candidate_role": candidate["candidate_role"],
        "residence_mode": candidate["residence_mode"],
        "config_index_debug_only": int(candidate["debug"]["config_index"]),
        "p4g_tir_sha256": candidate["tir_sha256"],
        "stable_id_check": "not_attempted",
        "source_hash_check": "not_attempted",
        "complete_config_entity_check": "not_attempted",
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
        "ssh_used": False,
        "rpc_used": False,
        "board_access": False,
        "performance_measurement": "not_collected",
    }


def fail(result, category, message, phase, subcategory, retryable=False, error=None):
    failure = make_failure(category, (message or category)[:4000], phase, retryable)
    failure["subcategory"] = subcategory
    if error is not None:
        failure["exception_type"] = type(error).__name__
    result.update(overall_status="failed", failure=failure)
    return result


def worker_result(candidate, source_hashes):
    result = base_result(candidate)
    try:
        env = vta.get_env()
        if env.TARGET != "axu5evb":
            raise RuntimeError("VTA TARGET must be axu5evb, got {}".format(env.TARGET))
        expected_host = "llvm -keys=arm_cpu,cpu -mtriple=aarch64-linux-gnu"
        if str(env.target_host) != expected_host:
            raise RuntimeError("unexpected AXU5EVB target_host: {}".format(env.target_host))
        options = cross_options()
    except Exception as error:
        return fail(
            result,
            "environment",
            str(error),
            "preflight",
            "cross_toolchain_environment",
            error=error,
        )

    try:
        recomputed = hashlib.sha256(canonical_json_bytes(candidate["identity"])).hexdigest()
        if recomputed != candidate["candidate_id"]:
            raise RuntimeError("stable candidate ID no longer matches identity")
        result["stable_id_check"] = "passed"
        if candidate["identity"]["schedule_source_sha256"] != source_hashes:
            raise RuntimeError("schedule/compiler source hashes changed")
        result["source_hash_check"] = "passed"
        task = create_task(candidate, env)
        config = task.config_space.get(int(candidate["debug"]["config_index"]))
        observed = semantic_config_key(config.to_json_dict())
        expected = semantic_config_key(candidate["identity"]["complete_config_entity"])
        if observed != expected:
            raise RuntimeError("complete ConfigEntity no longer matches stable identity")
        result["complete_config_entity_check"] = "passed"
    except Exception as error:
        return fail(
            result,
            "environment",
            str(error),
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
        relowered_hash = hashlib.sha256(tvm.ir.save_json(lowered).encode("utf-8")).hexdigest()
        result["relowered_tir_sha256"] = relowered_hash
        if relowered_hash != candidate["tir_sha256"]:
            raise RuntimeError("re-lowered TIR hash differs from P4g certificate")
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
        return fail(
            result,
            "compile",
            str(error),
            "vta_build",
            "vta_build_or_tir_mismatch",
            error=error,
        )

    export_start = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="c3_p4i_binary_") as temporary:
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
        return fail(
            result,
            "compile",
            str(error),
            "cross_export",
            "aarch64_shared_library_export",
            error=error,
        )


def timeout_result(candidate, seconds, phase):
    return fail(
        base_result(candidate),
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
    workloads = {}
    for workload_id in sorted({row["workload_id"] for row in records}):
        rows = [row for row in records if row["workload_id"] == workload_id]
        workloads[workload_id] = {
            "records": len(rows),
            "passed": sum(row["overall_status"] == "passed" for row in rows),
            "failed": sum(row["overall_status"] == "failed" for row in rows),
        }
    return {
        "schema": SUMMARY_SCHEMA,
        "status": "completed" if len(records) == expected else "partial",
        "scope": "isolated local AXU5EVB build/export deployability; no board or performance",
        "selected_p4g_fully_qualified": expected,
        "result_records": len(records),
        "passed": sum(row["overall_status"] == "passed" for row in records),
        "failed": sum(row["overall_status"] == "failed" for row in records),
        "failure_categories": dict(sorted(categories.items())),
        "failure_subcategories": dict(sorted(subcategories.items())),
        "workloads": workloads,
        "binary_retained_count": sum(row["export"]["binary_retained"] for row in records),
        "ssh_used": False,
        "rpc_used": False,
        "board_access": False,
        "performance_measurement": "not_collected",
    }


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def source_guards(paths):
    return {str(Path(path).resolve()): sha256_file(path) for path in paths}


def worker_main(args):
    repo_root = Path(__file__).resolve().parents[3]
    sources = expected_source_hashes(repo_root)
    selected = select_candidates(load_jsonl(args.p4g_results), sources)
    matches = [row for row in selected if row["candidate_id"] == args.worker_candidate_id]
    if len(matches) != 1:
        raise ValueError("worker candidate lookup returned {} rows".format(len(matches)))
    try:
        result = worker_result(matches[0], sources)
    except Exception as error:  # Last-resort process boundary.
        result = fail(
            base_result(matches[0]),
            "environment",
            str(error) or type(error).__name__,
            "worker_uncaught",
            "uncaught_worker_exception",
            error=error,
        )
    write_json(args.worker_output, result)


def main_run(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    repo_root = Path(__file__).resolve().parents[3]
    sources = expected_source_hashes(repo_root)
    p4g_evidence = validate_p4g_artifacts(args.p4g_results)
    candidates = select_candidates(load_jsonl(args.p4g_results), sources)
    guards_before = source_guards(args.guard_files)
    records, stdout_lines, stderr_lines = [], [], []
    started = time.monotonic()
    results_path = output / "results.jsonl"
    with open(results_path, "w", encoding="utf-8") as result_stream:
        for position, candidate in enumerate(candidates, 1):
            remaining = args.overall_timeout_seconds - (time.monotonic() - started)
            if remaining <= 0:
                result = timeout_result(candidate, args.overall_timeout_seconds, "global_deadline")
            else:
                with tempfile.TemporaryDirectory(prefix="c3_p4i_worker_") as temporary:
                    worker_output = Path(temporary) / "result.json"
                    command = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--p4g-results",
                        str(Path(args.p4g_results).resolve()),
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
                            result = fail(
                                base_result(candidate),
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
                "{}/{} {} {}".format(
                    position,
                    len(candidates),
                    candidate["workload_id"],
                    result["overall_status"],
                )
            )

    guards_after = source_guards(args.guard_files)
    if guards_before != guards_after:
        raise RuntimeError("guarded schedule/compiler/runtime source changed during qualification")
    summary = summarize(records, len(candidates))
    summary["wall_clock_seconds"] = time.monotonic() - started
    write_json(output / "summary.json", summary)
    (output / "stdout.log").write_text("\n".join(stdout_lines) + "\n", encoding="utf-8")
    (output / "stderr.log").write_text("".join(stderr_lines), encoding="utf-8")
    (output / "command.txt").write_text(" ".join([sys.executable] + sys.argv) + "\n")
    compiler = subprocess.run(
        ["aarch64-xilinx-linux-g++", "--version"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.splitlines()[0]
    write_json(
        output / "manifest.json",
        {
            "schema": "c3_p4i_weight_barrier_cross_compile_manifest_v1",
            "date": "2026-09-11",
            "environment": {
                "python": sys.executable,
                "python_version": platform.python_version(),
                "tvm_version": tvm.__version__,
                "vta_target": vta.get_env().TARGET,
                "target_host": str(vta.get_env().target_host),
                "cross_compiler": compiler,
                "sysroot": os.environ.get("SDKTARGETSYSROOT"),
                "ssh_used": False,
                "rpc_used": False,
                "board_contacted": False,
            },
            "inputs": {
                "p4g_results": {
                    "path": str(Path(args.p4g_results).resolve()),
                    "sha256": sha256_file(args.p4g_results),
                },
                "p4g_artifact_evidence": p4g_evidence,
            },
            "schedule_compiler_source_sha256": sources,
            "source_guards_before": guards_before,
            "source_guards_after": guards_after,
            "source": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(__file__)},
        },
    )
    write_json(
        output / "preregistered.json",
        {
            "schema": "c3_p4i_weight_barrier_cross_compile_protocol_v1",
            "selection": "all 32 P4g status/lower/dependency/FSim-passed weight_stationary_barrier candidates",
            "identity_checks": [
                "P4g artifact hash",
                "stable ID",
                "complete ConfigEntity",
                "schedule/compiler source hashes",
                "re-lowered TIR hash",
            ],
            "worker_isolation": "one subprocess and temporary binary directory per candidate",
            "candidate_timeout_seconds": args.candidate_timeout_seconds,
            "overall_timeout_seconds": args.overall_timeout_seconds,
            "binary_retention": "size/hash only; no candidate shared object retained",
            "failure_categories": ["lower", "compile", "wrong_answer", "timeout", "rpc", "environment"],
            "claim": "AXU5EVB build/export deployability only; no board correctness or performance",
        },
    )
    (output / "STATUS.md").write_text(
        "# C3-P4i weight-barrier AXU5EVB cross-compile\n\n"
        "- Status: `{}`\n"
        "- P4g fully qualified candidates: {}\n"
        "- Passed isolated build/export: {}\n"
        "- Failed: {}\n"
        "- Candidate binaries retained: {}\n"
        "- SSH/RPC/board: not used\n"
        "- Claim: deployability only; no latency or FPS result\n".format(
            summary["status"],
            summary["selected_p4g_fully_qualified"],
            summary["passed"],
            summary["failed"],
            summary["binary_retained_count"],
        ),
        encoding="utf-8",
    )
    (output / "HANDOFF.md").write_text(
        "# HANDOFF\n\n"
        "`results.jsonl` binds every AXU5EVB export result to the frozen P4g artifact, "
        "stable candidate ID, complete ConfigEntity, compiler/schedule hashes, and a fresh "
        "TIR hash. Shared objects were deleted with their temporary directories; only size "
        "and SHA-256 remain. Passing P4i proves local cross-build/export deployability only. "
        "It does not prove current-boot loadability, board correctness, latency, or FPS.\n",
        encoding="utf-8",
    )
    artifacts = {
        path.name: sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    test_source = Path(__file__).with_name(
        "test_qualify_vta_weight_barrier_axu_cross_compile.py"
    )
    source_artifacts = {str(Path(__file__).resolve()): sha256_file(__file__)}
    if test_source.is_file():
        source_artifacts[str(test_source.resolve())] = sha256_file(test_source)
    write_json(
        output / "artifact_hashes.json",
        {"output_sha256": artifacts, "source_sha256": source_artifacts},
    )
    print(json.dumps(summary, sort_keys=True))


def parse_args():
    base = Path(__file__).resolve().parent / "report_out" / "stage_tile_cotuning"
    p4 = base / "c3_dma_residency_autotune" / "04_dma_command_signatures"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--p4g-results",
        default=str(p4 / "20260911_p4g_weight_barrier_pool_run01" / "results.jsonl"),
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--candidate-timeout-seconds", type=int, default=180)
    parser.add_argument("--overall-timeout-seconds", type=int, default=3600)
    parser.add_argument(
        "--guard-files",
        nargs="+",
        default=[
            str(Path(__file__).parents[2] / "python/vta/top/vta_conv2d.py"),
            str(Path(__file__).parents[2] / "python/vta/top/vta_conv2d_residency.py"),
            str(Path(__file__).parents[2] / "python/vta/transform.py"),
            str(Path(__file__).parents[2] / "runtime/runtime.cc"),
            str(Path(__file__).parents[3] / "src/tir/transforms/coproc_sync.cc"),
            str(Path(__file__).parents[3] / "build/libtvm.so"),
        ],
    )
    parser.add_argument("--worker-candidate-id")
    parser.add_argument("--worker-output")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.worker_candidate_id:
        if not args.worker_output:
            raise ValueError("--worker-output is required in worker mode")
        worker_main(args)
    else:
        if not args.output_dir:
            raise ValueError("--output-dir is required")
        main_run(args)


if __name__ == "__main__":
    main()
