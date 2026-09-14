#!/usr/bin/env python3
"""Generate and locally qualify the isolated mode-4 weight-barrier pool.

The input is the frozen P4b mapped-tile pool.  This tool creates one new stable
candidate per unique mapped ConfigEntity, lowers it with residency mode 4,
extracts static DMA/sync/dependency evidence, and can optionally run three-seed
FSim correctness.  It never contacts a board and never records execution time.
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
from collections import Counter
from pathlib import Path

import numpy as np
import tvm
from tvm import autotvm, rpc, tir
from tvm.contrib import utils
import vta

import vta.top.vta_conv2d_residency  # pylint: disable=unused-import
from audit_vta_coproc_dependencies import audit_vta_coproc_dependencies
from c3_candidate_identity import canonical_json_bytes, make_failure, normalize_config_entity
from extract_static_vta_dma import extract_module_dma
from generate_vta_residency_candidates import (
    classify_lower_failure,
    distribution,
    dma_change,
    load_json,
    semantic_config_key,
    sha256_file,
    transfer_signature,
    unique_tensor_bytes,
)
from qualify_vta_residency_fsim import SEEDS, array_sha256, reference_data_from_workload


POOL_SCHEMA = "c3_weight_barrier_candidate_pool_v1"
RESULT_SCHEMA = "c3_weight_barrier_static_feature_v1"
SUMMARY_SCHEMA = "c3_weight_barrier_summary_v1"
IDENTITY_SCHEMA = "c3_weight_barrier_candidate_id_v1"
TEMPLATE_NAME = "conv2d_packed_residency.vta"
PUBLIC_MODE = "weight_stationary_barrier"
IMPLEMENTATION_MODE = 4
IMPLEMENTATION_NAME = "weight_stationary_sync_probe"


def schedule_sources(repo_root):
    paths = {
        "vta_conv2d.py": repo_root / "vta/python/vta/top/vta_conv2d.py",
        "transform.py": repo_root / "vta/python/vta/transform.py",
        "coproc_sync.cc": repo_root / "src/tir/transforms/coproc_sync.cc",
        "vta_conv2d_residency.py": repo_root / "vta/python/vta/top/vta_conv2d_residency.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def make_schedule_version(source_hashes):
    required = ("vta_conv2d.py", "transform.py", "coproc_sync.cc")
    if any(name not in source_hashes or len(source_hashes[name]) != 64 for name in required):
        raise ValueError("schedule version requires three compiler/schedule source hashes")
    return "weight_stationary_barrier@" + "+".join(
        "{}={}".format(name, source_hashes[name]) for name in sorted(source_hashes)
    )


def barrier_identity_record(
    hardware_fingerprint, schedule_version, source_hashes, workload, config, config_index
):
    payload = {
        "schema": IDENTITY_SCHEMA,
        "hardware_fingerprint": hardware_fingerprint,
        "template_name": TEMPLATE_NAME,
        "schedule_version": schedule_version,
        "schedule_source_sha256": dict(sorted(source_hashes.items())),
        "workload": workload,
        "residence_mode": PUBLIC_MODE,
        "implementation_mode": IMPLEMENTATION_MODE,
        "implementation_name": IMPLEMENTATION_NAME,
        "complete_config_entity": normalize_config_entity(config),
    }
    return {
        "candidate_id": hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
        "identity": payload,
        "debug": {"config_index": int(config_index)},
    }


def create_task(workload, env):
    return autotvm.task.create(
        TEMPLATE_NAME,
        args=tuple(workload[1:]) + (IMPLEMENTATION_MODE,),
        target=env.target,
        target_host=env.target_host,
    )


def sync_call_multiplicities(module):
    loop_extents = []
    multiplicities = []

    def preorder(node):
        if isinstance(node, tir.For):
            loop_extents.append(int(node.extent))
        if (
            isinstance(node, tir.Call)
            and isinstance(node.op, tvm.ir.Op)
            and node.op.name == "tir.vta.coproc_sync"
        ):
            product = 1
            for extent in loop_extents:
                product *= extent
            multiplicities.append(product)

    def postorder(node):
        if isinstance(node, tir.For):
            loop_extents.pop()

    tir.stmt_functor.ir_transform(module["main"].body, preorder, postorder, None)
    return multiplicities


def lower_barrier_candidate(task, config_index, expected_config, env, unique_bytes):
    phase = "instantiate"
    try:
        config = task.config_space.get(int(config_index))
        if semantic_config_key(config.to_json_dict()) != semantic_config_key(expected_config):
            raise ValueError("mode-4 ConfigEntity differs from frozen P4b mapped tile")
        with task.target:
            schedule, tensors = task.instantiate(config)
        phase = "tir_lower"
        with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
            module = tvm.lower(schedule, tensors, name="main")
        saved = tvm.ir.save_json(module)
        phase = "static_dma_extract"
        dma = extract_module_dma(module, env)
        phase = "dependency_audit"
        dependency = audit_vta_coproc_dependencies(module)
        sync_multiplicities = sync_call_multiplicities(module)
        if not dependency["valid"]:
            failure = make_failure(
                "lower", "mode-4 dependency topology audit failed", phase=phase
            )
            failure["subcategory"] = "dependency_topology"
            return {
                "status": "failed",
                "lower_status": "ok",
                "failure": failure,
                "tir_sha256": hashlib.sha256(saved.encode("utf-8")).hexdigest(),
                "tir_hash_encoding": "tvm.ir.save_json",
                "transfer_signature": transfer_signature(dma, unique_bytes),
                "sync": {
                    "static_callsites": len(sync_multiplicities),
                    "execution_multiplicities": sync_multiplicities,
                    "expanded_executions": sum(sync_multiplicities),
                },
                "dependency_audit": dependency,
            }
        return {
            "status": "ok",
            "lower_status": "ok",
            "failure": None,
            "tir_sha256": hashlib.sha256(saved.encode("utf-8")).hexdigest(),
            "tir_hash_encoding": "tvm.ir.save_json",
            "transfer_signature": transfer_signature(dma, unique_bytes),
            "sync": {
                "static_callsites": len(sync_multiplicities),
                "execution_multiplicities": sync_multiplicities,
                "expanded_executions": sum(sync_multiplicities),
                "interpretation": "mode-4 residency drains followed by existing function-final drain",
            },
            "dependency_audit": dependency,
        }
    except Exception as error:
        message = str(error) or type(error).__name__
        failure = make_failure("lower", message[:4000], phase=phase)
        failure["subcategory"] = classify_lower_failure(message, phase)
        failure["exception_type"] = type(error).__name__
        return {
            "status": "failed",
            "lower_status": "failed",
            "failure": failure,
            "tir_sha256": None,
            "tir_hash_encoding": "tvm.ir.save_json",
            "transfer_signature": None,
            "sync": None,
            "dependency_audit": None,
        }


def _qualification_base(candidate):
    return {
        "schema": "c3_weight_barrier_fsim_qualification_v1",
        "candidate_id": candidate["candidate_id"],
        "workload_id": candidate["workload_id"],
        "residence_mode": PUBLIC_MODE,
        "overall_status": "not_attempted",
        "build": {"status": "not_attempted", "count": 0},
        "seeds": [],
        "failure": None,
        "performance_measurement": "not_collected",
    }


def fsim_qualify(candidate):
    result = _qualification_base(candidate)
    env = vta.get_env()
    if env.TARGET != "sim":
        result.update(
            overall_status="failed",
            failure=make_failure("environment", "VTA TARGET must be sim", phase="preflight"),
        )
        return result
    try:
        from vta.testing import simulator

        if not simulator.enabled():
            raise RuntimeError("VTA simulator runtime is not enabled")
        task = create_task(candidate["identity"]["workload"], env)
        config_index = candidate["debug"]["config_index"]
        config = task.config_space.get(config_index)
        if semantic_config_key(config.to_json_dict()) != semantic_config_key(
            candidate["identity"]["complete_config_entity"]
        ):
            raise RuntimeError("ConfigEntity no longer matches stable identity")
        result["build"]["count"] = 1
        with task.target:
            schedule, tensors = task.instantiate(config)
        with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
            lowered = tvm.lower(schedule, tensors, name="main")
        relowered_hash = hashlib.sha256(tvm.ir.save_json(lowered).encode("utf-8")).hexdigest()
        result["relowered_tir_sha256"] = relowered_hash
        result["tir_hash_matches_static"] = relowered_hash == candidate["tir_sha256"]
        if not result["tir_hash_matches_static"]:
            raise RuntimeError("re-lowered TIR hash differs from static pool")
        with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
            module = vta.build(
                schedule,
                tensors,
                target=tvm.target.Target(env.target, host=env.target_host),
                name="main",
            )
        result["build"]["status"] = "ok"
        temporary = utils.tempdir()
        module_name = candidate["candidate_id"] + ".o"
        module_path = temporary.relpath(module_name)
        module.save(module_path)
        remote = rpc.LocalSession()
        remote.upload(module_path)
        function = remote.load_module(module_name)["main"]
        device = remote.ext_dev(0)
    except Exception as error:
        failure = make_failure(
            "compile", (str(error) or type(error).__name__)[:4000], phase="fsim_build"
        )
        failure["exception_type"] = type(error).__name__
        result["build"]["status"] = "failed"
        result.update(overall_status="failed", failure=failure)
        return result

    for seed in SEEDS:
        seed_result = {"seed": seed, "status": "not_attempted", "correct": None}
        try:
            data, weight, expected = reference_data_from_workload(
                candidate["identity"]["workload"], seed
            )
            output = tvm.nd.empty(expected.shape, "int8", device=device)
            function(tvm.nd.array(data, device), tvm.nd.array(weight, device), output)
            actual = output.numpy()
            mismatches = int(np.count_nonzero(actual != expected))
            seed_result.update(
                status="passed" if mismatches == 0 else "failed",
                correct=mismatches == 0,
                mismatch_count=mismatches,
                expected_sha256=array_sha256(expected),
                actual_sha256=array_sha256(actual),
            )
            if mismatches:
                seed_result["failure"] = make_failure(
                    "wrong_answer",
                    "{} output elements differ".format(mismatches),
                    phase="fsim_execute",
                )
        except Exception as error:
            failure = make_failure(
                "environment", (str(error) or type(error).__name__)[:4000], phase="fsim_execute"
            )
            failure["exception_type"] = type(error).__name__
            seed_result.update(status="failed", correct=False, failure=failure)
        result["seeds"].append(seed_result)
    result["overall_status"] = (
        "passed"
        if len(result["seeds"]) == len(SEEDS) and all(seed["correct"] for seed in result["seeds"])
        else "failed"
    )
    if result["overall_status"] == "failed":
        result["failure"] = next(
            (seed.get("failure") for seed in result["seeds"] if seed.get("failure")),
            make_failure("wrong_answer", "one or more seeds failed", phase="fsim_execute"),
        )
    return result


def worker_timeout(candidate, seconds):
    result = _qualification_base(candidate)
    failure = make_failure(
        "timeout", "FSim worker exceeded {} seconds".format(seconds), phase="fsim_worker", retryable=True
    )
    result.update(overall_status="failed", failure=failure)
    return result


def run_fsim_worker(candidate, timeout_seconds):
    with tempfile.TemporaryDirectory(prefix="c3_p4g_worker_") as directory:
        candidate_path = Path(directory) / "candidate.json"
        output_path = Path(directory) / "result.json"
        candidate_path.write_text(json.dumps(candidate))
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-candidate",
            str(candidate_path),
            "--worker-output",
            str(output_path),
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=os.environ.copy(),
            )
        except subprocess.TimeoutExpired:
            return worker_timeout(candidate, timeout_seconds), ""
        worker_stderr = completed.stderr[-4000:]
        if completed.returncode != 0 or not output_path.is_file():
            result = _qualification_base(candidate)
            failure = make_failure(
                "environment",
                "worker exit={} output_present={}".format(completed.returncode, output_path.is_file()),
                phase="fsim_worker",
            )
            result.update(overall_status="failed", failure=failure)
            return result, worker_stderr
        return json.loads(output_path.read_text()), worker_stderr


def validate_p4b_inputs(pool_path, results_path):
    pool_path, results_path = Path(pool_path), Path(results_path)
    if pool_path.parent != results_path.parent:
        raise ValueError("P4b pool and results must share a frozen run directory")
    hashes = load_json(pool_path.parent / "artifact_hashes.json")["output_sha256"]
    if sha256_file(pool_path) != hashes["candidate_pool.json"]:
        raise ValueError("P4b candidate pool hash mismatch")
    if sha256_file(results_path) != hashes["results.jsonl"]:
        raise ValueError("P4b results hash mismatch")
    pool = load_json(pool_path)
    rows = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]
    by_id = {row["candidate_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("P4b results contain duplicate candidate IDs")
    return pool, rows, by_id


def validate_p3e_run(p3e_run_dir, source_hashes):
    p3e_run_dir = Path(p3e_run_dir)
    ledger_path = p3e_run_dir / "artifact_hashes.json"
    manifest_path = p3e_run_dir / "manifest.json"
    summary_path = p3e_run_dir / "summary.json"
    ledger = load_json(ledger_path)
    artifacts = ledger.get("artifacts", {})
    if sha256_file(manifest_path) != artifacts.get("manifest.json"):
        raise ValueError("P3e manifest hash mismatch")
    if sha256_file(summary_path) != artifacts.get("summary.json"):
        raise ValueError("P3e summary hash mismatch")
    for name in ("vta_conv2d.py", "transform.py", "coproc_sync.cc"):
        matches = [value for path, value in ledger.get("sources", {}).items() if path.endswith(name)]
        if matches != [source_hashes[name]]:
            raise ValueError("P3e source hash mismatch for {}".format(name))
    summary = load_json(summary_path)
    if (
        summary.get("status") != "completed_positive_local_proof"
        or summary.get("dependency_gate_met") != 3
        or summary.get("fsim_seed_runs_correct") != 9
        or summary.get("board_contacted") is not False
        or summary.get("performance_timing_collected") is not False
    ):
        raise ValueError("P3e proof gate is not frozen GO")
    return {
        "run_dir": str(p3e_run_dir),
        "artifact_hashes_sha256": sha256_file(ledger_path),
        "manifest_sha256": sha256_file(manifest_path),
        "summary_sha256": sha256_file(summary_path),
        "decision": summary["decision"],
    }


def generate_candidates(pool, p4b_by_id, source_hashes, env, limit=None):
    version = make_schedule_version(source_hashes)
    output_pool = []
    results = []
    for workload_entry in pool["workloads"]:
        workload = workload_entry["workload"]
        workload_id = workload_entry["workload_id"]
        task = create_task(workload, env)
        mapped_output = []
        for mapped in workload_entry["mapping"]["records"]:
            if limit is not None and len(results) >= limit:
                break
            index = mapped["mapped_config_index"]
            config = mapped["complete_config_entity"]
            baseline_id = mapped["candidate_ids"]["original"]
            baseline = p4b_by_id[baseline_id]
            identity = barrier_identity_record(
                pool["hardware_fingerprint"], version, source_hashes, workload, config, index
            )
            lowered = lower_barrier_candidate(
                task, index, config, env, unique_tensor_bytes(workload)
            )
            result = {
                "schema": RESULT_SCHEMA,
                **identity,
                "workload_id": workload_id,
                "candidate_role": "weight_barrier_experiment",
                "residence_mode": PUBLIC_MODE,
                "implementation_mode": IMPLEMENTATION_MODE,
                "source_p4b_original_candidate_id": baseline_id,
                "source_candidates": mapped["sources"],
                **lowered,
                "fsim_qualification": _qualification_base(identity | {"workload_id": workload_id}),
                "performance_measurement": "not_collected",
            }
            result["relative_to_same_tile_original"] = dma_change(result, baseline)
            results.append(result)
            mapped_output.append(
                {
                    "mapped_config_index": index,
                    "complete_config_entity": config,
                    "source_p4b_original_candidate_id": baseline_id,
                    "candidate_id": identity["candidate_id"],
                }
            )
        if mapped_output:
            output_pool.append(
                {"workload_id": workload_id, "workload": workload, "records": mapped_output}
            )
        if limit is not None and len(results) >= limit:
            break
    ids = [result["candidate_id"] for result in results]
    if len(ids) != len(set(ids)):
        raise ValueError("weight-barrier pool contains duplicate stable IDs")
    return {
        "schema": POOL_SCHEMA,
        "template_name": TEMPLATE_NAME,
        "residence_mode": PUBLIC_MODE,
        "implementation_mode": IMPLEMENTATION_MODE,
        "schedule_version": version,
        "schedule_source_sha256": source_hashes,
        "hardware_fingerprint": pool["hardware_fingerprint"],
        "candidate_count": len(results),
        "workloads": output_pool,
    }, results


def summarize(results, run_fsim):
    lower_ok = [row for row in results if row["status"] == "ok"]
    reductions = [
        row["relative_to_same_tile_original"]["wgt"]["reduction_fraction"]
        for row in lower_ok
        if row["relative_to_same_tile_original"] is not None
    ]
    failure_counts = Counter(
        row["failure"]["subcategory"] for row in results if row["failure"] is not None
    )
    fsim_counts = Counter(row["fsim_qualification"]["overall_status"] for row in results)
    return {
        "schema": SUMMARY_SCHEMA,
        "status": "completed" if len(results) == 60 else "partial",
        "scope": "local static lowering/dependency audit and optional FSim correctness; no board or timing",
        "mode": PUBLIC_MODE,
        "implementation_mode": IMPLEMENTATION_MODE,
        "candidate_count": len(results),
        "lower_and_dependency_ok": len(lower_ok),
        "lower_or_dependency_failed": len(results) - len(lower_ok),
        "failure_subcategories": dict(sorted(failure_counts.items())),
        "dependency_valid": sum(
            (row.get("dependency_audit") or {}).get("valid", False) for row in results
        ),
        "forbidden_1_3_present": sum(
            (row.get("dependency_audit") or {}).get("summary", {}).get(
                "forbidden_1_3_present", False
            )
            for row in results
        ),
        "sync_static_callsites": distribution(
            [row["sync"]["static_callsites"] for row in lower_ok]
        ),
        "sync_expanded_executions": distribution(
            [row["sync"]["expanded_executions"] for row in lower_ok]
        ),
        "residency_drain_expanded_executions": distribution(
            [row["sync"]["expanded_executions"] - 1 for row in lower_ok]
        ),
        "weight_dma_reduction_fraction": distribution(reductions),
        "fsim_requested": bool(run_fsim),
        "frozen_seeds": list(SEEDS),
        "fsim_status_counts": dict(sorted(fsim_counts.items())),
        "performance_measurement": "not_collected",
        "board_contacted": False,
    }


def _write_new(path, value, jsonl=False):
    path = Path(path)
    if path.exists():
        raise FileExistsError("refusing to overwrite frozen artifact: {}".format(path))
    if jsonl:
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in value))
    else:
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p4b-candidate-pool")
    parser.add_argument("--p4b-results")
    parser.add_argument("--p3e-run-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--run-fsim", action="store_true")
    parser.add_argument("--candidate-timeout-seconds", type=int, default=180)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--worker-candidate")
    parser.add_argument("--worker-output")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.worker_candidate:
        candidate = load_json(args.worker_candidate)
        Path(args.worker_output).write_text(json.dumps(fsim_qualify(candidate), indent=2) + "\n")
        return
    if (
        not args.p4b_candidate_pool
        or not args.p4b_results
        or not args.p3e_run_dir
        or not args.output_dir
    ):
        raise ValueError("P4b pool/results, frozen P3e run, and output directory are required")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError("refusing to write into non-empty output directory")
    pool, _, p4b_by_id = validate_p4b_inputs(args.p4b_candidate_pool, args.p4b_results)
    env = vta.get_env()
    repo_root = Path(__file__).resolve().parents[3]
    sources = schedule_sources(repo_root)
    p3e_proof = validate_p3e_run(args.p3e_run_dir, sources)
    output_pool, results = generate_candidates(pool, p4b_by_id, sources, env, args.limit)
    stderr_lines = []
    if args.run_fsim:
        for position, candidate in enumerate(results):
            if candidate["status"] != "ok":
                continue
            qualification, worker_stderr = run_fsim_worker(
                candidate, args.candidate_timeout_seconds
            )
            candidate["fsim_qualification"] = qualification
            if worker_stderr.strip():
                stderr_lines.append(
                    "position={} candidate={} stderr={}\n".format(
                        position, candidate["candidate_id"], worker_stderr
                    )
                )
    summary = summarize(results, args.run_fsim)
    _write_new(output_dir / "candidate_pool.json", output_pool)
    _write_new(output_dir / "results.jsonl", results, jsonl=True)
    _write_new(output_dir / "summary.json", summary)
    command = " ".join([sys.executable] + sys.argv) + "\n"
    (output_dir / "command.txt").write_text(command)
    stdout = (
        "candidates={} lower_dependency_ok={} fsim_passed={} board=false timing=false\n".format(
            len(results),
            summary["lower_and_dependency_ok"],
            summary["fsim_status_counts"].get("passed", 0),
        )
    )
    (output_dir / "stdout.log").write_text(stdout)
    (output_dir / "stderr.log").write_text("".join(stderr_lines))
    preregistered = {
        "schema": "c3_p4g_preregistered_v1",
        "mode": PUBLIC_MODE,
        "implementation": "verified mode 4 weight_stationary_sync_probe",
        "population": "all 60 unique P4b mapped tile ConfigEntities",
        "admission": "static lower succeeds; dependency audit balanced/supported; no direct 1<->3",
        "fsim_seeds": list(SEEDS),
        "performance_rule": "FSim duration and static DMA are not performance measurements",
        "claim_boundary": "compiler full-sync experimental mode; not hardware prefetch",
    }
    _write_new(output_dir / "preregistered.json", preregistered)
    manifest = {
        "schema": "c3_p4g_local_manifest_v1",
        "date": "2026-09-11",
        "environment": {
            "python": sys.executable,
            "python_version": platform.python_version(),
            "tvm_version": tvm.__version__,
            "vta_target": env.TARGET,
            "board_contacted": False,
            "performance_timing_collected": False,
        },
        "inputs": {
            "p4b_candidate_pool": {
                "path": args.p4b_candidate_pool,
                "sha256": sha256_file(args.p4b_candidate_pool),
            },
            "p4b_results": {
                "path": args.p4b_results,
                "sha256": sha256_file(args.p4b_results),
            },
            "p3e_positive_local_proof": p3e_proof,
        },
        "sources": {
            "generator": str(Path(__file__).resolve()),
            "generator_sha256": sha256_file(Path(__file__)),
            "schedule_source_sha256": sources,
            "schedule_version": output_pool["schedule_version"],
        },
        "results": summary,
    }
    _write_new(output_dir / "manifest.json", manifest)
    (output_dir / "STATUS.md").write_text(
        "# C3-P4g weight-barrier pool\n\n"
        "Status: **{}**. {} candidates; {} lower/dependency valid; {} three-seed FSim passed. "
        "No board or timing was used. This is a compiler full-sync experimental mode, not "
        "hardware prefetch, and no speed benefit is established.\n".format(
            summary["status"],
            len(results),
            summary["lower_and_dependency_ok"],
            summary["fsim_status_counts"].get("passed", 0),
        )
    )
    (output_dir / "HANDOFF.md").write_text(
        "# HANDOFF\n\nUse `results.jsonl` for static DMA, explicit sync, dependency, and FSim "
        "evidence. Stable IDs bind the mode, full ConfigEntity, workload, hardware fingerprint, "
        "and schedule/compiler source hashes. Static DMA reduction and FSim correctness do not "
        "establish board speed. Do not merge this mode into production candidates without a "
        "separate admission decision.\n"
    )
    artifacts = {
        path.name: sha256_file(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    test_source = Path(__file__).with_name("test_generate_vta_weight_barrier_candidates.py")
    source_artifacts = {str(Path(__file__)): sha256_file(Path(__file__))}
    if test_source.is_file():
        source_artifacts[str(test_source)] = sha256_file(test_source)
    _write_new(
        output_dir / "artifact_hashes.json",
        {"output_sha256": artifacts, "source_sha256": source_artifacts},
    )
    print(stdout, end="")


if __name__ == "__main__":
    main()
