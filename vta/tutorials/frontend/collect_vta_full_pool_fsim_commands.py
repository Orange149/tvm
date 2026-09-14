#!/usr/bin/env python3
"""Collect timing-free local-FSim command signatures for the frozen C3 pool.

The population is fixed to 165 P4e unified-local-qualified mode-0..3
candidates plus 32 P4g candidates that also passed P4i cross compilation.
Every candidate is re-lowered, built, and checked with seed 0 in an isolated
local FSim worker.  This is never a board-capacity or performance experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import tvm
from tvm import autotvm
import vta
from vta.libinfo import find_libvta

import vta.top.vta_conv2d_residency  # pylint: disable=unused-import
from c3_candidate_identity import canonical_json_bytes, make_failure
from collect_vta_fsim_command_signatures import (
    STATUS,
    array_sha256,
    command_signature,
    parse_queue_records,
    reference_data_from_workload,
    semantic_config_key,
    sha256_file,
    write_json,
)


RESULT_SCHEMA = "c3_full_pool_fsim_command_signature_v1"
SUMMARY_SCHEMA = "c3_full_pool_fsim_command_summary_v1"
MODE_NUMBERS = {
    "original": 0,
    "input_stationary": 1,
    "weight_stationary": 2,
    "paper_inspired_hybrid": 3,
    "weight_stationary_barrier": 4,
}
TIMING_KEYS = {
    "finalize_us",
    "uop_pack_us",
    "uop_copy_us",
    "insn_copy_us",
    "queue_cache_us",
    "device_run_us",
    "submit_mmio_us",
    "poll_wait_us",
}


def load_json(path):
    return json.loads(Path(path).read_text())


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def verify_frozen_result(path):
    """Verify a results.jsonl against either historical ledger layout."""

    path = Path(path)
    ledger = load_json(path.parent / "artifact_hashes.json")
    hashes = ledger.get("output_sha256", ledger)
    expected = hashes.get(path.name)
    if expected is None:
        raise ValueError("frozen ledger does not cover {}".format(path))
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError("frozen result hash mismatch for {}".format(path))
    return {"path": str(path.resolve()), "sha256": observed}


def _stable_id_matches(row):
    return row["candidate_id"] == hashlib.sha256(
        canonical_json_bytes(row["identity"])
    ).hexdigest()


def select_full_pool(p4b_rows, p4e_rows, p4g_rows, p4i_rows):
    """Join qualifications before observing any P4j command signature."""

    p4b = {row["candidate_id"]: row for row in p4b_rows}
    p4g = {row["candidate_id"]: row for row in p4g_rows}
    p4i = {row["candidate_id"]: row for row in p4i_rows}
    if len(p4b) != len(p4b_rows) or len(p4g) != len(p4g_rows) or len(p4i) != len(p4i_rows):
        raise ValueError("duplicate ID in a frozen input")

    p4e_qualified = [row for row in p4e_rows if row.get("local_status") == "fsim_passed"]
    p4g_qualified = [
        row
        for row in p4g_rows
        if row.get("status") == "ok"
        and row.get("fsim_qualification", {}).get("overall_status") == "passed"
    ]
    if len(p4e_qualified) != 165 or len(p4g_qualified) != 32:
        raise ValueError(
            "frozen population differs from 165 P4e + 32 P4g: {} + {}".format(
                len(p4e_qualified), len(p4g_qualified)
            )
        )
    if set(row["candidate_id"] for row in p4g_qualified) != set(p4i):
        raise ValueError("P4i does not cover exactly the 32 P4g FSim-qualified candidates")

    selected = []
    for qualification in p4e_qualified:
        candidate_id = qualification["candidate_id"]
        if candidate_id not in p4b:
            raise ValueError("P4e candidate missing from P4b: {}".format(candidate_id))
        candidate = p4b[candidate_id]
        if qualification["tir_sha256"] != candidate.get("tir_sha256"):
            raise ValueError("P4e/P4b TIR mismatch: {}".format(candidate_id))
        if qualification.get("workload_id") != candidate.get("workload_id"):
            raise ValueError("P4e/P4b workload mismatch: {}".format(candidate_id))
        if qualification.get("residence_mode") != candidate.get("residence_mode"):
            raise ValueError("P4e/P4b mode mismatch: {}".format(candidate_id))
        if not _stable_id_matches(candidate):
            raise ValueError("P4b stable ID mismatch: {}".format(candidate_id))
        mode = candidate["residence_mode"]
        selected.append(
            {
                "candidate_id": candidate_id,
                "workload_id": candidate["workload_id"],
                "public_mode": mode,
                "implementation_mode": MODE_NUMBERS[mode],
                "identity": candidate["identity"],
                "debug": candidate["debug"],
                "expected_tir_sha256": candidate["tir_sha256"],
                "expected_residency_drains": 0,
                "candidate_role": candidate["candidate_role"],
                "qualification_reference": {
                    "phase": "P4e",
                    "local_status": qualification["local_status"],
                    "correctness_seeds": qualification["correctness_seeds"],
                    "evidence_source": qualification["evidence_source"],
                },
                "dependency_evidence_reference": {
                    "phase": "P4e/P4b",
                    "candidate_specific_push_pop_audit_available": False,
                    "claim": "local three-seed correctness and successful lowering only; no new dependency claim",
                },
            }
        )

    for candidate in p4g_qualified:
        candidate_id = candidate["candidate_id"]
        qualification = p4i[candidate_id]
        required_checks = (
            qualification.get("overall_status") == "passed"
            and qualification.get("stable_id_check") == "passed"
            and qualification.get("source_hash_check") == "passed"
            and qualification.get("tir_hash_check") == "passed"
            and qualification.get("complete_config_entity_check") == "passed"
            and qualification.get("p4g_tir_sha256") == candidate.get("tir_sha256")
        )
        if not required_checks:
            raise ValueError("P4i binding check failed: {}".format(candidate_id))
        if not _stable_id_matches(candidate):
            raise ValueError("P4g stable ID mismatch: {}".format(candidate_id))
        dependency = candidate.get("dependency_audit") or {}
        if not dependency.get("valid"):
            raise ValueError("P4g qualified candidate lacks valid dependency audit")
        multiplicities = candidate.get("sync", {}).get("execution_multiplicities", [])
        if len(multiplicities) < 2 or int(multiplicities[-1]) != 1:
            raise ValueError("P4g sync multiplicities lack function-final drain")
        selected.append(
            {
                "candidate_id": candidate_id,
                "workload_id": candidate["workload_id"],
                "public_mode": "weight_stationary_barrier",
                "implementation_mode": 4,
                "identity": candidate["identity"],
                "debug": candidate["debug"],
                "expected_tir_sha256": candidate["tir_sha256"],
                "expected_residency_drains": int(multiplicities[0]),
                "candidate_role": candidate["candidate_role"],
                "qualification_reference": {
                    "phase": "P4g/P4i",
                    "p4g_fsim_seeds": [seed["seed"] for seed in candidate["fsim_qualification"]["seeds"]],
                    "p4i_cross_compile": qualification["overall_status"],
                },
                "dependency_evidence_reference": {
                    "phase": "P4g",
                    "candidate_specific_push_pop_audit_available": True,
                    "valid": dependency["valid"],
                    "summary": dependency["summary"],
                },
            }
        )

    selected.sort(key=lambda row: (row["workload_id"], row["public_mode"], row["candidate_id"]))
    ids = [row["candidate_id"] for row in selected]
    if len(selected) != 197 or len(set(ids)) != 197:
        raise ValueError("selected pool must contain 197 unique stable IDs")
    return selected


def classify_failure(result):
    outcome = result.get("outcome", "negative_environment")
    if outcome == "passed_local_signature":
        return None
    category = {
        "negative_timeout": "timeout",
        "negative_wrong_answer": "wrong_answer",
        "negative_worker_failure": "compile"
        if result.get("build") != "passed"
        else "environment",
        "negative_environment": "environment",
        "negative_no_queue_diagnostics": "environment",
        "negative_signature_mismatch": "environment",
    }.get(outcome, "environment")
    return make_failure(
        category,
        result.get("error", "P4j local FSim dry-run outcome: {}".format(outcome)),
        phase="p4j_fsim_command_dry_run",
        retryable=category in ("timeout", "environment"),
    )


def strip_diagnostic_timing(result):
    """Delete emitted timing values so they cannot become performance evidence."""

    for record in result.get("queue_records", []):
        for key in TIMING_KEYS:
            record.pop(key, None)
    result["diagnostic_timing_fields"] = "discarded_not_performance_evidence"
    return result


def sanitize_stderr(stderr):
    lines = []
    for line in stderr.splitlines():
        if "[VTA_QUEUE] " in line:
            lines.append("[VTA_QUEUE] <parsed; timing fields discarded from P4j evidence>")
        else:
            lines.append(line)
    return "\n".join(lines)


def loaded_library_identity(library_basename):
    """Return the file identity of the library mapped by this worker process.

    Looking at ``/proc/self/maps`` deliberately binds evidence to the binary
    that served the execution, rather than to the first path returned by a
    library search performed in the parent process.
    """

    paths = set()
    for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines():
        fields = line.split(None, 5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        path_text = fields[5].removesuffix(" (deleted)")
        path = Path(path_text)
        if path.name == library_basename and path.is_file():
            paths.add(path.resolve())
    if len(paths) != 1:
        raise RuntimeError(
            "expected exactly one mapped {}, found {}".format(
                library_basename, sorted(str(path) for path in paths)
            )
        )
    path = paths.pop()
    return {"path": str(path), "sha256": sha256_file(path)}


def direct_worker_result(candidate):
    """Build and execute seed 0 through a direct in-process module call."""

    result = {
        "schema": RESULT_SCHEMA,
        "status": STATUS,
        "candidate_id": candidate["candidate_id"],
        "workload_id": candidate["workload_id"],
        "public_mode": candidate["public_mode"],
        "implementation_mode": candidate["implementation_mode"],
        "build": "not_attempted",
        "correctness": "not_attempted",
        "seed": 0,
        "performance_measurement": "not_collected",
        "execution_transport": "direct_local_module_no_rpc",
    }
    env = vta.get_env()
    if env.TARGET != "sim":
        result.update(outcome="negative_environment", error="VTA TARGET must be sim")
        return result
    try:
        from vta.testing import simulator

        if not simulator.enabled():
            raise RuntimeError("VTA FSim runtime is not enabled")
        result["execution_runtime"] = loaded_library_identity("libvta_fsim.so")
        workload = candidate["identity"]["workload"]
        task = autotvm.task.create(
            "conv2d_packed_residency.vta",
            args=tuple(workload[1:]) + (candidate["implementation_mode"],),
            target=env.target,
            target_host=env.target_host,
        )
        config = task.config_space.get(int(candidate["debug"]["config_index"]))
        if semantic_config_key(config.to_json_dict()) != semantic_config_key(
            candidate["identity"]["complete_config_entity"]
        ):
            raise RuntimeError("ConfigEntity differs from frozen identity")
        with task.target:
            schedule, tensors = task.instantiate(config)
        with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
            lowered = tvm.lower(schedule, tensors, name="main")
        tir_hash = hashlib.sha256(tvm.ir.save_json(lowered).encode("utf-8")).hexdigest()
        result["relowered_tir_sha256"] = tir_hash
        result["tir_hash_matches_frozen"] = tir_hash == candidate["expected_tir_sha256"]
        if not result["tir_hash_matches_frozen"]:
            raise RuntimeError("re-lowered TIR differs from frozen candidate")
        with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
            module = vta.build(
                schedule,
                tensors,
                target=tvm.target.Target(env.target, host=env.target_host),
                name="main",
            )
        result["build"] = "passed"
        function = module["main"]
        device = tvm.device("ext_dev", 0)
        data, weight, expected = reference_data_from_workload(workload, 0)
        output = tvm.nd.empty(expected.shape, "int8", device=device)
        function(tvm.nd.array(data, device), tvm.nd.array(weight, device), output)
        actual = output.numpy()
        mismatches = int((actual != expected).sum())
        result.update(
            correctness="passed" if mismatches == 0 else "failed",
            mismatch_count=mismatches,
            expected_sha256=array_sha256(expected),
            actual_sha256=array_sha256(actual),
            outcome="pending_parent_diagnostic_parse" if mismatches == 0 else "negative_wrong_answer",
        )
    except Exception as error:
        result.update(
            outcome="negative_worker_failure",
            error=(str(error) or type(error).__name__)[:4000],
        )
    return result


def run_candidate_no_rpc(candidate, timeout_seconds):
    """Isolate one direct-local FSim execution and parse its diagnostics."""

    with tempfile.TemporaryDirectory(prefix="c3_p4j_direct_worker_") as directory:
        candidate_path = Path(directory) / "candidate.json"
        output_path = Path(directory) / "result.json"
        write_json(candidate_path, candidate)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-candidate",
            str(candidate_path),
            "--worker-output",
            str(output_path),
        ]
        environment = os.environ.copy()
        environment["VTA_QUEUE_DIAGNOSTICS"] = "1"
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=environment,
            )
        except subprocess.TimeoutExpired as error:
            return {
                "schema": RESULT_SCHEMA,
                "status": STATUS,
                "candidate_id": candidate["candidate_id"],
                "workload_id": candidate["workload_id"],
                "public_mode": candidate["public_mode"],
                "implementation_mode": candidate["implementation_mode"],
                "seed": 0,
                "outcome": "negative_timeout",
                "error": "worker exceeded {} seconds".format(timeout_seconds),
                "performance_measurement": "not_collected",
                "execution_transport": "direct_local_module_no_rpc",
            }, error.stderr or ""
        records, malformed = parse_queue_records(completed.stderr)
        if completed.returncode != 0 or not output_path.is_file():
            result = {
                "schema": RESULT_SCHEMA,
                "status": STATUS,
                "candidate_id": candidate["candidate_id"],
                "workload_id": candidate["workload_id"],
                "public_mode": candidate["public_mode"],
                "implementation_mode": candidate["implementation_mode"],
                "seed": 0,
                "outcome": "negative_worker_failure",
                "error": "worker exit={} output_present={}".format(completed.returncode, output_path.is_file()),
                "performance_measurement": "not_collected",
                "execution_transport": "direct_local_module_no_rpc",
            }
        else:
            result = load_json(output_path)
        result["queue_records"] = records
        result["malformed_queue_records"] = malformed
        result["diagnostic_record_count"] = len(records)
        if result.get("correctness") == "passed" and records and not malformed:
            signature = command_signature(
                records,
                candidate["implementation_mode"],
                candidate["expected_residency_drains"],
            )
            result["command_signature"] = signature
            drain_ok = signature["structural"]["drain_check"]["matches_p3e"]
            timeout_ok = all(int(record["timeout"]) == 0 for record in records)
            result["outcome"] = "passed_local_signature" if drain_ok and timeout_ok else "negative_signature_mismatch"
        elif result.get("correctness") == "passed" and not records:
            result["outcome"] = "negative_no_queue_diagnostics"
        result["claim_boundary"] = "direct local FSim only; no RPC, board, capacity qualification, or performance"
        return result, completed.stderr


def run_one(candidate, timeout_seconds):
    result, stderr = run_candidate_no_rpc(candidate, timeout_seconds)
    strip_diagnostic_timing(result)
    result["schema"] = RESULT_SCHEMA
    result["status"] = STATUS
    result["candidate_role"] = candidate["candidate_role"]
    result["qualification_reference"] = candidate["qualification_reference"]
    result["dependency_evidence_reference"] = candidate["dependency_evidence_reference"]
    result["stable_id_check"] = "passed"
    result["source_and_tir_binding"] = (
        "passed" if result.get("tir_hash_matches_frozen") is True else "failed"
    )
    result["failure"] = classify_failure(result)
    return result, sanitize_stderr(stderr)


def metric_distribution(values):
    values = list(values)
    if not values:
        return {"count": 0, "sum": 0, "min": None, "median": None, "max": None}
    return {
        "count": len(values),
        "sum": sum(values),
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def group_summary(rows):
    passed = [row for row in rows if row.get("outcome") == "passed_local_signature"]

    def structural(row, *keys):
        value = row["command_signature"]["structural"]
        for key in keys:
            value = value[key]
        return value

    failures = Counter(
        row["failure"]["category"] for row in rows if row.get("failure") is not None
    )
    return {
        "candidates": len(rows),
        "passed": len(passed),
        "failed": len(rows) - len(passed),
        "failure_categories": dict(sorted(failures.items())),
        "submissions": metric_distribution(structural(row, "submissions") for row in passed),
        "insn_total_bytes": metric_distribution(
            structural(row, "totals", "insn_bytes") for row in passed
        ),
        "insn_peak_bytes": metric_distribution(
            structural(row, "peaks", "insn_bytes") for row in passed
        ),
        "uop_total_bytes": metric_distribution(
            structural(row, "totals", "uop_bytes") for row in passed
        ),
        "uop_peak_bytes": metric_distribution(
            structural(row, "peaks", "uop_bytes") for row in passed
        ),
        "load_total_bytes": metric_distribution(
            structural(row, "totals", "load_bytes") for row in passed
        ),
        "store_total_bytes": metric_distribution(
            structural(row, "totals", "store_bytes") for row in passed
        ),
        "finish_source_derived": metric_distribution(
            structural(row, "finish", "source_derived_count") for row in passed
        ),
        "replay": "unobservable_in_VTA_QUEUE_and_not_requested",
    }


def summarize(results):
    modes = {
        mode: group_summary([row for row in results if row["public_mode"] == mode])
        for mode in MODE_NUMBERS
    }
    workloads = {
        workload: group_summary([row for row in results if row["workload_id"] == workload])
        for workload in sorted({row["workload_id"] for row in results})
    }
    overall = group_summary(results)
    return {
        "schema": SUMMARY_SCHEMA,
        "status": STATUS,
        "scope": "197 frozen candidates, seed0, isolated local FSim structural command dry-run",
        "overall": overall,
        "modes": modes,
        "workloads": workloads,
        "unique_candidate_ids": len({row["candidate_id"] for row in results}),
        "elementwise_correct": sum(row.get("correctness") == "passed" for row in results),
        "mode4_drain_matches_existing_evidence": sum(
            row["implementation_mode"] == 4
            and row.get("command_signature", {})
            .get("structural", {})
            .get("drain_check", {})
            .get("matches_p3e", False)
            for row in results
        ),
        "claim_boundary": {
            "board_contacted": False,
            "board_qualified_capacity": False,
            "board_runtime_semantic_equivalence": "not_established",
            "fsim_timing_used": False,
            "performance_measurement": "not_collected",
            "replay": "unobservable_in_VTA_QUEUE",
            "finish": "source-derived, not opcode-observed",
        },
    }


def run_main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty evidence directory")
    input_paths = {
        "p4b_results": Path(args.p4b_results),
        "p4e_results": Path(args.p4e_results),
        "p4g_results": Path(args.p4g_results),
        "p4i_results": Path(args.p4i_results),
    }
    frozen_inputs = {name: verify_frozen_result(path) for name, path in input_paths.items()}
    candidates = select_full_pool(
        load_jsonl(input_paths["p4b_results"]),
        load_jsonl(input_paths["p4e_results"]),
        load_jsonl(input_paths["p4g_results"]),
        load_jsonl(input_paths["p4i_results"]),
    )
    write_json(
        output_dir / "preregistered.json",
        {
            "schema": "c3_p4j_preregistered_v1",
            "status": STATUS,
            "population": "165 P4e fsim_passed + 32 P4g fsim_passed and P4i cross-compile passed",
            "expected_unique_candidates": 197,
            "seed": 0,
            "correctness": "exact elementwise NumPy equality",
            "binding_gates": ["frozen artifact hash", "stable ID", "ConfigEntity", "TIR hash"],
            "signature": "address-free and timing-free structural [VTA_QUEUE] fields",
            "failure_categories": ["lower", "compile", "wrong_answer", "timeout", "rpc", "environment"],
            "performance_rule": "discard every *_us field; FSim time is never a ranking or performance signal",
        },
    )

    indexed_results = {}
    stderr_by_index = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_one, candidate, args.candidate_timeout_seconds): index
            for index, candidate in enumerate(candidates)
        }
        for future in as_completed(futures):
            index = futures[future]
            result, stderr = future.result()
            indexed_results[index] = result
            stderr_by_index[index] = stderr
            print(
                "{}/{} {} {} {}".format(
                    len(indexed_results),
                    len(candidates),
                    result["workload_id"],
                    result["public_mode"],
                    result["outcome"],
                ),
                flush=True,
            )
    results = [indexed_results[index] for index in range(len(candidates))]
    (output_dir / "results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in results)
    )
    summary = summarize(results)
    write_json(output_dir / "summary.json", summary)
    (output_dir / "stdout.log").write_text(
        "\n".join(
            "{} {} {} diagnostics={}".format(
                row["workload_id"], row["public_mode"], row["outcome"], row.get("diagnostic_record_count", 0)
            )
            for row in results
        )
        + "\n"
    )
    (output_dir / "stderr.log").write_text(
        "\n".join(
            "candidate={}\n{}".format(results[index]["candidate_id"], stderr_by_index[index])
            for index in range(len(results))
            if stderr_by_index[index].strip()
        )
    )
    (output_dir / "command.txt").write_text(" ".join([sys.executable] + sys.argv) + "\n")

    fsim_paths = find_libvta("libvta_fsim", optional=True)
    fsim_path = Path(fsim_paths[0]).resolve() if fsim_paths else None
    repo_root = Path(__file__).resolve().parents[3]
    source_paths = {
        "wrapper": Path(__file__).resolve(),
        "collector": Path(__file__).with_name("collect_vta_fsim_command_signatures.py").resolve(),
        "runtime": repo_root / "vta/runtime/runtime.cc",
    }
    manifest = {
        "schema": "c3_p4j_manifest_v1",
        "status": STATUS,
        "date": "2026-09-11",
        "environment": {
            "python": sys.executable,
            "python_version": platform.python_version(),
            "tvm_version": tvm.__version__,
            "vta_target": vta.get_env().TARGET,
            "vta_hw_path": os.environ.get("VTA_HW_PATH"),
            "workers": args.workers,
            "ssh_used": False,
            "local_rpc_api_used": False,
            "rpc_to_board_used": False,
            "board_contacted": False,
            "execution_transport": "direct_local_module_no_rpc",
            "performance_measurement": "not_collected",
        },
        "inputs": frozen_inputs,
        "sources": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in source_paths.items()
        },
        "libvta_fsim": {
            "path": str(fsim_path) if fsim_path else None,
            "sha256": sha256_file(fsim_path) if fsim_path else None,
        },
        "results": summary,
    }
    write_json(output_dir / "manifest.json", manifest)
    overall = summary["overall"]
    (output_dir / "STATUS.md").write_text(
        "# C3-P4j full-pool local FSim command dry-run\n\n"
        "Status: `fsim_dry_run`. {}/{} candidates passed seed-0 exact correctness and "
        "structural command-signature checks; failures: {}. Unique stable IDs: {}. "
        "Mode-4 drain checks: {}/32. All diagnostic timing fields were discarded. This is "
        "not board-qualified capacity, board-runtime equivalence, or performance evidence.\n".format(
            overall["passed"],
            overall["candidates"],
            overall["failure_categories"],
            summary["unique_candidate_ids"],
            summary["mode4_drain_matches_existing_evidence"],
        )
    )
    (output_dir / "HANDOFF.md").write_text(
        "# HANDOFF\n\n"
        "`results.jsonl` contains candidate-level, address-free and timing-free structural "
        "CommandSignature evidence. `summary.json` aggregates submission counts, instruction/"
        "UOP totals and peaks, source-derived FINISH, and unobservable replay by mode and "
        "workload. P4g dependency audits are referenced per candidate; P4e/P4b candidates "
        "are explicitly marked as lacking a candidate-specific push/pop audit rather than "
        "inventing one. Any board capacity, semantic-equivalence, or speed claim requires a "
        "separate board experiment.\n"
    )
    outputs = {
        path.name: sha256_file(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    test_path = Path(__file__).with_name("test_collect_vta_full_pool_fsim_commands.py")
    source_hashes = {str(path): sha256_file(path) for path in source_paths.values()}
    if test_path.is_file():
        source_hashes[str(test_path.resolve())] = sha256_file(test_path)
    write_json(output_dir / "artifact_hashes.json", {"output_sha256": outputs, "source_sha256": source_hashes})
    print(
        "status={} candidates={} passed={} failed={} unique={}".format(
            STATUS,
            overall["candidates"],
            overall["passed"],
            overall["failed"],
            summary["unique_candidate_ids"],
        )
    )


def parse_args():
    base = Path(__file__).resolve().parent / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/04_dma_command_signatures"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p4b-results", default=str(base / "20260910_p4b_local_residency_pool_run01/results.jsonl"))
    parser.add_argument("--p4e-results", default=str(base / "20260911_p4e_unified_local_qualification_run01/results.jsonl"))
    parser.add_argument("--p4g-results", default=str(base / "20260911_p4g_weight_barrier_pool_run01/results.jsonl"))
    parser.add_argument("--p4i-results", default=str(base / "20260911_p4i_weight_barrier_cross_compile_run01/results.jsonl"))
    parser.add_argument("--output-dir")
    parser.add_argument("--candidate-timeout-seconds", type=int, default=180)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--worker-candidate")
    parser.add_argument("--worker-output")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.worker_candidate:
        if not parsed.worker_output:
            raise ValueError("--worker-output is required in worker mode")
        write_json(parsed.worker_output, direct_worker_result(load_json(parsed.worker_candidate)))
    else:
        if not parsed.output_dir:
            raise ValueError("--output-dir is required")
        run_main(parsed)
