#!/usr/bin/env python3
"""Collect non-performance VTA command signatures from local FSim diagnostics.

This tool intentionally runs only the frozen P3e W00/W02/W09 representatives,
each in original mode and in the isolated mode-4 weight-barrier probe.  It uses
seed 0 and exact NumPy equality.  The resulting ``fsim_dry_run`` records are not
board-qualified queue capacities and contain no valid performance measurement.
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
from tvm import autotvm, rpc
from tvm.contrib import utils
import vta
from vta.libinfo import find_libvta

import vta.top.vta_conv2d_residency  # pylint: disable=unused-import
from c3_candidate_identity import canonical_json_bytes, normalize_config_entity
from qualify_vta_residency_fsim import array_sha256, reference_data_from_workload


RESULT_SCHEMA = "c3_fsim_command_signature_v1"
SUMMARY_SCHEMA = "c3_fsim_command_signature_summary_v1"
STATUS = "fsim_dry_run"
SEED = 0
WORKLOAD_IDS = ("W00", "W02", "W09")
IMPLEMENTATION_MODES = {"original": 0, "weight_stationary_barrier": 4}
REQUIRED_QUEUE_FIELDS = {
    "submit",
    "queue_id",
    "reason",
    "insn_capacity",
    "uop_capacity",
    "submit_threshold",
    "submit_threshold_configured",
    "insn_bytes",
    "uop_bytes",
    "insn_peak",
    "uop_peak",
    "load_bytes",
    "store_bytes",
    "timeout",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def semantic_config_key(config):
    return canonical_json_bytes(normalize_config_entity(config)).decode("utf-8")


def parse_queue_records(stderr):
    """Parse every complete JSON object tagged with ``[VTA_QUEUE]``."""

    records = []
    malformed = []
    marker = "[VTA_QUEUE] "
    for line_number, line in enumerate(stderr.splitlines(), 1):
        if marker not in line:
            continue
        payload = line.split(marker, 1)[1].strip()
        try:
            record = json.loads(payload)
        except json.JSONDecodeError as error:
            malformed.append({"line": line_number, "error": str(error), "payload": payload})
            continue
        missing = sorted(REQUIRED_QUEUE_FIELDS - set(record))
        if missing:
            malformed.append(
                {"line": line_number, "error": "missing required fields", "missing": missing}
            )
            continue
        records.append(record)
    return records, malformed


def command_signature(queue_records, implementation_mode, expected_residency_drains):
    """Return a timing-free, address-free structural command signature.

    ``runtime.cc`` appends one FINISH immediately before every non-empty device
    submission.  The diagnostic JSON does not expose an opcode histogram, so the
    FINISH count below is explicitly source-derived rather than log-observed.
    Replay bypasses the diagnostic print path and has no field in ``[VTA_QUEUE]``;
    it therefore remains not observable here.
    """

    normalized = [
        {
            "submit": int(row["submit"]),
            "reason": str(row["reason"]),
            "insn_capacity": int(row["insn_capacity"]),
            "uop_capacity": int(row["uop_capacity"]),
            "submit_threshold": int(row["submit_threshold"]),
            "submit_threshold_configured": bool(row["submit_threshold_configured"]),
            "insn_bytes": int(row["insn_bytes"]),
            "uop_bytes": int(row["uop_bytes"]),
            "insn_peak": int(row["insn_peak"]),
            "uop_peak": int(row["uop_peak"]),
            "load_bytes": int(row["load_bytes"]),
            "store_bytes": int(row["store_bytes"]),
            "timeout": int(row["timeout"]),
        }
        for row in queue_records
    ]
    reasons = Counter(row["reason"] for row in normalized)
    function_final_drains = 1 if normalized else 0
    observed_residency_drains = max(0, reasons.get("explicit_sync", 0) - function_final_drains)
    structural = {
        "schema": "c3_vta_command_signature_structural_v1",
        "submissions": len(normalized),
        "submit_reason_counts": dict(sorted(reasons.items())),
        "submit_sequence": normalized,
        "totals": {
            "insn_bytes": sum(row["insn_bytes"] for row in normalized),
            "uop_bytes": sum(row["uop_bytes"] for row in normalized),
            "load_bytes": sum(row["load_bytes"] for row in normalized),
            "store_bytes": sum(row["store_bytes"] for row in normalized),
        },
        "peaks": {
            "insn_bytes": max((row["insn_peak"] for row in normalized), default=0),
            "uop_bytes": max((row["uop_peak"] for row in normalized), default=0),
        },
        "finish": {
            "observed_in_queue_json": False,
            "source_derived_count": len(normalized),
            "derivation": "runtime.cc appends exactly one FINISH before each non-empty VTADeviceRun",
        },
        "replay": {
            "observed_in_queue_json": False,
            "runner_requested_capture_or_replay": False,
            "interpretation": "not_observable; replay path does not emit [VTA_QUEUE]",
        },
        "drain_check": {
            "implementation_mode": int(implementation_mode),
            "function_final_drain_count_assumed": function_final_drains,
            "expected_residency_drain_count_from_p3e": int(expected_residency_drains),
            "observed_residency_drain_count": observed_residency_drains,
            "matches_p3e": observed_residency_drains == int(expected_residency_drains),
        },
    }
    return {
        "structural": structural,
        "sha256": hashlib.sha256(canonical_json_bytes(structural)).hexdigest(),
        "excluded_as_nondeterministic_or_performance": [
            "queue_id",
            "finalize_us",
            "uop_pack_us",
            "uop_copy_us",
            "insn_copy_us",
            "queue_cache_us",
            "device_run_us",
            "submit_mmio_us",
            "poll_wait_us",
        ],
    }


def select_representatives(p4b_rows, p4g_rows, p3e_rows):
    """Join the six frozen candidates without selecting on FSim command outcomes."""

    p3e = {row["workload_id"]: row for row in p3e_rows}
    if set(p3e) != set(WORKLOAD_IDS):
        raise ValueError("P3e representative set must be exactly W00/W02/W09")
    selected = []
    for workload_id in WORKLOAD_IDS:
        evidence = p3e[workload_id]
        index = int(evidence["config_index"])
        originals = [
            row
            for row in p4b_rows
            if row.get("workload_id") == workload_id
            and row.get("residence_mode") == "original"
            and int(row.get("debug", {}).get("config_index", -1)) == index
            and row.get("status") == "ok"
        ]
        barriers = [
            row
            for row in p4g_rows
            if row.get("workload_id") == workload_id
            and row.get("residence_mode") == "weight_stationary_barrier"
            and int(row.get("implementation_mode", -1)) == 4
            and int(row.get("debug", {}).get("config_index", -1)) == index
            and row.get("status") == "ok"
        ]
        if len(originals) != 1 or len(barriers) != 1:
            raise ValueError(
                "{} expected one original and one mode4; got {}/{}".format(
                    workload_id, len(originals), len(barriers)
                )
            )
        original, barrier = originals[0], barriers[0]
        if barrier.get("source_p4b_original_candidate_id") != original["candidate_id"]:
            raise ValueError("{} mode4 is not linked to selected original".format(workload_id))
        if original.get("tir_sha256") != evidence["original"]["tir_sha256"]:
            raise ValueError("{} original TIR differs from P3e".format(workload_id))
        if barrier.get("tir_sha256") != evidence["mode4"]["tir_sha256"]:
            raise ValueError("{} mode4 TIR differs from P3e".format(workload_id))
        expected_mode4_drains = int(evidence["mode4"]["sync_static_execution_multiplicity"][0])
        for public_mode, implementation_mode, candidate, expected_drains in (
            ("original", 0, original, 0),
            ("weight_stationary_barrier", 4, barrier, expected_mode4_drains),
        ):
            selected.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "workload_id": workload_id,
                    "public_mode": public_mode,
                    "implementation_mode": implementation_mode,
                    "identity": candidate["identity"],
                    "debug": candidate["debug"],
                    "expected_tir_sha256": candidate["tir_sha256"],
                    "expected_residency_drains": expected_drains,
                    "p3e_static_dma": {
                        "input_bytes": evidence[public_mode if public_mode == "original" else "mode4"][
                            "input_bytes"
                        ],
                        "weight_bytes": evidence[public_mode if public_mode == "original" else "mode4"][
                            "weight_bytes"
                        ],
                        "store_calls": evidence[public_mode if public_mode == "original" else "mode4"][
                            "store_calls"
                        ],
                    },
                }
            )
    return selected


def _create_task(candidate, env):
    workload = candidate["identity"]["workload"]
    return autotvm.task.create(
        "conv2d_packed_residency.vta",
        args=tuple(workload[1:]) + (candidate["implementation_mode"],),
        target=env.target,
        target_host=env.target_host,
    )


def worker_result(candidate):
    """Build and execute exactly seed 0; diagnostics are captured by the parent."""

    result = {
        "schema": RESULT_SCHEMA,
        "status": STATUS,
        "candidate_id": candidate["candidate_id"],
        "workload_id": candidate["workload_id"],
        "public_mode": candidate["public_mode"],
        "implementation_mode": candidate["implementation_mode"],
        "build": "not_attempted",
        "correctness": "not_attempted",
        "seed": SEED,
        "performance_measurement": "not_collected",
    }
    env = vta.get_env()
    if env.TARGET != "sim":
        result.update(outcome="negative_environment", error="VTA TARGET must be sim")
        return result
    try:
        from vta.testing import simulator

        if not simulator.enabled():
            raise RuntimeError("VTA FSim runtime is not enabled")
        task = _create_task(candidate, env)
        config = task.config_space.get(int(candidate["debug"]["config_index"]))
        if semantic_config_key(config.to_json_dict()) != semantic_config_key(
            candidate["identity"]["complete_config_entity"]
        ):
            raise RuntimeError("ConfigEntity differs from frozen candidate identity")
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
        temporary = utils.tempdir()
        module_name = candidate["candidate_id"] + ".o"
        module_path = temporary.relpath(module_name)
        module.save(module_path)
        remote = rpc.LocalSession()
        remote.upload(module_path)
        function = remote.load_module(module_name)["main"]
        device = remote.ext_dev(0)
        data, weight, expected = reference_data_from_workload(
            candidate["identity"]["workload"], SEED
        )
        output = tvm.nd.empty(expected.shape, "int8", device=device)
        function(tvm.nd.array(data, device), tvm.nd.array(weight, device), output)
        actual = output.numpy()
        mismatches = int(np.count_nonzero(actual != expected))
        result.update(
            correctness="passed" if mismatches == 0 else "failed",
            mismatch_count=mismatches,
            expected_sha256=array_sha256(expected),
            actual_sha256=array_sha256(actual),
            outcome="pending_parent_diagnostic_parse" if mismatches == 0 else "negative_wrong_answer",
        )
    except Exception as error:  # worker must always leave a machine-readable record
        result.update(outcome="negative_worker_failure", error=(str(error) or type(error).__name__)[:4000])
    return result


def run_candidate(candidate, timeout_seconds):
    with tempfile.TemporaryDirectory(prefix="c3_p4h_worker_") as directory:
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
                "seed": SEED,
                "outcome": "negative_timeout",
                "error": "worker exceeded {} seconds".format(timeout_seconds),
                "performance_measurement": "not_collected",
            }, error.stderr or ""
        queue_records, malformed = parse_queue_records(completed.stderr)
        if completed.returncode != 0 or not output_path.is_file():
            result = {
                "schema": RESULT_SCHEMA,
                "status": STATUS,
                "candidate_id": candidate["candidate_id"],
                "workload_id": candidate["workload_id"],
                "public_mode": candidate["public_mode"],
                "implementation_mode": candidate["implementation_mode"],
                "seed": SEED,
                "outcome": "negative_worker_failure",
                "error": "worker exit={} output_present={}".format(
                    completed.returncode, output_path.is_file()
                ),
                "performance_measurement": "not_collected",
            }
        else:
            result = load_json(output_path)
        result["queue_records"] = queue_records
        result["malformed_queue_records"] = malformed
        result["diagnostic_record_count"] = len(queue_records)
        if result.get("correctness") == "passed" and queue_records and not malformed:
            signature = command_signature(
                queue_records,
                candidate["implementation_mode"],
                candidate["expected_residency_drains"],
            )
            result["command_signature"] = signature
            drain_ok = signature["structural"]["drain_check"]["matches_p3e"]
            timeouts_ok = all(int(row["timeout"]) == 0 for row in queue_records)
            result["outcome"] = (
                "passed_local_signature"
                if drain_ok and timeouts_ok
                else "negative_signature_mismatch"
            )
        elif result.get("correctness") == "passed" and not queue_records:
            result["outcome"] = "negative_no_queue_diagnostics"
        result["claim_boundary"] = (
            "local fsim dry-run only; not board-qualified capacity, not board-runtime semantic "
            "equivalence, and not performance"
        )
        return result, completed.stderr


def summarize(results):
    outcomes = Counter(row["outcome"] for row in results)
    passed = [row for row in results if row["outcome"] == "passed_local_signature"]
    return {
        "schema": SUMMARY_SCHEMA,
        "status": STATUS,
        "scope": "six local FSim seed-0 correctness command dry-runs; no board and no performance",
        "selected_candidates": len(results),
        "passed_local_signature": len(passed),
        "outcomes": dict(sorted(outcomes.items())),
        "elementwise_correct": sum(row.get("correctness") == "passed" for row in results),
        "diagnostic_records": sum(row.get("diagnostic_record_count", 0) for row in results),
        "mode4_drain_matches_p3e": sum(
            row.get("implementation_mode") == 4
            and row.get("command_signature", {})
            .get("structural", {})
            .get("drain_check", {})
            .get("matches_p3e", False)
            for row in results
        ),
        "claim_boundary": {
            "board_qualified_capacity": False,
            "board_runtime_semantic_equivalence": "not_established",
            "performance_measurement": "not_collected",
            "single_operator_fps": "not_measured",
            "end_to_end_fps": "not_measured",
        },
    }


def run_main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty evidence directory")
    input_paths = {
        "p4b_results": Path(args.p4b_results),
        "p4g_results": Path(args.p4g_results),
        "p3e_results": Path(args.p3e_results),
    }
    candidates = select_representatives(
        load_jsonl(input_paths["p4b_results"]),
        load_jsonl(input_paths["p4g_results"]),
        load_jsonl(input_paths["p3e_results"]),
    )
    write_json(
        output_dir / "preregistered.json",
        {
            "schema": "c3_p4h_preregistered_v1",
            "status": STATUS,
            "selection": "P3e W00/W02/W09 exact original/mode4 ConfigEntity and frozen TIR",
            "seed": SEED,
            "correctness": "exact elementwise equality against independent NumPy reference",
            "diagnostic": "VTA_QUEUE_DIAGNOSTICS=1; parse every [VTA_QUEUE] JSON line",
            "mode4_drain_gate": "explicit_sync submissions minus one function-final drain equals P3e 2/4/2",
            "negative_rule": "freeze no/malformed diagnostics or semantic mismatch; do not substitute timing",
            "performance_rule": "no FSim or wall-clock timing is a performance observation",
        },
    )
    results = []
    stderr_blocks = []
    stdout_lines = []
    for position, candidate in enumerate(candidates, 1):
        result, stderr = run_candidate(candidate, args.candidate_timeout_seconds)
        results.append(result)
        stderr_blocks.append(
            "candidate={} workload={} mode={}\n{}\n".format(
                candidate["candidate_id"], candidate["workload_id"], candidate["public_mode"], stderr
            )
        )
        stdout_lines.append(
            "{}/{} {} {} {} diagnostics={}".format(
                position,
                len(candidates),
                candidate["workload_id"],
                candidate["public_mode"],
                result["outcome"],
                result.get("diagnostic_record_count", 0),
            )
        )
    (output_dir / "results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in results)
    )
    summary = summarize(results)
    write_json(output_dir / "summary.json", summary)
    (output_dir / "stdout.log").write_text("\n".join(stdout_lines) + "\n")
    (output_dir / "stderr.log").write_text("\n".join(stderr_blocks))
    (output_dir / "command.txt").write_text(" ".join([sys.executable] + sys.argv) + "\n")

    repo_root = Path(__file__).resolve().parents[3]
    runtime_source = repo_root / "vta/runtime/runtime.cc"
    fsim_candidates = find_libvta("libvta_fsim", optional=True)
    fsim_library = Path(fsim_candidates[0]).resolve() if fsim_candidates else None
    manifest = {
        "schema": "c3_p4h_manifest_v1",
        "status": STATUS,
        "date": "2026-09-11",
        "environment": {
            "python": sys.executable,
            "python_version": platform.python_version(),
            "tvm_version": tvm.__version__,
            "vta_target": vta.get_env().TARGET,
            "vta_hw_path": os.environ.get("VTA_HW_PATH"),
            "VTA_QUEUE_DIAGNOSTICS": "set to 1 in each isolated worker",
            "ssh_used": False,
            "board_contacted": False,
            "performance_measurement": "not_collected",
        },
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in input_paths.items()
        },
        "sources": {
            "collector": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(__file__)},
            "runtime": {"path": str(runtime_source), "sha256": sha256_file(runtime_source)},
            "libvta_fsim": {
                "path": str(fsim_library) if fsim_library else None,
                "sha256": sha256_file(fsim_library) if fsim_library else None,
            },
        },
        "results": summary,
    }
    write_json(output_dir / "manifest.json", manifest)
    (output_dir / "STATUS.md").write_text(
        "# C3-P4h local FSim command signatures\n\n"
        "Status: `fsim_dry_run`. {}/{} candidates retained exact seed-0 correctness and a "
        "structurally valid queue signature; mode-4 drain checks matching P3e: {}/3. "
        "This is neither board-qualified capacity nor performance evidence. Replay is not "
        "observable in `[VTA_QUEUE]`; FINISH count is source-derived, not opcode-observed.\n".format(
            summary["passed_local_signature"], len(results), summary["mode4_drain_matches_p3e"]
        )
    )
    (output_dir / "HANDOFF.md").write_text(
        "# HANDOFF\n\n"
        "Use `results.jsonl` only as candidate-level local FSim command-shape evidence. "
        "The signature hash excludes addresses and all timing fields. A separate board run "
        "must establish board-runtime equivalence, safe queue capacity, and any performance "
        "claim. Negative or absent diagnostics are frozen rather than imputed.\n"
    )
    hashes = {
        path.name: sha256_file(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    test_path = Path(__file__).with_name("test_collect_vta_fsim_command_signatures.py")
    source_hashes = {str(Path(__file__).resolve()): sha256_file(__file__)}
    if test_path.is_file():
        source_hashes[str(test_path.resolve())] = sha256_file(test_path)
    write_json(
        output_dir / "artifact_hashes.json",
        {"output_sha256": hashes, "source_sha256": source_hashes},
    )
    print(
        "status={} candidates={} passed={} mode4_drain_matches={}/3".format(
            STATUS, len(results), summary["passed_local_signature"], summary["mode4_drain_matches_p3e"]
        )
    )


def parse_args():
    base = Path(__file__).resolve().parent / "report_out/stage_tile_cotuning/c3_dma_residency_autotune"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--p4b-results",
        default=str(
            base
            / "04_dma_command_signatures/20260910_p4b_local_residency_pool_run01/results.jsonl"
        ),
    )
    parser.add_argument("--p4g-results", required=False)
    parser.add_argument(
        "--p3e-results",
        default=str(
            base
            / "03_residency_schedules/20260910_p3e_weight_barrier_legalizer_run01/results.jsonl"
        ),
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--candidate-timeout-seconds", type=int, default=180)
    parser.add_argument("--worker-candidate")
    parser.add_argument("--worker-output")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.worker_candidate:
        if not args.worker_output:
            raise ValueError("--worker-output is required in worker mode")
        write_json(args.worker_output, worker_result(load_json(args.worker_candidate)))
        return
    if not args.p4g_results or not args.output_dir:
        raise ValueError("--p4g-results and --output-dir are required")
    run_main(args)


if __name__ == "__main__":
    main()
