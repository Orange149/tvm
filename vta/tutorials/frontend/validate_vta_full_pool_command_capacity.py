#!/usr/bin/env python3
"""Validate a compile-derived command capacity over the frozen 197-candidate pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from collect_vta_full_pool_fsim_commands import (
    load_jsonl,
    run_candidate_no_rpc,
    select_full_pool,
    strip_diagnostic_timing,
)
from validate_vta_p7r_command_resource_plan import (
    LEGACY_PER_QUEUE_BYTES,
    align_up,
)


RESULT_SCHEMA = "c3_full_pool_reduced_command_capacity_result_v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def verify_frozen_file(path):
    path = Path(path).resolve()
    ledger = load_json(path.parent / "artifact_hashes.json")
    hashes = ledger.get("artifacts", ledger.get("output_sha256", ledger))
    observed = sha256_file(path)
    if hashes.get(path.name) != observed:
        raise ValueError("frozen artifact hash mismatch: {}".format(path))
    return {"path": str(path), "sha256": observed}


def structural_peaks(row):
    peaks = row["command_signature"]["structural"]["peaks"]
    return int(peaks["insn_bytes"]), int(peaks["uop_bytes"])


def make_plan(rows):
    if len(rows) != 197 or len({row["candidate_id"] for row in rows}) != 197:
        raise ValueError("expected 197 unique frozen command signatures")
    if not all(row.get("outcome") == "passed_local_signature" for row in rows):
        raise ValueError("all baseline signatures must pass")
    max_insn = max(structural_peaks(row)[0] for row in rows)
    max_uop = max(structural_peaks(row)[1] for row in rows)
    insn_capacity = align_up(max_insn)
    uop_capacity = align_up(max_uop)
    total = insn_capacity + uop_capacity
    return {
        "schema": "c3_full_pool_command_resource_plan_v1",
        "status": "frozen_before_reduced_capacity_validation",
        "identity_set_size": len(rows),
        "workloads": sorted({row["workload_id"] for row in rows}),
        "modes": sorted({row["public_mode"] for row in rows}),
        "formula": "C_q(S)=align_4KiB(max_{candidate in S, submit in candidate} bytes_q(candidate, submit))",
        "insn_peak_bytes": max_insn,
        "uop_peak_bytes": max_uop,
        "insn_capacity_bytes": insn_capacity,
        "uop_capacity_bytes": uop_capacity,
        "total_capacity_bytes": total,
        "legacy_total_capacity_bytes": 2 * LEGACY_PER_QUEUE_BYTES,
        "requested_byte_reduction_fraction": 1.0
        - total / (2 * LEGACY_PER_QUEUE_BYTES),
        "finish_rule": "baseline peaks include the runtime-appended FINISH for every submission",
        "runtime_rule": "both instruction and UOP capacities are checked before device submission",
        "invalidation_key": [
            "hardware fingerprint",
            "workload shape",
            "residency mode",
            "complete ConfigEntity",
            "lowered TIR hash",
            "runtime source hash",
        ],
        "claim_boundary": (
            "ten workload identities W00--W09 and 197 frozen static identities; local FSim capacity safety, "
            "not board allocator occupancy, dynamic-shape, stage latency, or FPS"
        ),
    }


def worker(candidate, insn_capacity, uop_capacity, timeout_seconds):
    os.environ["VTA_INSN_BUFFER_BYTES"] = str(insn_capacity)
    os.environ["VTA_UOP_BUFFER_BYTES"] = str(uop_capacity)
    os.environ.pop("VTA_INSN_SUBMIT_THRESHOLD_BYTES", None)
    result, _ = run_candidate_no_rpc(candidate, timeout_seconds)
    strip_diagnostic_timing(result)
    records = result.get("queue_records", [])
    result.update(
        schema=RESULT_SCHEMA,
        candidate_role=candidate["candidate_role"],
        reported_capacity_matches=bool(records)
        and all(
            int(record["insn_capacity"]) == insn_capacity
            and int(record["uop_capacity"]) == uop_capacity
            for record in records
        ),
    )
    return result


def run_isolated(candidate, plan, timeout_seconds):
    with tempfile.TemporaryDirectory(prefix="c3_full_capacity_") as directory:
        directory = Path(directory)
        candidate_path = directory / "candidate.json"
        result_path = directory / "result.json"
        write_json(candidate_path, candidate)
        environment = os.environ.copy()
        environment["VTA_INSN_BUFFER_BYTES"] = str(plan["insn_capacity_bytes"])
        environment["VTA_UOP_BUFFER_BYTES"] = str(plan["uop_capacity_bytes"])
        environment.pop("VTA_INSN_SUBMIT_THRESHOLD_BYTES", None)
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker-candidate",
                str(candidate_path),
                "--worker-output",
                str(result_path),
                "--insn-capacity",
                str(plan["insn_capacity_bytes"]),
                "--uop-capacity",
                str(plan["uop_capacity_bytes"]),
                "--candidate-timeout-seconds",
                str(timeout_seconds),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds + 30,
            env=environment,
        )
        if completed.returncode != 0 or not result_path.is_file():
            return {
                "schema": RESULT_SCHEMA,
                "candidate_id": candidate["candidate_id"],
                "workload_id": candidate["workload_id"],
                "public_mode": candidate["public_mode"],
                "outcome": "negative_worker_failure",
                "error": "capacity worker exit={} output_present={}".format(
                    completed.returncode, result_path.is_file()
                ),
            }
        return load_json(result_path)


def main(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    output.mkdir(parents=True)
    input_paths = {
        "p4b": Path(args.p4b_results),
        "p4e": Path(args.p4e_results),
        "p4g": Path(args.p4g_results),
        "p4i": Path(args.p4i_results),
        "baseline": Path(args.baseline_results),
    }
    input_guards = {name: verify_frozen_file(path) for name, path in input_paths.items()}
    baseline = load_jsonl(input_paths["baseline"])
    plan = make_plan(baseline)
    plan["input_guards"] = input_guards
    root = Path(__file__).resolve().parents[3]
    source_paths = [
        root / "vta/runtime/runtime.cc",
        root / "vta/runtime/queue_capacity.h",
        root / "vta/tutorials/frontend/collect_vta_full_pool_fsim_commands.py",
    ]
    plan["source_guards_sha256"] = {
        str(path.relative_to(root)): sha256_file(path) for path in source_paths
    }
    plan_path = output / "resource_plan.json"
    write_json(plan_path, plan)

    candidates = select_full_pool(
        load_jsonl(input_paths["p4b"]),
        load_jsonl(input_paths["p4e"]),
        load_jsonl(input_paths["p4g"]),
        load_jsonl(input_paths["p4i"]),
    )
    baseline_by_id = {row["candidate_id"]: row for row in baseline}
    if {row["candidate_id"] for row in candidates} != set(baseline_by_id):
        raise ValueError("reconstructed candidate population differs from command baseline")
    candidate_path = output / "candidates.jsonl"
    candidate_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in candidates)
    )
    write_json(
        output / "pre_validation_hashes.json",
        {
            "artifacts": {
                "resource_plan.json": sha256_file(plan_path),
                "candidates.jsonl": sha256_file(candidate_path),
            }
        },
    )

    indexed = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_isolated, candidate, plan, args.candidate_timeout_seconds): index
            for index, candidate in enumerate(candidates)
        }
        for future in as_completed(futures):
            index = futures[future]
            result = future.result()
            baseline_row = baseline_by_id[result["candidate_id"]]
            result["peak_matches_baseline"] = (
                result.get("command_signature", {}).get("structural", {}).get("peaks")
                == baseline_row["command_signature"]["structural"]["peaks"]
            )
            result["capacity_validation_passed"] = (
                result.get("outcome") == "passed_local_signature"
                and result.get("reported_capacity_matches") is True
                and result["peak_matches_baseline"]
            )
            indexed[index] = result
            print(
                "{}/{} {} {} {}".format(
                    len(indexed), len(candidates), result["workload_id"],
                    result["public_mode"], result["capacity_validation_passed"]
                ),
                flush=True,
            )
    results = [indexed[index] for index in range(len(candidates))]
    results_path = output / "validation.jsonl"
    results_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in results))
    failures = [row for row in results if not row["capacity_validation_passed"]]
    summary = {
        "schema": "c3_full_pool_reduced_command_capacity_summary_v1",
        "status": "passed" if not failures else "failed",
        "selected": len(results),
        "passed": len(results) - len(failures),
        "failed": len(failures),
        "failure_outcomes": dict(Counter(row.get("outcome") for row in failures)),
        "workloads": {
            workload: {
                "selected": sum(row["workload_id"] == workload for row in results),
                "passed": sum(
                    row["workload_id"] == workload and row["capacity_validation_passed"]
                    for row in results
                ),
            }
            for workload in sorted({row["workload_id"] for row in results})
        },
        "capacity": {
            key: plan[key]
            for key in (
                "insn_peak_bytes", "uop_peak_bytes", "insn_capacity_bytes",
                "uop_capacity_bytes", "total_capacity_bytes",
                "legacy_total_capacity_bytes", "requested_byte_reduction_fraction",
            )
        },
        "performance_measurement": "not_collected",
        "board_contacted": False,
        "claim_boundary": plan["claim_boundary"],
    }
    write_json(output / "summary.json", summary)
    (output / "STATUS.md").write_text(
        "# Full-pool reduced command-capacity validation\n\n"
        "- Status: `{}` ({}/{})\n"
        "- Pool: {} workloads, {} modes, 197 frozen identities\n"
        "- Peak: instruction {} B; UOP {} B\n"
        "- Capacity: instruction {} B + UOP {} B = {} B\n"
        "- Requested-byte reduction from 64 MiB: {:.4%}\n"
        "- Local FSim only; no board allocator, stage latency, or FPS claim\n".format(
            summary["status"], summary["passed"], summary["selected"],
            len(plan["workloads"]), len(plan["modes"]), plan["insn_peak_bytes"],
            plan["uop_peak_bytes"], plan["insn_capacity_bytes"],
            plan["uop_capacity_bytes"], plan["total_capacity_bytes"],
            plan["requested_byte_reduction_fraction"],
        )
    )
    hashes = {
        path.name: sha256_file(path)
        for path in output.iterdir()
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    write_json(output / "artifact_hashes.json", {"artifacts": hashes})
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args():
    base = Path(__file__).resolve().parent / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/04_dma_command_signatures"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p4b-results", default=str(base / "20260910_p4b_local_residency_pool_run01/results.jsonl"))
    parser.add_argument("--p4e-results", default=str(base / "20260911_p4e_unified_local_qualification_run01/results.jsonl"))
    parser.add_argument("--p4g-results", default=str(base / "20260911_p4g_weight_barrier_pool_run01/results.jsonl"))
    parser.add_argument("--p4i-results", default=str(base / "20260911_p4i_weight_barrier_cross_compile_run01/results.jsonl"))
    parser.add_argument("--baseline-results", default=str(base / "20260911_p4j_full_pool_fsim_command_run02/results.jsonl"))
    parser.add_argument("--output-dir")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--candidate-timeout-seconds", type=int, default=180)
    parser.add_argument("--worker-candidate")
    parser.add_argument("--worker-output")
    parser.add_argument("--insn-capacity", type=int)
    parser.add_argument("--uop-capacity", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.worker_candidate:
        if not all((parsed.worker_output, parsed.insn_capacity, parsed.uop_capacity)):
            raise ValueError("worker mode requires output and both capacities")
        write_json(
            parsed.worker_output,
            worker(
                load_json(parsed.worker_candidate), parsed.insn_capacity,
                parsed.uop_capacity, parsed.candidate_timeout_seconds,
            ),
        )
    else:
        if not parsed.output_dir:
            raise ValueError("--output-dir is required")
        main(parsed)
