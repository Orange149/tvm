#!/usr/bin/env python3
"""Use qualified P5B stage telemetry to revise the CPU cost and rerun exact ranking."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
from pathlib import Path

from freeze_cpu_vta_pipeline_v1 import (
    PROTOCOL_ID,
    load_sealed_artifact,
    repo_root,
    seal_artifact,
    validate_artifact_sha256,
)
from build_cpu_vta_pipeline_v1_p5a import audit_package, candidate_dir_name
from solve_cpu_vta_pipeline_v1_p3 import (
    build_context,
    candidate_id,
    candidate_record,
    enumerate_oracle,
    solve_k_best_label_setting,
)


DEFAULT_OUTPUT = repo_root() / "vta/tutorials/frontend/report_out/resource_aware_maxplus"
CALIBRATED_SEGMENT = "cpu:00:02"
CONTROL_TOPOLOGY = "cpu-00-02__vta-03-17__cpu-18-20"


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def summarize_pipeline(path, warmup=2):
    rows = read_jsonl(path)
    if len(rows) <= warmup + 1:
        raise RuntimeError("pipeline result has too few rows: {}".format(path))
    scored = rows[warmup:]
    ends = [float(row["stage2_end_ms"]) for row in scored]
    intervals = [right - left for left, right in zip(ends, ends[1:])]
    cycle_ms = (ends[-1] - ends[0]) / (len(ends) - 1)
    return {
        "row_count": len(rows),
        "warmup_count": warmup,
        "scored_count": len(scored),
        "cycle_ms": cycle_ms,
        "fps": 1000.0 / cycle_ms,
        "completion_interval_median_ms": statistics.median(intervals),
        "completion_interval_min_ms": min(intervals),
        "completion_interval_max_ms": max(intervals),
        "stage0_run_ms_mean": statistics.mean(float(row["stage0_run_ms"]) for row in scored),
        "stage0_run_ms_median": statistics.median(
            float(row["stage0_run_ms"]) for row in scored
        ),
        "stage1_run_ms_median": statistics.median(
            float(row["stage1_run_ms"]) for row in scored
        ),
        "stage2_run_ms_median": statistics.median(
            float(row["stage2_run_ms"]) for row in scored
        ),
        "output_fingerprints": sorted(
            {
                output["fnv1a64"]
                for row in rows
                for output in row.get("raw_outputs", [])
            }
        ),
    }


def summarize_serial(path):
    rows = read_jsonl(path)
    if len(rows) != 1:
        raise RuntimeError("iteration expects one isolated serial correctness row")
    row = rows[0]
    return {
        "stage0_run_ms": float(row["stage0_run_ms"]),
        "stage0_run_process_cpu_ms": float(row["stage0_run_process_cpu_ms"]),
        "stage1_run_ms": float(row["stage1_run_ms"]),
        "stage2_run_ms": float(row["stage2_run_ms"]),
    }


def default_control_sources(output_dir):
    root = Path(output_dir)
    iteration = root / "v1_p5b_iteration1"
    return {
        1: {
            "candidate_id": "cpu-00-02_t1__vta-03-17_t1__cpu-18-20_t2",
            "serial": root / "v1_p5b_quick_smoke/serial_result.jsonl",
            "pipeline": root / "v1_p5b_quick_fps/pipeline_result_22.jsonl",
            "correctness": root / "v1_p5b_quick_smoke.json",
        },
        2: {
            "candidate_id": "cpu-00-02_t2__vta-03-17_t1__cpu-18-20_t2",
            "serial": iteration / "thread2/correctness/stage_serial_result.jsonl",
            "pipeline": iteration / "thread2/performance/native_result.jsonl",
            "correctness": iteration / "thread2/correctness/reference_comparison.json",
        },
        3: {
            "candidate_id": "cpu-00-02_t3__vta-03-17_t1__cpu-18-20_t2",
            "serial": iteration / "thread3/correctness/stage_serial_result.jsonl",
            "pipeline": iteration / "thread3/performance/native_result.jsonl",
            "correctness": iteration / "thread3/correctness/reference_comparison.json",
        },
        4: {
            "candidate_id": "cpu-00-02_t4__vta-03-17_t1__cpu-18-20_t2",
            "serial": iteration / "thread4/correctness/stage_serial_result.jsonl",
            "pipeline": iteration / "thread4/performance_rerun/native_result.jsonl",
            "correctness": iteration / "thread4/correctness/reference_comparison.json",
        },
    }


def correctness_passed(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("kind") == "cpu_vta_pipeline_v1_p5_tensor_comparison":
        return bool(payload.get("passed"))
    if payload.get("kind") == "cpu_vta_pipeline_v1_p5b_quick_smoke":
        return bool(payload.get("serial_pipeline_equivalence", {}).get("byte_exact")) and bool(
            payload.get("independent_reference", {}).get("frozen_numeric_gate_passed")
        )
    gate = payload.get("reference_comparison") or payload.get("qualification") or {}
    return bool(gate.get("passed", payload.get("correctness_passed", True)))


def build_measurements(output_dir):
    rows = []
    for threads, source in sorted(default_control_sources(output_dir).items()):
        if not correctness_passed(source["correctness"]):
            raise RuntimeError("correctness gate failed for threads={}".format(threads))
        serial = summarize_serial(source["serial"])
        pipeline = summarize_pipeline(source["pipeline"])
        rows.append(
            {
                "threads": threads,
                "candidate_id": source["candidate_id"],
                "topology_id": CONTROL_TOPOLOGY,
                "correctness_passed": True,
                "serial": serial,
                "pipeline": pipeline,
                "source_paths": {key: str(Path(value)) for key, value in source.items() if key != "candidate_id"},
            }
        )
    return rows


def apply_segment_observations(context, measurements):
    revised = copy.deepcopy(context)
    cost = revised["segment_costs"][CALIBRATED_SEGMENT]
    original = {
        "service_ms_by_threads": copy.deepcopy(cost["service_ms_by_threads"]),
        "host_core_demand_ms_by_threads": copy.deepcopy(
            cost["host_core_demand_ms_by_threads"]
        ),
    }
    for row in measurements:
        key = str(row["threads"])
        cost["service_ms_by_threads"][key] = row["serial"]["stage0_run_ms"]
        cost["host_core_demand_ms_by_threads"][key] = row["serial"][
            "stage0_run_process_cpu_ms"
        ]
    return revised, original


def apply_global_cpu_thread_correction(context, measurements):
    """Correct the global P2 CPU slope without changing the additive model family."""
    revised = copy.deepcopy(context)
    anchor = context["segment_costs"][CALIBRATED_SEGMENT]
    service_multiplier = {}
    core_multiplier = {}
    for row in measurements:
        key = str(row["threads"])
        service_multiplier[key] = (
            row["serial"]["stage0_run_ms"] / float(anchor["service_ms_by_threads"][key])
        )
        core_multiplier[key] = row["serial"]["stage0_run_process_cpu_ms"] / float(
            anchor["host_core_demand_ms_by_threads"][key]
        )
    updated = []
    for segment_id, cost in revised["segment_costs"].items():
        if revised["profile_segments"][segment_id]["device"] != "cpu":
            continue
        for key in cost["service_ms_by_threads"]:
            cost["service_ms_by_threads"][key] *= service_multiplier[key]
            cost["host_core_demand_ms_by_threads"][key] *= core_multiplier[key]
        updated.append(segment_id)
    return revised, {
        "service_multiplier_by_threads": service_multiplier,
        "core_multiplier_by_threads": core_multiplier,
        "updated_cpu_segment_count": len(updated),
        "anchor_segment": CALIBRATED_SEGMENT,
    }


def run_iteration(output_dir=DEFAULT_OUTPUT, top_k=20):
    output_dir = Path(output_dir)
    unit_schema = load_sealed_artifact(
        output_dir / "v1_unit_and_boundary_schema.json",
        "cpu_vta_pipeline_v1_unit_and_boundary_schema",
    )
    profile = load_sealed_artifact(
        output_dir / "v1_profile_manifest.json", "cpu_vta_pipeline_v1_profile_manifest"
    )
    local_cost = load_sealed_artifact(
        output_dir / "v1_local_cost_table.json", "cpu_vta_pipeline_v1_local_cost_table"
    )
    p4r = load_sealed_artifact(
        output_dir / "v1_p4r_candidate_manifest.json",
        "cpu_vta_pipeline_v1_p4r_candidate_manifest",
    )
    reference = load_sealed_artifact(
        output_dir / "v1_p5a_reference.json",
        "cpu_vta_pipeline_v1_p5a_independent_tensor_reference",
    )
    package_reaudit = []
    for row in p4r["rows"]:
        result = audit_package(
            row,
            output_dir / "v1_p5a_packages" / candidate_dir_name(row),
            reference,
        )
        package_reaudit.append(
            {
                "candidate_id": row["candidate_id"],
                "audit_passed": result["audit_passed"],
                "audit_failures": result["audit_failures"],
                "manifest_sha256": result["manifest_sha256"],
            }
        )
    if not all(row["audit_passed"] for row in package_reaudit):
        raise RuntimeError("P5A actual run-script re-audit failed")
    measurements = build_measurements(output_dir)
    base_context = build_context(profile, local_cost)
    exact_segment_context, original = apply_segment_observations(base_context, measurements)
    exact_segment_labels, _ = solve_k_best_label_setting(exact_segment_context, top_k=1)
    context, correction = apply_global_cpu_thread_correction(base_context, measurements)

    dp_labels, dp_stats = solve_k_best_label_setting(context, top_k=top_k)
    enum_labels, _, topology_count, execution_count = enumerate_oracle(
        unit_schema, context, top_k=top_k
    )
    dp_ids = [candidate_id(label.path) for label in dp_labels]
    enum_ids = [candidate_id(label.path) for label in enum_labels]
    if dp_ids != enum_ids:
        raise RuntimeError("revised DP Top-K differs from exact enumeration")

    measured_ids = {row["candidate_id"] for row in measurements}
    package_ids = {row["candidate_id"] for row in p4r["rows"]}
    records = []
    for rank, label in enumerate(dp_labels, 1):
        record = candidate_record(label, rank, context)
        record["scheme_cfg"] = [
            {
                "name": "stage{}".format(index),
                "device": context["profile_segments"][segment_id]["device"],
                "unit_names": list(context["profile_segments"][segment_id]["unit_names"]),
            }
            for index, (segment_id, _) in enumerate(label.path)
        ]
        record["stage_runtime_threads"] = [threads for _, threads in label.path]
        record["cost_provenance"]["candidate_measured_throughput_consumed"] = True
        record["cost_provenance"]["iteration_policy"] = (
            "global P2 CPU per-thread slope correction from isolated cpu:00:02 serial telemetry"
        )
        record["already_measured_control"] = record["candidate_id"] in measured_ids
        record["p5a_package_ready"] = record["candidate_id"] in package_ids
        records.append(record)
    validation = next(row for row in records if not row["already_measured_control"])
    validation_scheme = seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p5b_iteration_validation_scheme",
            "protocol_id": PROTOCOL_ID,
            "candidate_id": validation["candidate_id"],
            "scheme_name": validation["candidate_id"],
            "scheme_cfg": validation["scheme_cfg"],
            "stage_runtime_threads": validation["stage_runtime_threads"],
            "selection_policy": "revised global-CPU-slope Top-1 not used as a control measurement",
        }
    )

    measurement_artifact = seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p5b_iteration_measurements",
            "protocol_id": PROTOCOL_ID,
            "calibrated_segment": CALIBRATED_SEGMENT,
            "control_topology": CONTROL_TOPOLOGY,
            "measurement_count": len(measurements),
            "measurements": measurements,
            "p5a_actual_run_script_reaudit": package_reaudit,
            "rejected_sessions": [
                {
                    "threads": 4,
                    "reason": "one 2.44-second unexplained pause; whole session rejected before rerun",
                    "path": str(output_dir / "v1_p5b_iteration1/thread4/performance_rejected_pause/native_result.jsonl"),
                },
                {
                    "threads": 4,
                    "reason": "stale P5A package executed stage threads 4,1,4 instead of candidate 4,1,2",
                    "path": str(output_dir / "v1_p5b_iteration1/thread4/rejected_stale_package/native_result.jsonl"),
                },
            ],
        }
    )
    ranked = seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p5b_iteration_ranked_candidates",
            "protocol_id": PROTOCOL_ID,
            "measurement_artifact_sha256": measurement_artifact["artifact_sha256"],
            "original_segment_cost": original,
            "update_scope": "global P2 CPU per-thread slope correction from one representative compiled segment",
            "cpu_thread_correction": correction,
            "rejected_exact_segment_only_diagnostic": {
                "top1_candidate_id": candidate_id(exact_segment_labels[0].path),
                "reason": "optimizer escaped to adjacent uncorrected CPU segments generated by the same P2 slope",
            },
            "candidate_throughput_used_for_reranking": True,
            "validation_candidate": validation["candidate_id"],
            "validation_candidate_rank": validation["rank"],
            "validation_scheme_artifact_sha256": validation_scheme["artifact_sha256"],
            "rows": records,
        }
    )
    report = seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p5b_iteration_dp_report",
            "protocol_id": PROTOCOL_ID,
            "ranked_artifact_sha256": ranked["artifact_sha256"],
            "gate": "pass",
            "dp_top_k_equals_enumeration": True,
            "top_k": top_k,
            "topology_count": topology_count,
            "execution_configuration_count": execution_count,
            "dp_stats": dp_stats,
            "evidence_boundary": "implementation exactness under revised cost only; not hardware ranking accuracy",
        }
    )
    outputs = {
        "v1_p5b_iteration1_measurements.json": measurement_artifact,
        "v1_p5b_iteration1_ranked_candidates.json": ranked,
        "v1_p5b_iteration1_dp_report.json": report,
        "v1_p5b_iteration1_validation_scheme.json": validation_scheme,
    }
    for name, payload in outputs.items():
        validate_artifact_sha256(payload)
        write_json(output_dir / name, payload)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()
    outputs = run_iteration(args.output_dir, args.top_k)
    ranked = outputs["v1_p5b_iteration1_ranked_candidates.json"]
    print(
        json.dumps(
            {
                "top1": ranked["rows"][0]["candidate_id"],
                "top1_predicted_fps": ranked["rows"][0]["predicted_fps"],
                "validation_candidate": ranked["validation_candidate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
