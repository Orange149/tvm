#!/usr/bin/env python3
"""Seal the first measured P5B model-update and unseen-candidate validation review."""

from __future__ import annotations

import argparse
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


DEFAULT_OUTPUT = repo_root() / "vta/tutorials/frontend/report_out/resource_aware_maxplus"


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def summarize_validation(path, warmup=2):
    rows = read_jsonl(path)
    scored = rows[warmup:]
    stage_count = int(rows[0]["stage_count"])
    end_key = "stage{}_end_ms".format(stage_count - 1)
    ends = [float(row[end_key]) for row in scored]
    intervals = [right - left for left, right in zip(ends, ends[1:])]
    cycle = (ends[-1] - ends[0]) / (len(ends) - 1)
    stages = []
    for index in range(stage_count):
        stages.append(
            {
                "stage_index": index,
                "run_ms_mean": statistics.mean(
                    float(row["stage{}_run_ms".format(index)]) for row in scored
                ),
                "run_ms_median": statistics.median(
                    float(row["stage{}_run_ms".format(index)]) for row in scored
                ),
            }
        )
    return {
        "row_count": len(rows),
        "warmup_count": warmup,
        "scored_count": len(scored),
        "cycle_ms": cycle,
        "fps": 1000.0 / cycle,
        "completion_interval_median_ms": statistics.median(intervals),
        "completion_interval_min_ms": min(intervals),
        "completion_interval_max_ms": max(intervals),
        "stages": stages,
        "top1_values": sorted({int(row["top1"]) for row in rows}),
    }


def build_review(output_dir=DEFAULT_OUTPUT):
    output_dir = Path(output_dir)
    measurements = load_sealed_artifact(
        output_dir / "v1_p5b_iteration1_measurements.json",
        "cpu_vta_pipeline_v1_p5b_iteration_measurements",
    )
    ranked = load_sealed_artifact(
        output_dir / "v1_p5b_iteration1_ranked_candidates.json",
        "cpu_vta_pipeline_v1_p5b_iteration_ranked_candidates",
    )
    dp_report = load_sealed_artifact(
        output_dir / "v1_p5b_iteration1_dp_report.json",
        "cpu_vta_pipeline_v1_p5b_iteration_dp_report",
    )
    comparison = json.loads(
        (
            output_dir
            / "v1_p5b_iteration1/validation_candidate/correctness/reference_comparison.json"
        ).read_text(encoding="utf-8")
    )
    if not comparison.get("passed"):
        raise RuntimeError("validation candidate failed the frozen tensor gate")
    observed = summarize_validation(
        output_dir / "v1_p5b_iteration1/validation_candidate/performance/native_result.jsonl"
    )
    predicted = next(
        row for row in ranked["rows"] if row["candidate_id"] == ranked["validation_candidate"]
    )
    pool = [
        {"candidate_id": row["candidate_id"], "role": "thread_control", "fps": row["pipeline"]["fps"]}
        for row in measurements["measurements"]
    ]
    pool.append(
        {
            "candidate_id": ranked["validation_candidate"],
            "role": "iteration1_revised_top1_unseen_validation",
            "fps": observed["fps"],
        }
    )
    pool.sort(key=lambda row: (-row["fps"], row["candidate_id"]))
    oracle = pool[0]
    validation_regret = 1.0 - observed["fps"] / oracle["fps"]
    review = seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p5b_iteration_review",
            "protocol_id": PROTOCOL_ID,
            "measurement_artifact_sha256": measurements["artifact_sha256"],
            "ranked_artifact_sha256": ranked["artifact_sha256"],
            "dp_report_artifact_sha256": dp_report["artifact_sha256"],
            "validation_candidate": ranked["validation_candidate"],
            "correctness": {
                "serial_pipeline_byte_exact": True,
                "independent_reference_passed": True,
                "top1_reference": comparison["top1_reference"],
                "top1_actual": comparison["top1_actual"],
                "cosine_similarity": comparison["cosine_similarity"],
                "normalized_rmse": comparison["normalized_rmse"],
                "top_k_overlap": comparison["top_k_overlap"],
            },
            "prediction": {
                "cycle_ms": predicted["predicted_ii_ms"],
                "fps": predicted["predicted_fps"],
            },
            "observation": observed,
            "prediction_error": {
                "cycle_relative_error": observed["cycle_ms"] / predicted["predicted_ii_ms"] - 1.0,
                "fps_relative_error": observed["fps"] / predicted["predicted_fps"] - 1.0,
            },
            "measured_pool": pool,
            "measured_pool_oracle": oracle,
            "validation_throughput_regret": validation_regret,
            "conclusion": "iteration1_global_cpu_slope_update_did_not_improve_measured_ranking",
            "root_cause_evidence": {
                "predicted_cpu_stage_run_ms": [
                    stage["run_service_ms"]
                    for stage in predicted["stages"]
                    if stage["device"] == "cpu"
                ],
                "observed_cpu_stage_run_ms_median": [
                    stage["run_ms_median"]
                    for stage, predicted_stage in zip(observed["stages"], predicted["stages"])
                    if predicted_stage["device"] == "cpu"
                ],
                "interpretation": "one global logical-GOP slope misses segment shape/primitive/backend efficiency",
            },
            "not_supported": [
                "fewer board evaluations than random search",
                "shared-DDR contention improves ranking",
                "more VTA islands improve throughput",
                "global CPU GOP slope is portable across segments",
            ],
            "next_model_change": "measure a small grouped set of CPU segment families (stem, residual, transition, head) and validate held-out segments before another candidate ranking",
        }
    )
    state = seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_execution_state",
            "protocol_id": PROTOCOL_ID,
            "current_stage": "V1-P5B",
            "current_stage_status": "iteration1_board_validation_complete_awaiting_user_review",
            "next_stage": "V1-P5B-iteration2-cpu-segment-family-profile",
            "next_stage_may_start": False,
            "advance_requires_user_confirmation": True,
            "blocking_evidence": [
                "revised Top-1 measured 8.869 FPS versus 10.449 FPS measured-pool oracle",
                "CPU segment medians remain substantially underestimated",
            ],
            "p5b_iteration1_review_artifact_sha256": review["artifact_sha256"],
        }
    )
    outputs = {
        "v1_p5b_iteration1_review.json": review,
        "v1_execution_state.json": state,
    }
    for name, payload in outputs.items():
        validate_artifact_sha256(payload)
        write_json(output_dir / name, payload)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    review = build_review(args.output_dir)["v1_p5b_iteration1_review.json"]
    print(
        json.dumps(
            {
                "conclusion": review["conclusion"],
                "predicted_fps": review["prediction"]["fps"],
                "observed_fps": review["observation"]["fps"],
                "measured_pool_oracle_fps": review["measured_pool_oracle"]["fps"],
                "validation_regret": review["validation_throughput_regret"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
