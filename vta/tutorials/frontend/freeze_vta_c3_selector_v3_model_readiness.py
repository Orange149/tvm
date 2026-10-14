#!/usr/bin/env python3
"""Freeze an evidence-backed readiness audit for the selector-v3 model."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import c3_selector_v3_model
from c3_selector_v3_model import train_selector


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_V2 = GROUPED / "20260924_c3_selector_v2_dataset_run01"
DEFAULT_GRAPH_SET = GROUPED / "20260924_c3_yolo608_graph_candidate_set_run01"
DEFAULT_GRAPH_PLAN = GROUPED / "20260924_c3_selector_v3_graph_set_plan_run02"
DEFAULT_DEVELOPMENT = GROUPED / "20260924_c3_selector_v3_development_evidence_run01"
DEFAULT_OUTPUT = GROUPED / "20260924_c3_selector_v3_model_readiness_run02"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify_directory(root):
    root = Path(root)
    manifest = read(root / "artifact_hashes.json")
    entries = manifest.get("artifacts", manifest)
    failures = {name: {"expected": expected, "actual": sha256(root / name)}
                for name, expected in entries.items() if sha256(root / name) != expected}
    if failures:
        raise ValueError("artifact hash mismatch in {}: {}".format(root, failures))
    return {"root": str(root), "manifest_sha256": sha256(root / "artifact_hashes.json"),
            "verified_file_count": len(entries)}


def build_readiness(v2_rows, graph_set, graph_plan, development):
    mode_counts = dict(sorted(collections.Counter(
        str(row.get("candidate_mode", "unknown")) for row in v2_rows
    ).items()))
    task_count = len({str(row.get("execution_task_id", "")) for row in v2_rows})
    has_incumbent_metrics = isinstance(graph_set.get("incumbent_aggregate_metrics"), dict)
    sources = {
        "selector_v2_operator_table": {
            "observed_rows": len(v2_rows),
            "observed_execution_tasks": task_count,
            "candidate_mode_counts": mode_counts,
            "admitted_v3_graph_rows": 0,
            "exclusion_reasons": [
                "labels belong to isolated operator comparisons, not exact graph candidate sets",
                "legacy barrier rows cannot stand in for scoped-completion semantics",
            ],
        },
        "yolo608_exact_graph_set": {
            "observed_candidate_sets": 1,
            "observed_replacements": len(graph_set.get("candidate_ids") or []),
            "observed_distinct_boot_labels": int(
                development.get("distinct_boot_confirmation_count", 0)),
            "has_incumbent_aggregate_metrics": has_incumbent_metrics,
            "admitted_v3_graph_rows": 0,
            "exclusion_reasons": [
                reason for reason in (
                    None if has_incumbent_metrics else "missing matched incumbent aggregate metrics",
                    "only one candidate set; no within-task ranking pool",
                    "only one observed boot label",
                ) if reason is not None
            ],
        },
        "resnet18_graph_development": {
            "observed_candidate_sets": 1,
            "development_classification": development.get("resnet18_classification"),
            "admitted_v3_graph_rows": 0,
            "exclusion_reasons": [
                "development timing is not a frozen exact-set relative-feature row",
                "candidate fell back to incumbent and has no formal confirmation label",
            ],
        },
    }
    contract = train_selector([])
    return {
        "schema": "c3_selector_v3_model_readiness_v1",
        "status": "STATIC_FALLBACK_REQUIRED",
        "source_inventory": sources,
        "admitted_training_row_count": 0,
        "admitted_execution_task_count": 0,
        "current_graph_plan_ranking_model": graph_plan.get("ranking_model"),
        "current_graph_plan_performance_model_claim": graph_plan.get("performance_model_claim"),
        "model_contract": contract,
        "minimum_unlock_conditions": {
            "exact_graph_execution_tasks": 8,
            "candidate_set_incumbent_comparisons": 40,
            "target_coverage": 0.80,
            "scoped_completion_tasks": 3,
            "dependency_only_tasks": 3,
            "same_hardware_fingerprint": True,
            "matched_incumbent_aggregate_metrics": True,
        },
        "claim_boundary": (
            "implementation and fail-closed training contract exist; no completion-aware "
            "performance model is trained or claimed from current evidence"
        ),
    }


def run(args):
    roots = {
        "selector_v2_dataset": Path(args.selector_v2_dataset).resolve(),
        "graph_candidate_set": Path(args.graph_candidate_set).resolve(),
        "graph_set_plan": Path(args.graph_set_plan).resolve(),
        "development_evidence": Path(args.development_evidence).resolve(),
    }
    inputs = {name: verify_directory(root) for name, root in roots.items()}
    readiness = build_readiness(
        read_jsonl(roots["selector_v2_dataset"] / "training_rows.jsonl"),
        read(roots["graph_candidate_set"] / "graph_candidate_set.json"),
        read(roots["graph_set_plan"] / "measurement_plan.json"),
        read(roots["development_evidence"] / "summary.json"),
    )
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write(output / "contract.json", {
        "schema": "c3_selector_v3_model_readiness_freeze_v1",
        "status": "frozen_without_board_contact",
        "latency_values_created": False,
        "legacy_operator_rows_relabelled_as_graph_rows": False,
        "model_source_sha256": sha256(Path(c3_selector_v3_model.__file__).resolve()),
        "claim_boundary": readiness["claim_boundary"],
    })
    write(output / "input_hashes.json", inputs)
    write(output / "readiness.json", readiness)
    write(output / "selector_v3_model_contract.json", readiness["model_contract"])
    summary = {
        "schema": "c3_selector_v3_model_readiness_freeze_v1",
        "status": readiness["status"],
        "admitted_training_row_count": readiness["admitted_training_row_count"],
        "v2_rows_audited": len(read_jsonl(roots["selector_v2_dataset"] / "training_rows.jsonl")),
        "selector_kind": readiness["model_contract"]["selector_kind"],
        "performance_model_claim": False,
        "board_contacted": False,
    }
    write(output / "summary.json", summary)
    write(output / "artifact_hashes.json", {
        "artifacts": {path.name: sha256(path) for path in sorted(output.iterdir())
                      if path.is_file() and path.name != "artifact_hashes.json"},
        "source_sha256": sha256(Path(__file__).resolve()),
        "model_source_sha256": sha256(Path(c3_selector_v3_model.__file__).resolve()),
    })
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-v2-dataset", default=str(DEFAULT_V2))
    parser.add_argument("--graph-candidate-set", default=str(DEFAULT_GRAPH_SET))
    parser.add_argument("--graph-set-plan", default=str(DEFAULT_GRAPH_PLAN))
    parser.add_argument("--development-evidence", default=str(DEFAULT_DEVELOPMENT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
