#!/usr/bin/env python3
"""Freeze the real graph-set selector-v3 plan and current deployment gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from c3_graph_candidate_set import (
    enumerate_graph_candidate_sets,
    inherit_boot_result_identity,
)
from c3_selector_v3 import candidate_set_deployment_gate, graph_set_measurement_plan
from run_vta_c3_holdout_board_boot_v2 import verify_flat


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_GRAPH_SET = GROUPED / "20260924_c3_yolo608_graph_candidate_set_run01"
DEFAULT_CONFIRMATION = GROUPED / "20260924_c3_yolo608_composable_confirmation_boot1_run01"
DEFAULT_MODEL_READINESS = GROUPED / "20260924_c3_selector_v3_model_readiness_run02"
DEFAULT_OUTPUT = GROUPED / "20260924_c3_selector_v3_graph_set_plan_run04"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args):
    graph_root = Path(args.graph_set_root).resolve()
    confirmation_root = Path(args.confirmation_root).resolve()
    input_hashes = {
        "graph_set": verify_flat(graph_root),
        "confirmation": verify_flat(confirmation_root),
        "model_readiness": verify_flat(Path(args.model_readiness_root).resolve()),
    }
    candidate_set = read(graph_root / "graph_candidate_set.json")
    graph_certificate = read(graph_root / "migrated_graph_context_certificate.json")

    # Freeze selection from identities/static features before opening latency.
    graph_identity = candidate_set["identity"]["graph"]
    incumbent = enumerate_graph_candidate_sets(
        graph_identity, {}, {}, max_replacements=0
    )["accepted_sets"][0]
    pool = [incumbent, candidate_set]
    model_contract = read(
        Path(args.model_readiness_root).resolve() / "selector_v3_model_contract.json"
    )
    plan = graph_set_measurement_plan(pool, [graph_certificate], model_contract)

    legacy_boot = read(confirmation_root / "summary.json")
    migrated_boot = inherit_boot_result_identity(
        candidate_set, graph_certificate, legacy_boot
    )
    gate = candidate_set_deployment_gate(
        candidate_set["candidate_set_id"], [graph_certificate], [migrated_boot]
    )
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write(output / "contract.json", {
        "schema": "c3_selector_v3_graph_set_plan_freeze_v1",
        "status": "frozen",
        "selection_unit": "exact_graph_candidate_set",
        "plan_constructed_before_latency_read": True,
        "latency_exposure": "YOLO-608 remains development/exposed",
        "ranking_model": plan["ranking_model"],
        "performance_model_claim": plan["performance_model_claim"],
        "model_status": plan["model_status"],
        "model_fallback_reason": plan["model_fallback_reason"],
        "claim_boundary": "one migrated boot result cannot authorize deployment",
    })
    write(output / "candidate_set_pool.json", pool)
    write(output / "measurement_plan.json", plan)
    write(output / "migrated_boot1_result.json", migrated_boot)
    write(output / "deployment_gate.json", gate)
    write(output / "input_hashes.json", input_hashes)
    summary = {
        "schema": "c3_selector_v3_graph_set_plan_freeze_v1",
        "status": "REAL_GRAPH_SET_PLAN_FROZEN",
        "candidate_set_id": candidate_set["candidate_set_id"],
        "selection_unit": plan["selection_unit"],
        "shortlist_size": len(plan["shortlist_candidate_set_ids"]),
        "ranking_model": plan["ranking_model"],
        "performance_model_claim": plan["performance_model_claim"],
        "model_status": plan["model_status"],
        "model_fallback_reason": plan["model_fallback_reason"],
        "all_shortlisted_sets_graph_timing_eligible": (
            set(plan["shortlist_candidate_set_ids"])
            == set(plan["timing_eligible_candidate_set_ids"])
        ),
        "boot1_median_improvement_pct": migrated_boot["median_improvement_pct"],
        "boot1_wins": migrated_boot["wins"],
        "deployment_gate_status": gate["status"],
        "deployment_gate_reasons": gate["reasons"],
        "deployment_authorized": gate["status"] == "DEPLOY",
    }
    write(output / "summary.json", summary)
    write(output / "artifact_hashes.json", {
        "artifacts": {
            path.name: sha256(path) for path in sorted(output.iterdir())
            if path.is_file() and path.name != "artifact_hashes.json"
        },
        "source_sha256": sha256(Path(__file__).resolve()),
    })
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-set-root", default=str(DEFAULT_GRAPH_SET))
    parser.add_argument("--confirmation-root", default=str(DEFAULT_CONFIRMATION))
    parser.add_argument("--model-readiness-root", default=str(DEFAULT_MODEL_READINESS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
