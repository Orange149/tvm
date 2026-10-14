#!/usr/bin/env python3
"""Freeze selector-v3 graph-set decisions from already collected evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from c3_selector_v3 import candidate_set_deployment_gate
from run_vta_c3_holdout_board_boot_v2 import verify_flat


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_CERTIFICATES = GROUPED / "20260924_c3_composability_certificates_run01"
DEFAULT_YOLO_DEVELOPMENT = GROUPED / "20260924_c3_yolo608_completion_scope_timing_run01"
DEFAULT_YOLO_CONFIRMATION = GROUPED / "20260924_c3_yolo608_composable_confirmation_boot1_run01"
DEFAULT_R18_DEVELOPMENT = GROUPED / "20260924_c3_resnet18_h3_composable_timing_run01"
DEFAULT_OUTPUT = GROUPED / "20260924_c3_selector_v3_development_evidence_run01"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def yolo_decision(confirmation, graph_certificates):
    candidate_set_id = str(confirmation["candidate_set_id"])
    gate = candidate_set_deployment_gate(
        candidate_set_id, graph_certificates, [confirmation]
    )
    promising = (
        confirmation.get("status") == "ONE_BOOT_CONFIRMATION_COMPLETE"
        and confirmation.get("all_outputs_equal") is True
        and float(confirmation.get("median_improvement_pct", float("-inf"))) >= 2.0
        and int(confirmation.get("wins", -1)) >= 5
        and int(confirmation.get("rounds", -1)) == 6
    )
    return {
        "network": "yolov3_tiny",
        "input_resolution": 608,
        "deployment_unit": "exact_graph_candidate_set",
        "candidate_set_id": candidate_set_id,
        "candidate_ids": list(confirmation.get("candidate_ids") or []),
        "development_classification": (
            "PROMISING_REQUIRES_DISTINCT_BOOT_CONFIRMATION"
            if promising else "FALL_BACK_TO_INCUMBENT"
        ),
        "one_boot_result": {
            key: confirmation[key] for key in (
                "boot_id", "median_improvement_pct", "wins", "rounds",
                "all_outputs_equal", "timing_calls", "complete_graph_run_calls",
            )
        },
        "formal_deployment_gate": gate,
        "performance_claim": (
            "one-boot reproducible improvement only; no two-boot deployment claim"
            if promising else "no stable deployment improvement established"
        ),
    }


def resnet_decision(development):
    result = development["paired_improvement"]["composable_vs_stock"]
    improves = float(result["median_pct"]) >= 2.0 and int(result["wins"]) >= 5
    return {
        "network": "resnet18",
        "input_resolution": 224,
        "deployment_unit": "exact_graph_candidate_set",
        "development_classification": (
            "PROMISING_REQUIRES_FORMAL_CONFIRMATION" if improves
            else "FALL_BACK_TO_INCUMBENT"
        ),
        "development_result": {
            "boot_id": development["boot_id"],
            "median_improvement_pct": result["median_pct"],
            "wins": result["wins"],
            "pairs": result["pairs"],
            "all_outputs_equal": development["all_outputs_equal"],
        },
        "formal_confirmation_scheduled": bool(improves),
        "performance_claim": "development evidence only; candidate is not deployed",
    }


def run(args):
    roots = {
        "certificates": Path(args.certificate_root).resolve(),
        "yolo_development": Path(args.yolo_development).resolve(),
        "yolo_confirmation": Path(args.yolo_confirmation).resolve(),
        "resnet_development": Path(args.resnet_development).resolve(),
    }
    input_hashes = {name: verify_flat(path) for name, path in roots.items()}
    graph_certificates = read_jsonl(
        roots["certificates"] / "graph_context_certificates.jsonl"
    )
    yolo_development = read(roots["yolo_development"] / "summary.json")
    yolo_confirmation = read(roots["yolo_confirmation"] / "summary.json")
    resnet_development = read(roots["resnet_development"] / "summary.json")
    if yolo_development["boot_id"] != yolo_confirmation["boot_id"]:
        raise ValueError("expected existing development and boot-1 confirmation to share one boot")

    decisions = {
        "yolov3_tiny_608": yolo_decision(yolo_confirmation, graph_certificates),
        "resnet18_224": resnet_decision(resnet_development),
    }
    observed_boot = str(args.observed_current_boot_id or "")
    boot_observation = {
        "observed_current_boot_id": observed_boot or None,
        "confirmation_boot_id": yolo_confirmation["boot_id"],
        "is_distinct_from_confirmation_boot": (
            bool(observed_boot) and observed_boot != yolo_confirmation["boot_id"]
        ),
        "accepted_as_second_boot": False,
        "reason": (
            "no second confirmation result was provided"
            if observed_boot and observed_boot != yolo_confirmation["boot_id"]
            else "observed boot ID is unchanged or unavailable"
        ),
    }

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write(output / "contract.json", {
        "schema": "c3_selector_v3_development_evidence_v1",
        "status": "frozen",
        "selector_unit": "exact graph candidate set with composability certificate",
        "deployment_rule": "two distinct boots, each >=2% median improvement and >=5/6 wins",
        "latency_exposure": "all included performance tasks are development/exposed",
        "claim_boundary": (
            "one-boot results may prioritize confirmation but cannot authorize deployment; "
            "negative development evidence may stop an unpromising branch"
        ),
    })
    write(output / "decisions.json", decisions)
    write(output / "boot_observation.json", boot_observation)
    write(output / "input_hashes.json", input_hashes)
    summary = {
        "schema": "c3_selector_v3_development_evidence_v1",
        "status": "SELECTOR_V3_DEVELOPMENT_EVIDENCE_FROZEN",
        "yolo608_classification": decisions["yolov3_tiny_608"]["development_classification"],
        "yolo608_formal_gate": decisions["yolov3_tiny_608"]["formal_deployment_gate"]["status"],
        "resnet18_classification": decisions["resnet18_224"]["development_classification"],
        "distinct_boot_confirmation_count": 1,
        "deployment_authorized": False,
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
    parser.add_argument("--certificate-root", default=str(DEFAULT_CERTIFICATES))
    parser.add_argument("--yolo-development", default=str(DEFAULT_YOLO_DEVELOPMENT))
    parser.add_argument("--yolo-confirmation", default=str(DEFAULT_YOLO_CONFIRMATION))
    parser.add_argument("--resnet-development", default=str(DEFAULT_R18_DEVELOPMENT))
    parser.add_argument("--observed-current-boot-id", default="")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
