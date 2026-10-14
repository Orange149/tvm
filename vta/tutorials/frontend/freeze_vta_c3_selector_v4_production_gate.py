#!/usr/bin/env python3
"""Freeze the production-eligibility gate in front of selector v4.

This contract reads no candidate pool and no latency.  It binds the already
frozen label-free organizer to the versioned completion-policy/composability
gate for all future latency-unseen tasks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import c3_completion_policy
import c3_composability
import c3_selector_v4_deployment
import c3_selector_v4_diverse


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_SELECTOR = GROUPED / "20260924_c3_selector_v4_diverse_development_freeze_run01"
DEFAULT_OUTPUT = GROUPED / "20260924_c3_selector_v4_production_gate_freeze_run01"
SCHEMA = "c3_selector_v4_production_gate_freeze_v1"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify(directory):
    directory = Path(directory).resolve()
    manifest = read(directory / "artifact_hashes.json")
    for name, expected in manifest.get("artifacts", {}).items():
        path = directory / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError("artifact hash mismatch: {}".format(path))
    return manifest, sha256(directory / "artifact_hashes.json")


def build_contract(selector_contract, selector_manifest, manifest_sha256):
    if selector_contract.get("status") != "FROZEN_FOR_FUTURE_LATENCY_UNSEEN_TASKS":
        raise ValueError("upstream selector-v4 contract is not frozen")
    if (selector_contract.get("performance_labels_read_by_freeze") is not False
            or selector_contract.get("board_contacted") is not False):
        raise ValueError("upstream selector-v4 contract is not label isolated")
    if selector_manifest.get("selector_source_sha256") != sha256(
            Path(c3_selector_v4_diverse.__file__).resolve()):
        raise ValueError("upstream selector-v4 source drift")
    return {
        "schema": SCHEMA,
        "status": "FROZEN_FOR_FUTURE_LATENCY_UNSEEN_TASKS",
        "ordering_schema": c3_selector_v4_diverse.SCHEMA,
        "gate_schema": c3_selector_v4_deployment.SCHEMA,
        "pipeline": [
            "local_capacity_address_dependency_and_correctness_qualification",
            "completion_policy_and_composability_hard_gate",
            "label_free_low_movement_tile_diversity_order",
            "B1_B2_B3_prefix_measurement",
            "exact_graph_context_correctness_before_deployment",
        ],
        "reference_pool_policy": (
            "locally correct non-production programs may remain in a separately named "
            "isolated-operator reference pool but cannot enter deployment budgets"
        ),
        "legacy_completion_policy_action": "reject_and_regenerate_as_mode11",
        "performance_labels_read_by_freeze": False,
        "board_contacted": False,
        "performance_model_claim": False,
        "safe_pruning_claim": False,
        "upstream_selector_manifest_sha256": manifest_sha256,
        "claim_boundary": (
            "source-hash-bound eligibility and ordering protocol only; no candidate, "
            "correctness, latency, oracle, cross-boot, graph, or deployment result"
        ),
    }


def run(args):
    selector_dir = Path(args.selector_contract).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))
    selector_manifest, manifest_sha256 = verify(selector_dir)
    selector_contract = read(selector_dir / "selector_v4_contract.json")
    contract = build_contract(selector_contract, selector_manifest, manifest_sha256)
    output.mkdir(parents=True)
    write(output / "contract.json", contract)
    write(output / "summary.json", contract)
    write(output / "artifact_hashes.json", {
        "artifacts": {
            path.name: sha256(path) for path in sorted(output.iterdir())
            if path.is_file() and path.name != "artifact_hashes.json"
        },
        "source_sha256": sha256(Path(__file__).resolve()),
        "deployment_gate_source_sha256": sha256(
            Path(c3_selector_v4_deployment.__file__).resolve()
        ),
        "selector_source_sha256": sha256(Path(c3_selector_v4_diverse.__file__).resolve()),
        "composability_source_sha256": sha256(Path(c3_composability.__file__).resolve()),
        "completion_policy_source_sha256": sha256(
            Path(c3_completion_policy.__file__).resolve()
        ),
        "upstream_selector_manifest_sha256": manifest_sha256,
    })
    print(json.dumps(contract, indent=2, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-contract", type=Path, default=DEFAULT_SELECTOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
