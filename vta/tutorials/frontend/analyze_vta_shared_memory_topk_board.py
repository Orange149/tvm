#!/usr/bin/env python3
"""Analyze a three-boot frozen shared-memory-aware Top-K experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import tarfile
from pathlib import Path

from shared_memory_topk import read_json, sha256_file


def median(values):
    return float(statistics.median(float(value) for value in values))


def validate_boot(boot, contract, contract_sha256):
    if boot.get("schema") != "vta_shared_memory_topk_board_boot_v1":
        raise ValueError("unexpected board boot schema")
    if boot.get("selection_contract_sha256") != contract_sha256:
        raise ValueError("board boot was not produced from the frozen selection contract")
    if boot.get("status") not in (None, "complete_one_independent_boot"):
        raise ValueError("board boot is not complete")
    expected = set(contract["union_configuration_ids"])
    rows = boot.get("rows", [])
    actual = [row.get("configuration_id") for row in rows]
    if len(actual) != len(set(actual)):
        raise ValueError("duplicate configuration in board boot")
    if set(actual) != expected:
        raise ValueError("board boot configuration set differs from frozen union")
    for row in rows:
        for field in ("ii_ms", "framework_materialization_bytes", "additional_slot_bytes"):
            value = float(row[field])
            if not math.isfinite(value) or value < 0:
                raise ValueError("invalid {} for {}".format(field, row["configuration_id"]))
    return {row["configuration_id"]: row for row in rows}


def sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def tar_outputs(path):
    result = {}
    with tarfile.open(path, "r") as archive:
        for member in archive.getmembers():
            if member.isfile():
                result[member.name.lstrip("./")] = archive.extractfile(member).read()
    return result


def validate_boot_artifacts(path, deployment, deployment_sha256):
    """Reconstruct the evidence chain from a boot summary to raw artifacts."""
    path = Path(path)
    boot = read_json(path)
    if boot.get("status") != "complete_one_independent_boot":
        raise ValueError("boot artifact is not complete: {}".format(path))
    if boot.get("deployment_manifest_sha256") != deployment_sha256:
        raise ValueError("boot/deployment manifest hash mismatch")
    boot_index = boot.get("boot_index")
    if boot_index not in (0, 1, 2):
        raise ValueError("invalid boot index")
    if boot.get("configuration_order") != deployment["boot_configuration_orders"][boot_index]:
        raise ValueError("observed configuration order differs from preregistration")
    rows = {row["configuration_id"]: row for row in boot["rows"]}
    directory = path.parent

    reference_files = list(directory.glob("reference_*_outputs.tar"))
    reference_by_hash = {sha256_file(item): item for item in reference_files}
    references = {}
    for candidate_id, record in boot.get("correctness_references", {}).items():
        reference_path = reference_by_hash.get(record["output_archive_sha256"])
        if reference_path is None:
            raise ValueError("missing serial full-output reference for " + candidate_id)
        outputs = tar_outputs(reference_path)
        if len(outputs) != int(record["full_output_files"]) or len(outputs) != 3:
            raise ValueError("serial reference does not contain three complete outputs")
        references[candidate_id] = outputs

    for position, configuration_id in enumerate(boot["configuration_order"]):
        if configuration_id not in rows:
            raise ValueError("configuration missing from boot rows")
        row = rows[configuration_id]
        token = "cfg{:02d}_{}".format(position, sha256_text(configuration_id)[:12])
        artifacts = (
            (directory / (token + "_performance.jsonl"), row["performance_result_sha256"]),
            (directory / (token + "_profile.json"), row["profile_sha256"]),
            (
                directory / (token + "_correctness_outputs.tar"),
                row["correctness_full_output_archive_sha256"],
            ),
        )
        for artifact, digest in artifacts:
            if not artifact.is_file() or sha256_file(artifact) != digest:
                raise ValueError("raw artifact hash mismatch: {}".format(artifact))
        observed = tar_outputs(artifacts[2][0])
        if observed != references.get(row["candidate_id"]):
            raise ValueError("full-output bytes differ from serial reference: " + configuration_id)
    return boot


def validate_deployment(deployment_path, contract_sha256):
    deployment_path = Path(deployment_path)
    deployment = read_json(deployment_path)
    if deployment.get("status") != "board_ready_frozen_before_new_performance_labels":
        raise ValueError("deployment manifest is not a pre-label freeze")
    if deployment["selection_contract"]["sha256"] != contract_sha256:
        raise ValueError("deployment/selection contract hash mismatch")
    runner = deployment["board_runner"]
    if sha256_file(runner["path"]) != runner["sha256"]:
        raise ValueError("board runner changed after deployment freeze")
    for record in deployment["shared_plans"].values():
        if sha256_file(record["path"]) != record["sha256"]:
            raise ValueError("shared plan changed after deployment freeze")
    build = deployment["diagnostic_build"]
    for name, digest in build["artifacts"].items():
        if sha256_file(Path(build["path"]) / name) != digest:
            raise ValueError("diagnostic artifact changed after deployment freeze")
    return deployment, sha256_file(deployment_path)


def aggregate_boots(boots, contract, contract_sha256):
    boot_ids = [boot.get("boot_id") for boot in boots]
    if len(boots) < 3 or len(set(boot_ids)) != len(boots) or None in boot_ids:
        raise ValueError("at least three distinct nonempty boot ids are required")
    indexed = [validate_boot(boot, contract, contract_sha256) for boot in boots]
    rows = {}
    for configuration_id in contract["union_configuration_ids"]:
        observations = [boot[configuration_id] for boot in indexed]
        correctness = all(bool(row.get("correct", False)) for row in observations)
        rows[configuration_id] = {
            "configuration_id": configuration_id,
            "correct": correctness,
            "ii_ms_by_boot": [float(row["ii_ms"]) for row in observations],
            "ii_ms_median": median(row["ii_ms"] for row in observations),
            "framework_materialization_bytes": [
                int(row["framework_materialization_bytes"]) for row in observations
            ],
            "additional_slot_bytes": [int(row["additional_slot_bytes"]) for row in observations],
            "boundary_api_service_ms_median": median(
                row.get("boundary_api_service_ms", 0.0) for row in observations
            ),
            "pool_high_water_bytes_median": median(
                row.get("pool_high_water_bytes", 0) for row in observations
            ),
            "vta_internal_dma_bytes": [row.get("vta_internal_dma_bytes") for row in observations],
            "vta_dma_calls": [row.get("vta_dma_calls") for row in observations],
        }
    return boot_ids, rows


def strategy_metrics(order, rows, oracle_ms):
    valid = [rows[configuration_id] for configuration_id in order if rows[configuration_id]["correct"]]
    if not valid:
        return {
            "valid_count": 0,
            "selected_ii_ms": None,
            "regret_at_k": None,
            "first_oracle_plus_2pct_index": None,
        }
    selected = min(row["ii_ms_median"] for row in valid)
    threshold = oracle_ms * 1.02
    first = next(
        (
            index
            for index, configuration_id in enumerate(order, 1)
            if rows[configuration_id]["correct"]
            and rows[configuration_id]["ii_ms_median"] <= threshold
        ),
        None,
    )
    return {
        "valid_count": len(valid),
        "selected_ii_ms": selected,
        "regret_at_k": (selected - oracle_ms) / oracle_ms,
        "first_oracle_plus_2pct_index": first,
    }


def same_candidate_policy_invariants(rows):
    grouped = {}
    for configuration_id, row in rows.items():
        candidate_id, policy = configuration_id.rsplit("::", 1)
        grouped.setdefault(candidate_id, {})[policy] = row
    checks = []
    for candidate_id, policies in grouped.items():
        copy = policies.get("ordinary_copy")
        shared = [row for policy, row in policies.items() if policy != "ordinary_copy"]
        for row in shared:
            dma_known = all(value is not None for value in copy["vta_internal_dma_bytes"] + row["vta_internal_dma_bytes"]) if copy else False
            calls_known = all(value is not None for value in copy["vta_dma_calls"] + row["vta_dma_calls"]) if copy else False
            checks.append({
                "candidate_id": candidate_id,
                "shared_configuration_id": row["configuration_id"],
                "copy_present": copy is not None,
                "shared_materialization_zero_all_boots": all(
                    value == 0 for value in row["framework_materialization_bytes"]
                ),
                "copy_materialization_nonzero_all_boots": bool(copy) and all(
                    value > 0 for value in copy["framework_materialization_bytes"]
                ),
                "vta_internal_dma_equal": dma_known and copy["vta_internal_dma_bytes"] == row["vta_internal_dma_bytes"],
                "vta_dma_calls_equal": calls_known and copy["vta_dma_calls"] == row["vta_dma_calls"],
            })
    return checks


def analyze(contract, boots, contract_sha256):
    boot_ids, rows = aggregate_boots(boots, contract, contract_sha256)
    valid = [row for row in rows.values() if row["correct"]]
    if not valid:
        raise ValueError("no correct union configuration; oracle is undefined")
    oracle_ms = min(row["ii_ms_median"] for row in valid)
    metrics = {
        name: strategy_metrics(order, rows, oracle_ms)
        for name, order in contract["strategies"].items()
    }
    baseline_names = ("S0_m1_copy", "S1_m1_payload")
    s2 = metrics["S2_m1_certified_shared"]
    baseline_first = [metrics[name]["first_oracle_plus_2pct_index"] for name in baseline_names]
    baseline_regret = [metrics[name]["regret_at_k"] for name in baseline_names]
    ranking_gate = (
        s2["first_oracle_plus_2pct_index"] is not None
        and all(
            value is None or s2["first_oracle_plus_2pct_index"] <= value
            for value in baseline_first
        )
        and s2["regret_at_k"] is not None
        and all(value is not None for value in baseline_regret)
        and s2["regret_at_k"] <= min(baseline_regret) + 1.0e-12
    )
    invariants = same_candidate_policy_invariants(rows)
    mechanism_passed = bool(invariants) and all(
        row["copy_present"]
        and row["shared_materialization_zero_all_boots"]
        and row["copy_materialization_nonzero_all_boots"]
        and row["vta_internal_dma_equal"]
        and row["vta_dma_calls_equal"]
        for row in invariants
    )
    claim_allowed = ranking_gate and len(valid) == len(rows) and mechanism_passed
    return {
        "schema": "vta_shared_memory_topk_board_analysis_v1",
        "boot_ids": boot_ids,
        "independent_boot_count": len(boot_ids),
        "union_oracle_ii_ms": oracle_ms,
        "configurations": rows,
        "strategy_metrics": metrics,
        "same_candidate_policy_invariants": invariants,
        "gates": {
            "minimum_three_independent_boots": len(boot_ids) >= 3,
            "all_union_configurations_correct": len(valid) == len(rows),
            "shared_mechanism_invariants_passed": mechanism_passed,
            "selection_efficiency_claim_allowed": claim_allowed,
        },
        "conclusion": (
            "selection_efficiency_improved_under_frozen_protocol"
            if claim_allowed
            else "retain_shared_transfer_decomposition_and_report_negative_ranking_result"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--deployment-manifest", type=Path, required=True)
    parser.add_argument("--boots", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite {}".format(args.output_dir))
    contract_sha256 = sha256_file(args.contract)
    contract = read_json(args.contract)
    deployment, deployment_sha256 = validate_deployment(
        args.deployment_manifest, contract_sha256
    )
    boots = [
        validate_boot_artifacts(path, deployment, deployment_sha256)
        for path in args.boots
    ]
    if {boot["boot_index"] for boot in boots} != {0, 1, 2}:
        raise ValueError("the three preregistered boot indices 0,1,2 are required")
    result = analyze(contract, boots, contract_sha256)
    result["sources"] = {
        "contract": {"path": str(args.contract.resolve()), "sha256": contract_sha256},
        "deployment_manifest": {
            "path": str(args.deployment_manifest.resolve()),
            "sha256": deployment_sha256,
        },
        "boots": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in args.boots
        ],
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# C1 shared-memory-aware Top-K board result",
        "",
        "- Independent boots: {}.".format(result["independent_boot_count"]),
        "- Union oracle II: {:.6f} ms.".format(result["union_oracle_ii_ms"]),
        "- Mechanism invariants passed: {}.".format(
            str(result["gates"]["shared_mechanism_invariants_passed"]).lower()
        ),
        "- Selection-efficiency claim allowed: {}.".format(
            str(result["gates"]["selection_efficiency_claim_allowed"]).lower()
        ),
        "- Conclusion: `{}`.".format(result["conclusion"]),
        "",
        "| strategy | valid | first oracle+2% | Regret@K |",
        "|---|---:|---:|---:|",
    ]
    for name, row in result["strategy_metrics"].items():
        lines.append("| {} | {} | {} | {} |".format(
            name,
            row["valid_count"],
            row["first_oracle_plus_2pct_index"],
            "{:.6f}".format(row["regret_at_k"]) if row["regret_at_k"] is not None else "n/a",
        ))
    (args.output_dir / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result["gates"], indent=2))


if __name__ == "__main__":
    main()
