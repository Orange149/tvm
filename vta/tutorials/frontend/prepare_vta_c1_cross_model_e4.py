#!/usr/bin/env python3
"""Freeze and audit the twelve C1 E4 cross-model representative workloads.

This is a readiness audit, not the E4 result.  Existing C3 operator contracts
are useful for selecting geometrically diverse real workloads, but they do not
contain the normalized fused primitive/TIR contexts required to prove C1 tile
stability across partition placements.  The script therefore records exact
provenance and fails closed instead of treating an operator-level schedule as a
cross-model C1 template hit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from fusion_aware_transfer import digest_value, read_json, sha256_file


ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "vta/tutorials/frontend/report_out"
GROUPED = (
    REPORT
    / "stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
)
DEFAULT_C1_MODEL = (
    REPORT
    / "stage_tile_cotuning/compile_context_audit/c1_fusion_transfer_freeze_run1/model.json"
)

DEFAULT_CONTRACT_DIRS = (
    # Six geometrically diverse ResNet-50 operators.
    "20260912_p7r205_r50a_bottleneck_adaptive_holdout_contract_run01",
    "20260912_p7r216_r50b_bottleneck_adaptive_holdout_contract_run01",
    "20260913_p7r385_r50f_stride2_candidate_contract_run01",
    "20260913_p7r389_r50g_stride2_candidate_contract_run01",
    "20260913_p7r399_r50h_stride2_candidate_contract_run01",
    "20260913_p7r409_r50i_stride2_candidate_contract_run01",
    # Six YOLOv3-tiny operators spanning 104, 52, 26, 20, 13 and 10 grids.
    "20260912_p7r186_y06_conv4_adaptive_holdout_contract_run01",
    "20260912_p7r175_y05_conv6_multifidelity_holdout_contract_run01",
    "20260912_p7r146_y03_conv8_label_isolated_contract_run01",
    "20260913_p7r425_y09_yolov3_tiny_320_conv8_candidate_contract_run01",
    "20260912_p7r195_y07_conv14_adaptive_holdout_contract_run02",
    "20260913_p7r428_y10_yolov3_tiny_320_conv18_candidate_contract_run01",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def seal(payload):
    result = dict(payload)
    result.pop("artifact_sha256", None)
    result["artifact_sha256"] = hashlib.sha256(
        canonical(result).encode("utf-8")
    ).hexdigest()
    return result


def load_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def verify_contract_directory(directory):
    directory = Path(directory)
    hashes = read_json(directory / "artifact_hashes.json")["artifacts"]
    for name in ("contract.json", "candidates.jsonl"):
        if hashes.get(name) != sha256_file(directory / name):
            raise ValueError("contract source hash mismatch: {}".format(directory / name))
    contract = read_json(directory / "contract.json")
    candidates = load_jsonl(directory / "candidates.jsonl")
    if not candidates or any(row.get("performance_label") is not None for row in candidates):
        raise ValueError("representative selection source is not label-free")
    workload_ids = {row.get("workload_id") for row in candidates}
    workloads = {canonical(row["identity"]["workload"]) for row in candidates}
    if workload_ids != {contract.get("workload_id")} or len(workloads) != 1:
        raise ValueError("candidate workload identity differs from contract")
    if any(
        row["identity"].get("hardware_fingerprint")
        != contract.get("hardware_fingerprint")
        for row in candidates
    ):
        raise ValueError("candidate hardware fingerprint differs from contract")
    return contract, candidates, {
        "directory": str(directory.resolve()),
        "contract_sha256": hashes["contract.json"],
        "candidates_sha256": hashes["candidates.jsonl"],
        "artifact_hashes_sha256": sha256_file(directory / "artifact_hashes.json"),
    }


def model_family(source_model):
    name = str(source_model).lower().replace("_", "-")
    if "resnet50" in name:
        return "resnet50"
    if "yolo" in name:
        return "yolov3_tiny"
    raise ValueError("unsupported E4 model: " + str(source_model))


def audit_representative(contract, candidates, c1_model, source):
    original = [row for row in candidates if row["identity"]["residence_mode"] == "original"]
    if not original:
        raise ValueError("representative lacks the original schedule mode")
    workload = original[0]["identity"]["workload"]
    config_indices = sorted({int(row["debug"]["config_index"]) for row in original})
    candidate_keys = [
        digest_value({"workload": workload, "config_index": index})
        for index in config_indices
    ]
    workload_hits = sorted(set(candidate_keys) & set(c1_model["workloads"]))
    # The C3 contract does not capture C1 normalized primitive signatures or the
    # full compiler environment contract.  Both are mandatory for exact reuse.
    exact_environment_available = False
    normalized_fusion_context_available = False
    return {
        "model_family": model_family(contract["source_model"]),
        "source_model": contract["source_model"],
        "workload_id": contract["workload_id"],
        "source_layer": contract.get("source_layer"),
        "geometry_signature": contract.get("geometry_signature"),
        "workload": workload,
        "original_config_indices": config_indices,
        "resnet18_model_workload_hits": workload_hits,
        "checks": {
            "real_model_operator_provenance": bool(
                contract.get("model_source_geometry") or contract.get("geometry_signature")
            ),
            "label_free_representative_contract": all(
                row.get("performance_label") is None for row in candidates
            ),
            "target_hardware_shape_matches": contract.get("hardware_fingerprint", {}).get(
                "target"
            )
            == "axu5evb",
            "resnet18_workload_template_hit": bool(workload_hits),
            "exact_c1_environment_contract_available": exact_environment_available,
            "normalized_fusion_context_available": normalized_fusion_context_available,
            "cross_placement_tile_stability_proven": False,
            "exact_shared_edge_policy_certificate_available": False,
        },
        "reuse_decision": "fail_closed_full_context_build_required",
        "reuse_status": "environment_mismatch",
        "secondary_uncovered_reasons": [
            "new_workload" if not workload_hits else None,
            "new_fusion_class",
            "missing_cross_placement_context_pair",
            "missing_exact_shared_edge_policy_certificate",
        ],
        "source": source,
    }


def build_audit(contract_dirs, c1_model_path):
    c1_model = read_json(c1_model_path)
    if c1_model.get("schema") != "vta_fixed_tile_fusion_transfer_model_v1":
        raise ValueError("unexpected C1 model schema")
    rows = []
    for directory in contract_dirs:
        contract, candidates, source = verify_contract_directory(directory)
        rows.append(audit_representative(contract, candidates, c1_model, source))
    counts = {
        family: sum(row["model_family"] == family for row in rows)
        for family in ("resnet50", "yolov3_tiny")
    }
    if counts != {"resnet50": 6, "yolov3_tiny": 6}:
        raise ValueError("E4 requires exactly six representatives per model: {}".format(counts))
    workload_ids = [row["workload_id"] for row in rows]
    if len(workload_ids) != len(set(workload_ids)):
        raise ValueError("E4 representative workload ids must be unique")
    readiness_passed = all(
        row["checks"]["real_model_operator_provenance"]
        and row["checks"]["label_free_representative_contract"]
        and row["checks"]["target_hardware_shape_matches"]
        and row["reuse_decision"] == "fail_closed_full_context_build_required"
        for row in rows
    )
    return seal(
        {
            "schema": "vta_c1_cross_model_e4_readiness_v1",
            "status": (
                "representatives_frozen_full_context_builds_pending"
                if readiness_passed
                else "readiness_failed"
            ),
            "scope": (
                "label-free representative and fail-closed readiness audit only; "
                "not tile-stability, fusion-coverage, shared-policy, FPGA, or speed evidence"
            ),
            "c1_model": {
                "path": str(Path(c1_model_path).resolve()),
                "sha256": sha256_file(c1_model_path),
                "environment_sha256": c1_model["environment"]["sha256"],
            },
            "preparer": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__)),
            },
            "representative_counts": counts,
            "rows": rows,
            "gate_checks": {
                "six_representatives_per_model": counts
                == {"resnet50": 6, "yolov3_tiny": 6},
                "all_sources_hash_verified": True,
                "all_sources_label_free": all(
                    row["checks"]["label_free_representative_contract"] for row in rows
                ),
                "all_uncovered_cases_fail_closed": all(
                    row["reuse_decision"]
                    == "fail_closed_full_context_build_required"
                    for row in rows
                ),
            },
            "remaining_protocol": [
                "for each representative freeze one model-specific tile before placement comparison",
                "compile at least two partition/fusion contexts with identical workload and environment",
                "capture normalized primitive DAG, config entity, lowered TIR and DMA signature",
                "require exact tile identity across contexts or report instability",
                "query fusion and shared-edge certificates; every miss must remain fail-closed",
            ],
            "e4_complete": False,
            "readiness_passed": readiness_passed,
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c1-model", type=Path, default=DEFAULT_C1_MODEL)
    parser.add_argument(
        "--contract-dirs",
        type=Path,
        nargs="*",
        default=[GROUPED / name for name in DEFAULT_CONTRACT_DIRS],
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite {}".format(args.output_dir))
    result = build_audit(args.contract_dirs, args.c1_model)
    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "readiness.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "STATUS.md").write_text(
        "# C1 E4 cross-model readiness\n\n"
        "- Representatives: ResNet-50 6, YOLOv3-tiny 6.\n"
        "- Existing C3 operator contracts are provenance inputs, not C1 placement-context evidence.\n"
        "- 12/12 cases require fail-closed full context builds.\n"
        "- E4 complete: no.\n"
        "- Readiness artifact SHA256: `{}`.\n".format(result["artifact_sha256"]),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "representative_counts": result["representative_counts"],
                "e4_complete": result["e4_complete"],
                "artifact_sha256": result["artifact_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
