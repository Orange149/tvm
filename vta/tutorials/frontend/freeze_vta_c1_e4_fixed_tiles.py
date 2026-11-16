#!/usr/bin/env python3
"""Freeze one label-free, locally qualified tile for each C1 E4 workload."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from fusion_aware_transfer import digest_value, read_json, sha256_file
from prepare_vta_c1_cross_model_e4 import canonical, seal


ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "vta/tutorials/frontend/report_out"
GROUPED = REPORT / "stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_READINESS = (
    REPORT
    / "stage_tile_cotuning/compile_context_audit/c1_e4_cross_model_readiness_run3/readiness.json"
)
DEFAULT_QUALIFICATION_DIRS = {
    "R50A": "20260912_p7r209_r50a_full_local_oracle_completion_run01",
    "R50B": "20260912_p7r220_r50b_full_local_oracle_completion_run01",
    "R50F": "20260913_p7r387_r50f_full_local_qualification_run01",
    "R50G": "20260913_p7r391_r50g_full_local_qualification_run01",
    "R50H": "20260913_p7r401_r50h_full_local_qualification_run01",
    "R50I": "20260913_p7r411_r50i_full_local_qualification_run01",
    "Y06": "20260912_p7r190_y06_full_local_oracle_completion_run01",
    "Y05": "20260912_p7r179_y05_full_local_oracle_completion_run01",
    "Y03": "20260912_p7r151_y03_conv8_v2_local_pool_run01",
    "Y09": "20260913_p7r427_y09_full_local_qualification_run01",
    "Y07": "20260912_p7r199_y07_full_local_oracle_completion_run01",
    "Y10": "20260913_p7r430_y10_full_local_qualification_run01",
}


def validate_sealed(payload):
    expected = payload.get("artifact_sha256")
    unsealed = dict(payload)
    unsealed.pop("artifact_sha256", None)
    observed = hashlib.sha256(canonical(unsealed).encode("utf-8")).hexdigest()
    if expected != observed:
        raise ValueError("readiness artifact hash mismatch")


def load_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def verify_qualification(directory):
    directory = Path(directory)
    artifact_hashes = read_json(directory / "artifact_hashes.json")["artifacts"]
    required = (
        "candidates_v2.jsonl",
        "static_results.jsonl",
        "fsim_results.jsonl",
        "future_board_order.json",
        "summary.json",
    )
    for name in required:
        if artifact_hashes.get(name) != sha256_file(directory / name):
            raise ValueError("qualification hash mismatch: {}".format(directory / name))
    summary = read_json(directory / "summary.json")
    if summary.get("status") != "completed_local_no_board":
        raise ValueError("qualification is not a completed local pool")
    return {
        "candidates": load_jsonl(directory / "candidates_v2.jsonl"),
        "static": load_jsonl(directory / "static_results.jsonl"),
        "fsim": load_jsonl(directory / "fsim_results.jsonl"),
        "order": read_json(directory / "future_board_order.json"),
        "source": {
            "directory": str(directory.resolve()),
            "artifact_hashes_sha256": sha256_file(directory / "artifact_hashes.json"),
            **{name + "_sha256": artifact_hashes[name] for name in required},
        },
    }


def select_fixed_tile(workload_id, qualification):
    candidates = {row["candidate_id"]: row for row in qualification["candidates"]}
    static_pass = {
        row["candidate_id"]
        for row in qualification["static"]
        if row.get("public_mode") == "original"
        and row.get("status") == "ok"
        and row.get("failure") is None
        and row.get("performance_label") is None
    }
    fsim_pass = {
        row["candidate_id"]
        for row in qualification["fsim"]
        if row.get("public_mode") == "original"
        and row.get("status") == "passed"
        and row.get("failure") is None
        and row.get("performance_measurement") == "not_collected"
        and bool(row.get("tir_matches_static"))
    }
    eligible = static_pass & fsim_pass
    for family_id in qualification["order"]["full_structural_order"]:
        rows = [
            row
            for row in candidates.values()
            if row.get("family_id") == family_id
            and row.get("public_mode") == "original"
            and row["candidate_id"] in eligible
        ]
        if len(rows) > 1:
            raise ValueError("multiple original candidates in family " + family_id)
        if rows:
            row = rows[0]
            if row.get("workload_id") != workload_id:
                raise ValueError("qualification workload id mismatch")
            identity = row["identity"]
            config = identity["complete_config_entity"]
            return {
                "workload_id": workload_id,
                "candidate_id": row["candidate_id"],
                "family_id": family_id,
                "selection_rule": (
                    "first family in pre-existing label-free full_structural_order whose "
                    "original mode passed exact static lowering and three-seed FSim"
                ),
                "workload": identity["workload"],
                "config": config,
                "config_index": int(row["debug"]["config_index"]),
                "tile_identity_sha256": digest_value(config),
                "schedule_version": identity["schedule_version"],
                "hardware_fingerprint": identity["hardware_fingerprint"],
                "performance_label_used": False,
                "static_and_fsim_qualified": True,
                "qualification_source": qualification["source"],
            }
    raise ValueError("no original static+FSim-qualified tile for " + workload_id)


def build_freeze(readiness_path, qualification_dirs):
    readiness = read_json(readiness_path)
    validate_sealed(readiness)
    if not readiness.get("readiness_passed") or readiness.get("e4_complete"):
        raise ValueError("unexpected E4 readiness state")
    expected = [row["workload_id"] for row in readiness["rows"]]
    if set(expected) != set(qualification_dirs):
        raise ValueError("qualification workload set differs from readiness freeze")
    selected = []
    for workload_id in expected:
        selected.append(
            select_fixed_tile(
                workload_id, verify_qualification(qualification_dirs[workload_id])
            )
        )
    return seal(
        {
            "schema": "vta_c1_cross_model_e4_fixed_tile_freeze_v1",
            "status": "fixed_before_cross_context_lowering",
            "scope": (
                "one exact model-local tile per representative; no cross-context result, "
                "board label, shared-policy qualification, or speed claim"
            ),
            "readiness": {
                "path": str(Path(readiness_path).resolve()),
                "sha256": sha256_file(readiness_path),
                "artifact_sha256": readiness["artifact_sha256"],
            },
            "freezer": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__)),
            },
            "rows": selected,
            "gate_checks": {
                "representative_count": len(selected),
                "all_unique_workloads": len({row["workload_id"] for row in selected})
                == len(selected),
                "all_static_and_fsim_qualified": all(
                    row["static_and_fsim_qualified"] for row in selected
                ),
                "performance_labels_used": False,
                "cross_context_results_seen_before_freeze": False,
            },
            "e4_context_builds_started": False,
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite {}".format(args.output_dir))
    qualification_dirs = {
        key: GROUPED / value for key, value in DEFAULT_QUALIFICATION_DIRS.items()
    }
    result = build_freeze(args.readiness, qualification_dirs)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "fixed_tiles.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "representatives": len(result["rows"]),
                "artifact_sha256": result["artifact_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
