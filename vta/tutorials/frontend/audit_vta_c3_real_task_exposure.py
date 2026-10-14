#!/usr/bin/env python3
"""Bind real C3 dispatch tasks to exact historical latency evidence.

This is a read-only evidence join.  A workload alias is accepted only when its
predeclared geometry matches the inventory dispatch signature and the referenced
board summary reports completed, correct timing.  Tasks without a binding remain
unseen candidates or unresolved; no geometry-only incumbent is invented.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from build_vta_c3_real_task_inventory import canonical, digest, sha256_file, write_json, write_jsonl


HERE = Path(__file__).resolve().parent
C3 = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune"
GROUPED = C3 / "07_grouped_holdout"
DEFAULT_INVENTORY = GROUPED / "20260924_c3_real_task_inventory_run01"
DEFAULT_OUTPUT = GROUPED / "20260924_c3_real_task_exposure_audit_run01"
SCHEMA = "c3_real_task_exposure_audit_v1"


def sig(ci, h, w, co, k, stride=1, pad=None):
    pad = k // 2 if pad is None else pad
    return [1, ci, h, w, co, k, k, stride, stride, pad, pad, pad, pad]


ALIASES = {
    "Y00": {"model": "yolov3_tiny_416_darknet", "source_layer": "conv2", "signature": sig(16, 208, 208, 32, 3)},
    "Y01": {"model": "yolov3_tiny_416_darknet", "source_layer": "conv13", "signature": sig(1024, 13, 13, 256, 1, pad=0)},
    "Y02": {"model": "yolov3_tiny_416_darknet", "source_layer": "conv21", "signature": sig(384, 26, 26, 256, 3)},
    "Y03": {"model": "yolov3_tiny_416_darknet", "source_layer": "conv8", "signature": sig(128, 26, 26, 256, 3)},
    "Y04": {"model": "yolov3_tiny_416_darknet", "source_layer": "conv18", "signature": sig(256, 13, 13, 128, 1, pad=0)},
    "Y05": {"model": "yolov3_tiny_416_darknet", "source_layer": "conv6", "signature": sig(64, 52, 52, 128, 3)},
    "Y06": {"model": "yolov3_tiny_416_darknet", "source_layer": "conv4", "signature": sig(32, 104, 104, 64, 3)},
    "Y07": {"model": "yolov3_tiny_416_darknet", "source_layer": "conv14", "signature": sig(256, 13, 13, 512, 3)},
    "Y08": {"model": "yolov3_tiny_416_darknet", "source_layer": "conv12", "signature": sig(512, 13, 13, 1024, 3)},
    "R18-H1": {"model": "relay_testing_resnet18", "source_layer": None, "signature": sig(64, 56, 56, 64, 3)},
    "R18-H2": {"model": "relay_testing_resnet18", "source_layer": None, "signature": sig(256, 14, 14, 256, 3)},
    "R18-H3": {"model": "relay_testing_resnet18", "source_layer": None, "signature": sig(256, 14, 14, 512, 1, stride=2, pad=0)},
}


EVIDENCE = {
    "Y00": "20260912_p7r144_y00_layout_preserving_search_board_run01",
    "Y01": "20260912_p7r170_y01_six_point_clean_start_board_run01",
    "Y02": "20260912_p7r123_y02_b00_paired_timing_run01",
    "Y03": "20260912_p7r154_y03_layout_preserving_search_board_run01",
    "Y04": "20260912_p7r165_y04_clean_start_search_board_run01",
    "Y05": "20260912_p7r180_y05_full_board_oracle_completion_run01",
    "Y06": "20260912_p7r191_y06_full_board_oracle_completion_run01",
    "Y07": "20260912_p7r200_y07_full_board_oracle_completion_run01",
    "Y08": "20260913_p7r377_y08_frozen_board_pool_run01",
    "R18-H1": "20260914_p7r510_resnet18_merged_board_pool_run01",
    "R18-H2": "20260914_p7r510_resnet18_merged_board_pool_run01",
    "R18-H3": "20260914_p7r510_resnet18_merged_board_pool_run01",
}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def evidence_binding(workload_id, directory):
    directory = Path(directory)
    summary_path = directory / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = load_json(summary_path)
    status = str(summary.get("status", ""))
    correct = bool(
        summary.get("all_timed_correct")
        or summary.get("all_timed_calls_correct")
        or summary.get("fpga_correct_count")
        or summary.get("correct_candidate_count")
    )
    timing_samples = int(
        summary.get("timing_samples", summary.get("candidate_samples", 0))
    )
    candidate_id = None
    latency_ms = None
    source_kind = None
    if workload_id in summary.get("workload_oracles", {}):
        oracle = summary["workload_oracles"][workload_id]
        candidate_id, latency_ms = oracle["candidate_id"], oracle["latency_ms"]
        source_kind = "workload_oracle_in_complete_historical_pool"
    elif summary.get("pool_oracle_candidate_id"):
        candidate_id = summary["pool_oracle_candidate_id"]
        latency_ms = summary.get("pool_oracle_latency_ms")
        source_kind = "pool_oracle_in_historical_pool"
    elif summary.get("oracle_candidate_id"):
        candidate_id = summary["oracle_candidate_id"]
        latency_ms = summary.get("oracle_latency_ms")
        source_kind = "pool_oracle_in_historical_pool"
    elif summary.get("wave_best_candidate_id"):
        candidate_id = summary["wave_best_candidate_id"]
        latency_ms = summary.get("wave_best_latency_ms")
        source_kind = "best_in_historical_measured_wave"
    elif workload_id == "Y02":
        same_tile = summary.get("analysis", {}).get("same_tile", {})
        candidate_id = same_tile.get("weight_resident_barrier_id")
        latency_ms = same_tile.get("barrier_median_ms")
        source_kind = "best_of_historical_same_tile_pair"
    if not status or not correct or timing_samples <= 0 or not candidate_id or latency_ms is None:
        raise ValueError(
            "incomplete timing evidence for {}: {}".format(
                workload_id,
                {"status": status, "correct": correct, "timing_samples": timing_samples,
                 "candidate_id": candidate_id, "latency_ms": latency_ms},
            )
        )
    files = {}
    for name in ("summary.json", "timing_summary.json", "contract.json", "execution_contract.json"):
        path = directory / name
        if path.is_file():
            files[name] = sha256_file(path)
    return {
        "workload_id": workload_id,
        "directory": str(directory.resolve()),
        "files": files,
        "status": status,
        "timing_samples": timing_samples,
        "correct_timing": True,
        "incumbent_candidate_id": candidate_id,
        "incumbent_latency_ms": float(latency_ms),
        "incumbent_scope": source_kind,
        "incumbent_global_optimum_claimed": False,
    }


def alias_matches(task, workload_id):
    alias = ALIASES[workload_id]
    if task["model"] != alias["model"] or task["geometry_signature"] != alias["signature"]:
        return False
    return alias["source_layer"] is None or alias["source_layer"] in task["source_layers"]


def audit(inventory_dir, grouped_root):
    inventory_dir = Path(inventory_dir)
    dispatch = load_jsonl(inventory_dir / "dispatch_tasks.jsonl")
    initial = {
        row["dispatch_scope_id"]: row
        for row in load_jsonl(inventory_dir / "exposure_ledger.jsonl")
    }
    bindings = {
        workload_id: evidence_binding(workload_id, Path(grouped_root) / directory)
        for workload_id, directory in EVIDENCE.items()
    }
    rows = []
    for task in dispatch:
        aliases = sorted(
            workload_id for workload_id in ALIASES if alias_matches(task, workload_id)
        )
        prior = initial[task["dispatch_scope_id"]]
        if aliases:
            evidence = [bindings[workload_id] for workload_id in aliases]
            best = min(evidence, key=lambda row: (row["incumbent_latency_ms"], row["incumbent_candidate_id"]))
            exposure_status = "latency_exposed_exact_alias_binding"
            unseen = False
            incumbent = {
                "status": "bound_current_known_better_historical_candidate",
                "candidate_id": best["incumbent_candidate_id"],
                "latency_ms": best["incumbent_latency_ms"],
                "scope": best["incumbent_scope"],
                "workload_id": best["workload_id"],
                "global_optimum_claimed": False,
            }
        elif prior["possible_latency_source_count"]:
            evidence = []
            exposure_status = "latency_or_board_geometry_exposure_unresolved"
            unseen = False
            incumbent = {
                "status": "pending_exact_evidence_binding",
                "candidate_id": None,
                "reason": "historical board/timing-named geometry source exists without exact alias",
            }
        else:
            evidence = []
            exposure_status = "strict_latency_unseen_candidate_after_known_evidence_audit"
            unseen = True
            incumbent = {
                "status": "pending_production_or_tophub_binding",
                "candidate_id": None,
                "reason": "no comparable historical candidate latency was found",
            }
        rows.append(
            {
                "schema": "c3_dispatch_exposure_binding_v1",
                "dispatch_scope_id": task["dispatch_scope_id"],
                "model": task["model"],
                "source_layers": task["source_layers"],
                "geometry_signature": task["geometry_signature"],
                "call_count": task["call_count"],
                "execution_contract_count": len(task["execution_contract_ids"]),
                "known_workload_aliases": aliases,
                "exposure_status": exposure_status,
                "strict_latency_unseen_eligible": unseen,
                "incumbent": incumbent,
                "evidence": evidence,
                "initial_conservative_screen": {
                    "reserved_source_count": prior["reserved_source_count"],
                    "possible_latency_source_count": prior["possible_latency_source_count"],
                },
                "shared_dispatch_scope_note": (
                    "one exposed alias exposes all calls sharing this exact workload"
                    if aliases and len(task["source_layers"]) > len(aliases)
                    else None
                ),
            }
        )
    return rows


def render(rows):
    counts = {
        "dispatch_scopes": len(rows),
        "latency_exposed": sum(row["exposure_status"] == "latency_exposed_exact_alias_binding" for row in rows),
        "unresolved": sum(row["exposure_status"] == "latency_or_board_geometry_exposure_unresolved" for row in rows),
        "strict_unseen_candidates": sum(row["strict_latency_unseen_eligible"] for row in rows),
    }
    by_model = {}
    for model in sorted({row["model"] for row in rows}):
        subset = [row for row in rows if row["model"] == model]
        by_model[model] = {
            "dispatch_scopes": len(subset),
            "latency_exposed": sum(not row["strict_latency_unseen_eligible"] for row in subset),
            "strict_unseen_candidates": sum(row["strict_latency_unseen_eligible"] for row in subset),
        }
    lines = [
        "# C3 real-task exposure audit",
        "",
        "- Exact dispatch scopes: {}".format(counts["dispatch_scopes"]),
        "- Exact historical latency bindings: {}".format(counts["latency_exposed"]),
        "- Unresolved historical geometry exposures: {}".format(counts["unresolved"]),
        "- Strict latency-unseen candidates: {}".format(counts["strict_unseen_candidates"]),
        "",
        "| model | dispatch scopes | exposed/unresolved | strict unseen |",
        "|---|---:|---:|---:|",
    ]
    for model, values in by_model.items():
        lines.append(
            "| {} | {} | {} | {} |".format(
                model, values["dispatch_scopes"], values["latency_exposed"],
                values["strict_unseen_candidates"]
            )
        )
    lines.extend([
        "",
        "An alias-bound incumbent is only the best verified candidate in its stated historical pool.",
        "No global optimum, new FPGA result, or latency-unseen claim is inferred from geometry alone.",
    ])
    return counts, by_model, "\n".join(lines) + "\n"


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    rows = audit(args.inventory_dir, args.grouped_root)
    counts, by_model, status = render(rows)
    output.mkdir(parents=True)
    write_jsonl(output / "exposure_bindings.jsonl", rows)
    summary = {
        "schema": SCHEMA,
        "status": "exact_historical_exposure_bound",
        "counts": counts,
        "by_model": by_model,
        "inventory": {
            "path": str(Path(args.inventory_dir).resolve()),
            "artifact_hashes_sha256": sha256_file(Path(args.inventory_dir) / "artifact_hashes.json"),
        },
        "board_contacted": False,
        "new_performance_values_collected": False,
        "historical_performance_values_read": True,
        "claim_boundary": "historical exposure and current-known incumbent binding only",
    }
    write_json(output / "summary.json", summary)
    (output / "STATUS.md").write_text(status, encoding="utf-8")
    write_json(
        output / "artifact_hashes.json",
        {
            "artifacts": {
                path.name: sha256_file(path)
                for path in sorted(output.iterdir())
                if path.is_file() and path.name != "artifact_hashes.json"
            },
            "source_sha256": sha256_file(Path(__file__).resolve()),
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-dir", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--grouped-root", type=Path, default=GROUPED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
