#!/usr/bin/env python3
"""Resolve the remaining C3 ResNet18 exposure records with exact evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_AUDIT = GROUPED / "20260924_c3_real_task_exposure_audit_run02"
DEFAULT_OUTPUT = GROUPED / "20260924_c3_real_task_exposure_audit_v2_run01"
P7Q = GROUPED / "20260911_p7q_qualified_timing_contract_run01/contract.json"
P7Q_TIMING = {
    "W01": GROUPED / "20260911_p7q1_w01_timing_run01/summary.json",
    "W04": GROUPED / "20260911_p7q2_w04_timing_run01/summary.json",
    "W07": GROUPED / "20260911_p7q3_w07_timing_run01/summary.json",
    "W08": GROUPED / "20260911_p7q4_w08_timing_run01/summary.json",
}
NO_LATENCY = {
    "W03": {
        "contract": GROUPED / "20260911_p7r57_support_transfer_board_contracts_run01/w03_contract.json",
        "dispatch": GROUPED / "20260911_p7r63_w03_certified_dispatch_run02/dispatch.json",
    },
    "W06": {
        "contract": GROUPED / "20260911_p7r57_support_transfer_board_contracts_run01/w06_contract.json",
        "dispatch": GROUPED / "20260911_p7r65_w06_certified_dispatch_run02/dispatch.json",
    },
}
RECENT = GROUPED / "20260924_c3_real_task_cross_boot_analysis_run01/summary.json"
SCHEMA = "c3_real_task_exposure_audit_v2"


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def workload_signature(workload):
    data, weight = workload[1][1], workload[2][1]
    stride, padding = workload[3], workload[4]
    return [data[0] * data[4], data[1] * data[5], data[2], data[3],
            weight[0] * weight[4], weight[2], weight[3], stride[0], stride[1], *padding]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite immutable output: {args.output}")

    p7q = load(P7Q)
    p7q_by_signature = {}
    for wid, payload in p7q["workloads"].items():
        p7q_by_signature[tuple(workload_signature(payload["gross_candidates"][0]["workload"]))] = wid
    no_latency_by_signature = {}
    for wid, paths in NO_LATENCY.items():
        contract = load(paths["contract"])
        dispatch = load(paths["dispatch"])
        if "no latency" not in contract.get("purpose", ""):
            raise RuntimeError(f"{wid}: contract no longer certifies no latency")
        if dispatch.get("performance_labels_used") is not False:
            raise RuntimeError(f"{wid}: dispatch performance-label status changed")
        no_latency_by_signature[tuple(workload_signature(contract["workload"]))] = wid

    recent = load(RECENT)
    recent_signatures = {
        (1, 128, 28, 28, 128, 3, 3, 1, 1, 1, 1, 1, 1): "R18-U128C128H28K3S1",
        (1, 64, 56, 56, 64, 1, 1, 1, 1, 0, 0, 0, 0): "R18-U64C64H56K1S1",
    }

    rows = []
    for original in load_jsonl(args.audit / "exposure_bindings.jsonl"):
        row = dict(original)
        signature = tuple(row["geometry_signature"])
        row["schema"] = "c3_dispatch_exposure_binding_v2"
        if signature in recent_signatures:
            workload_id = recent_signatures[signature]
            result = recent["workloads"][workload_id]
            row.update({
                "exposure_status": "latency_exposed_current_c3_development_pool",
                "strict_latency_unseen_eligible": False,
                "incumbent": {
                    "status": "bound_current_c3_development_incumbent",
                    "candidate_id": result["incumbent_program_id"],
                    "latency_ms_by_boot": [result["boot1_incumbent_latency_ms"],
                                           result["boot2_incumbent_latency_ms"]],
                    "global_optimum_claimed": False,
                },
                "v2_resolution": {"kind": "recent_cross_boot_pool", "source": str(RECENT),
                                  "sha256": sha256(RECENT)},
            })
        elif signature in p7q_by_signature:
            workload_id = p7q_by_signature[signature]
            timing_path = P7Q_TIMING[workload_id]
            timing = load(timing_path)
            candidate_id, stats = min(timing["candidate_summaries"].items(),
                                      key=lambda item: (item[1]["median_ms"], item[0]))
            row.update({
                "exposure_status": "latency_exposed_exact_p7q_workload_binding",
                "strict_latency_unseen_eligible": False,
                "incumbent": {
                    "status": "bound_current_known_better_historical_candidate",
                    "candidate_id": candidate_id,
                    "latency_ms": stats["median_ms"],
                    "workload_id": workload_id,
                    "scope": "fastest candidate in qualified P7Q pool",
                    "global_optimum_claimed": False,
                },
                "v2_resolution": {"kind": "exact_blocked_workload_signature_and_timing",
                                  "contract": str(P7Q), "contract_sha256": sha256(P7Q),
                                  "timing": str(timing_path), "timing_sha256": sha256(timing_path)},
            })
        elif signature in no_latency_by_signature:
            workload_id = no_latency_by_signature[signature]
            paths = NO_LATENCY[workload_id]
            row.update({
                "exposure_status": "strict_latency_unseen_after_explicit_no_latency_audit",
                "strict_latency_unseen_eligible": True,
                "incumbent": {
                    "status": "pending_production_or_tophub_binding",
                    "candidate_id": None,
                    "reason": "exact historical task was correctness-only; no candidate latency was collected",
                },
                "v2_resolution": {
                    "kind": "exact_workload_explicit_no_latency_evidence",
                    "workload_id": workload_id,
                    "contract": str(paths["contract"]),
                    "contract_sha256": sha256(paths["contract"]),
                    "contract_purpose": load(paths["contract"])["purpose"],
                    "dispatch": str(paths["dispatch"]),
                    "dispatch_sha256": sha256(paths["dispatch"]),
                    "performance_labels_used": load(paths["dispatch"])["performance_labels_used"],
                },
            })
        rows.append(row)

    args.output.mkdir(parents=True)
    rows.sort(key=lambda row: row["dispatch_scope_id"])
    write_jsonl(args.output / "exposure_bindings.jsonl", rows)
    summary = {
        "schema": SCHEMA,
        "status": "six_unresolved_tasks_resolved_and_recent_pools_reclassified",
        "counts": {
            "dispatch_scopes": len(rows),
            "latency_exposed": sum("latency_exposed" in row["exposure_status"] for row in rows),
            "unresolved": sum("unresolved" in row["exposure_status"] for row in rows),
            "strict_unseen": sum(row["strict_latency_unseen_eligible"] for row in rows),
        },
        "strict_unseen_dispatch_scope_ids": [row["dispatch_scope_id"] for row in rows
                                             if row["strict_latency_unseen_eligible"]],
        "claim_boundary": "strict-unseen is based on exact historical evidence audit, not expected speedup",
        "board_contacted": False,
        "new_performance_values_collected": False,
    }
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "artifact_hashes.json", {
        "exposure_bindings.jsonl": sha256(args.output / "exposure_bindings.jsonl"),
        "summary.json": sha256(args.output / "summary.json"),
        "source": sha256(Path(__file__).resolve()),
    })
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
