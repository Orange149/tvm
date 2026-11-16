#!/usr/bin/env python3
"""Audit whether existing full-graph builds can close C1 E4 natural placement evidence.

The audit is intentionally read-only.  A full-graph record is reusable only when
its workload, complete ConfigEntity, implementation mode and public mode match
the pre-frozen E4 route and it retained a normalized primitive inventory.  A
workload-only or tile-only match remains diagnostic and fails closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "vta/tutorials/frontend/report_out"
DEFAULT_FIXED = (
    REPORT
    / "stage_tile_cotuning/compile_context_audit/c1_e4_fixed_tiles_run1/fixed_tiles.json"
)
DEFAULT_SEARCH_ROOT = (
    REPORT / "stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seal(payload):
    result = dict(payload)
    result.pop("artifact_sha256", None)
    result["artifact_sha256"] = hashlib.sha256(
        canonical(result).encode("utf-8")
    ).hexdigest()
    return result


def discover_builds(search_roots):
    builds = set()
    for root in search_roots:
        root = Path(root)
        for path in root.rglob("build.json"):
            if "fullgraph" in str(path).lower():
                builds.add(path.resolve())
    return sorted(builds, key=str)


def normalized_primitive_retained(build):
    inventory = build.get("primitive_inventory")
    return bool(inventory) and all(
        isinstance(item, dict)
        and item.get("signature")
        and item.get("normalized") is not None
        for item in inventory
    )


def classify_hit(frozen, hit, build):
    workload_exact = canonical(hit.get("workload")) == canonical(frozen["workload"])
    config = hit.get("complete_config_entity")
    config_recorded = config is not None
    config_exact = config_recorded and canonical(config) == canonical(frozen["config"])
    public_mode_exact = hit.get("public_mode") == "original"
    implementation_mode_exact = hit.get("implementation_mode") == 0
    primitive_retained = normalized_primitive_retained(build)
    exact_route = (
        workload_exact
        and config_exact
        and public_mode_exact
        and implementation_mode_exact
    )
    reusable = exact_route and primitive_retained
    if reusable:
        status = "exact_natural_context"
    elif exact_route:
        status = "missing_normalized_primitive"
    elif workload_exact and config_exact:
        status = "tile_only_schedule_mode_mismatch"
    elif workload_exact:
        status = "workload_only"
    else:
        status = "unrelated"
    return {
        "status": status,
        "workload_exact": workload_exact,
        "config_recorded": config_recorded,
        "config_exact": config_exact,
        "public_mode": hit.get("public_mode"),
        "public_mode_exact": public_mode_exact,
        "implementation_mode": hit.get("implementation_mode"),
        "implementation_mode_exact": implementation_mode_exact,
        "normalized_primitive_retained": primitive_retained,
        "reusable_as_natural_placement_evidence": reusable,
    }


def build_audit(fixed_path, build_paths):
    fixed_path = Path(fixed_path)
    fixed = json.loads(fixed_path.read_text(encoding="utf-8"))
    rows = fixed.get("rows", [])
    if len(rows) != 12:
        raise ValueError("E4 fixed tile freeze must contain exactly twelve workloads")
    frozen_by_workload = {canonical(row["workload"]): row for row in rows}
    if len(frozen_by_workload) != len(rows):
        raise ValueError("E4 frozen workload identities are not unique")

    findings = {row["workload_id"]: [] for row in rows}
    sources = []
    for build_path in sorted((Path(path).resolve() for path in build_paths), key=str):
        build = json.loads(build_path.read_text(encoding="utf-8"))
        sources.append({"path": str(build_path), "sha256": sha256_file(build_path)})
        hits = build.get("dispatch_audit", {}).get("schedule_hits", [])
        for index, hit in enumerate(hits):
            frozen = frozen_by_workload.get(canonical(hit.get("workload")))
            if frozen is None:
                continue
            classified = classify_hit(frozen, hit, build)
            findings[frozen["workload_id"]].append(
                {
                    "build_path": str(build_path),
                    "build_sha256": sources[-1]["sha256"],
                    "schedule_hit_index": index,
                    "candidate_id": hit.get("candidate_id"),
                    **classified,
                }
            )

    result_rows = []
    for frozen in rows:
        workload_id = frozen["workload_id"]
        matches = findings[workload_id]
        reusable = [
            item for item in matches if item["reusable_as_natural_placement_evidence"]
        ]
        result_rows.append(
            {
                "workload_id": workload_id,
                "frozen_candidate_id": frozen["candidate_id"],
                "frozen_tile_identity_sha256": frozen["tile_identity_sha256"],
                "match_count": len(matches),
                "reusable_context_count": len(reusable),
                "natural_cross_placement_ready": len(reusable) >= 2,
                "matches": matches,
            }
        )

    exact_contexts = sum(row["reusable_context_count"] for row in result_rows)
    ready_workloads = sum(row["natural_cross_placement_ready"] for row in result_rows)
    return seal(
        {
            "schema": "vta_c1_e4_natural_placement_reuse_audit_v1",
            "status": (
                "natural_placement_evidence_complete"
                if ready_workloads == len(result_rows)
                else "fail_closed_insufficient_existing_natural_placement_evidence"
            ),
            "scope": (
                "read-only reuse audit of pre-existing full-graph build records; "
                "no compiler, AutoTune, simulator or board execution"
            ),
            "fixed_tile_source": {
                "path": str(fixed_path.resolve()),
                "sha256": sha256_file(fixed_path),
                "artifact_sha256": fixed.get("artifact_sha256"),
            },
            "auditor": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__)),
            },
            "source_build_count": len(sources),
            "source_builds": sources,
            "rows": result_rows,
            "summary": {
                "frozen_workload_count": len(result_rows),
                "workloads_with_any_workload_match": sum(
                    bool(row["matches"]) for row in result_rows
                ),
                "exact_reusable_natural_context_count": exact_contexts,
                "natural_cross_placement_ready_workload_count": ready_workloads,
                "all_twelve_ready": ready_workloads == len(result_rows),
            },
            "claim_boundary": (
                "tile-only matches with a different schedule mode, or records without "
                "normalized primitive identity, cannot support cross-placement tile stability"
            ),
        }
    )


def write_readme(path, audit):
    summary = audit["summary"]
    lines = [
        "# C1 E4 natural-placement reuse audit",
        "",
        "- Status: `{}`.".format(audit["status"]),
        "- Existing full-graph build records scanned: {}.".format(
            audit["source_build_count"]
        ),
        "- Frozen workloads with any workload match: {}/12.".format(
            summary["workloads_with_any_workload_match"]
        ),
        "- Exact reusable natural contexts: {}.".format(
            summary["exact_reusable_natural_context_count"]
        ),
        "- Workloads with at least two exact natural placements: {}/12.".format(
            summary["natural_cross_placement_ready_workload_count"]
        ),
        "",
        "A workload/config match is insufficient when implementation/public mode differs or",
        "the normalized fused primitive was not retained. Such records remain fail-closed and",
        "are not relabeled as C1 cross-placement evidence.",
        "",
        "- Artifact SHA-256: `{}`.".format(audit["artifact_sha256"]),
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-tiles", type=Path, default=DEFAULT_FIXED)
    parser.add_argument(
        "--search-roots", type=Path, nargs="*", default=[DEFAULT_SEARCH_ROOT]
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite {}".format(args.output_dir))
    builds = discover_builds(args.search_roots)
    audit = build_audit(args.fixed_tiles, builds)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "analysis.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_readme(args.output_dir / "README.md", audit)
    print(
        json.dumps(
            {"status": audit["status"], **audit["summary"], "artifact_sha256": audit["artifact_sha256"]},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

