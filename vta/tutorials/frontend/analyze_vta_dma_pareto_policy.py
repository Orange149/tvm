#!/usr/bin/env python3
"""Evaluate a hardware-derived DMA Pareto gate for VTA input residency.

The gate is deliberately label-free.  For an exact same-tile original/residency
pair it requires that input traffic is reduced while neither total DMA bytes nor
total DMA requests increase.  Latency labels are used only after the decisions
have been made, to evaluate the policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_reduction(before, after):
    before = float(before)
    return None if before == 0 else (before - float(after)) / before


def p7q_pairs(contract, effects):
    entries = {
        row["candidate_id"]: row
        for workload in contract["workloads"].values()
        for row in workload["gross_candidates"]
    }
    rows = []
    for pair in effects["pairs"]:
        if pair["residence_mode"] != "input_stationary":
            continue
        original = entries[pair["control_candidate_id"]]
        residency = entries[pair["mechanism_candidate_id"]]
        before, after = original["static_metrics"], residency["static_metrics"]
        rows.append(
            evaluate_pair(
                workload_id=pair["workload_id"],
                pair_id="{}:{}".format(pair["workload_id"], pair["control_config_index"]),
                input_bytes_before=before["input_dma_bytes"],
                input_bytes_after=after["input_dma_bytes"],
                total_bytes_before=before["dma_total_bytes"],
                total_bytes_after=after["dma_total_bytes"],
                total_calls_before=before["dma_total_calls"],
                total_calls_after=after["dma_total_calls"],
                speedup_percent=pair["improvement_percent"],
                source="P7Q development labels",
            )
        )
    return rows


def unseen_pairs(shortlist, timing):
    candidates = {
        row["p7r_selection"]["stratum"]: row
        for row in shortlist
        if row["workload_id"] == timing["workload_id"]
    }
    rows = []
    for stratum, comparison_name in (
        ("request_shape", "shape"),
        ("contiguous_control", "control"),
    ):
        row = candidates[stratum]
        before, after = row["original"]["dma"], row["input_stationary"]["dma"]
        comparison = timing["comparisons"][comparison_name]
        rows.append(
            evaluate_pair(
                workload_id=row["workload_id"],
                pair_id="{}:{}".format(row["workload_id"], row["config_index"]),
                input_bytes_before=before["load_buffer_2d_inp_bytes"],
                input_bytes_after=after["load_buffer_2d_inp_bytes"],
                total_bytes_before=before["load_buffer_2d_bytes"]
                + before["store_buffer_2d_bytes"],
                total_bytes_after=after["load_buffer_2d_bytes"]
                + after["store_buffer_2d_bytes"],
                total_calls_before=before["load_buffer_2d_calls"]
                + before["store_buffer_2d_calls"],
                total_calls_after=after["load_buffer_2d_calls"]
                + after["store_buffer_2d_calls"],
                speedup_percent=comparison["paired_median_speedup_fraction"] * 100.0,
                source="P7R unseen E02 labels (retrospective only)",
            )
        )
    return rows


def evaluate_pair(
    *, workload_id, pair_id, input_bytes_before, input_bytes_after,
    total_bytes_before, total_bytes_after, total_calls_before, total_calls_after,
    speedup_percent, source
):
    reductions = {
        "input_bytes": relative_reduction(input_bytes_before, input_bytes_after),
        "total_dma_bytes": relative_reduction(total_bytes_before, total_bytes_after),
        "total_dma_calls": relative_reduction(total_calls_before, total_calls_after),
    }
    reasons = []
    if reductions["input_bytes"] is None or reductions["input_bytes"] <= 0:
        reasons.append("no_input_reuse")
    if reductions["total_dma_bytes"] is None or reductions["total_dma_bytes"] < 0:
        reasons.append("total_dma_bytes_increase")
    if reductions["total_dma_calls"] is None or reductions["total_dma_calls"] < 0:
        reasons.append("total_dma_calls_increase")
    return {
        "workload_id": workload_id,
        "pair_id": pair_id,
        "source": source,
        "reductions": reductions,
        "pareto_certified": not reasons,
        "reject_reasons": reasons,
        "measured_speedup_percent": float(speedup_percent),
        "measured_faster": float(speedup_percent) > 0.0,
    }


def summarize(rows):
    selected = [row for row in rows if row["pareto_certified"]]
    rejected = [row for row in rows if not row["pareto_certified"]]
    return {
        "pair_count": len(rows),
        "certified_count": len(selected),
        "certified_faster_count": sum(row["measured_faster"] for row in selected),
        "certified_regression_count": sum(not row["measured_faster"] for row in selected),
        "certified_median_speedup_percent": (
            statistics.median(row["measured_speedup_percent"] for row in selected)
            if selected else None
        ),
        "abstained_count": len(rejected),
        "abstained_regression_count": sum(not row["measured_faster"] for row in rejected),
    }


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p7q-contract", required=True)
    parser.add_argument("--p7q-effects", required=True)
    parser.add_argument("--unseen-shortlist", required=True)
    parser.add_argument("--unseen-timing", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    output.mkdir(parents=True)

    sources = {name: Path(value).resolve() for name, value in (
        ("p7q_contract", args.p7q_contract),
        ("p7q_effects", args.p7q_effects),
        ("unseen_shortlist", args.unseen_shortlist),
        ("unseen_timing", args.unseen_timing),
    )}
    development = p7q_pairs(
        json.loads(sources["p7q_contract"].read_text()),
        json.loads(sources["p7q_effects"].read_text()),
    )
    unseen = unseen_pairs(
        load_jsonl(sources["unseen_shortlist"]),
        json.loads(sources["unseen_timing"].read_text()),
    )
    result = {
        "schema": "c3_dma_pareto_policy_ablation_v1",
        "status": "completed",
        "rule": {
            "label_free": True,
            "same_tile_required": True,
            "conditions": [
                "input_dma_bytes_after < input_dma_bytes_before",
                "total_dma_bytes_after <= total_dma_bytes_before",
                "total_dma_calls_after <= total_dma_calls_before",
            ],
            "semantics": "certification is sufficient for priority, not proof of speedup; abstention retains original",
        },
        "claim_boundary": {
            "P7Q": "post-hoc development ablation",
            "E02": "retrospective falsification of the input-only rule; not prospective proof",
            "next": "freeze this rule before a new E03 board label",
        },
        "P7Q": {"summary": summarize(development), "pairs": development},
        "E02": {"summary": summarize(unseen), "pairs": unseen},
        "sources": {name: {"path": str(path), "sha256": sha256_file(path)}
                    for name, path in sources.items()},
    }
    (output / "analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    status = [
        "# DMA Pareto policy ablation",
        "",
        "- Status: `completed`",
        "- P7Q input-stationary pairs: {}".format(len(development)),
        "- P7Q Pareto-certified: {}; faster: {}; regressions: {}".format(
            result["P7Q"]["summary"]["certified_count"],
            result["P7Q"]["summary"]["certified_faster_count"],
            result["P7Q"]["summary"]["certified_regression_count"],
        ),
        "- E02 Pareto-certified: {}; abstained: {}; avoided regressions: {}".format(
            result["E02"]["summary"]["certified_count"],
            result["E02"]["summary"]["abstained_count"],
            result["E02"]["summary"]["abstained_regression_count"],
        ),
        "- Boundary: retrospective ablation only; a new E03 contract must be frozen before board timing.",
    ]
    (output / "STATUS.md").write_text("\n".join(status) + "\n")
    (output / "manifest.json").write_text(json.dumps({
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__)),
        "board_contacted": False,
    }, indent=2) + "\n")
    hashes = {path.name: sha256_file(path) for path in output.iterdir()
              if path.is_file() and path.name != "artifact_hashes.json"}
    (output / "artifact_hashes.json").write_text(
        json.dumps({"artifacts": hashes}, indent=2) + "\n"
    )
    print(json.dumps({"P7Q": result["P7Q"]["summary"], "E02": result["E02"]["summary"]}, indent=2))


if __name__ == "__main__":
    main()
