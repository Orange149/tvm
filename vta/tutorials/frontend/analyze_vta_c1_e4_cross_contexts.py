#!/usr/bin/env python3
"""Summarize the paired-context C1 E4 lowering without hiding fail-closed cases."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from prepare_vta_c1_cross_model_e4 import canonical, seal
from fusion_aware_transfer import read_json, sha256_file


DMA_FIELDS = (
    "load_buffer_2d_inp_bytes",
    "load_buffer_2d_wgt_bytes",
    "load_buffer_2d_acc_bytes",
    "store_buffer_2d_out_bytes",
    "load_buffer_2d_calls",
    "store_buffer_2d_calls",
)
BASE_FIELDS = (
    "load_buffer_2d_inp_bytes",
    "load_buffer_2d_wgt_bytes",
    "store_buffer_2d_out_bytes",
)


def validate_sealed(payload, label):
    expected = payload.get("artifact_sha256")
    unsealed = dict(payload)
    unsealed.pop("artifact_sha256", None)
    observed = hashlib.sha256(canonical(unsealed).encode("utf-8")).hexdigest()
    if expected != observed:
        raise ValueError(label + " artifact hash mismatch")


def dma_vector(build):
    result = {field: 0 for field in DMA_FIELDS}
    for snapshot in build["snapshots"]:
        for function in snapshot["functions"]:
            totals = function["dma"]["totals"]
            for field in result:
                result[field] += totals.get(field, 0)
    return result


def classify_failure(failure):
    if not failure:
        return None
    message = failure.get("message", "")
    if "Allocation exceed bound of memory tag local.acc_buffer" in message:
        return "acc_buffer_capacity_exceeded"
    if "Dynamic" in message and "DMA" in message:
        return "unsupported_dynamic_dma"
    return "other_lowering_failure"


def analyze(source):
    payload = read_json(source)
    validate_sealed(payload, "context lowering")
    rows = []
    for item in payload["rows"]:
        builds = item["builds"]
        if item["status"] != "complete":
            rows.append(
                {
                    "workload_id": item["workload_id"],
                    "status": "failed_closed",
                    "completed_contexts": [build["context"] for build in builds],
                    "failure_class": classify_failure(item["failure"]),
                    "failure_type": item["failure"]["exception_type"],
                    "failure_message": item["failure"]["message"].split("\n")[-1],
                }
            )
            continue
        by_context = {build["context"]: build for build in builds}
        plain = dma_vector(by_context["conv_quant"])
        residual = dma_vector(by_context["conv_residual_quant"])
        delta = {field: residual[field] - plain[field] for field in DMA_FIELDS}
        exact_dispatch = all(
            build["dispatch_identity"]["exact"] for build in builds
        )
        rows.append(
            {
                "workload_id": item["workload_id"],
                "status": "complete",
                "exact_frozen_dispatch": exact_dispatch,
                "same_tile_identity": item["same_tile_identity_across_contexts"],
                "distinct_fusion_signatures": item["fusion_contexts_distinct"],
                "conv_quant_dma": plain,
                "conv_residual_quant_dma": residual,
                "fusion_dma_delta": delta,
                "base_dma_invariant": all(delta[field] == 0 for field in BASE_FIELDS),
                "acc_load_delta_positive": delta["load_buffer_2d_acc_bytes"] > 0,
            }
        )
    complete = [row for row in rows if row["status"] == "complete"]
    failed = [row for row in rows if row["status"] == "failed_closed"]
    result = seal(
        {
            "schema": "vta_c1_cross_model_e4_context_analysis_v1",
            "scope": (
                "synthetic paired compiler contexts with pre-frozen model-local tiles; "
                "not natural cross-placement stability, shared-memory certification, "
                "board performance, or end-to-end model evidence"
            ),
            "source": {
                "path": str(source.resolve()),
                "sha256": sha256_file(source),
                "artifact_sha256": payload["artifact_sha256"],
            },
            "analyzer": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__)),
            },
            "rows": rows,
            "summary": {
                "requested_workloads": len(rows),
                "complete_paired_contexts": len(complete),
                "failed_closed": len(failed),
                "exact_frozen_dispatch": sum(
                    row["exact_frozen_dispatch"] for row in complete
                ),
                "same_tile_identity": sum(row["same_tile_identity"] for row in complete),
                "distinct_fusion_signatures": sum(
                    row["distinct_fusion_signatures"] for row in complete
                ),
                "base_dma_invariant": sum(row["base_dma_invariant"] for row in complete),
                "positive_acc_load_residual": sum(
                    row["acc_load_delta_positive"] for row in complete
                ),
                "failure_classes": {
                    name: sum(row["failure_class"] == name for row in failed)
                    for name in sorted({row["failure_class"] for row in failed})
                },
            },
            "conclusion": (
                "The frozen tile is mechanically reusable across both synthetic fusion "
                "contexts only for the complete rows. A fused residual is represented by "
                "a measured static-DMA delta. Any capacity failure remains fail-closed and "
                "requires context-local recompilation or retuning."
            ),
        }
    )
    return result


def write_markdown(path, payload):
    summary = payload["summary"]
    lines = [
        "# C1 E4 paired fusion-context analysis",
        "",
        "This is a compiler-only mechanism check. It does not prove natural tile "
        "stability across model partition placements and does not measure shared-memory "
        "or board speed.",
        "",
        "- Paired contexts complete: {}/{}".format(
            summary["complete_paired_contexts"], summary["requested_workloads"]
        ),
        "- Exact frozen dispatch / same tile / distinct fusion signatures: {}/{}/{}".format(
            summary["exact_frozen_dispatch"],
            summary["same_tile_identity"],
            summary["distinct_fusion_signatures"],
        ),
        "- Base input/weight/output DMA invariant: {}/{}".format(
            summary["base_dma_invariant"], summary["complete_paired_contexts"]
        ),
        "- Positive fused ACC-load residual: {}/{}".format(
            summary["positive_acc_load_residual"], summary["complete_paired_contexts"]
        ),
        "- Fail-closed classes: `{}`".format(json.dumps(summary["failure_classes"], sort_keys=True)),
        "",
        "| Workload | Result | Base DMA invariant | ACC residual | Failure class |",
        "|---|---|---:|---:|---|",
    ]
    for row in payload["rows"]:
        if row["status"] == "complete":
            lines.append(
                "| {} | complete | {} | {} B |  |".format(
                    row["workload_id"],
                    "yes" if row["base_dma_invariant"] else "no",
                    row["fusion_dma_delta"]["load_buffer_2d_acc_bytes"],
                )
            )
        else:
            lines.append(
                "| {} | fail-closed |  |  | {} |".format(
                    row["workload_id"], row["failure_class"]
                )
            )
    lines += ["", payload["conclusion"], ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite {}".format(args.output_dir))
    args.output_dir.mkdir(parents=True)
    result = analyze(args.source)
    (args.output_dir / "analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_markdown(args.output_dir / "README.md", result)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
