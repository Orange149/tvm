#!/usr/bin/env python3
"""Build and freeze the fixed-tile fusion-aware transfer model and E1 predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fusion_aware_transfer import (
    build_model,
    canonical,
    digest_value,
    estimate_domain,
    read_json,
    sha256_file,
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    root = Path("vta/tutorials/frontend/report_out/stage_tile_cotuning")
    audit = root / "compile_context_audit"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, default=audit / "precodegen87_run1")
    parser.add_argument("--archive", type=Path, default=audit / "representatives23_run1")
    parser.add_argument("--census", type=Path, default=audit / "representatives23_run1/logical_dma_census_v2.json")
    parser.add_argument("--verification", type=Path, default=audit / "representatives23_run1/context_verification_v2.json")
    parser.add_argument("--workloads", type=Path, default=root / "stage_memory_experiments/static_workload_dma.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite {}".format(args.output_dir))
    args.output_dir.mkdir(parents=True)

    model = build_model(
        scan_dir=args.scan,
        archive_dir=args.archive,
        census_path=args.census,
        verification_path=args.verification,
        workloads_path=args.workloads,
    )
    estimates = estimate_domain(model=model, scan_dir=args.scan)
    calibration = set(model["calibration_segments"])
    blind_rows = [row for row in estimates if row["segment_id"] not in calibration]
    if len(blind_rows) != 64:
        raise ValueError("expected 64 blind segments, got {}".format(len(blind_rows)))
    prediction_hash = digest_value(blind_rows)

    write_json(args.output_dir / "model.json", model)
    write_json(args.output_dir / "estimates.json", {
        "schema": "vta_fixed_tile_fusion_transfer_estimates_v1",
        "model_sha256": sha256_file(args.output_dir / "model.json"),
        "rows": estimates,
        "status_counts": {status: sum(row["status"] == status for row in estimates)
                          for status in sorted({row["status"] for row in estimates})},
    })
    write_json(args.output_dir / "blind64_predictions.json", {
        "schema": "vta_fixed_tile_fusion_transfer_blind_predictions_v1",
        "status": "predictions_frozen_before_full_build_truth",
        "calibration_segments": sorted(calibration),
        "blind_segments": [row["segment_id"] for row in blind_rows],
        "prediction_sha256": prediction_hash,
        "rows": blind_rows,
    })
    write_json(args.output_dir / "blind64_selection.json", {
        "status": "blind validation selection; predictions already frozen",
        "source_summary_sha256": sha256_file(args.scan / "summary.json"),
        "level2_proposal": {"selected_segments": [row["segment_id"] for row in blind_rows]},
        "prediction_sha256": prediction_hash,
    })
    write_json(args.output_dir / "full_build_queue.json", {
        "schema": "vta_fixed_tile_fusion_transfer_full_build_queue_v1",
        "policy": "every non-exact_reuse segment must be fully built; estimates must not be used for ranking",
        "rows": [
            {
                "segment_id": row["segment_id"],
                "status": row["status"],
                "all_status_reasons": row["all_status_reasons"],
            }
            for row in estimates
            if row["full_build_required"]
        ],
    })
    write_json(args.output_dir / "manifest.json", {
        "schema": "vta_fixed_tile_fusion_transfer_freeze_manifest_v1",
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__)),
        "sources": {name: {"path": str(path.resolve()), "sha256": sha256_file(path)} for name, path in (
            ("scan_summary", args.scan / "summary.json"),
            ("archive_summary", args.archive / "summary.json"),
            ("census", args.census),
            ("verification", args.verification),
            ("workloads", args.workloads),
        )},
        "board_contacted": False,
        "full_blind_builds_executed": False,
    })
    status = [
        "# C1 fixed-tile fusion-aware transfer status",
        "",
        "- Model: frozen from 23 complete representative segments.",
        "- Domain: 87 audited ResNet-18 VTA segments.",
        "- Blind predictions: 64/64 frozen before full-build truth.",
        "- Exact-reuse estimates: {}/87.".format(sum(row["status"] == "exact_reuse" for row in estimates)),
        "- Prediction hash: `{}`.".format(prediction_hash),
        "- Board contacted: no.",
        "- Claim boundary: logical compiler-visible DMA/CPU accesses, not physical DDR or latency.",
        "- Next gate: build the frozen 64-segment selection in a new archive, then run the blind validator.",
    ]
    (args.output_dir / "STATUS.md").write_text("\n".join(status) + "\n")
    hashes = {
        path.name: sha256_file(path)
        for path in args.output_dir.iterdir()
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    write_json(args.output_dir / "artifact_hashes.json", {"artifacts": hashes})
    print(json.dumps({
        "model_coverage": model["coverage"],
        "status_counts": {status: sum(row["status"] == status for row in estimates)
                          for status in sorted({row["status"] for row in estimates})},
        "blind_segments": len(blind_rows),
        "prediction_sha256": prediction_hash,
    }, indent=2))


if __name__ == "__main__":
    main()
