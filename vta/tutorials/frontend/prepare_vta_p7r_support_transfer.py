#!/usr/bin/env python3
"""Freeze unseen workload candidates by transferring a proven mapping structure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from prepare_vta_p7r_e03_confirmation import annotate


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def knobs_from_contract(entry):
    return {
        name: int(value[-1] if kind == "sp" else value)
        for name, kind, value in entry["complete_config_entity"]["entity"]
    }


def knob_signature(knobs):
    return tuple(int(knobs[name]) for name in (
        "tile_b", "tile_h", "tile_w", "tile_ci", "tile_co", "oc_nthread", "h_nthread"
    ))


def parse_transfer(value):
    try:
        target, anchor = value.split("=", 1)
        target_workload, target_index = target.split(":", 1)
        anchor_workload, anchor_index = anchor.split(":", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("transfer must be TARGET:INDEX=ANCHOR:INDEX") from error
    return target_workload.upper(), int(target_index), anchor_workload.upper(), int(anchor_index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", required=True)
    parser.add_argument("--development-contract", required=True)
    parser.add_argument("--development-effects", required=True)
    parser.add_argument("--transfer", action="append", type=parse_transfer, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    output.mkdir(parents=True)

    paths = {name: Path(value).resolve() for name, value in (
        ("scan", args.scan),
        ("development_contract", args.development_contract),
        ("development_effects", args.development_effects),
    )}
    scan = {(row["workload_id"], int(row["config_index"])): row
            for row in load_jsonl(paths["scan"])}
    contract = json.loads(paths["development_contract"].read_text())
    effects = json.loads(paths["development_effects"].read_text())
    development_entries = {
        (row["workload_id"], int(row["config_index"]), row["residence_mode"]): row
        for workload in contract["workloads"].values()
        for row in workload["gross_candidates"]
    }
    speedups = {
        (row["workload_id"], int(row["mechanism_config_index"])): row["improvement_percent"]
        for row in effects["pairs"] if row["residence_mode"] == "input_stationary"
    }

    selected = []
    for target_workload, target_index, anchor_workload, anchor_index in args.transfer:
        target = annotate(scan[(target_workload, target_index)])
        anchor = development_entries[(anchor_workload, anchor_index, "input_stationary")]
        speedup = float(speedups[(anchor_workload, anchor_index)])
        target_signature = knob_signature(target["knobs"])
        anchor_signature = knob_signature(knobs_from_contract(anchor))
        if target_signature != anchor_signature:
            raise ValueError("target and development anchor have different mapping structures")
        if speedup <= 0:
            raise ValueError("development anchor was not faster")
        if not target["locally_eligible"] or not target["dma_pareto"]["certified"]:
            raise ValueError("target is not locally eligible and DMA-Pareto certified")
        target["candidate_role"] = "p7r_support_transfer"
        target["p7r_selection"] = {
            "stratum": "support_transfer",
            "prediction_frozen_before_target_fsim_or_board": "positive_same_tile_speedup",
            "mapping_signature": list(target_signature),
            "development_anchor": {
                "workload_id": anchor_workload,
                "config_index": anchor_index,
                "measured_same_tile_speedup_percent": speedup,
            },
            "target_label_free_features": target["dma_pareto"],
        }
        selected.append(target)

    shortlist = output / "shortlist.jsonl"
    shortlist.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected), encoding="utf-8"
    )
    protocol = {
        "schema": "c3_p7r_support_transfer_confirmation_v1",
        "status": "frozen_before_target_fsim_fpga_or_latency_labels",
        "performance_label_use": "development anchors only; no W03/W05/W06 residency labels",
        "selection_rule": "exact mapping-structure support transfer plus target-local analytic/lowering and DMA-Pareto certificates",
        "mapping_signature_fields": [
            "tile_b", "tile_h", "tile_w", "tile_ci", "tile_co", "oc_nthread", "h_nthread"
        ],
        "primary_hypothesis": "each transferred candidate has positive same-tile speedup on its target workload",
        "secondary_hypothesis": "larger target total-DMA Pareto margin tends to yield larger speedup",
        "candidate_count": len(selected),
        "candidates": [
            {"workload_id": row["workload_id"], "config_index": row["config_index"],
             "candidate_id": row["candidate_id"], "selection": row["p7r_selection"]}
            for row in selected
        ],
        "gates": [
            "residency and same-tile original three-seed FSim",
            "residency and same-tile original three-seed FPGA correctness",
            "paired timing only for workloads whose complete correctness contract passes",
            "final comparison against protected TopHub incumbent",
        ],
        "sources": {name: {"path": str(path), "sha256": sha256_file(path)}
                    for name, path in paths.items()},
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (output / "STATUS.md").write_text(
        "# P7R support-transfer confirmation\n\n"
        "- Status: `frozen_before_target_fsim_fpga_or_latency_labels`\n"
        "- Target workloads: {}\n- Candidates: {}\n"
        "- Selection: exact known-positive mapping structure plus target-local DMA Pareto.\n".format(
            ", ".join(row["workload_id"] for row in selected), len(selected)
        )
    )
    hashes = {path.name: sha256_file(path) for path in output.iterdir()
              if path.is_file() and path.name != "artifact_hashes.json"}
    (output / "artifact_hashes.json").write_text(
        json.dumps({"artifacts": hashes}, indent=2) + "\n"
    )
    print(json.dumps(protocol["candidates"], indent=2))


if __name__ == "__main__":
    main()
