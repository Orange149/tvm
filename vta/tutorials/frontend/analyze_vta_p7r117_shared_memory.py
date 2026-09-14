#!/usr/bin/env python3
"""Summarize how frozen YOLO tiles change data and command shared-memory demand."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


SCHEMA = "c3_p7r117_tile_shared_memory_analysis_v1"
MODES = (
    "original",
    "input_stationary",
    "weight_stationary",
    "paper_inspired_hybrid",
)


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aligned(value, alignment=4096):
    return int(math.ceil(int(value) / alignment) * alignment)


def distribution(values):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "min": None, "median": None, "max": None}
    n = len(ordered)
    median = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2
    return {"count": n, "min": ordered[0], "median": median, "max": ordered[-1]}


def knobs(candidate):
    answer = {}
    for name, kind, value in candidate["identity"]["complete_config_entity"]["entity"]:
        answer[name] = int(value[-1] if kind == "sp" else value)
    return answer


def analyze(candidates_path, static_path, fsim_path):
    candidates = load_jsonl(candidates_path)
    static_rows = load_jsonl(static_path)
    fsim_rows = load_jsonl(fsim_path)
    candidate_by_id = {row["candidate_id"]: row for row in candidates}
    static_by_id = {row["candidate_id"]: row for row in static_rows}
    fsim_by_id = {row["candidate_id"]: row for row in fsim_rows}
    if len(candidate_by_id) != len(candidates) or len(static_by_id) != len(static_rows):
        raise ValueError("duplicate candidate identity")
    if set(static_by_id) != set(candidate_by_id):
        raise ValueError("static results do not cover the frozen candidate pool exactly")

    groups = []
    detailed = []
    for workload_id in sorted({row["workload_id"] for row in candidates}):
        for mode in MODES:
            selected = [
                row for row in candidates
                if row["workload_id"] == workload_id and row["residence_mode"] == mode
            ]
            static_ok = [static_by_id[row["candidate_id"]] for row in selected
                         if static_by_id[row["candidate_id"]]["status"] == "ok"]
            fsim_passed = [fsim_by_id[row["candidate_id"]] for row in selected
                           if row["candidate_id"] in fsim_by_id
                           and fsim_by_id[row["candidate_id"]]["status"] == "passed"]
            static_features = [row["static_feature_vector"] for row in static_ok]
            command_values = [row["command_features"]["values"] for row in fsim_passed]
            group = {
                "workload_id": workload_id,
                "residence_mode": mode,
                "gross_candidates": len(selected),
                "static_ok": len(static_ok),
                "fsim_passed": len(fsim_passed),
                "data_dma_bytes": distribution(
                    feature["load_dma_bytes"] + feature["store_dma_bytes"]
                    for feature in static_features
                ),
                "data_dma_calls": distribution(
                    feature["load_dma_calls"] + feature["store_dma_calls"]
                    for feature in static_features
                ),
                "insn_peak_bytes": distribution(
                    value["peaks"]["insn_bytes"] for value in command_values
                ),
                "uop_peak_bytes": distribution(
                    value["peaks"]["uop_bytes"] for value in command_values
                ),
                "requested_command_backing_bytes": distribution(
                    aligned(value["peaks"]["insn_bytes"])
                    + aligned(value["peaks"]["uop_bytes"])
                    for value in command_values
                ),
            }
            groups.append(group)

    for candidate_id, result in sorted(fsim_by_id.items()):
        if result["status"] != "passed":
            continue
        candidate = candidate_by_id[candidate_id]
        static = static_by_id[candidate_id]
        command = result["command_features"]["values"]
        feature = static["static_feature_vector"]
        detailed.append({
            "candidate_id": candidate_id,
            "workload_id": candidate["workload_id"],
            "family_id": candidate["family_id"],
            "residence_mode": candidate["residence_mode"],
            "knobs": knobs(candidate),
            "data_dma_bytes": feature["load_dma_bytes"] + feature["store_dma_bytes"],
            "data_dma_calls": feature["load_dma_calls"] + feature["store_dma_calls"],
            "input_dma_bytes": feature["input_dma_bytes"],
            "weight_dma_bytes": feature["weight_dma_bytes"],
            "insn_peak_bytes": command["peaks"]["insn_bytes"],
            "uop_peak_bytes": command["peaks"]["uop_bytes"],
            "requested_command_backing_bytes": (
                aligned(command["peaks"]["insn_bytes"])
                + aligned(command["peaks"]["uop_bytes"])
            ),
        })

    paired = []
    by_family = defaultdict(dict)
    for row in detailed:
        by_family[(row["workload_id"], row["family_id"])][row["residence_mode"]] = row
    for (workload_id, family_id), modes in sorted(by_family.items()):
        control = modes.get("original")
        if control is None:
            continue
        for mode, row in sorted(modes.items()):
            if mode == "original":
                continue
            paired.append({
                "workload_id": workload_id,
                "family_id": family_id,
                "residence_mode": mode,
                "data_dma_delta_bytes": row["data_dma_bytes"] - control["data_dma_bytes"],
                "input_dma_delta_bytes": row["input_dma_bytes"] - control["input_dma_bytes"],
                "weight_dma_delta_bytes": row["weight_dma_bytes"] - control["weight_dma_bytes"],
                "insn_peak_delta_bytes": row["insn_peak_bytes"] - control["insn_peak_bytes"],
                "uop_peak_delta_bytes": row["uop_peak_bytes"] - control["uop_peak_bytes"],
                "command_backing_delta_bytes": (
                    row["requested_command_backing_bytes"]
                    - control["requested_command_backing_bytes"]
                ),
            })

    failures = Counter()
    for row in static_rows:
        if row["status"] != "ok":
            failure = row.get("failure") or {}
            failures[(row["workload_id"], failure.get("subcategory", failure.get("category")))] += 1
    for row in fsim_rows:
        if row["status"] != "passed":
            failure = row.get("failure") or {}
            failures[(row["workload_id"], failure.get("phase", failure.get("category")))] += 1

    command_backings = [row["requested_command_backing_bytes"] for row in detailed]
    return {
        "schema": SCHEMA,
        "scope": "P7R117 36-point label-free local pilot; no board latency/FPS",
        "sources": {
            "candidates": {"path": str(Path(candidates_path).resolve()), "sha256": sha256(candidates_path)},
            "static": {"path": str(Path(static_path).resolve()), "sha256": sha256(static_path)},
            "fsim": {"path": str(Path(fsim_path).resolve()), "sha256": sha256(fsim_path)},
        },
        "counts": {
            "gross_candidates": len(candidates),
            "static_ok": sum(row["status"] == "ok" for row in static_rows),
            "fsim_attempted": len(fsim_rows),
            "fsim_passed": len(detailed),
        },
        "failure_counts": [
            {"workload_id": key[0], "reason": key[1], "count": value}
            for key, value in sorted(failures.items())
        ],
        "groups": groups,
        "passed_candidates": detailed,
        "same_tile_pairs": paired,
        "global_command_backing": distribution(command_backings),
        "interpretation": [
            "tile and residence mode jointly change u-dma-buf tensor traffic and command backing",
            "per-candidate aligned command backing is not a board-wide constant",
            "compiler/FSim failures are evidence about legal shared-memory mappings, not performance labels",
            "weight_stationary mode 2 is feasibility-only; its zero weight-DMA delta is not weight reuse",
        ],
    }


def format_range(item):
    if item["count"] == 0:
        return "--"
    return "{:.0f}/{:.0f}/{:.0f}".format(item["min"], item["median"], item["max"])


def markdown(result):
    lines = [
        "# P7R117 tile—shared-memory analysis",
        "",
        "> Local no-latency pilot. Values are min/median/max over candidates that passed the corresponding gate.",
        "",
        "| workload | mode | static | FSim | data bytes | DMA calls | insn peak B | uop peak B | aligned command backing B |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["groups"]:
        lines.append(
            "| {workload_id} | {residence_mode} | {static_ok}/{gross_candidates} | {fsim_passed} | {data} | {calls} | {insn} | {uop} | {backing} |".format(
                data=format_range(row["data_dma_bytes"]),
                calls=format_range(row["data_dma_calls"]),
                insn=format_range(row["insn_peak_bytes"]),
                uop=format_range(row["uop_peak_bytes"]),
                backing=format_range(row["requested_command_backing_bytes"]),
                **row,
            )
        )
    overall = result["global_command_backing"]
    lines += [
        "",
        "## Result",
        "",
        "- Gross/static/FSim-pass: {}/{}/{}.".format(
            result["counts"]["gross_candidates"],
            result["counts"]["static_ok"],
            result["counts"]["fsim_passed"],
        ),
        "- Per-candidate 4-KiB-aligned instruction+UOP backing spans {:.0f}--{:.0f} B (median {:.0f} B).".format(
            overall["min"], overall["max"], overall["median"]
        ),
        "- The old 8 KiB W05 result is therefore an exact-allowlist result, not a universal VTA constant.",
        "- Mode-2 weight-stationary has no realized weight-DMA reduction and stays a negative control. The next formal pool must use the explicit barrier weight-residency mechanism.",
        "- No latency or FPGA claim is made from this analysis.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--static-results", type=Path, required=True)
    parser.add_argument("--fsim-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite {}".format(args.output_dir))
    result = analyze(args.candidates, args.static_results, args.fsim_results)
    args.output_dir.mkdir(parents=True)
    analysis_path = args.output_dir / "analysis.json"
    report_path = args.output_dir / "RESULTS.md"
    analysis_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    report_path.write_text(markdown(result))
    hashes = {path.name: sha256(path) for path in (analysis_path, report_path)}
    (args.output_dir / "artifact_hashes.json").write_text(
        json.dumps({"artifacts": hashes}, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
