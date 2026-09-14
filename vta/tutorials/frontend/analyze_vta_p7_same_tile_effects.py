#!/usr/bin/env python3
"""Exploratory same-ConfigEntity residency effects from P7Q FPGA labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def knob_key(entry):
    values = []
    for name, kind, value in entry["complete_config_entity"]["entity"]:
        values.append((name, tuple(value) if kind == "sp" else value))
    return tuple(values)


def summarize(pairs):
    improvements = [row["improvement_percent"] for row in pairs]
    if not improvements:
        return {"pair_count": 0}
    return {
        "pair_count": len(pairs),
        "median_improvement_percent": statistics.median(improvements),
        "min_improvement_percent": min(improvements),
        "max_improvement_percent": max(improvements),
        "positive_count": sum(value > 0.0 for value in improvements),
        "at_least_2_percent_count": sum(value >= 2.0 for value in improvements),
        "at_least_5_percent_count": sum(value >= 5.0 for value in improvements),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--timing-summary", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("refusing to overwrite analysis")

    contract_path = Path(args.contract).resolve()
    contract = json.loads(contract_path.read_text())
    summaries = {}
    summary_inputs = {}
    for value in args.timing_summary:
        path = Path(value).resolve()
        summary = json.loads(path.read_text())
        workload_id = summary["workload_id"]
        summaries[workload_id] = summary
        summary_inputs[workload_id] = {"path": str(path), "sha256": sha256(path)}

    all_pairs = []
    per_workload = {}
    for workload_id in contract["workload_order"]:
        workload = contract["workloads"][workload_id]
        latency = {
            candidate_id: float(row["median_ms"])
            for candidate_id, row in summaries[workload_id]["candidate_summaries"].items()
        }
        originals = {
            knob_key(entry): entry
            for entry in workload["gross_candidates"]
            if entry["residence_mode"] == "original"
        }
        workload_pairs = []
        for entry in workload["gross_candidates"]:
            candidate_id = entry["candidate_id"]
            if entry["residence_mode"] == "original" or candidate_id not in latency:
                continue
            control = originals.get(knob_key(entry))
            if control is None or control["candidate_id"] not in latency:
                continue
            control_ms = latency[control["candidate_id"]]
            mechanism_ms = latency[candidate_id]
            pair = {
                "workload_id": workload_id,
                "residence_mode": entry["residence_mode"],
                "mechanism_candidate_id": candidate_id,
                "control_candidate_id": control["candidate_id"],
                "mechanism_config_index": entry["config_index"],
                "control_config_index": control["config_index"],
                "control_median_ms": control_ms,
                "mechanism_median_ms": mechanism_ms,
                "improvement_percent": (control_ms - mechanism_ms) / control_ms * 100.0,
            }
            workload_pairs.append(pair)
            all_pairs.append(pair)
        per_workload[workload_id] = summarize(workload_pairs)

    modes = sorted({row["residence_mode"] for row in all_pairs})
    per_mode = {
        mode: summarize([row for row in all_pairs if row["residence_mode"] == mode])
        for mode in modes
    }
    result = {
        "schema": "c3_p7q_same_tile_effects_v1",
        "analysis_status": "post_hoc_secondary; not the preregistered P7Q primary search endpoint",
        "contract": {"path": str(contract_path), "sha256": sha256(contract_path)},
        "timing_summaries": summary_inputs,
        "comparison": "same workload and identical seven ConfigEntity knob values; median-of-five FPGA latency",
        "overall": summarize(all_pairs),
        "per_workload": per_workload,
        "per_mode": per_mode,
        "pairs": all_pairs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
