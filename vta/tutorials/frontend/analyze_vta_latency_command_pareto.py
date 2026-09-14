#!/usr/bin/env python3
"""Build an offline latency-versus-command-backing Pareto analysis for P7Q."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


PAGE_BYTES = 4096
EQUIVALENCE_BANDS_PERCENT = (0, 2, 5, 10)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aligned(value: int) -> int:
    return max(PAGE_BYTES, math.ceil(value / PAGE_BYTES) * PAGE_BYTES)


def command_backing(row):
    structural = row["command_signature"]["structural"]
    peaks = structural["peaks"]
    return {
        "instruction_bytes": aligned(int(peaks["insn_bytes"])),
        "uop_bytes": aligned(int(peaks["uop_bytes"])),
        "total_bytes": aligned(int(peaks["insn_bytes"])) + aligned(int(peaks["uop_bytes"])),
        "raw_instruction_peak_bytes": int(peaks["insn_bytes"]),
        "raw_uop_peak_bytes": int(peaks["uop_bytes"]),
        "submissions": int(structural["submissions"]),
    }


def pareto_front(rows):
    front = []
    for row in rows:
        dominated = any(
            other["median_ms"] <= row["median_ms"]
            and other["command_backing"]["total_bytes"] <= row["command_backing"]["total_bytes"]
            and (
                other["median_ms"] < row["median_ms"]
                or other["command_backing"]["total_bytes"] < row["command_backing"]["total_bytes"]
            )
            for other in rows
        )
        if not dominated:
            front.append(row)
    return sorted(front, key=lambda row: (row["median_ms"], row["command_backing"]["total_bytes"], row["candidate_id"]))


def analyze_workload(entries, latency, signatures):
    rows = []
    for candidate_id, median_ms in latency.items():
        if candidate_id not in signatures:
            raise ValueError(f"missing command signature: {candidate_id}")
        entry = entries[candidate_id]
        rows.append({
            "candidate_id": candidate_id,
            "config_index": entry["config_index"],
            "residence_mode": entry["residence_mode"],
            "median_ms": float(median_ms),
            "command_backing": command_backing(signatures[candidate_id]),
        })
    oracle = min(rows, key=lambda row: (row["median_ms"], row["candidate_id"]))
    bands = {}
    for band in EQUIVALENCE_BANDS_PERCENT:
        eligible = [row for row in rows if row["median_ms"] <= oracle["median_ms"] * (1.0 + band / 100.0)]
        winner = min(eligible, key=lambda row: (row["command_backing"]["total_bytes"], row["median_ms"], row["candidate_id"]))
        bands[str(band)] = {
            "eligible_count": len(eligible),
            "candidate_id": winner["candidate_id"],
            "config_index": winner["config_index"],
            "residence_mode": winner["residence_mode"],
            "median_ms": winner["median_ms"],
            "latency_over_oracle_percent": (winner["median_ms"] / oracle["median_ms"] - 1.0) * 100.0,
            "command_backing": winner["command_backing"],
        }
    return {
        "candidate_count": len(rows),
        "oracle": oracle,
        "pareto_front": pareto_front(rows),
        "equivalence_bands_percent": bands,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--command-signatures", required=True)
    parser.add_argument("--timing-summary", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("refusing to overwrite analysis")
    contract_path = Path(args.contract).resolve()
    signatures_path = Path(args.command_signatures).resolve()
    contract = json.loads(contract_path.read_text())
    signatures = {
        row["candidate_id"]: row
        for row in (json.loads(line) for line in signatures_path.read_text().splitlines() if line.strip())
    }
    timing_inputs = {}
    timing = {}
    for value in args.timing_summary:
        path = Path(value).resolve()
        summary = json.loads(path.read_text())
        workload_id = summary["workload_id"]
        timing_inputs[workload_id] = {"path": str(path), "sha256": sha256(path)}
        timing[workload_id] = {
            candidate_id: row["median_ms"]
            for candidate_id, row in summary["candidate_summaries"].items()
        }
    result = {
        "schema": "c3_p7q_latency_command_pareto_v1",
        "status": "development_only_not_prospective_confirmation",
        "capacity_model": "one live instruction queue plus one live UOP queue; each peak rounded up to 4096 bytes",
        "claim_boundary": "host command backing only; not tensor SRAM, physical fragmentation, concurrency, replay, FPS, or FPGA area",
        "inputs": {
            "contract": {"path": str(contract_path), "sha256": sha256(contract_path)},
            "command_signatures": {"path": str(signatures_path), "sha256": sha256(signatures_path)},
            "timing_summaries": timing_inputs,
        },
        "workloads": {},
    }
    for workload_id in contract["workload_order"]:
        entries = {row["candidate_id"]: row for row in contract["workloads"][workload_id]["gross_candidates"]}
        result["workloads"][workload_id] = analyze_workload(entries, timing[workload_id], signatures)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
