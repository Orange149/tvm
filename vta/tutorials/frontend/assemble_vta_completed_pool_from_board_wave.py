#!/usr/bin/env python3
"""Assemble the canonical completed candidate pool from a full board wave."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import prepare_vta_p7r132_y00_search_confirmation as prepare
import run_vta_p7r132_y00_search_confirmation as board


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def verify(directory):
    directory = Path(directory)
    ledger = read_json(directory / "artifact_hashes.json")
    for name, expected in ledger["artifacts"].items():
        if board.sha256(directory / name) != expected:
            raise ValueError("artifact hash mismatch: " + str(directory / name))
    return board.sha256(directory / "artifact_hashes.json")


def run(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    local = Path(args.local_dir)
    board_dir = Path(args.board_dir)
    local_ledger = verify(local)
    board_ledger = verify(board_dir)
    candidates = read_jsonl(local / "candidates_v2.jsonl")
    static_rows = read_jsonl(local / "static_results.jsonl")
    fsim_rows = read_jsonl(local / "fsim_results.jsonl")
    prospective, eligible = prepare.build_pool(
        candidates, static_rows, fsim_rows, args.workload_id
    )
    correctness = read_jsonl(board_dir / "correctness.jsonl")
    timing_rows = read_jsonl(board_dir / "timing.jsonl")
    timing_summary = read_json(board_dir / "timing_summary.json")
    if set(eligible) != {row["candidate_id"] for row in correctness}:
        raise ValueError("board wave is not the complete FSim-pass candidate set")
    static = {row["candidate_id"]: row for row in static_rows}
    fsim = {row["candidate_id"]: row for row in fsim_rows}
    qualification = {
        cid: {
            "lower_wall_ms": float(static[cid]["diagnostic_wall_seconds"]) * 1000,
            "fsim_wall_ms": float(fsim[cid]["diagnostic_wall_seconds"]) * 1000,
        }
        for cid in eligible
    }
    certificates = read_json(board_dir / "cross_certificates.json")
    compile_walls = {cid: float(certificates[cid]["cross_compile_wall_ms"])
                     for cid in eligible}
    completed = board.completed_pool(
        prospective, correctness, timing_summary, qualification, compile_walls, timing_rows
    )
    completed["claim_status"] = "oracle_completion_after_prospective_first_wave"
    completed["source_bindings"] = {
        "local": {"path": str(local.resolve()), "ledger_sha256": local_ledger},
        "board": {"path": str(board_dir.resolve()), "ledger_sha256": board_ledger},
    }
    output.mkdir(parents=True)
    board.write_json(output / "completed_pool.json", completed)
    board.write_json(output / "summary.json", {
        "status": "canonical_completed_pool_assembled",
        "workload_id": args.workload_id,
        "gross_candidates": len(candidates),
        "static_pass": sum(row["status"] == "ok" for row in static_rows),
        "fsim_pass": len(eligible),
        "fpga_correct": sum(row["status"] == "passed" for row in correctness),
        "pool_oracle_candidate_id": timing_summary["pool_oracle_candidate_id"],
        "pool_oracle_latency_ms": timing_summary["pool_oracle_latency_ms"],
        "claim_boundary": "pool completion only; first-wave prospective evidence stays separate",
    })
    (output / "command.txt").write_text(
        " ".join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]) + "\n",
        encoding="utf-8",
    )
    board.finalize(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--board-dir", type=Path, required=True)
    parser.add_argument("--workload-id", default="Y05")
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
