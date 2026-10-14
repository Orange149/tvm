#!/usr/bin/env python3
"""Test mode-11 successor correctness after normal VTA programs.

Each case uses a fresh RPC process and bitstream reload.  The same frozen
matrix can be run independently for every execution task in the correctness
pool.  No timing API is called.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_vta_c3_module_transition_diagnostic as diagnostic


SCHEMA = "c3_transition_safe_successor_matrix_v1"
SEEDS = diagnostic.SEEDS
MODES = (
    "original",
    "input_stationary",
    "weight_resident_barrier",
    "input_weight_resident_barrier",
    "input_weight_resident_transition_safe",
)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def task_families(board_pool_dir, workload_id):
    rows = [row for row in read_jsonl(Path(board_pool_dir) / "candidates.jsonl")
            if row["workload_id"] == workload_id]
    families = {}
    for row in rows:
        family = families.setdefault(row["family_id"], {})
        mode = row["residence_mode"]
        if mode in family:
            raise ValueError("duplicate family mode")
        family[mode] = row["candidate_id"]
    if len(families) != 3 or any(tuple(sorted(value)) != tuple(sorted(MODES))
                                 for value in families.values()):
        raise ValueError("workload does not contain exactly three complete five-mode families")
    return [families[name] for name in sorted(families)]


def control(case_id, successor):
    return {
        "case_id": case_id,
        "preload": [successor],
        "calls": [(successor, seed) for seed in SEEDS],
        "successor_ids": [successor],
        "question": "Is mode 11 correct for three seeds in a fresh session?",
    }


def pair(case_id, predecessor, successor, question):
    return {
        "case_id": case_id,
        "preload": [predecessor, successor],
        "calls": [(predecessor, 0)] + [(successor, seed) for seed in SEEDS],
        "successor_ids": [successor],
        "question": question,
    }


def build_cases(families):
    result = []
    for index, family in enumerate(families):
        successor = family["input_weight_resident_transition_safe"]
        result.append(control("f{}_mode11_control".format(index), successor))
        result.append(pair(
            "f{}_input_then_mode11".format(index), family["input_stationary"], successor,
            "Can mode 11 follow a normal input-stationary program at the same Tile?",
        ))
        result.append(pair(
            "f{}_original_then_mode11".format(index), family["original"], successor,
            "Can mode 11 follow a normal original program at the same Tile?",
        ))
    # Cross-Tile transition and a longer mixed prefix prevent the result from
    # depending only on same-family address layouts.
    result.append(pair(
        "f0_input_then_f1_mode11",
        families[0]["input_stationary"],
        families[1]["input_weight_resident_transition_safe"],
        "Can mode 11 drain a normal predecessor generated at another Tile?",
    ))
    mixed_successor = families[0]["input_weight_resident_transition_safe"]
    mixed_prefix = [
        families[0]["input_stationary"],
        families[1]["original"],
        families[2]["input_stationary"],
    ]
    result.append({
        "case_id": "mixed_three_program_prefix_then_f0_mode11",
        "preload": mixed_prefix + [mixed_successor],
        "calls": [(candidate_id, 0) for candidate_id in mixed_prefix]
        + [(mixed_successor, seed) for seed in SEEDS],
        "successor_ids": [mixed_successor],
        "question": "Can one entry drain recover after a longer mixed normal-program prefix?",
    })
    return result


def classify(results):
    by_case = {}
    for row in results:
        # Every frozen case ends with exactly the three seeded mode-11 calls.
        successor_calls = row["calls"][-len(SEEDS):]
        by_case[row["case_id"]] = (
            len(successor_calls) == len(SEEDS)
            and len({call["candidate_id"] for call in successor_calls}) == 1
            and [call["seed"] for call in successor_calls] == list(SEEDS)
            and all(call["correct"] for call in successor_calls)
        )
    return {
        "successor_correct_by_case": by_case,
        "all_mode11_successors_correct": all(by_case.values()),
        "interpretation_rule": (
            "all cases passing supports transition safety for the exact tested programs, Tiles, "
            "workload, runtime, and bitstream; it does not prove all possible VTA programs"
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-pool-dir", type=Path, required=True)
    parser.add_argument("--cross-compile-dir", type=Path, required=True)
    parser.add_argument("--runtime-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workload-id", required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9400)
    parser.add_argument("--session-timeout", type=int, default=7200)
    parser.add_argument("--known-hosts", default="/home/orange/.ssh/known_hosts")
    return parser.parse_args()


def run(args):
    families = task_families(args.board_pool_dir, args.workload_id)
    cases = build_cases(families)
    # The shared runner validates these four IDs as a same-task frozen subset
    # and uses TARGET to allocate the task's reusable buffers.
    diagnostic.A1 = families[0]["input_stationary"]
    diagnostic.A2 = families[1]["original"]
    diagnostic.A3 = families[2]["input_stationary"]
    diagnostic.TARGET = families[0]["input_weight_resident_transition_safe"]
    diagnostic.SCHEMA = SCHEMA
    diagnostic.frozen_cases = lambda _all_ids: cases
    diagnostic.classify = classify
    diagnostic.__file__ = __file__
    diagnostic.run(args)

    output = Path(args.output_dir).resolve()
    records = read_jsonl(output / "cases.jsonl")
    summary = diagnostic.read(output / "summary.json")
    summary.pop("target_correct_cases", None)
    summary.pop("target_wrong_cases", None)
    successor_ok = summary["classification"]["successor_correct_by_case"]
    summary["workload_id"] = args.workload_id
    summary["successor_correct_cases"] = sum(successor_ok.values())
    summary["successor_wrong_cases"] = sum(not value for value in successor_ok.values())
    summary["all_outputs_correct_cases"] = sum(row["all_outputs_correct"] for row in records)
    diagnostic.write(output / "summary.json", summary)
    diagnostic.seal(output)


if __name__ == "__main__":
    run(parse_args())
