#!/usr/bin/env python3
"""Qualify successor modes after the minimal A3 execution prefix.

All programs belong to the same frozen YOLO-576 execution task.  Each case is
correctness-only and independently reloads the bitstream through the shared
transition-diagnostic protocol.
"""

from __future__ import annotations

import json
from pathlib import Path

import run_vta_c3_module_transition_diagnostic as diagnostic


diagnostic.SCHEMA = "c3_transition_successor_matrix_v1"

A3 = diagnostic.A3
B1 = diagnostic.TARGET
B2_SAME_TILE = "3acb5b1b7b9cb5fc03c404781cec062f6d4c941e07673f35d8d2775c071d64d2"
ORIGINAL_SAME_TILE = "f9d0c3343e7ee3978edf6b572ac35d83cb49ceaa9cb5b582dd6865f327888aec"
W_BARRIER_SAME_TILE = "42f0014e76aba06a407adff56a58f5f67f1d2e11dda8046b1385323e6b51e9cd"
IW_BARRIER_SAME_TILE = "885cdc561b6ab3a18333e3aeb42acd312c1a4109191087d0df3282f04db46898"
INPUT_TILE_8_1_1 = "201c24e53737f194b7c128cbb042da066a1b63d333fe86e530771bc68934578c"
INPUT_TILE_16_8_4 = "0f38c2773974d4f1b925217750b5de643dcfac7690119cb91a737d5b0312acef"
ORIGINAL_TILE_16_8_4 = "16dc76f9af68894a52e66124680f93983277c7338a46213281e06e16291e2fa5"


def pair(case_id, predecessor, successor, question):
    return {
        "case_id": case_id,
        "preload": [predecessor, successor],
        "calls": [(predecessor, 0), (successor, 0)],
        "question": question,
    }


def control(case_id, candidate_id):
    return {
        "case_id": case_id,
        "preload": [candidate_id],
        "calls": [(candidate_id, 0)],
        "question": "Fresh-session successor control.",
    }


def cases(_all_ids):
    return [
        control("b1_control", B1),
        control("b2_same_tile_control", B2_SAME_TILE),
        pair("a3_then_a3", A3, A3, "Can input-stationary follow itself?"),
        pair("a3_then_original_same_tile", A3, ORIGINAL_SAME_TILE,
             "Can the normal original schedule follow A3 at the same Tile?"),
        pair("a3_then_w_barrier_same_tile", A3, W_BARRIER_SAME_TILE,
             "Can weight-resident barrier mode follow A3 at the same Tile?"),
        pair("a3_then_iw_barrier_same_tile", A3, IW_BARRIER_SAME_TILE,
             "Can input+weight barrier mode follow A3 at the same Tile?"),
        pair("a3_then_b2_same_tile", A3, B2_SAME_TILE,
             "Can force-serial mode10 follow A3 without changing Tile?"),
        pair("a3_then_b1_different_tile", A3, B1,
             "Positive control: A3 followed by the known failing mode10 Tile."),
        pair("input_tile_8_1_1_then_b1", INPUT_TILE_8_1_1, B1,
             "Does another input-stationary Tile corrupt B1?"),
        pair("input_tile_16_8_4_then_b1", INPUT_TILE_16_8_4, B1,
             "Does a third input-stationary Tile corrupt B1?"),
        pair("original_same_tile_then_b1", ORIGINAL_SAME_TILE, B1,
             "Does the original schedule at A3's Tile corrupt B1?"),
        pair("original_tile_16_8_4_then_b1", ORIGINAL_TILE_16_8_4, B1,
             "Does an original schedule at another Tile corrupt B1?"),
        pair("a3_then_a1_barrier", A3, diagnostic.A1,
             "Can a non-force-serial resident barrier successor follow A3?"),
        pair("a3_then_a2_barrier", A3, diagnostic.A2,
             "Can a second non-force-serial resident barrier successor follow A3?"),
    ]


def classify(results):
    return {
        "all_outputs_correct_by_case": {
            row["case_id"]: row["all_outputs_correct"] for row in results
        },
        "successor_correct_by_case": {
            row["case_id"]: row["calls"][-1]["correct"] for row in results
        },
        "interpretation_rule": (
            "Controls establish standalone correctness; pair failures are scoped to the exact "
            "predecessor-successor transition and do not alone establish a hardware bottleneck"
        ),
    }


if __name__ == "__main__":
    diagnostic.frozen_cases = cases
    diagnostic.classify = classify
    diagnostic.__file__ = __file__
    args = diagnostic.parse_args()
    diagnostic.run(args)
    # The shared runner's legacy target counters refer specifically to B1.
    # Replace them with explicit successor counters for this heterogeneous
    # matrix, then reseal the completed artifact.
    output = Path(args.output_dir).resolve()
    records = [
        json.loads(line)
        for line in (output / "cases.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    summary = diagnostic.read(output / "summary.json")
    summary.pop("target_correct_cases", None)
    summary.pop("target_wrong_cases", None)
    summary["successor_correct_cases"] = sum(row["calls"][-1]["correct"] for row in records)
    summary["successor_wrong_cases"] = sum(not row["calls"][-1]["correct"] for row in records)
    diagnostic.write(output / "summary.json", summary)
    diagnostic.seal(output)
