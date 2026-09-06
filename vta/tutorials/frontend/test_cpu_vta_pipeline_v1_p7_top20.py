#!/usr/bin/env python3
"""Local tests for the P7 natural Top-20 board runner."""

import json

import run_cpu_vta_pipeline_v1_p7_top20 as p7


def test_parse_ranks_accepts_mixed_ranges():
    assert p7.parse_ranks("1-3,7,9-10") == {1, 2, 3, 7, 9, 10}


def test_topology_key_excludes_runtime_threads():
    base = {
        "scheme_cfg": [
            {"name": "stage0", "device": "cpu", "unit_names": ["a"]},
            {"name": "stage1", "device": "vta", "unit_names": ["b"]},
        ],
        "stage_runtime_threads": [1, 1],
    }
    variant = dict(base, stage_runtime_threads=[4, 1])
    assert p7.topology_key(base) == p7.topology_key(variant)


def test_aggregate_preserves_static_order_and_adds_measured_rank(tmp_path):
    rows = [
        {
            "natural_static_rank": 1,
            "candidate_id": "a",
            "measurement": {"pipeline_fps": 9.0, "pipeline_cycle_ms": 111.0},
            "prediction": {"raw_ii_ms": 80.0, "fixed_residual_ii_ms": 114.0},
            "correctness": {"independent_reference_passed": True},
        },
        {
            "natural_static_rank": 2,
            "candidate_id": "b",
            "measurement": {"pipeline_fps": 10.0, "pipeline_cycle_ms": 100.0},
            "prediction": {"raw_ii_ms": 70.0, "fixed_residual_ii_ms": 104.0},
            "correctness": {"independent_reference_passed": True},
        },
    ]
    p7.write_aggregate(tmp_path, rows)
    payload = json.loads((tmp_path / "top20_board_summary.json").read_text(encoding="utf-8"))
    assert [row["candidate_id"] for row in payload["rows"]] == ["a", "b"]
    assert [row["measured_rank_within_top20"] for row in payload["rows"]] == [2, 1]
