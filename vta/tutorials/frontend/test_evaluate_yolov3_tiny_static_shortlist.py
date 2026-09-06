#!/usr/bin/env python3
"""Tests for the host-only YOLOv3-tiny V1 shortlist audit."""

from evaluate_yolov3_tiny_static_shortlist import (
    build_minimal_yolo_anchor_profile,
    build_transferred_profile,
    enumerate_topologies,
    rank_candidates,
    topology_id,
)


def test_yolo_branch_aware_search_space_is_stable():
    topologies = enumerate_topologies()
    assert len(topologies) == 684
    assert sum(4 ** (len(item) + 1) for item in topologies) == 105696
    assert len({topology_id(item) for item in topologies}) == len(topologies)


def test_yolo_static_ranking_is_label_free_and_diverse():
    transferred = build_transferred_profile()
    profile = build_minimal_yolo_anchor_profile(transferred)
    assert profile["target_calibration"]["configuration_count"] == 6
    assert not profile["target_calibration"]["pipeline_throughput_fields_consumed"]
    raw, diverse, topology_count, execution_count = rank_candidates(profile, top_k=20)
    assert topology_count == 684
    assert execution_count == 105696
    assert len(raw) == len(diverse) == 20
    assert len({row["topology_id"] for row in diverse}) == 20
    assert all(row["predicted_ii_ms"] > 0 for row in raw + diverse)
    assert all(len(row["cpu_threads"]) == row["island_count"] + 1 for row in diverse)
