"""Tests for the local-only residency candidate pool generator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import vta

import vta.top.vta_conv2d_residency  # pylint: disable=unused-import
from generate_vta_residency_candidates import (
    EXPERIMENTAL_MODES,
    build_config_lookup,
    create_residency_task,
    distribution,
    force_experimental_knobs,
    map_selected_candidates,
    semantic_config_key,
)


BASE = Path(__file__).resolve().parent / "report_out" / "stage_tile_cotuning"
SELECTED_PATH = BASE / "iteration6_safe_overlay_final" / "selected_tasks.json"


def test_force_knobs_preserves_full_tile_entity_and_input():
    source = {
        "index": 91,
        "code_hash": None,
        "entity": [
            ["tile_h", "sp", [-1, 7]],
            ["tile_w", "sp", [-1, 14]],
            ["oc_nthread", "ot", 2],
            ["h_nthread", "ot", 2],
        ],
    }
    forced = force_experimental_knobs(source)
    assert source["index"] == 91
    assert source["entity"][-1][-1] == 2
    assert "index" not in forced
    assert forced["entity"][:2] == source["entity"][:2]
    assert forced["entity"][-2][-1] == 1
    assert forced["entity"][-1][-1] == 1


def test_semantic_key_ignores_only_debug_index():
    left = {"index": 1, "code_hash": None, "entity": [["tile_h", "sp", [-1, 7]]]}
    right = {"index": 99, "code_hash": None, "entity": [["tile_h", "sp", [-1, 7]]]}
    changed = {"index": 1, "code_hash": None, "entity": [["tile_h", "sp", [-1, 14]]]}
    assert semantic_config_key(left) == semantic_config_key(right)
    assert semantic_config_key(left) != semantic_config_key(changed)


def test_distribution_is_deterministic_and_counts_signs():
    result = distribution([-0.5, 0.0, 0.25, 1.0])
    assert result["count"] == 4
    assert result["median"] == pytest.approx(0.125)
    assert (result["positive"], result["zero"], result["negative"]) == (2, 1, 1)


def test_all_iteration6_candidates_exact_map_and_deduplicate():
    selected = json.loads(SELECTED_PATH.read_text())
    env = vta.get_env()
    totals = {"source": 0, "unique": 0, "deduplicated": 0}
    for item in selected:
        task = create_residency_task(item["workload"], 0, env)
        mapping = map_selected_candidates(item, build_config_lookup(task.config_space))
        assert mapping["source_count"] == 8
        assert mapping["unique_count"] > 0
        assert mapping["source_count"] == mapping["unique_count"] + mapping["deduplicated_count"]
        for record in mapping["records"]:
            entity = {row[0]: row[2] for row in record["complete_config_entity"]["entity"]}
            assert entity["oc_nthread"] == 1
            assert entity["h_nthread"] == 1
            assert record["sources"]
            expected_key = semantic_config_key(record["complete_config_entity"])
            for mode_number in range(1, 4):
                mode_task = create_residency_task(item["workload"], mode_number, env)
                actual = mode_task.config_space.get(record["mapped_config_index"]).to_json_dict()
                assert semantic_config_key(actual) == expected_key
        totals["source"] += mapping["source_count"]
        totals["unique"] += mapping["unique_count"]
        totals["deduplicated"] += mapping["deduplicated_count"]
    assert totals["source"] == 80
    assert totals["source"] == totals["unique"] + totals["deduplicated"]
    assert len(EXPERIMENTAL_MODES) == 3


if __name__ == "__main__":
    pytest.main([__file__])
