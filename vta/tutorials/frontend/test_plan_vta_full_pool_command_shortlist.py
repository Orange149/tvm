"""Tests for the P5c label-free full-pool command-aware shortlist."""

from __future__ import annotations

import copy

import pytest

import plan_vta_full_pool_command_shortlist as planner


def real_inputs():
    paths = planner.default_paths()
    static = planner.join_qualified_pool(
        planner.load_jsonl(paths["p4b"]),
        planner.load_jsonl(paths["p4e"]),
        planner.load_jsonl(paths["p4f"]),
        planner.load_jsonl(paths["p4g"]),
        planner.load_jsonl(paths["p4i"]),
    )
    return paths, static, planner.load_jsonl(paths["p4j"])


def test_frozen_inputs_join_exactly_197_and_preserve_incumbents():
    paths, static, p4j = real_inputs()
    for path in paths.values():
        planner.verify_run_artifacts(path.parent)
    joined = planner.join_p4j_commands(static, p4j)
    shortlist, rows = planner.build_shortlist(joined)
    assert len(static) == len(joined) == len(rows) == 197
    assert len(shortlist["workloads"]) == 10
    for workload_id, workload in shortlist["workloads"].items():
        incumbent = workload["protected_original_incumbent"]
        for budget in planner.DEFAULT_BUDGETS:
            prefix = workload["budgets"][str(budget)]["candidate_ids"]
            assert prefix[0] == incumbent
            assert len(prefix) == len(set(prefix))
        # Apart from the explicitly protected incumbent, mode exploration never
        # moves a later Pareto front ahead of an earlier one.
        fronts = [
            row["pareto_front"]
            for row in rows
            if row["workload_id"] == workload_id
            and row["candidate_id"] != incumbent
        ]
        assert fronts == sorted(fronts)


def test_ranking_does_not_change_when_forbidden_labels_are_poisoned():
    _, static, p4j = real_inputs()
    original, _ = planner.build_shortlist(planner.join_p4j_commands(static, p4j))
    poisoned = copy.deepcopy(p4j)
    for index, record in enumerate(poisoned):
        record["latency_ms"] = -10**9 + index
        record["fps"] = 10**9 - index
        record["correctness"] = "deliberately_poisoned"
        record["measurement_result"] = {"cost": -index}
    modified, _ = planner.build_shortlist(planner.join_p4j_commands(static, poisoned))
    assert modified == original


def test_feature_firewall_rejects_label_if_it_reaches_rank_payload():
    with pytest.raises(ValueError, match="forbidden rank label"):
        planner.assert_rank_payload_label_free({"metrics": {"latency_ms": 1.0}})


def test_p5b_comparison_records_32_additions_without_performance_claim():
    paths, static, p4j = real_inputs()
    shortlist, rows = planner.build_shortlist(planner.join_p4j_commands(static, p4j))
    comparison = planner.compare_to_p5b(shortlist, planner.load_json(paths["p5b"]))
    summary = planner.summarize(shortlist, rows, comparison)
    assert comparison["p5b_candidate_space"] == 165
    assert comparison["p5c_candidate_space"] == 197
    assert comparison["shared_candidates"] == 165
    assert comparison["added_candidates"] == 32
    assert comparison["removed_candidates"] == 0
    assert summary["search_efficiency_claim"] is False
    assert summary["performance_measurement"] == "not_collected"
    assert summary["g5_board_labels_required"] is True


if __name__ == "__main__":
    pytest.main([__file__])
