import copy

import pytest

from c3_selector_v4_diverse import diverse_static_order, project_static_candidate


def row(candidate_id, request_bytes, tile_h, tile_w, calls=10):
    return {
        "program_id": candidate_id,
        "performance_label": None,
        "timing_freeze_metrics": {
            "total_request_bytes": request_bytes,
            "total_request_calls": calls,
            "submissions": 3,
            "uop_load_bytes": 20,
            "instruction_bytes": 100,
        },
        "visible_features": {
            "tile_b": 1,
            "tile_h": tile_h,
            "tile_w": tile_w,
            "tile_ci": 1,
            "tile_co": 1,
            "oc_nthread": 1,
            "h_nthread": 1,
        },
    }


def test_projection_consumes_only_frozen_static_fields_and_rejects_latency():
    candidate = row("a", 100, 1, 4)
    projected = project_static_candidate(candidate)
    assert projected["candidate_id"] == "a"
    assert projected["metrics"]["total_request_bytes"] == 100
    leaked = copy.deepcopy(candidate)
    leaked["candidate_latency_ms"] = 1.25
    with pytest.raises(ValueError, match="labels are forbidden"):
        project_static_candidate(leaked)


def test_executable_program_identity_wins_over_pre_lowering_candidate_identity():
    candidate = row("program", 100, 1, 4)
    candidate["candidate_id"] = "configuration"
    assert project_static_candidate(candidate)["candidate_id"] == "program"


def test_historical_command_uop_alias_is_name_only_and_conflicts_fail_closed():
    candidate = row("a", 100, 1, 4)
    candidate["timing_freeze_metrics"]["command_uop_bytes"] = 20
    del candidate["timing_freeze_metrics"]["uop_load_bytes"]
    assert project_static_candidate(candidate)["metrics"]["uop_load_bytes"] == 20
    candidate["timing_freeze_metrics"]["uop_load_bytes"] = 21
    with pytest.raises(ValueError, match="conflicting metric aliases"):
        project_static_candidate(candidate)


def test_budget_orders_are_prefixes_of_one_frozen_order():
    rows = [row("a", 100, 1, 4), row("b", 100, 8, 4), row("c", 100, 1, 16),
            row("d", 300, 2, 2), row("e", 500, 4, 4), row("f", 700, 8, 8)]
    result = diverse_static_order(rows)
    order = result["ordered_candidate_ids"]
    assert result["budget_prefixes"]["B1"] == order[:1]
    assert result["budget_prefixes"]["B2"] == order[:2]
    assert result["budget_prefixes"]["B3"] == order[:3]
    assert result["window_size"] == 6


def test_farthest_first_preserves_tile_diversity_inside_low_movement_tier():
    rows = [
        row("anchor", 100, 1, 1, calls=1),
        row("near", 100, 1, 2, calls=2),
        row("far", 100, 16, 16, calls=3),
        row("higher_movement", 250, 32, 32, calls=1),
    ]
    result = diverse_static_order(rows)
    assert result["ordered_candidate_ids"][:3] == ["anchor", "far", "near"]
    assert result["selection_trace"][1]["minimum_tile_distance"] > 0.0


def test_invalid_or_missing_static_evidence_fails_closed():
    candidate = row("a", 100, 1, 4)
    del candidate["timing_freeze_metrics"]["instruction_bytes"]
    with pytest.raises(ValueError, match="instruction_bytes"):
        diverse_static_order([candidate])
    duplicate = row("a", 200, 2, 4)
    with pytest.raises(ValueError, match="unique"):
        diverse_static_order([row("a", 100, 1, 4), duplicate])


def test_input_order_does_not_change_result():
    rows = [row("a", 100, 1, 4), row("b", 100, 8, 4), row("c", 120, 4, 8),
            row("d", 400, 2, 2)]
    forward = diverse_static_order(rows)
    reverse = diverse_static_order(list(reversed(rows)))
    assert forward["ordered_candidate_ids"] == reverse["ordered_candidate_ids"]
    assert forward["budget_prefixes"] == reverse["budget_prefixes"]
