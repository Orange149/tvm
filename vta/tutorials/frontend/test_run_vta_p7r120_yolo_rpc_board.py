"""Pure tests for the P7R120 RPC-only result policy."""

from run_vta_p7r120_yolo_rpc_board import promotion_decision, same_tile_comparisons


def _canary(value=10.0):
    return [
        {"round": round_index, "bracket": bracket, "latency_ms": value}
        for round_index in range(7)
        for bracket in ("before", "after")
    ]


def test_incumbent_is_retained_inside_two_percent_guardband():
    stats = {
        "candidate": {
            "family_id": "Y02B00",
            "public_mode": "input_stationary",
            "values_ms": [9.9] * 7,
            "median_ms": 9.9,
        }
    }
    result = promotion_decision(stats, _canary())
    assert result["fallback_retained"] is True
    assert result["deployment_choice"] == "sealed_tophub_canary"


def test_candidate_requires_both_two_percent_and_five_round_wins():
    stats = {
        "winner": {
            "family_id": "Y02B00",
            "public_mode": "weight_resident_barrier",
            "values_ms": [9.0] * 7,
            "median_ms": 9.0,
        }
    }
    result = promotion_decision(stats, _canary())
    assert result["fallback_retained"] is False
    assert result["deployment_choice"] == "winner"
    assert result["candidate_stats"]["winner"]["round_midpoint_wins"] == 7


def test_same_tile_comparison_never_calls_policy_a_hybrid():
    stats = {
        "o": {"family_id": "Y02B00", "public_mode": "original", "median_ms": 10.0},
        "i": {"family_id": "Y02B00", "public_mode": "input_stationary", "median_ms": 9.0},
        "w": {"family_id": "Y02B00", "public_mode": "weight_resident_barrier", "median_ms": 8.0},
    }
    result = same_tile_comparisons(stats)
    assert set(result["Y02B00"]) == {"input_stationary", "weight_resident_barrier"}
    assert result["Y02B00"]["weight_resident_barrier"]["same_tile_improvement_fraction"] == 0.2
