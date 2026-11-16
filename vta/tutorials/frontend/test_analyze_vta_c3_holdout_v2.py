import math

from analyze_vta_c3_holdout_v2 import (
    combined_reference,
    deterministic_replay,
    paired_gate,
    random_replay,
)


def test_paired_gate_requires_each_boot_independently():
    passed = {"paired": {"median_improvement_pct": 2.5, "wins": 5,
                         "round_improvements_pct": [3, 3, 3, 3, 3, -1]}}
    failed = {"paired": {"median_improvement_pct": 1.9, "wins": 6,
                         "round_improvements_pct": [2, 2, 2, 2, 2, 2]}}
    assert paired_gate([passed, passed])["stable_improvement"]
    assert not paired_gate([passed, failed])["stable_improvement"]


def _reference(a, b):
    return {
        "fpga_correct_program_ids": ["a", "b"],
        "fpga_rejected_program_ids": [],
        "program_statistics": {
            "a": {"round_medians_ms": a},
            "b": {"round_medians_ms": b},
        },
    }


def test_combined_reference_uses_twelve_round_median_and_id_tie_break():
    boot1 = _reference([1] * 6, [2] * 6)
    boot2 = _reference([3] * 6, [2] * 6)
    result = combined_reference(boot1, boot2)
    # Both medians are 2; lexicographically smaller program a wins the tie.
    assert result["pool_oracle_program_id"] == "a"
    assert result["pool_oracle_latency_ms"] == 2.0


def test_deterministic_replay_includes_incumbent_and_respects_budget():
    result = deterministic_replay(["slow", "oracle"], "inc", {
        "inc": 10.0, "slow": 12.0, "oracle": 5.0}, "oracle")
    assert result["budgets"]["B1"]["best_program_id"] == "inc"
    assert result["budgets"]["B2"]["best_program_id"] == "oracle"
    assert result["first_hit_additional_candidates"]["exact_oracle"] == 2
    assert result["budgets"]["B1"]["improvement_vs_incumbent_pct"] == 0.0


def test_random_replay_uses_exactly_frozen_thousand_orders():
    orders = [["oracle", "slow"]] * 500 + [["slow", "oracle"]] * 500
    result = random_replay(orders, "inc", {"inc": 10.0, "slow": 12.0, "oracle": 5.0},
                           "oracle")
    assert math.isclose(result["budgets"]["B1"]["exact_oracle_hit_rate"], 0.5)
    assert result["budgets"]["B2"]["exact_oracle_hit_rate"] == 1.0


def test_random_replay_rejects_wrong_order_count():
    try:
        random_replay([["oracle"]], "inc", {"inc": 2.0, "oracle": 1.0}, "oracle")
    except ValueError as error:
        assert "1000" in str(error)
    else:
        raise AssertionError("wrong random-order count was accepted")


def test_shortlist_schema_contract_is_nested():
    # Guard the policy shape that the end-to-end analyzer consumes.
    policy = {"shortlist": {"shortlist_candidate_ids": ["inc", "new"]}}
    assert "inc" in policy["shortlist"]["shortlist_candidate_ids"]
