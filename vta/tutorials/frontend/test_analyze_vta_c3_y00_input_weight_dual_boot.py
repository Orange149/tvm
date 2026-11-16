"""Regression checks for the Y00 input-plus-weight dual-boot classification."""

import analyze_vta_c3_y00_input_weight_dual_boot as analysis


def _observation(gain, wins, losses):
    return {
        "paired_median_gain_pct": gain,
        "wins": wins,
        "losses": losses,
    }


def test_dual_boot_stable_improvement_requires_both_boots():
    rows = [_observation(2.28, 6, 0), _observation(2.12, 6, 0)]
    assert analysis._classification(rows, True) == "stable_improvement"
    assert analysis._classification(rows[:1], True) == "awaiting_two_distinct_boots"


def test_dual_boot_mixed_direction_does_not_pass():
    rows = [_observation(2.28, 6, 0), _observation(1.99, 6, 0)]
    assert analysis._classification(rows, True) == "no_stable_direction_threshold"
    assert analysis._classification(rows, False) == "control_not_matched"


def test_dual_boot_stable_regression_requires_both_boots():
    rows = [_observation(-2.4, 0, 6), _observation(-3.1, 0, 6)]
    assert analysis._classification(rows, True) == "stable_regression"
