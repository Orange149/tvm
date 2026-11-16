"""Regression tests for controlled C1 memory-feature replay."""

import pytest

import analyze_vta_c1_memory_features as study


def candidate(name, score, nbytes, calls):
    row = {field: 0 for field in study.unified.UNIFIED_DIMENSIONS}
    row.update(candidate_id=name, topology_id=name, m1_score_ms=score,
               vta_load_inp_bytes=nbytes, vta_load_inp_calls=calls,
               vta_islands=1, unified_feature_exact=True,
               realization={"realized_policy": "ordinary_copy"})
    return row


def test_bytes_and_request_ablations_preserve_first_and_band():
    rows = [candidate("a", 100, 1000, 1000), candidate("b", 100.5, 10, 100),
            candidate("c", 101, 100, 1), candidate("d", 104, 0, 0)]
    orders = study.freeze_orders(rows, 4)
    get = lambda name: [r["candidate_id"] for r in orders[name]["selected_rows"]]
    assert get("no_memory") == ["a", "b", "c", "d"]
    assert get("bytes") == ["a", "b", "c", "d"]
    assert get("requests_only") == ["a", "c", "b", "d"]
    assert all(get(name)[0] == "a" and get(name)[-1] == "d" for name in study.GROUPS)
    assert "vta_load_inp_calls" not in orders["full_no_requests"]["dimensions"]
    assert "framework_materialization_bytes" not in orders["full_no_boundary"]["dimensions"]


def test_unknown_compiler_features_are_not_zero_filled():
    row = candidate("a", 100, 10, 1)
    row["unified_feature_exact"] = False
    with pytest.raises(ValueError, match="unknown is not zero"):
        study.freeze_orders([row])


@pytest.mark.parametrize("value", [float("nan"), -1])
def test_invalid_memory_features_are_rejected(value):
    row = candidate("a", 100, 10, 1)
    row["vta_load_inp_calls"] = value
    with pytest.raises(ValueError, match="invalid memory feature"):
        study.freeze_orders([row])


def test_missing_label_does_not_shorten_or_improve_the_budget_prefix():
    result = study.prefix_metrics(["a", "missing", "b"], {"a": 10, "b": 8}, 8)
    assert result["status"] == "missing_exact_labels"
    assert result["missing"] == ["missing"]
    assert result["curves"] is None


def test_exact_matched_first_random_uses_permutations_not_extra_hardware_tasks():
    result = study.exact_fixed_first_random(
        ["a", "b", "c", "d", "e"], "a", {"a": 10, "b": 10, "c": 10, "d": 10, "e": 8}, 8
    )
    assert result["permutations"] == 24
    assert result["mean_first_hit"] == 3.5
    assert result["first_hit_distribution"] == {"2": 6, "3": 6, "4": 6, "5": 6}


def test_prefix_metrics_retains_early_failure_and_uses_the_same_oracle():
    result = study.prefix_metrics(["a", "b", "c"], {"a": 10, "b": 9, "c": 8}, 8)
    assert result["first_oracle_plus_2pct_index"] == 3
    assert [r["regret"] for r in result["curves"]] == pytest.approx([0.25, 0.125, 0])
