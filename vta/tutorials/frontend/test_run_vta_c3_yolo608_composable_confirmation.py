from run_vta_c3_yolo608_composable_confirmation import ROUND_ORDERS, SEEDS, summarize_rounds


def test_round_order_is_balanced_for_two_variants():
    assert len(ROUND_ORDERS) == 6
    assert sum(row[0] == "stock" for row in ROUND_ORDERS) == 3
    assert sum(row[0] == "scoped_composable" for row in ROUND_ORDERS) == 3


def test_summary_uses_one_seed_median_per_round():
    rows = []
    for round_index in range(6):
        for variant, center in (("stock", 100.0), ("scoped_composable", 90.0)):
            for offset, seed in zip((-1.0, 0.0, 50.0), SEEDS):
                rows.append({"round": round_index, "variant": variant,
                             "seed": seed, "latency_ms": center + offset})
    result = summarize_rounds(rows)
    assert result["rounds"] == 6
    assert result["wins"] == 6
    assert abs(result["median_improvement_pct"] - 10.0) < 1e-12


def test_summary_rejects_incomplete_round():
    rows = []
    for round_index in range(6):
        for variant in ("stock", "scoped_composable"):
            for seed in SEEDS:
                rows.append({"round": round_index, "variant": variant,
                             "seed": seed, "latency_ms": 1.0})
    rows.pop()
    try:
        summarize_rounds(rows)
    except ValueError as error:
        assert "incomplete seed coverage" in str(error)
    else:
        raise AssertionError("incomplete round was accepted")
