"""Pure protocol checks for the node24 whole-group DMA experiment."""

from run_vta_c3_node24_group_dma_board import ORDERS, VERSIONS, analyze


def test_frozen_order_and_pair_direction():
    assert len(ORDERS) == 6
    assert all(set(row) == set(VERSIONS) and len(row) == 5 for row in ORDERS)
    assert ORDERS[-1] == tuple(reversed(VERSIONS))
    first = sum(row.index("f13") < row.index("f13_b") for row in ORDERS)
    assert first == 3


def test_paired_round_gate_uses_six_seed_medians():
    rows = []
    base = {"a13": 10.0, "f13": 7.0, "a13_b": 9.0,
            "f13_b": 6.65, "input_alt": 5.0}
    for round_id in range(6):
        for version, value in base.items():
            for seed in range(3):
                rows.append({"phase": "timing", "round": round_id,
                             "variant": version, "latency_ms": value + (seed - 1) * 0.01})
    result = analyze(rows)
    assert result["primary_engineering_gate_passed"]
    assert not result["beats_alternative_input"]
    assert result["comparisons"]["f13_to_f13_b"]["wins"] == 6
