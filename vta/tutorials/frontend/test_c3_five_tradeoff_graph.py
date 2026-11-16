import pytest

from run_c3_five_tradeoff_graph import VERSIONS, evaluate, ORDERS


def rows(times):
    return [{"program": name, "call_kind": "graph_timing", "block": block,
             "correct": True, "status": "PASS", "latency_ms": times[name]}
            for name in VERSIONS for block in range(2) for _ in range(5)]


def test_graph_gain_vs_weak_current_does_not_imply_practical_increment():
    r = evaluate(rows({"current": 400., "Input": 375., "rule": 374.5}))
    assert r["single_boot_improvement_vs_current"]
    assert not r["single_boot_practical_increment_vs_Input"]
    assert r["single_graph_equivalent_fps"]["rule"] == 1000 / 374.5


def test_graph_mixed_block_direction_does_not_pass():
    data = rows({"current": 400., "Input": 375., "rule": 365.})
    for r in data:
        if r["program"] == "rule" and r["block"] == 1:
            r["latency_ms"] = 376.
    result = evaluate(data)
    assert not result["single_boot_practical_increment_vs_Input"]


def test_missing_and_wrong_outputs_do_not_produce_graph_results():
    data = rows({"current": 400., "Input": 375., "rule": 365.})
    with pytest.raises(ValueError, match="incomplete"):
        evaluate(data[:-1])
    data[0]["correct"] = False
    with pytest.raises(ValueError, match="incorrect"):
        evaluate(data)


def test_graph_order_is_symmetric():
    assert ORDERS[1] == tuple(reversed(ORDERS[0]))
