from replay_vta_hierarchical_bundle_negative_control import edge


def test_edge_uses_paired_rounds_and_strict_all_win_gate():
    rows = []
    for round_index, delta in enumerate([-2.0, -1.0, -3.0, -0.5, -2.5, -1.5, -4.0]):
        rows.extend([
            {"round": round_index, "deployment_variant": "a", "latency_ms": 10.0},
            {"round": round_index, "deployment_variant": "b", "latency_ms": 10.0 + delta},
        ])
    result = edge(rows, "a", "b")
    assert result["accepted"]
    rows[-1]["latency_ms"] = 11.0
    result = edge(rows, "a", "b")
    assert not result["accepted"]
    assert result["right_wins"] == 6
