from freeze_vta_c3_selector_v3_evaluation_protocol import protocol


def test_protocol_freezes_equal_budget_and_oracle_boundaries_without_labels():
    value = protocol()
    assert value["budgets"]["B1"] == 1
    assert value["budgets"]["B3"] == 3
    assert value["random_baseline"]["replicates"] == 1000
    assert "attempt consumes budget" in value["invalid_candidate_policy"]
    assert "fully measured declared reference pool" in value["oracle_scope"]
    assert "no new latency" in value["claim_boundary"]
