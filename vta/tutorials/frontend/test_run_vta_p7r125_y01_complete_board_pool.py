import run_vta_p7r125_y01_complete_board_pool as p7r125


def test_selects_exact_p7r124_rank0_six():
    rows = p7r125.select_six(p7r125.DEFAULT_P7R124)
    assert len(rows) == len({row["candidate_id"] for row in rows}) == 6
    assert {row["family_id"] for row in rows} == {"Y01F02", "Y01F07"}
    assert {row["public_mode"] for row in rows} == {
        "original", "input_stationary", "weight_resident_barrier"
    }


def test_seven_orders_have_equal_budget_and_latin_first_six():
    rows = p7r125.select_six(p7r125.DEFAULT_P7R124)
    orders = p7r125.candidate_orders(rows)
    budgets = p7r125.verify_orders(orders, rows)
    assert all(row["samples"] == 7 for row in budgets.values())
    assert sorted(row["first_positions"] for row in budgets.values()) == [1, 1, 1, 1, 1, 2]


def test_y01_tophub_is_not_an_input():
    rows = p7r125.select_six(p7r125.DEFAULT_P7R124)
    assert all(row["identity"]["template_name"] == "conv2d_packed_residency.vta" for row in rows)
    assert all(row["performance_label"] is None for row in rows)
