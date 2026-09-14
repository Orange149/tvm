#!/usr/bin/env python3

from run_vta_resnet50_fused_program_pareto_board import choose_ids, expected_delta


def test_choose_ids_separates_prospective_front_and_later_completion():
    pool = {
        "pareto_front_candidate_ids": ["b", "a"],
        "programs": [{"candidate_id": name} for name in ("a", "b", "c", "d")],
    }
    assert choose_ids(pool, "front") == ["b", "a"]
    assert choose_ids(pool, "remaining") == ["c", "d"]


def test_expected_delta_is_candidate_minus_stock_for_all_runtime_keys():
    keys = (
        "load_buffer_2d_bytes", "load_buffer_2d_calls",
        "load_buffer_2d_acc_bytes", "load_buffer_2d_acc_calls",
        "load_buffer_2d_inp_bytes", "load_buffer_2d_inp_calls",
        "load_buffer_2d_wgt_bytes", "load_buffer_2d_wgt_calls",
        "store_buffer_2d_bytes", "store_buffer_2d_calls",
    )
    program = {"traffic": {key: 17 for key in keys}}
    stock = {"traffic": {key: 5 for key in keys}}
    assert expected_delta(program, stock) == {key: 12 for key in keys}
