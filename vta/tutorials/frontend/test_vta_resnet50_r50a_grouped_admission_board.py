from run_vta_resnet50_r50a_grouped_admission_board import (
    add_delta,
    expected_mask_delta,
    graph_file,
)


def test_expected_mask_delta_sums_only_newly_admitted_routes():
    routes = [
        {"expected_delta": {"a": -3, "b": -1}},
        {"expected_delta": {"a": -3, "b": -1}},
        {"expected_delta": {"a": -3, "b": -1}},
    ]
    assert add_delta([routes[0]["expected_delta"], routes[1]["expected_delta"]]) == {
        "a": -6,
        "b": -2,
    }
    # Exercise with the real profile-key shape.
    complete = []
    from run_vta_resnet50_fused_tir_pair_board_v2 import PROFILE_KEYS

    for _ in routes:
        complete.append({"expected_delta": {key: -1 for key in PROFILE_KEYS}})
    assert expected_mask_delta(complete, [1, 0, 0], [1, 1, 0]) == {
        key: -1 for key in PROFILE_KEYS
    }
    assert graph_file([1, 0, 1]) == "mask101.json"
