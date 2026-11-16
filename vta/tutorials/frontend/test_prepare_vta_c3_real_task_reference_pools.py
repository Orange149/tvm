"""Tests for compact real-task reference pool preparation."""

import prepare_vta_c3_real_task_reference_pools as prepare


def test_workload_id_and_geometry_are_complete():
    workload = [
        "conv2d_packed.vta",
        ["TENSOR", [1, 8, 28, 28, 1, 16], "int8"],
        ["TENSOR", [8, 8, 3, 3, 16, 16], "int8"],
        [1, 1], [1, 1, 1, 1], [1, 1], "NCHW1n16c", "int32",
    ]
    assert prepare.geometry(workload) == {
        "ci": 128, "co": 128, "height": 28, "width": 28, "kernel": 3, "stride": 1,
    }
    assert prepare.workload_id(workload) == "R18-U128C128H28K3S1"


def test_knobs_keeps_threads_in_candidate_identity():
    entity = {
        "entity": [
            ["tile_b", "sp", [1, 1]], ["tile_h", "sp", [2, 14]],
            ["tile_w", "sp", [1, 28]], ["tile_ci", "sp", [8, 1]],
            ["tile_co", "sp", [4, 2]], ["oc_nthread", "ot", 1],
            ["h_nthread", "ot", 2],
        ]
    }
    values = prepare.knobs(entity)
    assert values["h_nthread"] == 2
    assert values["oc_nthread"] == 1
    assert values["tile_h"] == 14
