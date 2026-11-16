from freeze_vta_c3_yolo608_candidate_pools import geometry


def test_geometry_is_resolution_agnostic():
    workload = [
        "conv2d_packed.vta",
        ["TENSOR", [1, 32, 18, 18, 1, 16], "int8"],
        ["TENSOR", [16, 32, 1, 1, 16, 16], "int8"],
        [1, 1], [0, 0, 0, 0], [1, 1], "NCHW1n16c", "int32",
    ]
    assert geometry(workload) == {
        "ci": 512, "co": 256, "height": 18, "width": 18,
        "kernel": 1, "stride": 1,
    }
