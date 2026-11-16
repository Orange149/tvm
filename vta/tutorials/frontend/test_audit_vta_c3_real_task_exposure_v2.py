from audit_vta_c3_real_task_exposure_v2 import workload_signature


def test_packed_workload_signature_includes_stride_and_padding():
    workload = [
        "conv2d_packed.vta",
        ["TENSOR", [1, 4, 56, 56, 1, 16], "int8"],
        ["TENSOR", [8, 4, 3, 3, 16, 16], "int8"],
        [2, 2], [1, 1, 1, 1], [1, 1], "NCHW1n16c", "int32",
    ]
    assert workload_signature(workload) == [1, 64, 56, 56, 128, 3, 3, 2, 2, 1, 1, 1, 1]
