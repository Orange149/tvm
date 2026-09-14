from analyze_vta_resnet50_fused_tir_occurrence_delta import vector_delta


def test_vector_delta_fills_missing_fused_fields_with_zero():
    left = {"load_buffer_2d_bytes": 100, "load_buffer_2d_calls": 4}
    right = {"load_buffer_2d_bytes": 70, "load_buffer_2d_calls": 2}
    result = vector_delta(left, right)
    assert result["load_buffer_2d_bytes"] == -30
    assert result["load_buffer_2d_calls"] == -2
    assert result["load_buffer_2d_acc_calls"] == 0
    assert result["store_buffer_2d_calls"] == 0
