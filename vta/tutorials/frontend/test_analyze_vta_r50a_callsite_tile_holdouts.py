from analyze_vta_r50a_callsite_tile_holdouts import (
    aggregate_occurrences,
    percent_change,
)


def test_aggregate_occurrences_sums_exact_fields():
    fused = {
        "graph_occurrences": [
            {"original_dma": {"load_buffer_2d_bytes": 10}},
            {"original_dma": {"load_buffer_2d_bytes": 20}},
        ]
    }
    result = aggregate_occurrences(fused, "original_dma")
    assert result["load_buffer_2d_bytes"] == 30
    assert result["load_buffer_2d_calls"] == 0


def test_percent_change_keeps_direction():
    assert percent_change(75, 100) == -25
    assert percent_change(150, 100) == 50
