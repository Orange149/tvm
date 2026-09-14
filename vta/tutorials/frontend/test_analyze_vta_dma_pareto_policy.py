"""Unit tests for the label-free VTA DMA Pareto gate."""

from analyze_vta_dma_pareto_policy import evaluate_pair, summarize


def pair(input_after, total_after, calls_after, speedup):
    return evaluate_pair(
        workload_id="T00",
        pair_id="T00:0",
        input_bytes_before=100,
        input_bytes_after=input_after,
        total_bytes_before=200,
        total_bytes_after=total_after,
        total_calls_before=20,
        total_calls_after=calls_after,
        speedup_percent=speedup,
        source="test",
    )


def test_certifies_only_net_dma_pareto_improvement():
    assert pair(50, 150, 10, 4.0)["pareto_certified"]
    assert not pair(50, 220, 10, -3.0)["pareto_certified"]
    assert not pair(50, 150, 21, -3.0)["pareto_certified"]
    assert not pair(100, 150, 10, -3.0)["pareto_certified"]


def test_summary_keeps_measurement_out_of_selection():
    rows = [pair(50, 150, 10, 4.0), pair(50, 220, 10, -3.0)]
    result = summarize(rows)
    assert result["certified_count"] == 1
    assert result["certified_faster_count"] == 1
    assert result["abstained_regression_count"] == 1
