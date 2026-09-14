from analyze_vta_same_tile_completed_pool import measured_latency


def test_measured_latency_accepts_only_positive_finite_numbers():
    assert measured_latency({"oracle": {"latency_ms": 1.25}}) == 1.25
    assert measured_latency({"oracle": {"latency_ms": None}}) is None
    assert measured_latency({"oracle": {"latency_ms": 0}}) is None
    assert measured_latency({"oracle": {"latency_ms": float("inf")}}) is None
    assert measured_latency({"oracle": {"latency_ms": True}}) is None
