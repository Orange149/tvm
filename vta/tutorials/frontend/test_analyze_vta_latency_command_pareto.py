"""Tests for latency/command-backing Pareto analysis."""

from analyze_vta_latency_command_pareto import aligned, pareto_front


def row(candidate_id, latency, backing):
    return {"candidate_id": candidate_id, "median_ms": latency, "command_backing": {"total_bytes": backing}}


def test_alignment_has_one_page_minimum():
    assert aligned(0) == 4096
    assert aligned(4096) == 4096
    assert aligned(4097) == 8192


def test_pareto_removes_latency_and_space_dominated_points():
    rows = [row("fast-large", 1.0, 12288), row("small", 1.1, 8192), row("dominated", 1.2, 12288)]
    assert [item["candidate_id"] for item in pareto_front(rows)] == ["fast-large", "small"]
