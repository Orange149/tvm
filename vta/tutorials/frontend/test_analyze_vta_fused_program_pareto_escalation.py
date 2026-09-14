#!/usr/bin/env python3

from analyze_vta_fused_program_pareto_escalation import dominates, pareto_front


def row(name, bytes_, calls):
    return {"program_id": name, "dma_bytes": bytes_, "dma_calls": calls}


def test_pareto_front_eliminates_only_jointly_dominated_points():
    low_bytes = row("low_bytes", 1, 10)
    low_calls = row("low_calls", 10, 1)
    dominated = row("dominated", 11, 11)
    assert not dominates(low_bytes, low_calls)
    assert dominates(low_bytes, dominated)
    assert {item["program_id"] for item in pareto_front(
        [low_bytes, low_calls, dominated]
    )} == {"low_bytes", "low_calls"}
