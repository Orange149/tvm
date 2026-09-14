#!/usr/bin/env python3

from build_vta_resnet50_fused_program_pool import dominates, pareto_front


def program(name, bytes_, calls, submissions):
    return {
        "candidate_id": name,
        "dma_bytes": bytes_,
        "dma_calls": calls,
        "extra_submissions": submissions,
    }


def test_three_axis_pareto_retains_tradeoffs_and_removes_dominated():
    a = program("a", 1, 10, 0)
    b = program("b", 10, 1, 0)
    c = program("c", 1, 10, 1)
    assert dominates(a, c)
    assert not dominates(a, b)
    assert {row["candidate_id"] for row in pareto_front([a, b, c])} == {"a", "b"}
