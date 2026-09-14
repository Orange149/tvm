#!/usr/bin/env python3

from analyze_vta_operator_proxy_pareto_development import operator_proxy_program


def test_operator_proxy_uses_final_operator_descriptors_and_barrier_class():
    program = {
        "candidate_id": "x", "family_id": "f",
        "public_mode": "weight_resident_barrier",
    }
    static = {"transfer_signature": {"descriptor_aggregates": {
        "expanded_bytes": 123, "expanded_calls": 45,
    }}}
    assert operator_proxy_program(program, static) == {
        "candidate_id": "x", "family_id": "f",
        "public_mode": "weight_resident_barrier",
        "dma_bytes": 123, "dma_calls": 45, "extra_submissions": 1,
    }
