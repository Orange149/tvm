#!/usr/bin/env python3

from run_vta_resnet50_fused_program_reranker_board import profile_delta


def test_profile_delta_is_program_b_minus_a():
    programs = ["a", "b"]
    keys = {"load_buffer_2d_bytes": 100, "load_buffer_2d_calls": 10}
    rows = [
        {"program_id": "a", "runtime_profile_complete": keys},
        {"program_id": "b", "runtime_profile_complete": {
            "load_buffer_2d_bytes": 70, "load_buffer_2d_calls": 13,
        }},
    ]
    assert profile_delta(rows, programs) == {
        "load_buffer_2d_bytes": -30,
        "load_buffer_2d_calls": 3,
    }
    assert profile_delta(rows, programs, divisor=2.0) == {
        "load_buffer_2d_bytes": -15,
        "load_buffer_2d_calls": 1.5,
    }
