"""Focused normalization tests for the post-board literature replay adapter."""

from __future__ import annotations

import pytest

import prepare_vta_c3_resnet18_literature_replay as adapter


def test_hidden_compiler_feature_aliases_are_canonicalized():
    value = adapter.normalize_hidden(
        {
            "loop_count": 5,
            "loop_extent_sum": 20,
            "loop_extent_log2_product": 7.5,
            "branch_count": 2,
            "partial_tile_count": 1,
            "allocation_bytes": 4096,
            "tensorize_uop_sites": 3,
        }
    )
    assert value["loop_extent_product_log2"] == 7.5
    assert value["tensorize_count"] == 3


def test_static_features_include_load_store_and_command_peaks():
    candidate = {
        "dma_features": {
            "totals": {
                "load_buffer_2d_bytes": 100,
                "store_buffer_2d_bytes": 20,
                "load_buffer_2d_calls": 4,
                "store_buffer_2d_calls": 2,
                "load_buffer_2d_inp_bytes": 60,
                "load_buffer_2d_wgt_bytes": 40,
                "store_buffer_2d_out_bytes": 20,
            },
            "padded_calls": 1,
        },
        "command_features": {
            "values": {
                "submissions": 2,
                "peaks": {"insn_bytes": 8192, "uop_bytes": 512},
            }
        },
        "sync": {"explicit_residency_drains": 1},
    }
    value = adapter.normalize_static(candidate)
    assert value["total_dma_bytes"] == 120
    assert value["total_dma_calls"] == 6
    assert value["instruction_peak_bytes"] == 8192
    assert value["uop_peak_bytes"] == 512


def test_runtime_profile_cost_counts_logical_dma_and_real_driver_runs():
    value = adapter.profile_cost(
        [
            {
                "runtime_profile_complete": {
                    "load_buffer_2d_bytes": 100,
                    "store_buffer_2d_bytes": 20,
                    "load_buffer_2d_calls": 4,
                    "store_buffer_2d_calls": 2,
                    "driver_run_calls": 2,
                }
            }
        ]
    )
    assert value == {
        "logical_dma_bytes": 120,
        "logical_dma_calls": 6,
        "fpga_kernel_invocations": 2,
    }


def test_pure_instruction_time_divides_complete_operator_invocations_not_submissions():
    row = {
        "time_evaluator_total_kernel_invocations": 2,
        "runtime_profile_complete": {
            "driver_run_calls": 34,
            "driver_run_total_us": 210000,
        },
    }
    assert adapter.pure_instruction_ms_per_operator(row) == 105.0


def test_missing_pure_instruction_profile_is_rejected_instead_of_becoming_zero():
    with pytest.raises(ValueError, match="lacks driver_run_total_us"):
        adapter.pure_instruction_ms_per_operator(
            {"time_evaluator_total_kernel_invocations": 2}
        )
