"""Tests for the offline P6b mechanism-canary contract."""

import collections
from pathlib import Path

import pytest

from prepare_vta_mechanism_canary import (
    ORDER_SEED,
    TIMED_REPEATS,
    WORKLOADS,
    _balanced_orders,
    choose_best,
    ensure_empty_output_dir,
    prepare_contract,
)


BASE = (
    Path(__file__).resolve().parent
    / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/04_dma_command_signatures"
)
P4B = BASE / "20260910_p4b_local_residency_pool_run01"
P4E = BASE / "20260911_p4e_unified_local_qualification_run01"
P4F = BASE / "20260911_p4f_axu_cross_compile_run01"
P4G = BASE / "20260911_p4g_weight_barrier_pool_run01"
P4I = BASE / "20260911_p4i_weight_barrier_cross_compile_run01"


def _contract():
    return prepare_contract(P4B, P4E, P4F, P4G)


def _qualified_contract():
    return prepare_contract(P4B, P4E, P4F, P4G, P4I)


def test_selects_expected_three_mechanisms_per_workload():
    manifest, _ = _contract()
    expected = {
        "W00": [
            "b046098ba4198415ec8f8be17ed62ac7f4c59198a3b0f262cb945e31bac46dc2",
            "674e3264beecb2eebc7f6d3380a18e7bb02b2bf80c1eeff49ca1978bfad74b38",
            "df9ef8675c4e1bc453463bbb8c9954837070d50ab9e9847c51fe312f11ed9908",
        ],
        "W02": [
            "37c662cfab07879956fbf12bb9f60f1461b1a197e4a4f1826f4033f0adc1fe96",
            "b298636a650419939c72ff49cbbee99be2b60e09f60c2a248c64b5133b41c25f",
            "b975b5e12b53db2d6ef72f8ce10b839cca0c9a7495115f8d7172deaf0996f77a",
        ],
        "W09": [
            "1ea905d902b7345d88692916e59a1239623e337a1b5b7e7c48d6b6ae749a3846",
            "77bf9c711ccd704418941cdfface6e8941f5e5f1e1c6cac3fc31fd7cb905297c",
            "24953281a2465eacf558acf79ba57ad12c654060a60790cc46dde705c8369d22",
        ],
    }
    for workload_id in WORKLOADS:
        rows = manifest["workloads"][workload_id]
        assert len(rows) == len({row["candidate_id"] for row in rows}) == 3
        assert [row["candidate_id"] for row in rows] == expected[workload_id]
        assert [row["mechanism_role"] for row in rows] == [
            "original_incumbent",
            "input_stationary_mechanism",
            "weight_barrier_mechanism",
        ]


def test_selected_existing_qualifications_and_pending_p4i_are_explicit():
    manifest, _ = _contract()
    assert manifest["p4i_barrier_axu"]["status"] == "pending"
    assert manifest["board_executed"] is False
    for rows in manifest["workloads"].values():
        for row in rows:
            assert row["local_fsim_evidence"]["status"] in ("fsim_passed", "passed")
            assert len(row["local_fsim_evidence"]["seeds"]) == 3
            assert row["complete_config_entity"]["index"] == row["config_index"]
        assert rows[0]["axu_cross_compile_evidence"]["status"] == "passed"
        assert rows[1]["axu_cross_compile_evidence"]["status"] == "passed"
        assert rows[2]["axu_cross_compile_evidence"]["status"] == "pending"


def test_completed_p4i_is_bound_as_passing_barrier_axu_evidence():
    manifest, _ = _qualified_contract()
    assert "results_sha256" in manifest["p4i_barrier_axu"]
    for rows in manifest["workloads"].values():
        barrier = rows[2]
        evidence = barrier["axu_cross_compile_evidence"]
        assert evidence["status"] == evidence["build_status"] == evidence["export_status"] == "passed"
        assert evidence["tir_sha256"] == barrier["tir_sha256"]
        assert len(evidence["binary_sha256"]) == 64
        assert barrier["qualification_source_guards_sha256"]


def test_selection_metrics_follow_max_reduction_then_calls_then_id():
    candidates = [
        {
            "candidate_id": "b",
            "relative_to_same_tile_original": {"inp": {"reduction_fraction": 0.5}},
            "transfer_signature": {"totals": {"load_buffer_2d_inp_calls": 3}},
        },
        {
            "candidate_id": "a",
            "relative_to_same_tile_original": {"inp": {"reduction_fraction": 0.5}},
            "transfer_signature": {"totals": {"load_buffer_2d_inp_calls": 3}},
        },
        {
            "candidate_id": "c",
            "relative_to_same_tile_original": {"inp": {"reduction_fraction": 0.5}},
            "transfer_signature": {"totals": {"load_buffer_2d_inp_calls": 4}},
        },
    ]
    assert choose_best(candidates, "inp")["candidate_id"] == "a"


def test_complete_crossover_is_deterministic_and_position_balanced():
    first = _balanced_orders(["a", "b", "c"], TIMED_REPEATS, str(ORDER_SEED))
    second = _balanced_orders(["a", "b", "c"], TIMED_REPEATS, str(ORDER_SEED))
    assert first == second and len(first) == 30
    positions = collections.Counter((item, position) for order in first for position, item in enumerate(order))
    assert set(positions.values()) == {10}
    assert len({tuple(order) for order in first}) == 6


def test_dispatch_plan_freezes_correctness_timing_counters_and_thresholds():
    _, plan = _contract()
    assert plan["board_executed"] is False
    assert plan["correctness"]["seeds"] == [0, 20250901, 20260910]
    assert plan["timing"]["warmup_runs_per_candidate"] == 3
    assert plan["timing"]["timed_repeats_per_candidate"] == 30
    assert len(plan["ordering"]["rounds"]) == 30
    assert "load_buffer_2d_wgt_bytes" in plan["runtime_counters"]["required"]
    assert "0.02" in plan["decision_rule"]["equivalent"]
    assert "0.05" in plan["decision_rule"]["improved"]


def test_output_writer_refuses_nonempty_directory(tmp_path):
    (tmp_path / "already_frozen").write_text("x")
    with pytest.raises(FileExistsError, match="non-empty frozen output"):
        ensure_empty_output_dir(tmp_path)
