"""Contract, leakage and accounting tests for the unified C3 policy runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import run_vta_c3_literature_baselines as runner


def _id(text):
    return hashlib.sha256(text.encode()).hexdigest()


def candidate(family, mode, coordinate=1, dma=100, fits=True):
    return {
        "candidate_id": _id("{}:{}:{}".format(family, mode, coordinate)),
        "workload_id": "T0",
        "family_id": family,
        "model_hash": _id("model"),
        "hardware_fingerprint": {"target": "test"},
        "complete_config_entity": {
            "entity": [["tile_h", "sp", [-1, coordinate]]],
        },
        "residence_mode": mode,
        "visible_features": {
            "tile_b": 1,
            "tile_h": coordinate,
            "tile_w": 1,
            "tile_ci": 1,
            "tile_co": 1,
            "oc_nthread": 1,
            "h_nthread": 1,
        },
        "workload_features": {
            "ci": 16,
            "co": 16,
            "height": 8,
            "width": 8,
            "kernel": 3,
            "stride": 1,
        },
        "static_features": {
            "total_dma_bytes": dma,
            "total_dma_calls": dma // 10,
            "padding_dma_calls": 0,
            "submissions": 1,
        },
        "applicability": {
            "full_weight_fits_sram": fits,
            "accumulator_fits_sram": True,
        },
    }


def workload(candidates):
    return {
        "schema": "c3_literature_workload_contract_v1",
        "workload_id": "T0",
        "candidates": candidates,
    }


def phase_outcome(valid=True, latency=1.0):
    lower = {
        "status": "ok" if valid else "invalid",
        "wall_seconds": 1.0,
        "compiler_attempts": 1,
        "hidden_features": {
            "loop_count": 3,
            "loop_extent_sum": 10,
            "loop_extent_product_log2": 5,
            "branch_count": 0,
            "partial_tile_count": 0,
            "allocation_bytes": 1024,
            "tensorize_count": 1,
        },
    }
    result = {"lower": lower}
    if valid:
        result.update(
            {
                "fsim": {"status": "ok", "wall_seconds": 2.0},
                "compile": {
                    "status": "ok",
                    "wall_seconds": 3.0,
                    "compiler_attempts": 1,
                },
                "fpga_correctness": {
                    "status": "ok",
                    "wall_seconds": 4.0,
                    "fpga_kernel_invocations": 3,
                    "logical_dma_bytes": 100,
                    "logical_dma_calls": 10,
                },
                "timing": {
                    "status": "ok",
                    "latency_ms": latency,
                    "wall_seconds": 5.0,
                    "fpga_kernel_invocations": 7,
                    "logical_dma_bytes": 200,
                    "logical_dma_calls": 20,
                },
            }
        )
    return result


def test_future_label_leak_is_rejected():
    row = candidate("F0", "original")
    row["latency_ms"] = 1.0
    with pytest.raises(ValueError, match="future target label leaked"):
        runner.validate_workload_contract(workload([row]))


def test_rieber_presampling_uses_feedback_and_balanced_e0():
    rows = [candidate("F{}".format(i), "original", i) for i in range(60)]
    calls = []

    def valid(candidate_id):
        calls.append(candidate_id)
        return True

    order = runner.rieber_presampling_order(rows, 7, valid, limit=20, parallel=1)
    by_id = {row["candidate_id"]: row for row in rows}
    assert calls == order
    assert len(order) == len(set(order)) == 20
    assert runner.manhattan(by_id[order[0]], by_id[order[1]]) == 1

    labelled = []
    for index, row in enumerate(rows):
        labelled.append({**row, "is_valid": index < 30})
    e0 = runner.balanced_e0(labelled, per_class=25)
    assert len(e0) == 50
    assert sum(row["is_valid"] for row in e0) == 25


def test_neighbourhood_uses_adjacent_discrete_factor_levels_not_raw_distance():
    rows = [
        candidate("F{}".format(index), "original", tile_h)
        for index, tile_h in enumerate((1, 2, 4, 7, 14))
    ]
    neighbours = runner.neighbor_map(rows)
    assert rows[2]["candidate_id"] in neighbours[rows[1]["candidate_id"]]  # 2 -> 4
    assert rows[4]["candidate_id"] in neighbours[rows[3]["candidate_id"]]  # 7 -> 14
    assert rows[4]["candidate_id"] not in neighbours[rows[0]["candidate_id"]]


def test_parallel_presampling_exhausts_odd_sized_space_without_duplicates():
    rows = [candidate("F{}".format(index), "original", index) for index in range(31)]
    order = runner.rieber_presampling_order(rows, 57004, lambda _: True, limit=31, parallel=8)
    assert len(order) == len(set(order)) == 31


def test_invalid_rows_never_enter_performance_regression():
    good = {**candidate("F0", "original"), "is_valid": True, "latency_ms": 1.0}
    bad = {**candidate("F1", "original"), "is_valid": False, "latency_ms": 0.01}
    unknown = candidate("F2", "original")
    assert runner.performance_training_rows([good, bad, unknown], []) == [good]
    assert len(runner.validity_training_rows([good, bad, unknown], [])) == 2


def test_compile_success_remains_censored_until_fsim_and_fpga_observation():
    row = candidate("F0", "original")
    compile_only = runner.compile_stage_observation(
        row,
        {
            "lower": {
                "status": "ok",
                "hidden_features": phase_outcome(True)["lower"]["hidden_features"],
            },
            "compile": {"status": "ok"},
        },
    )
    assert compile_only["compile_valid"] is True
    assert compile_only["final_validity_observed"] is False
    assert "is_valid" not in compile_only
    assert runner.validity_training_rows([], [compile_only]) == []


def test_compile_failure_is_a_terminal_negative_validity_label():
    row = candidate("F0", "original")
    failed = runner.compile_stage_observation(
        row,
        {"lower": {"status": "ok"}, "compile": {"status": "invalid"}},
    )
    assert failed["compile_valid"] is False
    assert failed["final_validity_observed"] is True
    assert failed["is_valid"] is False
    assert runner.validity_training_rows([], [failed]) == [failed]


def test_cheng_four_modes_and_resource_fallback_are_explicit():
    rows = [
        candidate("F0", "original", dma=100),
        candidate("F0", "input_stationary", dma=80),
        candidate("F0", "weight_resident_barrier", dma=70, fits=False),
        candidate("F0", "input_weight_resident_barrier", dma=50, fits=False),
    ]
    selected, fallback = runner.cheng_minimum_access_order(rows)
    assert [row["residence_mode"] for row in selected] == ["input_stationary"]
    assert len(fallback) == 2
    assert all(row["status"] == "not_applicable" for row in fallback)
    assert all(row["fallback_candidate_id"] == rows[0]["candidate_id"] for row in fallback)


def test_cheng_accumulator_bound_does_not_pre_reject_original():
    original = candidate("F0", "original")
    input_mode = candidate("F0", "input_stationary")
    original["applicability"]["accumulator_fits_sram"] = False
    input_mode["applicability"]["accumulator_fits_sram"] = False
    assert runner.cheng_applicability(original) == (True, None)
    assert runner.cheng_applicability(input_mode) == (
        False,
        "accumulator_exceeds_sram",
    )


def test_failures_remain_in_gross_and_wall_cost():
    good = candidate("F0", "original", dma=10)
    bad = candidate("F1", "original", dma=20)
    ledger = runner.OutcomeLedger(
        {
            "schema": "c3_literature_outcome_ledger_v1",
            "session_status": "complete",
            "pool_complete": True,
            "outcomes": {
                good["candidate_id"]: phase_outcome(True, 1.0),
                bad["candidate_id"]: phase_outcome(False),
            },
        }
    )
    timeline, results, _, _ = runner.replay_standard(
        "ours_dma_multifidelity", [good, bad], 2, 1, [], ledger
    )
    cost = runner.summarize_cost(timeline, results)
    assert cost["gross_candidates"] == 2
    assert cost["invalid_candidates"] == 1
    assert cost["wall_seconds"] == 16.0
    assert cost["timed_valid_candidates"] == 1
    assert cost["phase_wall_seconds"] == {
        "lower": 2.0,
        "fsim": 2.0,
        "compile": 3.0,
        "fpga_correctness": 4.0,
        "timing": 5.0,
    }
    assert cost["qualification_wall_seconds"] == 4.0
    assert cost["cross_compile_wall_seconds"] == 3.0
    assert cost["correctness_wall_seconds"] == 4.0
    assert cost["timing_wall_seconds"] == 5.0


def test_oracle_is_unavailable_before_complete_pool():
    row = candidate("F0", "original")
    ledger = runner.OutcomeLedger(
        {
            "schema": "c3_literature_outcome_ledger_v1",
            "session_status": "complete",
            "pool_complete": False,
            "outcomes": {row["candidate_id"]: phase_outcome(True)},
        }
    )
    with pytest.raises(RuntimeError, match="before complete-pool"):
        ledger.oracle_after_completion([row["candidate_id"]])


def test_ml2tuner_compiles_2n_and_profiles_at_most_n():
    rows = [candidate("F{}".format(i), "original", i + 1, 100 + i) for i in range(24)]
    outcomes = {
        row["candidate_id"]: phase_outcome(valid=(index % 5 != 0), latency=index + 1)
        for index, row in enumerate(rows)
    }
    ledger = runner.OutcomeLedger(
        {
            "schema": "c3_literature_outcome_ledger_v1",
            "session_status": "complete",
            "pool_complete": True,
            "outcomes": outcomes,
        }
    )
    development = [
        {
            **row,
            "is_valid": index % 2 == 0,
            "latency_ms": float(index + 10) if index % 2 == 0 else None,
            "hidden_features": phase_outcome(True)["lower"]["hidden_features"],
        }
        for index, row in enumerate(rows[:8])
    ]
    timeline, results, diagnostic = runner.replay_ml2tuner(
        rows, 10, 9, development, ledger, include_dma=True
    )
    assert diagnostic["compiled_candidates"] == 20
    assert diagnostic["model_a_promoted"] <= 10
    assert len(results) <= 10
    profiled = {event["candidate_id"] for event in timeline if event["phase"] == "timing"}
    compile_invalid = {
        row["candidate_id"] for index, row in enumerate(rows) if index % 5 == 0
    }
    assert not (profiled & compile_invalid)


def test_ml2tuner_repeats_fixed_ten_candidate_waves_for_larger_budget():
    rows = [candidate("F{}".format(i), "original", i + 1, 100 + i) for i in range(45)]
    ledger = runner.OutcomeLedger(
        {
            "schema": "c3_literature_outcome_ledger_v1",
            "session_status": "complete",
            "pool_complete": True,
            "outcomes": {
                row["candidate_id"]: phase_outcome(True, index + 1)
                for index, row in enumerate(rows)
            },
        }
    )
    _, results, diagnostic = runner.replay_ml2tuner(
        rows, 25, 11, [], ledger, include_dma=False
    )
    assert len(results) == 25
    assert [wave["compile_candidates"] for wave in diagnostic["waves"]] == [20, 20, 5]
    assert [wave["model_a_promoted"] for wave in diagnostic["waves"]] == [10, 10, 5]
    assert diagnostic["compiled_candidates"] == 45


def test_ml2tuner_does_not_train_v_on_unprofiled_compile_successes():
    rows = [candidate("F{}".format(i), "original", i + 1, 100 + i) for i in range(20)]
    ledger = runner.OutcomeLedger(
        {
            "schema": "c3_literature_outcome_ledger_v1",
            "session_status": "complete",
            "pool_complete": True,
            "outcomes": {
                row["candidate_id"]: phase_outcome(True, index + 1)
                for index, row in enumerate(rows)
            },
        }
    )
    _, results, diagnostic = runner.replay_ml2tuner(
        rows, 10, 17, [], ledger, include_dma=False
    )
    assert len(results) == 10
    assert diagnostic["censored_compile_valid_not_profiled"] == 10
    assert diagnostic["invalid_during_profiling"] == 0
    assert diagnostic["invalid_profiling_ratio"] == 0.0


def test_freeze_writes_required_files_without_touching_input(tmp_path):
    rows = [candidate("F0", mode, dma=100 - index * 10) for index, mode in enumerate(runner.CHENG_MODES)]
    source = tmp_path / "workload.json"
    source.write_text(json.dumps(workload(rows)), encoding="utf-8")
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    output = tmp_path / "frozen"
    args = SimpleNamespace(
        policy="cheng_minimum_access",
        workload_contract=str(source),
        candidate_budget=4,
        seed=1,
        phase="freeze",
        output_dir=str(output),
        outcome_ledger=None,
        development_data=None,
        ml2_n=10,
        ml2_alpha=1,
        ml2_include_dma=False,
        rieber_level="balanced_e0_validity_bias",
    )
    runner.run(args)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    for name in (
        "contract.json",
        "timeline.jsonl",
        "results.jsonl",
        "summary.json",
        "artifact_hashes.json",
    ):
        assert (output / name).is_file()
    with pytest.raises(FileExistsError):
        runner.run(args)


def test_interrupted_session_is_invalid_and_cannot_be_spliced(tmp_path):
    row = candidate("F0", "original")
    source = tmp_path / "workload.json"
    source.write_text(json.dumps(workload([row])), encoding="utf-8")
    outcome = tmp_path / "outcome.json"
    outcome.write_text(
        json.dumps(
            {
                "schema": "c3_literature_outcome_ledger_v1",
                "session_status": "rpc_interrupted",
                "pool_complete": False,
                "outcomes": {},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "invalid"
    args = SimpleNamespace(
        policy="random",
        workload_contract=str(source),
        candidate_budget=1,
        seed=1,
        phase="replay",
        output_dir=str(output),
        outcome_ledger=str(outcome),
        development_data=None,
        ml2_n=10,
        ml2_alpha=1,
        ml2_include_dma=False,
        rieber_level="balanced_e0_validity_bias",
    )
    with pytest.raises(RuntimeError, match="cannot be replayed or spliced"):
        runner.run(args)
    invalid = json.loads((output / "invalid_session.json").read_text())
    assert invalid["cannot_splice_with_later_session"] is True
