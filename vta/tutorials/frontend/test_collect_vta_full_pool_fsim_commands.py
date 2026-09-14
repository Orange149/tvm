"""Tests for the frozen 197-candidate P4j command dry-run."""

import json
from pathlib import Path

from collect_vta_full_pool_fsim_commands import (
    MODE_NUMBERS,
    classify_failure,
    direct_worker_result,
    group_summary,
    load_jsonl,
    sanitize_stderr,
    select_full_pool,
    strip_diagnostic_timing,
    verify_frozen_result,
)


BASE = (
    Path(__file__).resolve().parent
    / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/04_dma_command_signatures"
)


def test_frozen_inputs_and_full_selection_are_exact():
    paths = {
        "p4b": BASE / "20260910_p4b_local_residency_pool_run01/results.jsonl",
        "p4e": BASE / "20260911_p4e_unified_local_qualification_run01/results.jsonl",
        "p4g": BASE / "20260911_p4g_weight_barrier_pool_run01/results.jsonl",
        "p4i": BASE / "20260911_p4i_weight_barrier_cross_compile_run01/results.jsonl",
    }
    for path in paths.values():
        assert verify_frozen_result(path)["sha256"]
    selected = select_full_pool(*(load_jsonl(paths[name]) for name in ("p4b", "p4e", "p4g", "p4i")))
    assert len(selected) == len({row["candidate_id"] for row in selected}) == 197
    counts = {mode: sum(row["public_mode"] == mode for row in selected) for mode in MODE_NUMBERS}
    assert counts == {
        "original": 55,
        "input_stationary": 32,
        "weight_stationary": 43,
        "paper_inspired_hybrid": 35,
        "weight_stationary_barrier": 32,
    }
    barriers = [row for row in selected if row["implementation_mode"] == 4]
    assert all(row["dependency_evidence_reference"]["valid"] for row in barriers)


def test_timing_fields_are_deleted_and_stderr_is_sanitized():
    result = {
        "queue_records": [
            {"insn_bytes": 10, "device_run_us": 99.0, "finalize_us": 2.0, "queue_id": "0x1"}
        ]
    }
    strip_diagnostic_timing(result)
    assert result["queue_records"] == [{"insn_bytes": 10, "queue_id": "0x1"}]
    assert "99.0" not in sanitize_stderr('[VTA_QUEUE] {"device_run_us":99.0}')


def test_failure_categories_are_machine_readable():
    assert classify_failure({"outcome": "negative_timeout"})["category"] == "timeout"
    assert classify_failure({"outcome": "negative_wrong_answer"})["category"] == "wrong_answer"
    assert classify_failure({"outcome": "negative_worker_failure", "build": "not_attempted"})["category"] == "compile"
    assert classify_failure({"outcome": "passed_local_signature"}) is None


def test_group_summary_aggregates_structural_not_timing_metrics():
    structural = {
        "submissions": 2,
        "totals": {"insn_bytes": 100, "uop_bytes": 20, "load_bytes": 80, "store_bytes": 40},
        "peaks": {"insn_bytes": 60, "uop_bytes": 12},
        "finish": {"source_derived_count": 2},
    }
    rows = [
        {
            "outcome": "passed_local_signature",
            "command_signature": {"structural": structural},
            "failure": None,
        },
        {
            "outcome": "negative_wrong_answer",
            "failure": {"category": "wrong_answer"},
        },
    ]
    summary = group_summary(rows)
    assert summary["passed"] == 1 and summary["failed"] == 1
    assert summary["submissions"]["sum"] == 2
    assert summary["insn_total_bytes"]["sum"] == 100
    assert summary["failure_categories"] == {"wrong_answer": 1}
    assert not any("us" in key for key in json.dumps(summary).split('"'))


def test_direct_worker_source_does_not_reference_rpc_api():
    import inspect

    source = inspect.getsource(direct_worker_result)
    assert "LocalSession" not in source
    assert "rpc." not in source
    assert 'module["main"]' in source
