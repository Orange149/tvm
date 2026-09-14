"""Unit tests for the bounded local FSim qualification orchestrator."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from qualify_vta_residency_fsim import (
    MODE_NUMBERS,
    SEEDS,
    reference_data_from_workload,
    select_candidates,
    summarize,
)


BASE = Path(__file__).resolve().parent / "report_out" / "stage_tile_cotuning"
P4B_RESULTS = (
    BASE
    / "c3_dma_residency_autotune"
    / "04_dma_command_signatures"
    / "20260910_p4b_local_residency_pool_run01"
    / "results.jsonl"
)


def test_p4b_selection_is_exact_and_unique():
    rows = [json.loads(line) for line in P4B_RESULTS.read_text().splitlines() if line]
    selected = select_candidates(rows)
    assert len(selected) == 110
    assert len({row["candidate_id"] for row in selected}) == 110
    assert Counter(row["residence_mode"] for row in selected) == {
        "input_stationary": 32,
        "weight_stationary": 43,
        "paper_inspired_hybrid": 35,
    }


def test_same_tile_original_control_selection_is_separate_and_complete():
    rows = [json.loads(line) for line in P4B_RESULTS.read_text().splitlines() if line]
    selected = select_candidates(rows, roles=("same_tile_original_control",))
    assert len(selected) == 45
    assert len({row["candidate_id"] for row in selected}) == 45
    assert {row["residence_mode"] for row in selected} == {"original"}


def test_numpy_reference_is_seeded_independent_and_shape_correct():
    rows = [json.loads(line) for line in P4B_RESULTS.read_text().splitlines() if line]
    workload = select_candidates(rows)[0]["identity"]["workload"]
    first = reference_data_from_workload(workload, SEEDS[0])
    repeat = reference_data_from_workload(workload, SEEDS[0])
    other = reference_data_from_workload(workload, SEEDS[1])
    assert all(np.array_equal(left, right) for left, right in zip(first, repeat))
    assert not np.array_equal(first[0], other[0])
    assert first[2].dtype == np.int8
    assert first[2].shape[:2] == (workload[1][1][0], workload[2][1][0])


def test_summary_preserves_mode_and_workload_counts():
    records = []
    for index, mode in enumerate(MODE_NUMBERS):
        records.append(
            {
                "candidate_id": str(index),
                "workload_id": "W00",
                "residence_mode": mode,
                "overall_status": "passed",
                "failure": None,
            }
        )
    result = summarize(records, len(MODE_NUMBERS))
    assert result["status"] == "completed"
    assert result["passed_all_seeds"] == len(MODE_NUMBERS)
    assert all(result["modes"][mode]["selected"] == 1 for mode in MODE_NUMBERS)
    assert result["workloads"]["W00"]["input_stationary"]["passed_all_seeds"] == 1


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
