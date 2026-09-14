"""Tests for the unified local C3 candidate qualification join."""

import copy
import json
from pathlib import Path

import pytest

from join_vta_local_candidate_qualification import join_qualifications, load_json, load_jsonl


BASE = Path(__file__).resolve().parent / "report_out" / "stage_tile_cotuning" / "c3_dma_residency_autotune"
P2 = BASE / "02_baselines" / "20260910_p2_local_baseline_run02"
P4 = BASE / "04_dma_command_signatures"


def _inputs():
    return (
        load_jsonl(P4 / "20260910_p4b_local_residency_pool_run01" / "results.jsonl"),
        load_jsonl(P4 / "20260910_p4c_local_fsim_qualification_run01" / "results.jsonl"),
        load_jsonl(P4 / "20260910_p4d_original_controls_fsim_run02" / "results.jsonl"),
        load_json(P2 / "fsim_results.json"),
        load_json(P2 / "tir_results.json"),
    )


def test_real_join_covers_every_lower_success():
    records, summary = join_qualifications(*_inputs())
    assert len(records) == summary["unique_candidate_ids"] == 250
    assert summary["fsim_passed_all_three_seeds"] == 165
    assert summary["lower_failed"] == 85
    assert summary["unqualified_lower_success"] == 0


def test_missing_experimental_certificate_is_rejected():
    values = list(_inputs())
    values[1] = values[1][:-1]
    with pytest.raises(ValueError, match="missing P4c certificate"):
        join_qualifications(*values)


def test_relowered_tir_mismatch_is_rejected():
    values = list(_inputs())
    values[1] = copy.deepcopy(values[1])
    values[1][0]["relowered_tir_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="re-lowered TIR hash mismatch"):
        join_qualifications(*values)


def test_protected_incumbent_config_mismatch_is_rejected():
    values = list(_inputs())
    values[3] = copy.deepcopy(values[3])
    values[3][0]["config_index"] += 1
    with pytest.raises(ValueError, match="P2 config mismatch"):
        join_qualifications(*values)
