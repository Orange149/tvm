"""Tests for the C3 P6a offline board-dispatch contract."""

import copy
import json
from pathlib import Path

import pytest

from prepare_vta_board_dispatch import (
    DEFAULT_CORRECTNESS_SEEDS,
    build_dispatch_manifest,
    prepare_dispatch,
    validate_frozen_inputs,
    write_dispatch_manifest,
)


BASE = (
    Path(__file__).resolve().parent
    / "report_out"
    / "stage_tile_cotuning"
    / "c3_dma_residency_autotune"
)
P4B = BASE / "04_dma_command_signatures" / "20260910_p4b_local_residency_pool_run01"
P5B = BASE / "05_grouped_replay" / "20260910_p5b_local_predispatch_shortlist_run01"
P4E = BASE / "04_dma_command_signatures" / "20260911_p4e_unified_local_qualification_run01"
SHORTLIST = P5B / "shortlist.json"
POOL = P4B / "candidate_pool.json"
RESULTS = P4B / "results.jsonl"
QUALIFICATION_RESULTS = P4E / "results.jsonl"
QUALIFICATION_SUMMARY = P4E / "summary.json"


def _prepare(workload_id="W00"):
    return prepare_dispatch(
        SHORTLIST,
        POOL,
        RESULTS,
        QUALIFICATION_RESULTS,
        QUALIFICATION_SUMMARY,
        workload_id,
        "B7",
        4,
    )


def _validate():
    return validate_frozen_inputs(
        SHORTLIST, POOL, RESULTS, QUALIFICATION_RESULTS, QUALIFICATION_SUMMARY
    )


@pytest.mark.parametrize("workload_id", ["W00", "W02", "W09"])
def test_real_b7_budget4_contract_is_complete_offline_and_incumbent_first(workload_id):
    manifest = _prepare(workload_id)
    assert manifest["board_executed"] is False
    assert manifest["claim_boundary"]["G6_passed"] is False
    assert manifest["selection"]["effective_budget"] == 4
    assert manifest["selection"]["ordered_candidate_ids"][0] == manifest["selection"][
        "protected_original_incumbent"
    ]
    assert len(set(manifest["selection"]["ordered_candidate_ids"])) == 4
    assert len(manifest["correctness_protocol"]["seeds"]) == 3
    assert manifest["timing_protocol"]["warmup_runs"] is None
    assert manifest["board_fingerprint"]["actual"]["target"] is None
    assert {candidate["residence_mode"] for candidate in manifest["candidates"]} == {
        "original",
        "input_stationary",
        "weight_stationary",
        "paper_inspired_hybrid",
    }
    for order, candidate in enumerate(manifest["candidates"]):
        assert candidate["dispatch_order"] == order
        config = candidate["complete_config_entity"]
        assert config["index"] == candidate["config_index"]
        assert config["entity"]
        assert candidate["template_name"]
        assert candidate["local_qualification"]["status"] == "fsim_passed"
        assert candidate["local_qualification"]["correctness_seeds"] == list(
            DEFAULT_CORRECTNESS_SEEDS
        )


def test_frozen_input_validation_reproduces_feature_hash():
    validated = _validate()
    assert validated["shortlist"]["feature_set_sha256"] == (
        "ee48faf94f166074777dfae0e22f157062ef0c7d34efd1876972c394e3cbf0c7"
    )
    assert len(validated["features"]) == 165
    assert len(validated["failures"]) == 85


def test_tampered_candidate_pool_is_rejected_by_hash(tmp_path):
    pool = json.loads(POOL.read_text())
    pool["hardware_fingerprint"]["target"] = "tampered"
    tampered = tmp_path / "candidate_pool.json"
    tampered.write_text(json.dumps(pool))
    with pytest.raises(ValueError, match="candidate pool SHA-256 mismatch"):
        validate_frozen_inputs(
            SHORTLIST, tampered, RESULTS, QUALIFICATION_RESULTS, QUALIFICATION_SUMMARY
        )


def test_tampered_qualification_summary_is_rejected_by_hash(tmp_path):
    summary = json.loads(QUALIFICATION_SUMMARY.read_text())
    summary["fsim_passed_all_three_seeds"] = 164
    tampered = tmp_path / "summary.json"
    tampered.write_text(json.dumps(summary))
    copied_results = tmp_path / "results.jsonl"
    copied_results.write_bytes(QUALIFICATION_RESULTS.read_bytes())
    (tmp_path / "artifact_hashes.json").write_bytes((P4E / "artifact_hashes.json").read_bytes())
    with pytest.raises(ValueError, match="qualification summary SHA-256 mismatch"):
        validate_frozen_inputs(SHORTLIST, POOL, RESULTS, copied_results, tampered)


def test_duplicate_dispatch_candidate_is_rejected():
    validated = _validate()
    broken = copy.deepcopy(validated)
    selection = broken["shortlist"]["workloads"]["W00"]["strategies"]["B7"]["budgets"]["4"]
    selection["candidate_ids"][1] = selection["candidate_ids"][0]
    with pytest.raises(ValueError, match="not unique/complete"):
        build_dispatch_manifest(broken, "W00", "B7", 4, DEFAULT_CORRECTNESS_SEEDS)


def test_incumbent_must_be_first():
    validated = _validate()
    broken = copy.deepcopy(validated)
    selection = broken["shortlist"]["workloads"]["W00"]["strategies"]["B7"]["budgets"]["4"]
    selection["candidate_ids"][0], selection["candidate_ids"][1] = (
        selection["candidate_ids"][1],
        selection["candidate_ids"][0],
    )
    with pytest.raises(ValueError, match="incumbent must be first"):
        build_dispatch_manifest(broken, "W00", "B7", 4, DEFAULT_CORRECTNESS_SEEDS)


def test_requires_exactly_three_unique_correctness_seeds():
    validated = _validate()
    with pytest.raises(ValueError, match="three unique"):
        build_dispatch_manifest(validated, "W00", "B7", 4, (0, 0, 1))


def test_incomplete_config_entity_is_rejected():
    validated = _validate()
    broken = copy.deepcopy(validated)
    del broken["candidate_pool"]["workloads"][0]["protected_original_incumbent"][
        "complete_config_entity"
    ]["entity"]
    with pytest.raises(ValueError, match="incomplete ConfigEntity"):
        build_dispatch_manifest(broken, "W00", "B7", 4, DEFAULT_CORRECTNESS_SEEDS)


def test_writer_refuses_to_overwrite_frozen_manifest(tmp_path):
    manifest = _prepare()
    output = write_dispatch_manifest(tmp_path, manifest)
    assert output.name == "dispatch_manifest.json"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_dispatch_manifest(tmp_path, manifest)
