"""Tests for the P4g mode-4 weight-barrier pool generator."""

import json
from pathlib import Path

import pytest
import vta

from generate_vta_weight_barrier_candidates import (
    IMPLEMENTATION_MODE,
    PUBLIC_MODE,
    barrier_identity_record,
    create_task,
    generate_candidates,
    lower_barrier_candidate,
    make_schedule_version,
    schedule_sources,
    validate_p3e_run,
    unique_tensor_bytes,
    validate_p4b_inputs,
)


ROOT = Path(__file__).resolve().parents[3]
P4B = (
    Path(__file__).resolve().parent
    / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/04_dma_command_signatures"
    / "20260910_p4b_local_residency_pool_run01"
)
P3E = (
    Path(__file__).resolve().parent
    / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/03_residency_schedules"
    / "20260910_p3e_weight_barrier_legalizer_run01"
)


def _inputs():
    return validate_p4b_inputs(P4B / "candidate_pool.json", P4B / "results.jsonl")


def test_schedule_version_explicitly_binds_three_required_sources():
    sources = schedule_sources(ROOT)
    version = make_schedule_version(sources)
    for name in ("vta_conv2d.py", "transform.py", "coproc_sync.cc"):
        assert "{}={}".format(name, sources[name]) in version


def test_p3e_positive_proof_and_source_hashes_are_frozen_inputs():
    proof = validate_p3e_run(P3E, schedule_sources(ROOT))
    assert proof["decision"] == "go_local_compiler_explicit_drain_proof"
    assert len(proof["artifact_hashes_sha256"]) == 64


def test_identity_is_stable_across_debug_index_and_new_mode_namespace():
    pool, _, _ = _inputs()
    mapped = pool["workloads"][0]["mapping"]["records"][0]
    sources = schedule_sources(ROOT)
    version = make_schedule_version(sources)
    one = barrier_identity_record(
        pool["hardware_fingerprint"],
        version,
        sources,
        pool["workloads"][0]["workload"],
        mapped["complete_config_entity"],
        1,
    )
    two = barrier_identity_record(
        pool["hardware_fingerprint"],
        version,
        sources,
        pool["workloads"][0]["workload"],
        mapped["complete_config_entity"],
        999,
    )
    assert one["candidate_id"] == two["candidate_id"]
    assert one["identity"]["residence_mode"] == PUBLIC_MODE
    assert one["identity"]["implementation_mode"] == IMPLEMENTATION_MODE


def test_p4b_mapping_contains_exactly_sixty_unique_tiles():
    pool, rows, _ = _inputs()
    assert sum(len(workload["mapping"]["records"]) for workload in pool["workloads"]) == 60
    assert len({row["candidate_id"] for row in rows}) == 250


@pytest.mark.parametrize("workload_id,config_index", [("W00", 252), ("W02", 177), ("W09", 83)])
def test_verified_mode4_representatives_have_sync_and_legal_dependencies(
    workload_id, config_index
):
    pool, _, _ = _inputs()
    workload_entry = next(row for row in pool["workloads"] if row["workload_id"] == workload_id)
    mapped = next(
        row for row in workload_entry["mapping"]["records"]
        if row["mapped_config_index"] == config_index
    )
    result = lower_barrier_candidate(
        create_task(workload_entry["workload"], vta.get_env()),
        config_index,
        mapped["complete_config_entity"],
        vta.get_env(),
        unique_tensor_bytes(workload_entry["workload"]),
    )
    assert result["status"] == "ok"
    assert result["sync"]["static_callsites"] >= 2
    assert result["sync"]["expanded_executions"] == sum(
        result["sync"]["execution_multiplicities"]
    )
    assert result["dependency_audit"]["valid"] is True
    assert result["dependency_audit"]["summary"]["push_pop_balanced"] is True
    assert result["dependency_audit"]["summary"]["forbidden_1_3_present"] is False
    assert result["transfer_signature"] is not None


def test_scan_keeps_stable_unique_ids_and_classifies_early_failures():
    pool, rows, by_id = _inputs()
    output_pool, results = generate_candidates(
        pool, by_id, schedule_sources(ROOT), vta.get_env(), limit=3
    )
    assert output_pool["candidate_count"] == 3
    assert len({row["candidate_id"] for row in results}) == 3
    assert not {row["candidate_id"] for row in results}.intersection(
        row["candidate_id"] for row in rows
    )
    for row in results:
        if row["status"] != "ok":
            assert row["failure"]["category"] == "lower"
            assert row["failure"]["subcategory"]


def test_tampered_p4b_pool_is_rejected_by_frozen_hash(tmp_path):
    pool = json.loads((P4B / "candidate_pool.json").read_text())
    pool["mapping_diagnostics"]["mapped_unique_tile_entities"] = 59
    tampered = tmp_path / "candidate_pool.json"
    tampered.write_text(json.dumps(pool))
    (tmp_path / "results.jsonl").write_bytes((P4B / "results.jsonl").read_bytes())
    (tmp_path / "artifact_hashes.json").write_bytes((P4B / "artifact_hashes.json").read_bytes())
    with pytest.raises(ValueError, match="candidate pool hash mismatch"):
        validate_p4b_inputs(tampered, tmp_path / "results.jsonl")
