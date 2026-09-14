"""Tests for the P7R115 label-isolated YOLO confirmation contract."""

from __future__ import annotations

from collections import Counter

import pytest

from c3_candidate_identity import canonical_json_bytes
from prepare_vta_p7r115_yolo_confirmation import (
    EXPOSURE_SOURCES,
    SELECTED_CONVS,
    YOLO_SOURCE,
    build_contract,
    canonical_sha256,
    exact_tophub_workload_audit,
    exposure_audit,
    parse_yolo_convs,
)


@pytest.fixture(scope="module")
def pilot():
    return build_contract(3)


def test_selected_geometries_are_literal_real_yolo_layers():
    rows = parse_yolo_convs(YOLO_SOURCE)
    assert set(SELECTED_CONVS.values()) <= set(rows)
    assert (rows["conv2"]["ci"], rows["conv2"]["co"], rows["conv2"]["height"], rows["conv2"]["kernel"]) == (16, 32, 208, 3)
    assert (rows["conv13"]["ci"], rows["conv13"]["co"], rows["conv13"]["height"], rows["conv13"]["kernel"]) == (1024, 256, 13, 1)
    assert (rows["conv21"]["ci"], rows["conv21"]["co"], rows["conv21"]["height"], rows["conv21"]["kernel"]) == (384, 256, 26, 3)


def test_exposure_audit_covers_all_frozen_namespaces():
    audit = exposure_audit(EXPOSURE_SOURCES)
    assert {row["label_namespace"] for row in audit} == {
        "W00_W09", "P7Q", "E00", "E01", "E02", "E03"
    }
    assert next(row for row in audit if row["label_namespace"] == "W00_W09")["unique_geometry_count"] == 10
    assert next(row for row in audit if row["label_namespace"] == "P7Q")["unique_geometry_count"] == 4


def test_pilot_is_exactly_balanced_and_label_disjoint(pilot):
    contract, domain, candidates = pilot
    assert contract["study_tier"] == "p7r115_pilot"
    assert not contract["sufficient_for_final_ccf_b_search_claim"]
    assert len(domain) == 800 + 560 + 2560
    assert len(candidates) == 36
    assert Counter(row["workload_id"] for row in candidates) == {"Y00": 12, "Y01": 12, "Y02": 12}
    assert Counter(row["residence_mode"] for row in candidates) == {
        "original": 9,
        "input_stationary": 9,
        "weight_stationary": 9,
        "paper_inspired_hybrid": 9,
    }
    assert all(not item["exposed_performance_label_collision"] for item in contract["geometries"])


def test_complete_mode_domains_are_identical_and_indices_are_debug_only(pilot):
    contract, _, candidates = pilot
    for geometry in contract["geometries"]:
        domains = geometry["config_domain"]
        assert len({row["ordered_domain_sha256"] for row in domains.values()}) == 1
        assert len({row["set_domain_sha256"] for row in domains.values()}) == 1
    for row in candidates:
        assert "config_index" not in canonical_json_bytes(row["identity"]).decode()
        assert "config_index" in row["debug"]


def test_pool_is_committed_before_boolean_tophub_audit(pilot):
    contract, _, candidates = pilot
    assert contract["candidate_pool_commitment_sha256"] == canonical_sha256(candidates)
    audit = contract["post_commit_exact_tophub_audit"]
    assert audit["audit_phase"] == "after_candidate_pool_commitment"
    assert audit["pool_commitment_sha256"] == contract["candidate_pool_commitment_sha256"]
    serialized = canonical_json_bytes(audit).decode()
    assert "config_index" not in serialized
    assert "complete_config_entity" not in serialized
    for row in audit["workloads"].values():
        assert not ({"cost", "latency", "config", "config_index"} & set(row))


def test_missing_tophub_package_fails_closed(monkeypatch, tmp_path):
    import tvm.autotvm.tophub as tophub

    monkeypatch.setattr(tophub, "AUTOTVM_TOPHUB_ROOT_PATH", tmp_path)
    result = exact_tophub_workload_audit({"YXX": ["conv2d_packed.vta"]}, "abc")
    assert result["status"] == "closed_no_local_package"
    assert not result["all_exact_hits"]
    assert not result["workloads"]["YXX"]["exact_hit"]
    assert not result["network_download_attempted"]


def test_scale_up_contract_freezes_n8_interface(pilot):
    contract, _, _ = pilot
    scale = contract["scale_up_contract"]
    assert "N=8" in scale["minimum_next_confirmation"]
    assert "--families-per-geometry 8" in scale["command"]
    assert "exact prefix" in scale["prefix_stability"]


if __name__ == "__main__":
    pytest.main([__file__])
