"""Tests for shared-DDR evidence ingestion by the Top-K freezer."""

import copy

import pytest

import run_cpu_vta_pipeline_v1_p7_profile as p7
from run_vta_shared_memory_topk import validate_multiboot_contention


def _multiboot_payload(single_summary):
    return p7.seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p7c_three_boot_reproducibility",
            "board_boot_ids": ["boot-a", "boot-b", "boot-c"],
            "source_sessions": [
                {
                    "artifact_sha256": single_summary[
                        "source_session_artifact_sha256"
                    ]
                }
            ],
            "ddr_specific_contrast": {
                "compute_heavy": {"ddr_contention_multiplier": 1.04},
                "dma_heavy": {"ddr_contention_multiplier": 1.06},
            },
            "exact_protocol_contention_profile_ready": True,
            "generalized_candidate_contention_surface_ready": False,
        }
    )


def test_multiboot_contention_is_recorded_but_not_applied_to_ranking():
    single = p7.load_artifact(
        p7.DEFAULT_P7C_SUMMARY,
        "cpu_vta_pipeline_v1_p7c_shared_ddr_qualification_summary",
    )
    result = validate_multiboot_contention(_multiboot_payload(single), single)

    assert result["independent_boot_count"] == 3
    assert result["formal_shared_ddr_parameter_admitted"]
    assert not result["generalized_candidate_contention_surface_ready"]
    assert "do not alter candidate ranking" in result["model_action"]


def test_multiboot_contention_rejects_hash_and_lineage_mismatch():
    single = p7.load_artifact(
        p7.DEFAULT_P7C_SUMMARY,
        "cpu_vta_pipeline_v1_p7c_shared_ddr_qualification_summary",
    )
    bad_hash = _multiboot_payload(single)
    bad_hash["artifact_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="artifact_sha256 mismatch"):
        validate_multiboot_contention(bad_hash, single)

    wrong_lineage = copy.deepcopy(single)
    wrong_lineage["source_session_artifact_sha256"] = "1" * 64
    with pytest.raises(ValueError, match="does not include"):
        validate_multiboot_contention(_multiboot_payload(single), wrong_lineage)
