import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("freeze_vta_c1_unified_structured_order.py")
SPEC = importlib.util.spec_from_file_location("c1_unified_order", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_real_domain_has_exact_unified_features_and_stays_diagnostic():
    result = MODULE.analyze(
        MODULE.DEFAULT_TOPOLOGIES,
        MODULE.DEFAULT_EDGE_CONTEXTS,
        MODULE.DEFAULT_CERTIFICATES,
        MODULE.DEFAULT_LEGACY_CONTRACT,
        MODULE.DEFAULT_FUSION_ESTIMATES,
        MODULE.DEFAULT_FUSION_CONTRACT,
        0.02,
        5,
    )
    assert result["domain"]["exact_unified_features"] == 4623
    assert result["domain"]["inexact_unified_features"] == 0
    assert result["label_free_inputs_only"]
    assert not result["board_artifacts_read"]
    assert not result["eligible_for_ongoing_s0_s1_s2_claim"]
    assert not result["comparison_to_frozen_s2"][
        "existing_boots_reclassified_as_prospective_s3_evidence"
    ]
    assert result["selected_configuration_ids"][0] == result[
        "comparison_to_frozen_s2"
    ]["legacy_configuration_ids"][0]


def test_fusion_summary_fails_closed_when_segment_is_missing():
    topology = {"path": [["cpu:00:00", 1], ["vta:01:01", 1]]}
    result = MODULE.fusion_summary(topology, {})
    assert not result["exact_reuse"]
    assert result["missing_segments"] == ["vta:01:01"]
