from freeze_vta_c3_selector_v3_model_readiness import build_readiness


def test_readiness_never_relabels_operator_rows_or_single_boot_graph_evidence():
    v2_rows = [
        {"execution_task_id": "t1", "candidate_mode": "original"},
        {"execution_task_id": "t2", "candidate_mode": "weight_stationary_barrier"},
    ]
    graph_set = {"candidate_ids": ["a", "b"], "incumbent_aggregate_metrics": None}
    plan = {"ranking_model": "static_command_fallback_no_completion_aware_model",
            "performance_model_claim": False}
    development = {"distinct_boot_confirmation_count": 1,
                   "resnet18_classification": "FALL_BACK_TO_INCUMBENT"}
    result = build_readiness(v2_rows, graph_set, plan, development)
    assert result["status"] == "STATIC_FALLBACK_REQUIRED"
    assert result["admitted_training_row_count"] == 0
    assert result["model_contract"]["selector_kind"] == "static_fallback"
    assert result["source_inventory"]["selector_v2_operator_table"][
        "admitted_v3_graph_rows"] == 0
    assert result["source_inventory"]["yolo608_exact_graph_set"][
        "has_incumbent_aggregate_metrics"] is False
