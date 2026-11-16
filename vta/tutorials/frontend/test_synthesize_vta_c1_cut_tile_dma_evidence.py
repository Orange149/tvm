import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("synthesize_vta_c1_cut_tile_dma_evidence.py")
SPEC = importlib.util.spec_from_file_location("c1_synthesis", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_real_frozen_evidence_synthesis_is_fail_closed_and_sealed():
    result = MODULE.synthesize(MODULE.DEFAULT_SOURCES)

    assert result["artifact_sha256"]
    assert result["deduplication_policy"]["new_board_experiment_run_by_this_synthesis"] is False
    assert result["evidence"]["placement_fixed_tile"]["fixed_tile_reuse_audit_pass"]
    assert result["evidence"]["policy_axis_at_fixed_cut_and_tile"][
        "vta_dma_invariant_across_all_modes"
    ]
    assert result["prospective_progress"]["topk_board"]["completed_independent_boots"] == 2
    assert result["prospective_progress"]["topk_board"]["formal_ranking_claim_ready"] is False
    assert result["prospective_progress"]["endpoint_service"][
        "completed_independent_boots"
    ] == 1
    assert result["prospective_progress"]["endpoint_service"][
        "formal_service_claim_ready"
    ] is False


def test_tile_and_cut_axes_are_kept_separate():
    result = MODULE.synthesize(MODULE.DEFAULT_SOURCES)
    tile_w = result["evidence"]["tile_axis_at_fixed_cut"]["axes"]["tile_w"]
    assert tile_w["median_ratios_to_incumbent"]["load_buffer_2d_calls"] == 4.5

    cuts = result["evidence"]["cut_axis_at_fixed_tile"]["transitions"]
    assert cuts[0]["observed_stage_dma_delta"]["load_buffer_2d_calls"] == 66
    assert cuts[1]["observed_stage_dma_delta"]["load_buffer_2d_calls"] == 0
