import json
from pathlib import Path

import run_cpu_vta_pipeline_v1_p8_top20 as top20


ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "vta/tutorials/frontend/report_out/resource_aware_maxplus"


def test_all_frozen_candidates_resolve_to_four_compiled_topologies():
    packages = top20.discover_packages(
        REPORT / "v1_p7_top20_board_20260904/packages"
    )
    rows = json.loads(
        (REPORT / "v1_p7_top20_board_20260904/top20_board_summary.json").read_text()
    )["rows"]
    assert len(packages) == 4
    assert len(rows) == 20
    assert all(top20.candidate_signature(row["candidate_id"]) in packages for row in rows)


def test_candidate_manifest_applies_runtime_threads_and_affinity():
    packages = top20.discover_packages(
        REPORT / "v1_p7_top20_board_20260904/packages"
    )
    row = json.loads(
        (REPORT / "v1_p7_top20_board_20260904/top20_board_summary.json").read_text()
    )["rows"][19]
    _, manifest = packages[top20.candidate_signature(row["candidate_id"])]
    candidate = top20.manifest_for_candidate(manifest, row)
    assert candidate["candidate_id"] == row["candidate_id"]
    assert candidate["pipeline_stage_runtime_threads"]["stage2"] == 2
    assert candidate["pipeline_stage_runtime_threads"]["stage4"] == 4
    assert candidate["stage_cpu_affinity"]["pipeline"]["stage2"] == [0, 1]
    assert candidate["stage_cpu_affinity"]["pipeline"]["stage4"] == [0, 1, 2, 3]
    assert candidate["stage_cpu_affinity"]["pipeline"]["stage3"] == []


def test_balanced_mode_order_reverses_for_even_rank():
    assert top20.block_order(1) == ("B0", "B2", "B2", "B0")
    assert top20.block_order(2) == ("B2", "B0", "B0", "B2")


def test_rank_filter_parser():
    assert top20.parse_ranks("") is None
    assert top20.parse_ranks("20,5") == {5, 20}


def test_block_window_ii_uses_elapsed_steady_state_window():
    rows = [
        {"completion_index": 0, "completion_ms": 0.0},
        {"completion_index": 1, "completion_ms": 1.0},
        {"completion_index": 2, "completion_ms": 2.0},
        {"completion_index": 3, "completion_ms": 102.0},
    ]
    assert top20.block_window_ii_ms(rows) == 34.0


def test_residual_boundary_is_a_two_tensor_bundle():
    packages = top20.discover_packages(
        REPORT / "v1_p7_top20_board_20260904/packages"
    )
    rows = json.loads(
        (REPORT / "v1_p7_top20_board_20260904/top20_board_summary.json").read_text()
    )["rows"]
    row = rows[4]
    _, manifest = packages[top20.candidate_signature(row["candidate_id"])]
    contract = top20.boundary_contract(manifest, 1)
    assert contract["tensor_count"] == 2
    assert [tensor["bytes"] for tensor in contract["tensors"]] == [100352, 200704]
    assert contract["bytes"] == 301056


def test_multi_tensor_slot_range_validation_detects_overlap():
    boundary = {
        "edge_index": 0,
        "slot_id": 0,
        "physical_address": "0x1000",
        "boundary_bytes": 0x180,
        "physical_addresses": ["0x1000", "0x2000"],
        "tensor_bytes": [0x100, 0x80],
    }
    assert top20.block_ranges_non_overlapping([{"p8_boundaries": [boundary]}])
    boundary["physical_addresses"][1] = "0x1080"
    assert not top20.block_ranges_non_overlapping([{"p8_boundaries": [boundary]}])


def test_candidate_estimate_gives_each_block_equal_weight():
    result = {
        "mode_summaries": {
            "B0": {"pipeline_ii_ms_median": 1.0, "block_ii_ms_medians": [80.0, 100.0]},
            "B2": {"pipeline_ii_ms_median": 1.0, "block_ii_ms_medians": [80.0, 80.0]},
        },
        "effects": {},
    }
    top20.refresh_candidate_estimates(result)
    assert result["mode_summaries"]["B0"]["pipeline_ii_ms_median"] == 90.0
    assert result["mode_summaries"]["B2"]["pipeline_ii_ms_median"] == 80.0
    assert result["effects"]["fps_increase_percent"] == 12.5
