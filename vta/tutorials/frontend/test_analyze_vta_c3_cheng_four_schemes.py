"""Tests for Cheng-style four-scheme resolution and fallback."""

import analyze_vta_c3_cheng_four_schemes as cheng


def candidate(mode, candidate_id, weight=True, acc=True):
    return {
        "candidate_id": candidate_id,
        "workload_id": "W",
        "family_id": "F",
        "residence_mode": mode,
        "complete_config_entity": {"entity": []},
        "applicability": {
            "full_weight_fits_sram": weight,
            "accumulator_fits_sram": acc,
        },
    }


def observed(candidate_id, dma, latency, correct=True):
    return {
        "candidate_id": candidate_id,
        "fpga_correct": correct,
        "total_dma_bytes": dma,
        "operator_latency_ms": latency if correct else None,
        "pure_instruction_ms": latency if correct else None,
    }


def test_resource_not_applicable_modes_fall_back_to_original():
    original = candidate("original", "o", weight=False)
    inp = candidate("input_stationary", "i", weight=False)
    rows = [original, inp]
    values = {"o": observed("o", 100, 10), "i": observed("i", 80, 9)}
    result = cheng.analyze_family(rows, values)
    by_mode = {row["mode"]: row for row in result["schemes"]}
    assert by_mode["weight_resident_barrier"]["status"] == "not_applicable_fallback_original"
    assert by_mode["input_weight_resident_barrier"]["candidate_id"] == "o"
    assert result["all_four_semantically_resolved"]


def test_compiler_missing_applicable_scheme_is_not_silently_fallback():
    original = candidate("original", "o", weight=True)
    result = cheng.analyze_family([original], {"o": observed("o", 100, 10)})
    by_mode = {row["mode"]: row for row in result["schemes"]}
    assert by_mode["weight_resident_barrier"]["status"].startswith("unavailable")
    assert not result["all_four_semantically_resolved"]


def test_minimum_access_need_not_be_fastest():
    rows = [candidate(mode, mode) for mode in cheng.MODES]
    values = {
        "original": observed("original", 100, 10),
        "input_stationary": observed("input_stationary", 70, 8),
        "weight_resident_barrier": observed("weight_resident_barrier", 50, 9),
        "input_weight_resident_barrier": observed(
            "input_weight_resident_barrier", 40, 12
        ),
    }
    result = cheng.analyze_family(rows, values)
    assert result["minimum_access_mode"] == "input_weight_resident_barrier"
    assert result["fastest_mode"] == "input_stationary"
    assert not result["minimum_access_is_fastest"]
