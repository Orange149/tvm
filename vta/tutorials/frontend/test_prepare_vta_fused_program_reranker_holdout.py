#!/usr/bin/env python3

from prepare_vta_fused_program_reranker_holdout import aggregate_residency, choose


def test_aggregate_residency_sums_occurrences():
    from analyze_vta_resnet50_fused_tir_occurrence_delta import RUNTIME_KEYS
    vectors = []
    for first, second in ((2, 3), (5, 7)):
        vector = {key: 0 for key in RUNTIME_KEYS}
        vector[RUNTIME_KEYS[0]] = first
        vector[RUNTIME_KEYS[1]] = second
        vectors.append({"residency_dma": vector})
    contract = {"graph_occurrences": vectors}
    got = aggregate_residency(contract)
    assert got[RUNTIME_KEYS[0]] == 7
    assert got[RUNTIME_KEYS[1]] == 10
    assert all(got[key] == 0 for key in RUNTIME_KEYS[2:])


def test_choose_uses_service_then_stable_id():
    programs = [
        {"program_id": "b", "service_score_byte_equivalent": 10},
        {"program_id": "a", "service_score_byte_equivalent": 10},
        {"program_id": "c", "service_score_byte_equivalent": 11},
    ]
    assert choose(programs)["program_id"] == "a"
