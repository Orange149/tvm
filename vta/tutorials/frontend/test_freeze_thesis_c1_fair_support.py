import copy

import pytest

from freeze_thesis_c1_fair_support import RANDOM_SEEDS, SEMANTIC_DIMENSIONS, freeze
from shared_memory_topk import pareto_layers


def candidate(topology, variant, score, islands, threads, reasonable_a=False, **metrics):
    row = {
        "candidate_id": "{}::{}".format(topology, variant),
        "topology_id": topology,
        "m1_score_ms": score,
        "vta_islands": islands,
        "stage_devices": ["cpu", "vta", "cpu"],
        "thread_vector": threads,
        "static_legal": True,
        "realized_policy": "ordinary_copy",
        "reasonable_a": reasonable_a,
        "framework_materialization_bytes": 1000 + int(score),
        "vta_dma_calls": 100 + int(score),
        "vta_internal_dma_bytes": 10000 + int(score),
        "additional_slot_bytes": 0,
    }
    row.update(metrics)
    return row


def payload():
    rows = []
    for index in range(10):
        topology = "topology_{:02d}".format(index)
        islands = index % 3 + 1
        best_threads = [1, 1, 1] if index == 0 else [2 + index % 3, 1, 2]
        rows.append(
            candidate(
                topology,
                "best",
                10.0 + index * 0.02,
                islands,
                best_threads,
                reasonable_a=index == 7,
                framework_materialization_bytes=5000 - index * 100,
            )
        )
        prescribed = [4, 1, 4] if best_threads == [1, 1, 1] else [1, 1, 1]
        rows.append(
            candidate(topology, "prescribed", 11.0 + index, islands, prescribed)
        )
        rows.append(candidate(topology, "other", 20.0 + index, islands, [3, 1, 3]))
    return {
        "schema": "thesis_c1_fair_support_input_v1",
        "new_performance_labels_read": False,
        "candidates": rows,
    }


def test_freezes_exact_support_and_all_orders():
    result = freeze(payload())
    assert result["support_size"] == 16
    assert result["topology_count"] == 8
    support = {row["candidate_id"] for row in result["support"]}
    orders = [
        result["orders"]["O0_m1"],
        result["orders"]["O1_topology_first"],
        result["orders"]["O2_semantic_band"],
        *result["orders"]["O3_random"].values(),
    ]
    assert len(result["orders"]["O3_random"]) == 20
    assert list(map(int, result["orders"]["O3_random"])) == list(RANDOM_SEEDS)
    assert all(len(order) == 16 and set(order) == support for order in orders)
    assert len({order[0] for order in orders}) == 1
    assert result["measurement_rotation_offsets"] == [0, 5, 10]
    assert len({tuple(value) for value in result["measurement_orders"].values()}) == 3


def test_topology_selection_includes_reasonable_a_and_island_categories():
    result = freeze(payload())
    assert any(row["reasonable_a"] for row in result["topologies"])
    assert {row["vta_islands"] for row in result["topologies"]} == {1, 2, 3}


def test_duplicate_best_all_cpu_one_uses_all_cpu_four_variant():
    result = freeze(payload())
    first = next(row for row in result["topologies"] if row["topology_id"] == "topology_00")
    assert first["best_candidate_id"].endswith("::best")
    assert first["second_candidate_id"].endswith("::prescribed")
    second = next(row for row in result["support"] if row["candidate_id"] == first["second_candidate_id"])
    assert second["thread_vector"] == [4, 1, 4]


def test_unknown_semantic_band_falls_back_without_dropping_candidate():
    value = payload()
    value["candidates"][3]["vta_dma_calls"] = None
    result = freeze(value)
    assert result["semantic_audit"]["fallback_to_m1"]
    assert result["semantic_audit"]["unknown_candidate_ids"]
    assert len(result["orders"]["O2_semantic_band"]) == 16


def test_missing_prescribed_variant_fails_closed():
    value = payload()
    value["candidates"] = [
        row
        for row in value["candidates"]
        if not (row["topology_id"] == "topology_01" and row["thread_vector"] == [1, 1, 1])
    ]
    with pytest.raises(ValueError, match="prescribed second thread vector"):
        freeze(value)


def test_new_performance_label_or_shared_policy_is_rejected():
    value = payload()
    value["candidates"][0]["observed_ii_ms"] = 1.0
    with pytest.raises(ValueError, match="forbidden new performance label"):
        freeze(value)
    value = payload()
    value["candidates"][0]["realized_policy"] = "external_k2_shared"
    with pytest.raises(ValueError, match="ordinary-copy"):
        freeze(value)


def test_candidate_id_breaks_pareto_ties():
    base = {field: 1 for field in SEMANTIC_DIMENSIONS}
    rows = [
        {**base, "candidate_id": "b", "topology_id": "same", "m1_score_ms": 1.0},
        {**base, "candidate_id": "a", "topology_id": "same", "m1_score_ms": 1.0},
    ]
    assert [row["candidate_id"] for row in pareto_layers(rows, SEMANTIC_DIMENSIONS)[0]] == ["a", "b"]


def test_input_is_not_mutated_and_result_is_deterministic():
    value = payload()
    original = copy.deepcopy(value)
    first = freeze(value)
    second = freeze(value)
    assert value == original
    assert first == second
