"""Focused tests for the P7R112 Rieber-2022 sparse-pool replay."""

from analyze_vta_p7r112_rieber2022_reproduction import (
    KNOBS,
    balanced_maxmin_e0,
    calls_to_k_valid,
    coordinate_system,
    observed_neighbor_map,
    rieber_presample,
    same_task_manhattan_one,
    validity_biased_proxy_order,
)


def candidate(candidate_id, tile_h, tile_w, status="ok", mode="original"):
    values = {
        "tile_b": 1,
        "tile_h": tile_h,
        "tile_w": tile_w,
        "tile_ci": 1,
        "tile_co": 4,
        "oc_nthread": 1,
        "h_nthread": 1,
    }
    return {
        "candidate_id": candidate_id,
        "workload_id": "W00",
        "residence_mode": mode,
        "status": status,
        "_source_ordinal": len(candidate_id),
        "identity": {
            "complete_config_entity": {
                "code_hash": None,
                "entity": [
                    [name, "sp" if name.startswith("tile_") else "ot", [-1, values[name]] if name.startswith("tile_") else values[name]]
                    for name in KNOBS
                ],
            }
        },
    }


def test_manhattan_one_is_same_task_and_one_knob_rank_only():
    center = candidate("center", 2, 2)
    h_neighbor = candidate("h", 4, 2)
    diagonal = candidate("diagonal", 4, 4)
    other_mode = candidate("mode", 2, 2, mode="input_stationary")
    rows = [center, h_neighbor, diagonal, other_mode]
    _, coordinates = coordinate_system(rows)
    assert same_task_manhattan_one(center, h_neighbor, coordinates)
    assert not same_task_manhattan_one(center, diagonal, coordinates)
    assert not same_task_manhattan_one(center, other_mode, coordinates)


def test_valid_presample_expands_observed_manhattan_neighbor():
    center = candidate("center", 2, 2, "ok")
    neighbor = candidate("neighbor", 4, 2, "failed")
    far = candidate("far", 4, 4, "ok")
    rows = [center, neighbor, far]
    _, coordinates = coordinate_system(rows)
    replay = rieber_presample(
        rows, coordinates, n_samples=2, n_parallel=1, seed=7, initial_ids=["center"]
    )
    assert [row["candidate_id"] for row in replay["order"]] == ["center", "neighbor"]


def test_presample_has_equal_unique_gross_calls_and_records_sparse_fill():
    rows = [
        candidate("a", 1, 1, "ok"),
        candidate("b", 2, 2, "failed"),
        candidate("c", 4, 4, "ok"),
        candidate("d", 8, 8, "failed"),
    ]
    _, coordinates = coordinate_system(rows)
    replay = rieber_presample(
        rows, coordinates, n_samples=4, n_parallel=2, seed=3, initial_ids=["a"]
    )
    ids = [row["candidate_id"] for row in replay["order"]]
    assert len(ids) == len(set(ids)) == 4
    assert replay["sparse_frontier_random_fill_count"] > 0


def test_balanced_e0_uses_both_classes_and_maximizes_spread():
    rows = [
        candidate("v0", 1, 1, "ok"),
        candidate("v1", 2, 1, "ok"),
        candidate("v2", 4, 1, "ok"),
        candidate("i0", 1, 2, "failed"),
        candidate("i1", 1, 4, "failed"),
        candidate("i2", 1, 8, "failed"),
    ]
    _, coordinates = coordinate_system(rows)
    e0 = balanced_maxmin_e0(rows, 4, coordinates)
    assert e0["achieved_size"] == 4
    assert e0["valid"] == e0["invalid"] == 2
    ids = {row["candidate_id"] for row in e0["order"]}
    assert {"v0", "v2"}.issubset(ids)
    assert {"i0", "i2"}.issubset(ids)


def test_validity_bias_never_ranks_known_invalid_before_known_valid():
    rows = [
        candidate("valid-a", 1, 1, "ok"),
        candidate("invalid", 2, 1, "failed"),
        candidate("valid-b", 4, 1, "ok"),
    ]
    order = validity_biased_proxy_order(rows, seed=11)
    assert order[-1]["candidate_id"] == "invalid"


def test_calls_to_k_valid_counts_gross_invalid_attempts():
    rows = [
        candidate("i0", 1, 1, "failed"),
        candidate("v0", 2, 1, "ok"),
        candidate("i1", 4, 1, "failed"),
        candidate("v1", 8, 1, "ok"),
    ]
    assert calls_to_k_valid(rows, 2) == 4
    assert calls_to_k_valid(rows, 3) is None


def test_observed_neighbor_map_does_not_invent_cross_mode_edges():
    left = candidate("left", 1, 1, mode="original")
    right = candidate("right", 1, 1, mode="input_stationary")
    rows = [left, right]
    _, coordinates = coordinate_system(rows)
    neighbours = observed_neighbor_map(rows, coordinates)
    assert neighbours == {"left": set(), "right": set()}
