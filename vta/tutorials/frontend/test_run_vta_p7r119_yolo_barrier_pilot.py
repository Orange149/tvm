"""Contract-level tests for the P7R119 explicit-barrier pilot."""

from __future__ import annotations

import copy

from run_vta_p7r119_yolo_barrier_pilot import (
    IDENTITY_SCHEMA,
    LEGACY_ABLATIONS,
    MODES,
    build_candidates,
    candidate_identity,
    minimum_access_policy,
    select_structural_probe_rows,
)


def _entity(tile_co):
    return {
        "code_hash": None,
        "entity": [
            ["tile_b", "sp", [-1, 1]],
            ["tile_h", "sp", [-1, 26]],
            ["tile_w", "sp", [-1, 2]],
            ["tile_ci", "sp", [-1, 1]],
            ["tile_co", "sp", [-1, tile_co]],
            ["oc_nthread", "ot", 1],
            ["h_nthread", "ot", 1],
        ],
    }


def _legality(index, tile_co, legal=True):
    return {
        "debug": {"config_index": index},
        "knobs": {
            "tile_b": 1,
            "tile_h": 26,
            "tile_w": 2,
            "tile_ci": 1,
            "tile_co": tile_co,
            "oc_nthread": 1,
            "h_nthread": 1,
        },
        "all_four_modes_legal": legal,
    }


def _template():
    return {
        "identity": {
            "hardware_fingerprint": {"target": "axu5evb"},
            "template_name": "conv2d_packed_residency.vta",
            "schedule_version": "schedule-hash",
            "workload": ["conv2d_packed.vta", "placeholder"],
        }
    }


def test_mode_vocabulary_separates_formal_mechanisms_from_legacy_ablations():
    assert MODES == {
        "original": 0,
        "input_stationary": 1,
        "weight_resident_barrier": 4,
    }
    assert LEGACY_ABLATIONS == {
        "weight_stationary": 2,
        "paper_inspired_hybrid": 3,
    }
    assert not set(MODES) & set(LEGACY_ABLATIONS)


def test_structural_selection_is_exact_full_height_tile_co_sweep():
    rows = [_legality(7, 1), _legality(135, 2), _legality(391, 8)]
    rows.extend([_legality(999, 4), _legality(1000, 1, legal=False)])
    selected = select_structural_probe_rows(rows)
    assert [row["debug"]["config_index"] for row in selected] == [7, 135, 391]


def test_v2_identity_is_stable_under_debug_reindex_and_changes_with_mode():
    base = {
        **_template(),
        "complete_config_entity": _entity(1),
        "debug": {"config_index": 7},
    }
    original = candidate_identity(base, "original", 0)
    reindexed = copy.deepcopy(base)
    reindexed["debug"]["config_index"] = 7007
    assert candidate_identity(reindexed, "original", 0)["candidate_id"] == original[
        "candidate_id"
    ]
    barrier = candidate_identity(base, "weight_resident_barrier", 4)
    assert barrier["candidate_id"] != original["candidate_id"]
    assert barrier["identity"]["schema"] == IDENTITY_SCHEMA
    assert "config_index" not in barrier["identity"]


def test_candidate_pool_is_three_modes_per_same_tile_family():
    candidates = [_template()]
    domains = [
        {
            "workload_id": "Y02",
            "debug": {"config_index": index},
            "complete_config_entity": _entity(tile_co),
        }
        for index, tile_co in ((7, 1), (135, 2), (391, 8))
    ]
    candidates[0].update(workload_id="Y02")
    result = build_candidates(
        candidates,
        domains,
        [_legality(7, 1), _legality(135, 2), _legality(391, 8)],
    )
    assert len(result) == 9
    assert len({row["candidate_id"] for row in result}) == 9
    assert all(
        {row["public_mode"] for row in result if row["family_id"] == family}
        == set(MODES)
        for family in {row["family_id"] for row in result}
    )


def test_minimum_access_is_parameterized_choice_not_fake_hybrid():
    rows = []
    for mode, candidate_id, load_bytes, drains in (
        ("original", "o", 100, 0),
        ("input_stationary", "i", 70, 0),
        ("weight_resident_barrier", "w", 40, 3),
    ):
        rows.append(
            {
                "family_id": "Y02B00",
                "public_mode": mode,
                "candidate_id": candidate_id,
                "status": "ok",
                "dma_summary": {"load_bytes": load_bytes, "load_calls": 10},
                "sync": {"residency_drains": drains},
            }
        )
    policy = minimum_access_policy(rows)[0]
    assert set(policy["choices"]) == {
        "input_stationary",
        "weight_resident_barrier",
    }
    assert policy["lambda_sync_value"] is None
    assert policy["selection"].startswith("deferred")
    assert policy["is_schedule_candidate"] is False
    assert "hybrid" not in policy["choices"]
