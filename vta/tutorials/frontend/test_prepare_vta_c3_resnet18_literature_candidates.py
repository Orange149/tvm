"""Tests for label-blind ResNet18 literature-study preregistration."""

from __future__ import annotations

import hashlib

import vta

import prepare_vta_c3_resnet18_literature_candidates as prepare


def _entity(value):
    return {
        "entity": [
            ["tile_b", "sp", [-1, 1]],
            ["tile_h", "sp", [-1, value]],
            ["tile_w", "sp", [-1, 1]],
            ["tile_ci", "sp", [-1, 1]],
            ["tile_co", "sp", [-1, 1]],
            ["oc_nthread", "ot", 1],
            ["h_nthread", "ot", 1],
        ]
    }


def test_max_min_is_deterministic_unique_and_label_free():
    entities = [_entity(index + 1) for index in range(10)]
    first = prepare.max_min_select(entities, 4, "TEST")
    second = prepare.max_min_select(entities, 4, "TEST")
    assert first == second
    assert len({prepare.digest_value(row) for row in first}) == 4
    assert all("latency" not in str(row).lower() for row in first)


def test_paper_weight_applicability_uses_full_layer_bytes():
    env = vta.get_env()
    knobs = prepare.knobs(_entity(1))
    h1 = prepare.analytic_applicability(prepare.GEOMETRIES["R18-H1"], knobs, env)
    h2 = prepare.analytic_applicability(prepare.GEOMETRIES["R18-H2"], knobs, env)
    h3 = prepare.analytic_applicability(prepare.GEOMETRIES["R18-H3"], knobs, env)
    assert h1["full_weight_bytes"] == 36864 and h1["full_weight_fits_sram"]
    assert h2["full_weight_bytes"] == 589824 and not h2["full_weight_fits_sram"]
    assert h3["full_weight_bytes"] == 131072 and h3["full_weight_fits_sram"]


def test_real_h3_space_and_four_mode_pool_cardinality():
    env = vta.get_env()
    fingerprint = prepare.hardware_fingerprint(env)
    version = hashlib.sha256(b"test-schedule").hexdigest()
    complete, selected = prepare.build_geometry(
        "R18-H3", prepare.GEOMETRIES["R18-H3"], 24, env, version, fingerprint
    )
    assert complete["complete_config_space_count"] == 480
    assert len(complete["candidates"]) == 480
    assert len(selected["candidates"]) == 96
    assert {row["residence_mode"] for row in selected["candidates"]} == set(prepare.MODES)
    assert len({row["family_id"] for row in selected["candidates"]}) == 24
    assert all(row["lowered_tir_sha256"] is None for row in selected["candidates"])

