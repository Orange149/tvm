"""Tests for the scoped P7R118 Y02 legality rules and frozen report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from analyze_vta_p7r118_yolo_legality import (
    DEFAULT_OUTPUT,
    confusion_matrix,
    four_mode_intersection_rule,
    original_legality_rule,
)


def knobs(tile_h, tile_w, tile_ci, tile_co):
    return {
        "tile_h": tile_h,
        "tile_w": tile_w,
        "tile_ci": tile_ci,
        "tile_co": tile_co,
    }


def test_original_rule_explains_padding_then_2d_dma_constraints():
    assert not original_legality_rule(knobs(1, 1, 2, 1), 26)
    assert not original_legality_rule(knobs(13, 1, 1, 2), 26)
    assert original_legality_rule(knobs(13, 1, 1, 1), 26)
    assert original_legality_rule(knobs(26, 13, 1, 8), 26)


def test_four_mode_rule_keeps_odd_outer_width_or_degenerate_layout():
    assert four_mode_intersection_rule(knobs(13, 2, 1, 1), 26)
    assert four_mode_intersection_rule(knobs(26, 2, 1, 8), 26)
    assert four_mode_intersection_rule(knobs(1, 13, 1, 1), 26)
    assert not four_mode_intersection_rule(knobs(26, 13, 1, 8), 26)


def test_confusion_matrix_and_length_guard():
    assert confusion_matrix([True, True, False, False], [True, False, True, False]) == {
        "tp": 1,
        "tn": 1,
        "fp": 1,
        "fn": 1,
        "total": 4,
        "accuracy": 0.5,
        "precision": 0.5,
        "recall": 0.5,
        "specificity": 0.5,
    }
    with pytest.raises(ValueError, match="length mismatch"):
        confusion_matrix([True], [])


def test_frozen_report_has_complete_coverage_and_exact_scoped_rules():
    report = json.loads((DEFAULT_OUTPUT / "report.json").read_text())
    population = report["population"]
    assert population["p7r115_hardware_eligible"] == 343
    assert population["original_lower_legal"] == 29
    assert population["four_mode_legal"] == 12
    assert report["replacement_decision"]["sufficient"]
    assert report["replacement_decision"]["other_yolo_geometry_scan"] == "not_triggered"
    assert report["rules"]["original_legality_v1"]["confusion"]["fp"] == 0
    assert report["rules"]["original_legality_v1"]["confusion"]["fn"] == 0
    assert report["rules"]["four_mode_intersection_v1"]["confusion"]["fp"] == 0
    assert report["rules"]["four_mode_intersection_v1"]["confusion"]["fn"] == 0
    assert not report["tophub_entity_or_cost_read"]


def test_frozen_artifact_hashes_and_row_counts():
    ledger = json.loads((DEFAULT_OUTPUT / "artifact_hashes.json").read_text())["artifacts"]
    assert all(
        hashlib.sha256((DEFAULT_OUTPUT / name).read_bytes()).hexdigest() == expected
        for name, expected in ledger.items()
    )
    assert sum(1 for line in (DEFAULT_OUTPUT / "original_scan.jsonl").open() if line.strip()) == 343
    assert sum(1 for line in (DEFAULT_OUTPUT / "four_mode_scan.jsonl").open() if line.strip()) == 29


if __name__ == "__main__":
    pytest.main([__file__])
