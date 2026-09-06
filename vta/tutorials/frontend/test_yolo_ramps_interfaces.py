#!/usr/bin/env python3
"""Tests for frozen-prior and prospective-order YOLO search interfaces."""

import argparse

import search_yolov3_tiny_stage_splits as search


def args(**overrides):
    values = {
        "resource_model_json": "",
        "exclude_candidate_id_file": [],
        "candidate_prior_file": "",
        "cpu_thread_policy": "tuned",
        "candidate_count": 20,
        "queue_depth": 2,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_prior_order_exclusion_and_single_thread_policy(tmp_path):
    prior = tmp_path / "prior.txt"
    prior.write_text("c3\nc1\nc2\n", encoding="utf-8")
    excluded = tmp_path / "excluded.txt"
    excluded.write_text("c1\n", encoding="utf-8")
    candidates = [
        {"candidate_id": "c1"},
        {"candidate_id": "c2"},
        {"candidate_id": "c3"},
    ]
    selected = search.apply_candidate_selection(
        args(
            candidate_prior_file=str(prior),
            exclude_candidate_id_file=[str(excluded)],
            cpu_thread_policy="single",
        ),
        candidates,
    )
    assert [item["candidate_id"] for item in selected] == ["c3", "c2"]
    assert all(item["runtime_config"]["stage0_threads"] == 1 for item in selected)
    assert all(item["runtime_config"]["stage2_threads"] == 1 for item in selected)


def test_measurement_order_is_independent_from_build_order(tmp_path):
    order = tmp_path / "order.txt"
    order.write_text("c2\nc3\nc1\n", encoding="utf-8")
    rows = [{"candidate_id": "c1"}, {"candidate_id": "c2"}, {"candidate_id": "c3"}]
    measured = search.order_measurement_rows(rows, str(order))
    assert [item["candidate_id"] for item in measured] == ["c2", "c3", "c1"]
