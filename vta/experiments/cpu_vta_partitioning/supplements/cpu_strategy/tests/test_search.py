"""Exact resource-state DP versus independent exhaustive scalar scoring."""
from copy import deepcopy

import pytest

from cpu_strategy.domain import components, configuration, matrix
from cpu_strategy.search import enumerate_reference, rank_dp


def costs():
    rows = {}
    for item in components():
        stage = item["stage"]
        base = 180. if stage["begin"] == 0 else 120.
        if stage["device"] == "vta":
            rows[stage["signature"]] = dict(service_ms=66., run_core_ms=14., boundary_core_ms=.8)
        else:
            rows[stage["signature"]] = dict(service_ms=base/stage["threads"],
                run_core_ms=base*(1+.02*(stage["threads"]-1)), boundary_core_ms=.7)
    return {"source": "fresh_isolated_board_calibration", "costs": rows}


@pytest.mark.parametrize("k", [1, 5, 20, 32, 99])
def test_dp_matches_every_terminal_of_full_frozen_domain(k):
    document = costs()
    result = rank_dp(document, k)
    assert [(r["score_ms"],r["id"]) for r in result["rankings"]] == enumerate_reference(document,k)
    assert result["candidate_count"] == 32 and result["scalar_enumeration_checked"]
    assert result["state_widths"][-1] == 32
    assert not result["intermediate_pruning"] and not result["ranking_uses_pipeline_feedback"]


def test_overlap_work_and_shared_host_boundary_are_counted_once():
    document = costs()
    for item in components():
        stage = item["stage"]
        document["costs"][stage["signature"]] = dict(service_ms=3. if stage["device"]=="vta" else 1.,
            run_core_ms=4. if stage["device"]=="vta" else 8., boundary_core_ms=2.)
    scores = {r["id"]:r["score_ms"] for r in rank_dp(document,32)["rankings"]}
    plans = matrix()
    assert scores[plans["interventions"]["overlap_low"]] == 10.
    assert scores[plans["interventions"]["separate_low"]] == 6.5


def test_dp_retains_a_worse_prefix_that_wins_after_core_overlap():
    document = costs()
    for row in document["costs"].values():
        row.update(service_ms=100.,run_core_ms=100.,boundary_core_ms=0.)
    settings = {"cpu_00_02_t2_a0-1":(1.,10.), "cpu_00_02_t2_a2-3":(6.,2.),
                "cpu_18_20_t2_a0-1":(1.,14.), "cpu_18_20_t2_a2-3":(30.,30.),
                "vta_03_17_t1_a0-1-2-3":(1.,0.)}
    for name,(service,work) in settings.items():
        document["costs"][name] = dict(service_ms=service,run_core_ms=work,boundary_core_ms=0.)
    best = rank_dp(document,1)["rankings"][0]
    assert best["id"] == configuration((2,2),((2,3),(0,1)))["id"]
    assert best["score_ms"] == 7.


def test_ranking_is_frozen_without_pipeline_feedback_and_has_stable_ties():
    document = costs()
    original = deepcopy(document)
    a = rank_dp(document,32)
    assert document == original and rank_dp(document,32) == a
    for x,y in zip(a["rankings"],a["rankings"][1:]):
        assert (x["score_ms"],x["id"]) <= (y["score_ms"],y["id"])


@pytest.mark.parametrize("k", [0,-1,True,1.5])
def test_invalid_topk_rejected(k):
    with pytest.raises(ValueError):rank_dp(costs(),k)


def test_missing_stale_and_nonfinite_costs_rejected():
    doc = costs();doc["source"] = "historical"
    with pytest.raises(ValueError,match="fresh"):rank_dp(doc)
    doc = costs();doc["costs"].pop("cpu_00_02_t1_a0")
    with pytest.raises(ValueError,match="missing"):rank_dp(doc)
    doc = costs();doc["costs"]["cpu_00_02_t1_a0"]["run_core_ms"] = float("nan")
    with pytest.raises(ValueError,match="nonfinite"):rank_dp(doc)
