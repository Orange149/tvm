from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from cpu_policies import pilot_search as pilot
from cpu_policies.pilot_common import DEFAULT_BUNDLE, load_legacy


def test_two_domain_top20_merge_uses_score_then_complete_identity():
    sources = {
        "shared4": {"rankings": [{"id": f"a{i:02}", "score_ms": float(i)}
                                 for i in range(20)]},
        "reserved3": {"rankings": [{"id": f"b{i:02}", "score_ms": float(i)}
                                   for i in range(20)]},
    }
    rows = pilot.merge_rankings(sources)
    assert len(rows) == 20
    assert [r["id"] for r in rows[:4]] == ["a00", "b00", "a01", "b01"]
    assert rows[0]["resource_policy"] == "shared4"
    assert rows[1]["resource_policy"] == "reserved3"
    assert "resource_policy" not in sources["shared4"]["rankings"][0]


def test_merge_rejects_corrupt_duplicate_unsorted_and_nonfinite_scores():
    with pytest.raises(ValueError, match="differs"):
        pilot.merge_rankings({"a": {"rankings": [{"id": "x", "score_ms": 1.}]},
                              "b": {"rankings": [{"id": "x", "score_ms": 2.}]}})
    with pytest.raises(ValueError, match="unsorted"):
        pilot.merge_rankings({"a": {"rankings": [{"id": "x", "score_ms": 2.},
                                                 {"id": "y", "score_ms": 1.}]}})
    with pytest.raises(ValueError, match="invalid static"):
        pilot.merge_rankings({"a": {"rankings": [{"id": "x", "score_ms": float("nan")}]}})


@pytest.fixture
def legacy():
    load_legacy()
    from cpu_strategy.policy import domain, search
    return domain, search


def fixture_programs(domain):
    rows = {}
    for a, b in domain.CUTS.values():
        for device, lo, hi in (("cpu", 0, a), ("vta", a, b), ("cpu", b, 21)):
            row = domain.stage(device, lo, hi, 1, (0,))
            rows[row["program"]] = {k: row[k] for k in ("device", "begin", "end")}
    return rows


def fixture_cost(configuration):
    s = configuration["stages"][0]
    if s["device"] == "cpu":
        work = 180. if s["begin"] == 0 else 3. if s["begin"] == 20 else 120.
        wall = work / s["threads"]
    else:
        work = 14.
        wall = 60. if s["mask"] == [3] else 66.
    return {"configuration": configuration,
            "cost": {"service_ms": wall + .75, "run_core_ms": work,
                     "boundary_core_ms": .45, "set_ms": .5, "run_ms": wall,
                     "get_ms": .25, "set_core_ms": .3, "get_core_ms": .15}}


def test_full_frozen_program_inventory_has_514_real_distinct_requirements(legacy):
    programs = json.loads((DEFAULT_BUNDLE / "plans/programs.json").read_text())
    rows = pilot.requirements(programs)
    cpu = [r for r in rows.values() if r["stages"][0]["device"] == "cpu"]
    vta = [r for r in rows.values() if r["stages"][0]["device"] == "vta"]
    assert len(rows) == 514 and len(cpu) == 340 and len(vta) == 174
    assert all(r["affinity_mode"] == "pinned" for r in cpu)
    assert all(r["affinity_mode"] == "group" for r in vta)
    assert {tuple(r["host_mask"]) for r in vta} == {(0, 1, 2, 3), (3,)}
    assert all(r["host_mask"] == r["control_mask"] for r in vta)


def test_small_complete_graph_ranks_both_policies_for_all_methods(legacy):
    domain, _ = legacy
    programs = fixture_programs(domain)
    costs = {key: fixture_cost(cfg) for key, cfg in pilot.requirements(programs).items()}
    result = pilot.rank_all(programs, costs)
    assert set(result["methods"]) == set(pilot.METHODS)
    assert result["all_methods_search_both_host_policies"]
    assert not result["ranking_uses_pipeline_feedback"]
    for method, entry in result["methods"].items():
        assert len(entry["rankings"]) == 20
        assert set(entry["per_environment"]) == {"shared4", "reserved3"}
        assert entry["rankings"] == sorted(entry["rankings"], key=lambda x: (x["score_ms"], x["id"]))
        for name, environment in entry["per_environment"].items():
            assert environment["verification"]["passed"]
            for row in environment["rankings"]:
                domain.validate_candidate({k: row[k] for k in pilot.CONFIGURATION_FIELDS}, programs)
                assert row["execution_mode"] == ("off" if method == "prefix" else "resource_group")
                if name == "reserved3":
                    assert row["host_mask"] == [3] and row["control_mask"] == [3]
                    assert all(3 not in s["mask"] for s in row["stages"] if s["device"] == "cpu")
    pool = pilot.select_pool(result)
    assert 20 <= len(pool) <= 30
    for cfg in pool.values():
        assert set(cfg) == set(pilot.CONFIGURATION_FIELDS)
        domain.validate_candidate(cfg, programs)


def test_missing_cost_and_independent_enumeration_mismatch_fail(legacy, monkeypatch):
    domain, search = legacy
    programs = fixture_programs(domain)
    costs = {key: fixture_cost(cfg) for key, cfg in pilot.requirements(programs).items()}
    incomplete = dict(costs)
    incomplete.pop(next(iter(incomplete)))
    with pytest.raises(ValueError, match="missing 1"):
        pilot.rank_all(programs, incomplete)
    original = search.enumerate_reference

    def altered(*args, **kwargs):
        result = original(*args, **kwargs)
        result["rankings"][0]["score_ms"] += .001
        return result

    monkeypatch.setattr(search, "enumerate_reference", altered)
    with pytest.raises(ValueError, match="mismatch"):
        pilot.rank_all(programs, costs)


def test_same_cpu_group_and_all_vta_islands_accumulate_without_boundary_duplicate(legacy):
    domain, search = legacy
    rows = [domain.stage("cpu", 0, 3, 1, (0,)),
            domain.stage("vta", 3, 8, 1, (3,)),
            domain.stage("cpu", 8, 12, 1, (0,)),
            domain.stage("vta", 12, 20, 1, (3,)),
            domain.stage("cpu", 20, 21, 1, (1,))]
    cfg = domain.candidate(rows, "pinned", "resource_group", (3,), (3,))
    costs = {}
    for row in rows:
        c = domain.component(row, "pinned", (3,))
        costs[domain.component_key(c)] = {"configuration": c, "cost": {
            "service_ms": 10., "run_core_ms": 4., "boundary_core_ms": 1.}}
    value = search.evaluate(cfg, search.Costs(costs))
    assert value["vta_service_sum_ms"] == 20.
    assert value["group_services_ms"]["0"] == 20.
    assert value["boundary_core_ms"] == 5.
    assert value["host_core_ms"] == 8.
    # Reserved core3 receives VTA run+boundary once for each of the two islands.
    assert value["subset_capacity_ms"][7] == 10.


def test_pool_caps_each_baseline_at_five_and_keeps_enhanced_twenty():
    def row(prefix, index):
        cfg = {key: [] for key in pilot.CONFIGURATION_FIELDS}
        cfg.update(id=f"{prefix}{index}", score_ms=float(index), static_rank=index + 1,
                   resource_policy="shared4")
        return cfg
    rankings = {"methods": {method: {"rankings": [row(method, i) for i in range(20)]}
                             for method in pilot.METHODS}}
    pool = pilot.select_pool(rankings)
    assert len(pool) == 30
    assert "enhanced19" in pool
    assert "single_core4" in pool and "single_core5" not in pool
    assert "prefix4" in pool and "prefix5" not in pool
