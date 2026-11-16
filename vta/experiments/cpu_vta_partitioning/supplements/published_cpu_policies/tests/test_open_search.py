"""Adversarial model/search tests; every small full domain uses the C++ oracle."""
from copy import deepcopy
from itertools import combinations, product
import json
from pathlib import Path

import numpy as np
import pytest

from cpu_policies.open_resources import domain
from cpu_policies.open_resources.costs import Costs, score_configuration
from cpu_policies.open_resources import search


def programs_for(intervals):
    result = {}
    for device, begin, end in intervals:
        name = f"{device}_{begin:02}_{end - 1:02}"
        result[name] = {"device": device, "begin": begin, "end": end, "name": name}
    return result


def records_for(programs, *, equal=False, weights=False):
    result = {}
    for key, configuration in domain.requirements(programs).items():
        row = configuration["stages"][0]
        n = row["threads"]
        if row["device"] == "cpu":
            wall = 10. if equal else (41 + row["begin"] * 2) / n + .17 * n
            run = 0. if equal else 40. + row["begin"] * 1.7 + n * .1
            boundary = 0. if equal else .07 * (row["begin"] + 1)
        else:
            wall = 10. if equal else 5. + row["begin"] * .13
            run = 0. if equal else .5 + row["mask"][0] * .03
            boundary = 0. if equal else .13
        result[key] = {"configuration": configuration, "kind": "synthetic",
                       "cost": {"service_ms": wall, "run_core_ms": run,
                                "boundary_core_ms": boundary}}
        if weights and row["device"] == "cpu":
            values = np.arange(1, n + 1, dtype=float)
            result[key]["session_thread_weights"] = list(map(float, values / values.sum()))
    return result


@pytest.fixture
def three():
    programs = programs_for([("cpu", 0, 1), ("vta", 1, 2), ("cpu", 2, 3)])
    return programs, records_for(programs)


def test_nine_context_full_small_domain_and_oracle(three, tmp_path):
    programs, records = three
    result = search.rank_all(programs, records, unit_count=3, max_vta_islands=1,
                             output_dir=tmp_path)
    assert result["verification"]["passed"]
    assert result["counts"] == {"topologies": 1, "contexts": 9,
                                 "enhanced": 5 * (225 + 65) + 4 * (49 + 19),
                                 "single_core": 5 * 16 + 4 * 9,
                                 "prefix": 5 * 16 + 4 * 9}
    assert len(result["pure_static20"]) == len(result["diverse20"]) == 20
    assert len(search.select_pool(result)) <= 40
    assert not result["ranking_uses_pipeline_feedback"]
    # Reuse a complete context checkpoint instead of re-running a changed search.
    again = search.rank_all(programs, records, unit_count=3, max_vta_islands=1,
                            output_dir=tmp_path)
    assert again["pure_static20"] == result["pure_static20"]
    checkpoint = tmp_path / "shared4.json"
    original = checkpoint.read_text()
    tampered = json.loads(original)
    tampered["counts"]["enhanced"] += 1
    checkpoint.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="checkpoint content hash"):
        search.rank_all(programs, records, unit_count=3, max_vta_islands=1,
                        output_dir=tmp_path)
    checkpoint.write_text(original)
    changed = deepcopy(records)
    next(iter(changed.values()))["cost"]["run_core_ms"] += 1.
    with pytest.raises(ValueError, match="checkpoint inputs changed"):
        search.rank_all(programs, changed, unit_count=3, max_vta_islands=1,
                        output_dir=tmp_path)


def test_four_stage_half_dp_preserves_complementary_resources(tmp_path):
    programs = programs_for([("cpu", 0, 1), ("vta", 1, 2), ("cpu", 2, 3),
                             ("vta", 3, 4), ("cpu", 4, 5), ("vta", 5, 6), ("cpu", 6, 7)])
    records = records_for(programs, weights=True)
    costs = Costs(records, distribution="session_weights")
    ddr = {"stage_service_ms": {name: {str(n): (p["end"] - p["begin"]) * (1 + n / 11)
                for n in (range(1, 5) if p["device"] == "cpu" else [1])}
                for name, p in programs.items()}}
    result = search.rank_all(programs, costs, ddr=ddr, contexts=[domain.context("shared4")],
                             unit_count=7, output_dir=tmp_path)
    assert result["counts"]["enhanced"] == 15**4 + 749
    assert result["verification"]["passed"]
    assert len(json.loads((tmp_path / "shared4.json").read_text())["resource_strata"]) > 1000
    for row in result["pure_static20"]:
        assert score_configuration(row, costs, ddr)["score_ms"] == row["score_ms"]


def test_ties_use_execution_tuple_not_hash(three, tmp_path):
    programs, _ = three
    records = records_for(programs, equal=True)
    result = search.rank_all(programs, records, contexts=[domain.context("shared4")],
                             unit_count=3, max_vta_islands=1, output_dir=tmp_path)
    rows = result["pure_static20"]
    assert len({row["score_ms"] for row in rows}) == 1
    assert rows == sorted(rows, key=lambda row: domain.canonical_tuple(row))
    assert result["verification"]["passed"]


def test_partial_overlap_hotspot_and_serial_constraint(three):
    programs, records = three
    for record in records.values():
        row = record["configuration"]["stages"][0]
        record["cost"] = {"service_ms": 100. if row["device"] == "cpu" else 1.,
                          "run_core_ms": 200. if row["device"] == "cpu" else 0.,
                          "boundary_core_ms": 0.}
    costs = Costs(records)
    context = domain.context("shared4")
    table = search._prepare(programs, costs, context, None)
    graph = tuple(programs)
    allocation = search._allocation_domain((0, 1, 2, 3), 2)
    values = search._join_values((graph[0], graph[2]), graph, table, context, allocation)
    masks = ((0, 1), (1, 2))
    selected = tuple(allocation["options"].index(mask) for mask in masks)
    index = np.flatnonzero((allocation["allocations"] == selected).all(axis=1))[0]
    assert values["enhanced"][index] / search.SCORE_SCALE == 200.
    assert not allocation["compatible"][index]
    config = domain.candidate([domain.stage("cpu", 0, 1, 2, masks[0]),
                               domain.stage("vta", 1, 2, 1, (0, 1, 2, 3)),
                               domain.stage("cpu", 2, 3, 2, masks[1])], context)
    assert score_configuration(config, costs)["score_ms"] == 200.


def test_stratified_selection_reaches_outside_pure_top20():
    def row(cut, n, mask, score):
        config = domain.candidate([domain.stage("cpu", 0, cut, n, mask),
                                   domain.stage("vta", cut, 20, 1, (0, 1, 2, 3)),
                                   domain.stage("cpu", 20, 21, 1, (3,))])
        return {**config, "score_ms": score}
    pure = [row(3, 4, (0, 1, 2, 3), 10.)]
    # Distinct physical states of the same cut/shape make a deliberately
    # homogeneous pure Top-20; use host positions and gates to get unique IDs.
    for ctx in domain.contexts()[:5]:
        for gate in search.GATES:
            for mask in ((0,), (1,)):
                cfg = domain.candidate([domain.stage("cpu", 0, 3, 1, mask),
                    domain.stage("vta", 3, 20, 1, ctx["host_mask"]),
                    domain.stage("cpu", 20, 21, 1, (3,))], ctx, gate)
                pure.append({**cfg, "score_ms": 10.01 + len(pure) / 1000})
    pure = sorted(pure, key=search._row_key)[:20]
    alternatives = [row(cut, 2, (0, 1), 10.5 + cut / 1000) for cut in range(4, 10)]
    alternatives.append(row(10, 3, (0, 1, 2), 1000.))
    selected = search.select_diverse(pure, alternatives)
    assert len(selected) == 20
    assert [r["id"] for r in selected[:10]] == [r["id"] for r in pure[:10]]
    assert len({search.topology_key(r) for r in selected}) >= 6
    assert not any(r["score_ms"] == 1000. for r in selected)
    assert any(r["id"] not in {p["id"] for p in pure} for r in selected)


def test_ddr_missing_stage_rejected(three):
    programs, records = three
    with pytest.raises(ValueError, match="missing DDR"):
        search.rank_all(programs, records, ddr={"stage_service_ms": {}}, unit_count=3,
                        max_vta_islands=1, contexts=[domain.context("shared4")], verify=False)


def test_counting_complete_open_domain():
    # Counts depend only on the compiler-qualified topology CPU-stage counts.
    topologies_by_cpu = {2: 87, 3: 990, 4: 3546}
    total = 0
    for pool, contexts in (((0, 1, 2, 3), 5), ((0, 1, 2), 4)):
        count = 0
        for n, topologies in topologies_by_cpu.items():
            allocation = search._allocation_domain(pool, n)
            count += topologies * (len(allocation["allocations"]) + int(allocation["compatible"].sum()))
        total += count * contexts
    assert total == 966_240_306


def test_quantized_group_top_two_tie_order():
    values = np.array([3, 2, 2, 1, 2, 1], dtype=np.int64)
    labels = np.array([0, 0, 0, 1, 1, 1])
    first, second = search._group_first_two(values, np.ones(6, dtype=bool), labels, 2)
    assert list(first) == [1, 3]
    assert list(second) == [2, 5]


def test_real_fractional_costs_have_exact_stratum_ties(tmp_path):
    fixture = json.loads((Path(__file__).parent / "fixtures/open_real_rounding.json").read_text())
    programs, records = fixture["programs"], fixture["records"]
    result = search.rank_all(programs, records, contexts=[domain.context("shared4")],
                             output_dir=tmp_path)
    assert result["verification"]["passed"]
    assert result["counts"]["enhanced"] == 15**4 + 749
    assert result["score_quantization_ms"] == 1e-6
    assert result["integer_accounting"]["capacity_division"].startswith("ceil once")
    for row in result["pure_static20"]:
        assert row["score_ms"] == score_configuration(row, Costs(records))["score_ms"]
    # A repeated, one-nanosecond remainder always goes to the same logical slot.
    assert search._allocate_integer(10, (1, 1, 1)) == [4, 3, 3]
    assert sum(search._allocate_integer(179738641, (333333333333,) * 3)) == 179738641


def test_oracle_preserves_int64_costs_above_binary64_exact_range(three, tmp_path):
    programs, records = three
    for value in records.values():
        row = value["configuration"]["stages"][0]
        if row["device"] == "cpu":
            value["cost"] = {"service_ms": 10000000000.123455 / row["threads"],
                             "run_core_ms": 10000000000.123455,
                             "boundary_core_ms": .0000015}
    context = domain.context("shared4")
    table = search._prepare(programs, Costs(records), context, None)
    assert table["cpu_00_00"]["costs"][0][0] > 2**53
    result = search.rank_all(programs, records, contexts=[context], unit_count=3,
                             max_vta_islands=1, output_dir=tmp_path)
    assert result["verification"]["passed"]
    assert all(type(row["score_ns"]) is int for row in result["pure_static20"])
