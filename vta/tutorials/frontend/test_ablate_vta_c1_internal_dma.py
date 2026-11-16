"""Check isolation, fixed ranking rules, and the 15/16/17 counterfactual."""

import copy

import pytest

import ablate_vta_c1_internal_dma as study
import analyze_vta_c1_memory_features as memory
import shared_memory_topk as topk
import solve_cpu_vta_pipeline_v1_p3 as p3


def row(name, score, nbytes, calls, boundary=0):
    item = {d: 0 for d in study.unified.UNIFIED_DIMENSIONS}
    item.update(candidate_id=name, topology_id=name, m1_score_ms=score,
                vta_load_inp_bytes=nbytes, vta_load_inp_calls=calls,
                framework_materialization_bytes=boundary, unified_feature_exact=True,
                realization={"realized_policy": "ordinary_copy"})
    return item


def test_groups_partition_existing_features_without_altering_deployed_rule():
    assert set(study.INTERNAL).isdisjoint(study.CONTROL)
    assert set(study.INTERNAL) | set(study.CONTROL) == set(study.unified.UNIFIED_DIMENSIONS)
    assert study.GROUPS["deployed_full"] == study.unified.UNIFIED_DIMENSIONS
    assert study.GROUPS["off"] == ()
    assert all("framework_materialization_bytes" not in study.GROUPS[group]
               for group in ("off", "bytes", "calls", "shape", "bytes_calls", "internal_all"))


def test_internal_features_change_only_in_band_order_and_preserve_costs():
    rows = [row("a", 100, 1000, 1000), row("b", 100.5, 100, 10),
            row("c", 101, 10, 100), row("outside", 105, 0, 0)]
    saved = copy.deepcopy(rows)
    orders = study.frozen_orders(rows)
    get = lambda name: [r["candidate_id"] for r in orders[name]["top5"]]
    assert get("off") == ["a", "b", "c", "outside"]
    assert get("bytes") == ["a", "c", "b", "outside"]
    assert get("calls") == ["a", "b", "c", "outside"]
    assert rows == saved
    assert all(order["top5"][0]["candidate_id"] == "a" for order in orders.values())
    assert all(order["top5"][-1]["candidate_id"] == "outside" for order in orders.values())
    assert all(order["top5"][1]["m1_score_ms"] in (100.5, 101) for order in orders.values())


def test_full_rule_retains_tradeoff_instead_of_forcing_less_dma_to_win():
    rows = [row("anchor", 100, 1000, 1000), row("15", 100.5, 100, 10, boundary=1000),
            row("16", 101, 200, 20, boundary=500)]
    orders = study.frozen_orders(rows)
    assert topk.dominates(rows[1], rows[2], study.INTERNAL)
    assert not topk.dominates(rows[1], rows[2], study.unified.UNIFIED_DIMENSIONS)
    assert [r["candidate_id"] for r in orders["internal_all"]["top5"]] == ["anchor", "15", "16"]
    assert [r["candidate_id"] for r in orders["deployed_full"]["top5"]] == ["anchor", "16", "15"]


@pytest.mark.parametrize("bad", [float("nan"), -1, True])
def test_invalid_internal_features_are_rejected(bad):
    item = row("a", 100, 10, 10)
    item["vta_load_inp_calls"] = bad
    with pytest.raises(ValueError):
        study.frozen_orders([item])


def test_unknown_and_duplicate_features_are_rejected():
    item = row("a", 100, 10, 10)
    item["unified_feature_exact"] = False
    with pytest.raises(ValueError, match="unknown"):
        study.frozen_orders([item])
    item["unified_feature_exact"] = True
    with pytest.raises(ValueError, match="duplicate"):
        study.frozen_orders([item, copy.deepcopy(item)])


def test_real_endpoint_features_and_all_thread_pairs():
    topologies = topk.topology_diverse(topk.read_json(study.unified.DEFAULT_TOPOLOGIES))
    estimates = {r["segment_id"]: r for r in topk.read_json(study.unified.DEFAULT_FUSION_ESTIMATES)["rows"]}
    context = p3.build_context(
        topk.read_json(p3.DEFAULT_OUTPUT / "v1_profile_manifest.json"),
        topk.read_json(p3.DEFAULT_OUTPUT / "v1_local_cost_table.json"))
    checks = study.endpoint_rows(topologies, estimates, context)
    assert {tuple(r["cpu_threads"]) for r in checks} == {(a, b) for a in range(1, 5) for b in range(1, 5)}
    for check in checks:
        first, second, third = check["rows"]
        assert second["vta_load_bytes"] - first["vta_load_bytes"] == 219648
        assert second["vta_load_calls"] - first["vta_load_calls"] == 66
        assert second["vta_store_bytes"] - first["vta_store_bytes"] == 25088
        assert second["logical_boundary_bytes"] - first["logical_boundary_bytes"] == -100352
        assert second["framework_materialization_bytes"] - first["framework_materialization_bytes"] == -200704
        assert third["vta_internal_dma_bytes"] == second["vta_internal_dma_bytes"]
        assert all(r["realization"]["realized_policy"] == "ordinary_copy" for r in check["rows"])
        assert check["pairwise"][0]["internal_from_dominates_to"]
        assert not check["pairwise"][0]["full_from_dominates_to"]


def test_compiled_feature_mismatch_is_detected():
    topology = {"path": [("vta:03:15", 1)], "vta_logical_dma": {"load_buffer_2d_calls": 2}}
    estimates = {"vta:03:15": {"status": "exact_reuse", "usable_for_ranking": True,
                                "total_dma": {"load_buffer_2d_calls": 1}}}
    with pytest.raises(ValueError, match="mismatch"):
        study.validate_compiled_topology(topology, estimates)
