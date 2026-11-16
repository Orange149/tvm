from copy import deepcopy
import json
from pathlib import Path

import pytest

from cpu_policies.open_resources import costs, domain, history


def programs(count=3):
    return {f"{dev}_{i:02}_{i:02}": {"device": dev, "begin": i, "end": i + 1}
            for i in range(count) for dev in (("cpu",) if i % 2 == 0 else ("vta",))}


def records(p, cpu_work=200., cpu_service=100., boundary=0., host=0., vta_service=1.):
    result = {}
    for key, cfg in domain.requirements(p).items():
        is_cpu = cfg["stages"][0]["device"] == "cpu"
        result[key] = {"configuration": cfg, "kind": "measured",
                       "cost": {"service_ms": cpu_service if is_cpu else vta_service,
                                "run_core_ms": cpu_work if is_cpu else host,
                                "boundary_core_ms": boundary if is_cpu else 0.}}
    return result


def cfg(masks, ctx="shared4", gate="off", count=3):
    c = domain.context(ctx)
    allocations = iter(masks)
    rows = []
    for i in range(count):
        dev = "cpu" if i % 2 == 0 else "vta"
        mask = next(allocations) if dev == "cpu" else c["host_mask"]
        rows.append(domain.stage(dev, i, i + 1, len(mask) if dev == "cpu" else 1, mask))
    return domain.candidate(rows, c, gate)


def test_nine_contexts_define_roles_and_reservation_independently():
    contexts = domain.contexts()
    assert len(contexts) == 9
    assert len({c["id"] for c in contexts}) == 9
    shared = domain.context("host_shared_3")
    reserved = domain.context("host_reserved_3")
    assert shared["host_mask"] == reserved["host_mask"] == [3]
    assert shared["compute_pool"] == [0, 1, 2, 3]
    assert reserved["compute_pool"] == [0, 1, 2]
    assert shared["control_mask"] == [0, 1, 2, 3]
    assert reserved["control_mask"] == [3]
    valid = cfg(((3,), (0,)), "host_shared_3")
    domain.validate_candidate(valid, programs(), unit_count=3)
    invalid = cfg(((3,), (0,)), "host_reserved_3")
    with pytest.raises(ValueError, match="reserved"):
        domain.validate_candidate(invalid, programs(), unit_count=3)


def test_all_225_ordered_mask_assignments_and_four_plus_four_are_legal():
    masks = domain.nonempty_masks()
    assert len(masks) == 15
    ids = set()
    thread_pairs = set()
    for first in masks:
        for last in masks:
            value = cfg((first, last))
            domain.validate_candidate(value, programs(), unit_count=3)
            ids.add(value["id"])
            thread_pairs.add((len(first), len(last)))
    assert len(ids) == 225 and len(thread_pairs) == 16
    domain.validate_candidate(cfg(((0, 1, 2, 3), (0, 1, 2, 3))), unit_count=3)
    # A genuine three-core computation pool can still share two three-thread pools.
    domain.validate_candidate(cfg(((0, 1, 3), (0, 1, 3)), "host_reserved_2"), unit_count=3)


def test_serial_accepts_reused_groups_rejects_partial_overlap():
    domain.validate_candidate(cfg(((0, 1), (0, 1)), gate="resource_group"), unit_count=3)
    domain.validate_candidate(cfg(((0, 1), (2, 3)), gate="resource_group"), unit_count=3)
    with pytest.raises(ValueError, match="partially overlap"):
        domain.validate_candidate(cfg(((0, 1), (1, 2)), gate="resource_group"), unit_count=3)


def test_worker_caller_context_and_complete_identity_are_checked():
    value = cfg(((0, 1), (2, 3)))
    for field, replacement in (("caller_core", 1), ("worker_core_map", [1, 0])):
        altered = deepcopy(value)
        altered["stages"][0][field] = replacement
        with pytest.raises(ValueError, match="caller|worker"):
            domain.validate_candidate(altered, unit_count=3)
    altered = deepcopy(value)
    altered["host_mask"] = [2]
    with pytest.raises(ValueError, match="context"):
        domain.validate_candidate(altered, unit_count=3)
    altered = deepcopy(value)
    altered["id"] = "cfg_bad"
    with pytest.raises(ValueError, match="identity"):
        domain.validate_candidate(altered, unit_count=3)


def test_component_keys_preserve_effective_work_but_alias_control_and_reservation():
    row = domain.stage("vta", 1, 2, 1, (3,))
    shared = domain.component(row, "host_shared_3")
    reserved = domain.component(row, "host_reserved_3")
    assert shared["id"] != reserved["id"]
    assert domain.component_key(shared) == domain.component_key(reserved)
    domain.validate_candidate(shared, component=True, unit_count=3)
    domain.validate_candidate(reserved, component=True, unit_count=3)
    other = domain.component(domain.stage("vta", 1, 2, 1, (2,)), "host_shared_2")
    assert domain.component_key(other) != domain.component_key(shared)
    isolated = domain.candidate([domain.stage("cpu", 0, 1, 1, (0,))], gate="resource_group")
    with pytest.raises(ValueError, match="isolated components"):
        domain.validate_candidate(isolated, component=True, unit_count=3)


def test_frozen_stage_inventory_needs_775_costs_and_87_traffic_signatures():
    bundle = Path(__file__).parents[1] / "artifacts/bundles/be7a3c600c2ed47f245dfb9f"
    if not bundle.exists():
        pytest.skip("local frozen deployment assets not installed")
    p = json.loads((bundle / "plans/programs.json").read_text())
    required = domain.requirements(p)
    assert len(required) == 775
    assert sum(c["stages"][0]["device"] == "cpu" for c in required.values()) == 340
    assert len(domain.traffic_requirements(p)) == 87


def test_partial_overlap_counts_shared_core_hotspot_not_only_confined_masks():
    c = costs.Costs(records(programs()))
    overlapping = costs.score_configuration(cfg(((0, 1), (1, 2))), c)
    separate = costs.score_configuration(cfg(((0, 1), (2, 3))), c)
    assert overlapping["cpu_core_demand_ms"] == [100., 200., 100., 0.]
    assert overlapping["score_ms"] == 200.
    assert separate["score_ms"] == 100.
    assert not overlapping["ddr_available"]
    assert overlapping["control_cost_model_status"] == "not_separately_profiled"


def test_single_host_and_boundary_charge_once_on_correct_physical_cores():
    c = costs.Costs(records(programs(), boundary=10., host=20.))
    result = costs.score_configuration(cfg(((0, 1), (1, 2)), "host_shared_1"), c)
    assert result["cpu_core_demand_ms"] == [110., 210., 100., 0.]
    assert result["host_core_ms"] == 20.
    assert result["boundary_core_ms"] == 20.
    assert result["score_ms"] == 230.


def test_same_group_service_and_all_vta_islands_accumulate():
    p = programs(5)
    c = costs.Costs(records(p, cpu_work=1., cpu_service=10., vta_service=8.))
    closed = costs.score_configuration(cfg(((0,), (0,), (1,)), gate="resource_group", count=5), c)
    opened = costs.score_configuration(cfg(((0,), (0,), (1,)), count=5), c)
    assert closed["group_services_ms"] == {"0": 20., "1": 10.}
    assert closed["vta_service_sum_ms"] == 16.
    assert closed["score_ms"] == 20.
    assert opened["score_ms"] == 16.


def test_homogeneous_mask_alias_is_explicit_and_new_host_needs_real_measurement():
    values = records(programs())
    c = costs.Costs(values)
    result = c.get(domain.stage("cpu", 0, 1, 2, (1, 3)))
    assert result["source_key"] != result["requested_key"]
    assert result["assumptions"] == ["homogeneous_core_permutation_not_per_mask_measurement"]
    assert c.get(domain.stage("cpu", 0, 1, 2, (0, 1)))["assumptions"] == []
    host = domain.component(domain.stage("vta", 1, 2, 1, (2,)), "host_shared_2")
    values.pop(domain.component_key(host))
    with pytest.raises(ValueError, match="missing matched"):
        c.get(host["stages"][0], "host_shared_2")


def test_session_shares_are_separate_explicit_approximation_and_ddr_is_max_bound():
    values = records(programs())
    key = domain.component_key(domain.component(domain.stage("cpu", 0, 1, 2, (0, 1))))
    values[key]["session_thread_weights"] = [.9, .1]
    c = costs.Costs(values, distribution="session_weights")
    ddr = {"stage_service_ms": {p: {str(n): 90. for n in range(1, 5)} for p in programs()}}
    result = costs.score_configuration(cfg(((0, 1), (2, 3))), c, ddr)
    assert result["cpu_core_demand_ms"] == [180., 20., 100., 100.]
    assert result["ddr_service_ms"] == 270.
    assert result["score_ms"] == 270.
    assert "whole_session" in c.get(domain.stage("cpu", 0, 1, 2, (0, 1)))["assumptions"][0]


@pytest.mark.parametrize("shares", [[.9, .9], [.5], [float("nan"), .1], [-.1, 1.1], [True, 0.]])
def test_session_share_corruption_cannot_change_total_scored_core_work(shares):
    values = records(programs())
    row = domain.stage("cpu", 0, 1, 2, (0, 1))
    key = domain.component_key(domain.component(row))
    values[key]["session_thread_weights"] = shares
    with pytest.raises(ValueError, match="normalized"):
        costs.Costs(values, distribution="session_weights").get(row)


def test_invalid_cost_and_asset_hash_fail_before_import(monkeypatch, tmp_path):
    values = records(programs())
    key = next(iter(values))
    values[key]["cost"]["run_core_ms"] = float("nan")
    with pytest.raises(ValueError, match="invalid isolated"):
        costs.Costs(values).get(values[key]["configuration"]["stages"][0])
    path = tmp_path / "programs/x/module.so"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"wrong")
    manifest = {"files": {"programs/x/module.so": {"bytes": 5, "sha256": "0" * 64}}}
    with pytest.raises(ValueError, match="asset differs"):
        costs._verify_assets(manifest, tmp_path, tmp_path)


def test_historical_references_stay_external_and_rules_explain_reused_groups():
    bundle = Path(__file__).parents[1] / "artifacts/bundles/be7a3c600c2ed47f245dfb9f"
    if not bundle.exists() or not history.RECOVERY.exists():
        pytest.skip("archival source assets are not installed")
    p = json.loads((bundle / "plans/programs.json").read_text())
    refs = history.historical_references(p)
    assert len(refs) == 3 and len({r["id"] for r in refs}) == 3
    assert refs[0]["context_id"] == "shared4"
    assert refs[1]["context_id"] == "host_shared_2"
    assert refs[0]["execution_mode"] == "resource_group"
    assert [r["threads"] for r in refs[0]["stages"] if r["device"] == "cpu"] == [3, 1, 3, 1]


def test_integer_nanosecond_half_up_and_conserving_logical_slot_split():
    assert costs.quantize_ns(.0000005) == 1
    assert costs.quantize_ns(1.0000005) == 1000001
    assert costs.quantize_ns(.0000004999999) == 0
    assert costs.allocate_core_ns(10, threads=3) == (4, 3, 3)
    assert costs.allocate_core_ns(1, [.5, .5]) == (1, 0)
    assert costs.allocate_core_ns(3, [.5, .3, .2]) == (1, 1, 1)
    assert costs.integer_weights([1 / 3] * 3) == (333333333333,) * 3
    assert costs.allocate_integer_ns(9, (3, 3, 4)) == (3, 3, 3)
    for n in range(1, 5):
        for work in (0, 1, 2, 7, 104488846, (1 << 52) + 7):
            allocation = costs.allocate_core_ns(work, threads=n)
            assert sum(allocation) == work
            assert max(allocation) - min(allocation) <= 1
    assert costs.ceil_div_ns(1 + 1, 3) == 1
    assert costs.ceil_div_ns(1, 3) + costs.ceil_div_ns(1, 3) == 2


def test_fractional_nanosecond_components_do_not_depend_on_float_addition_order():
    # These half-nanosecond components exposed resource-stratum tie changes
    # when the vector DP and scalar oracle summed binary floats differently.
    values = records(programs(), cpu_work=.0000015, cpu_service=.0000005,
                     boundary=.0000005, host=.0000005, vta_service=.0000005)
    for value in values.values():
        if value["configuration"]["stages"][0]["device"] == "vta":
            value["cost"]["boundary_core_ms"] = .0000005
    config = cfg(((0, 1, 2), (1, 2, 3)), "host_shared_2")
    c = costs.Costs(values)
    result = costs.score_configuration(config, c)
    assert result["cpu_core_demand_ns"] == [2, 3, 1, 0]
    assert result["cpu_compute_core_ns"] == 4
    assert result["host_core_ns"] == 2
    assert result["score_ns"] == 3
    assert result["score_ms"] == .000003
    assert result["subset_capacity_ns"][1] == 3
    assert result["subset_capacity_ns"][3] == 3
    assert sum(result["cpu_core_demand_ns"]) == result["cpu_compute_core_ns"] + 2
    # Source evidence retains the original unrounded millisecond values.
    assert all(v["cost"]["run_core_ms"] == (.0000015 if v["configuration"]["stages"][0]["device"] == "cpu" else .0000005)
               for v in values.values())
