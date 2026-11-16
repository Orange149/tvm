"""Physical CPU budgets, caller accounting, explicit sharing, and identity."""
from copy import deepcopy

import pytest

from cpu_policies.runtime_contract import build_runtime_contract, validate_runtime_contract


def test_four_core_partition_counts_caller_once():
    result = build_runtime_contract([[0, 1], [2, 3]])
    first, second = result["stages"]
    assert first["threads"] == second["threads"] == 2
    assert first["caller_core"] == 0
    assert second["caller_core"] == 2
    assert first["worker_core_map"] == [0, 1]
    assert second["worker_core_map"] == [2, 3]
    assert sum(s["threads"] for s in result["stages"]) == 4
    assert result["caller_included_in_threads"] is True
    assert result["affinity_mode"] == "pinned"
    assert result["execution_mode"] == "resource_group"
    assert result["project_extension"] is False
    validate_runtime_contract(result)


@pytest.mark.parametrize("groups", [[[0, 1], [1, 2]], [[0, 1], [0, 1]], [[0, 1, 2, 3, 4]]])
def test_overlapping_or_over_budget_groups_are_rejected(groups):
    with pytest.raises(ValueError):
        build_runtime_contract(groups)


def test_single_group_four_threads_includes_its_caller():
    result = build_runtime_contract([[0, 1, 2, 3]])
    stage = result["stages"][0]
    assert stage["threads"] == len(stage["worker_core_map"]) == 4
    assert stage["caller_slot"] == 0
    assert stage["caller_core"] == stage["worker_core_map"][0]


def test_nonadjacent_stage_reuse_needs_explicit_extension_and_serial_gate():
    with pytest.raises(ValueError, match="allow_stage_group_reuse"):
        build_runtime_contract([[0, 1], [2, 3]], [0, 1, 0])
    result = build_runtime_contract([[0, 1], [2, 3]], [0, 1, 0], allow_stage_group_reuse=True)
    assert result["project_extension"] is True
    assert "nonadjacent_cpu_stage_group_reuse" in result["project_extensions"]
    assert result["same_group_serialization"] == [
        {"group_index": 0, "stage_indices": [0, 2]},
        {"group_index": 1, "stage_indices": [1]},
    ]
    assert result["serialization_scope"] == "complete_cpu_stage_service"
    assert result["stages"][0]["worker_core_map"] == result["stages"][2]["worker_core_map"]
    validate_runtime_contract(result)


def test_host_and_control_do_not_overwrite_computation_affinity():
    with pytest.raises(ValueError, match="allow_host_compute_overlap"):
        build_runtime_contract([[0, 1], [2, 3]], host_mask=[2])
    result = build_runtime_contract([[0, 1], [2, 3]], host_mask=[2], control_mask=[0, 1, 2, 3],
                                    allow_host_compute_overlap=True)
    assert result["host_mask"] == [2]
    assert result["control_mask"] == [0, 1, 2, 3]
    assert result["host_compute_overlap"] == [2]
    assert result["control_compute_overlap"] == [0, 1, 2, 3]
    assert result["stages"][0]["worker_core_map"] == [0, 1]
    assert result["stages"][1]["worker_core_map"] == [2, 3]
    assert result["process_cpu_affinity"] is None
    assert result["forbid_process_affinity_override"] is True
    validate_runtime_contract(result)


def test_host_may_use_a_core_not_assigned_to_a_cpu_stage():
    result = build_runtime_contract([[0, 1], [2, 3]], [0], host_mask=[2])
    assert result["host_shares_compute"] is False
    assert result["assigned_cpu_cores"] == [0, 1]
    assert result["unassigned_group_indices"] == [1]


def test_host_and_control_positions_are_part_of_identity():
    def contract(host, control):
        return build_runtime_contract([[0, 1], [2, 3]], host_mask=host, control_mask=control,
                                      allow_host_compute_overlap=True)
    a, b, d = contract([0], [3]), contract([2], [3]), contract([0], [1])
    assert len({x["configuration_id"] for x in (a, b, d)}) == 3
    assert a["stages"] == b["stages"] == d["stages"]


def test_group_membership_order_is_canonical_without_mutating_inputs():
    groups = [[1, 0], [3, 2]]
    result = build_runtime_contract(groups)
    assert result["configuration_id"] == build_runtime_contract([[0, 1], [2, 3]])["configuration_id"]
    assert groups == [[1, 0], [3, 2]]
    groups[0].append(3)
    assert result["cpu_groups"] == [[0, 1], [2, 3]]


@pytest.mark.parametrize("kwargs", [
    {"stage_group_indices": [True]}, {"stage_group_indices": [2]},
    {"stage_group_indices": []}, {"host_mask": []}, {"control_mask": [4]},
    {"physical_cores": [0, 1, 1, 2]}, {"allow_stage_group_reuse": 1},
    {"allow_host_compute_overlap": "yes"},
])
def test_invalid_role_or_index_data_is_rejected(kwargs):
    with pytest.raises(ValueError):
        build_runtime_contract([[0, 1], [2, 3]], **kwargs)


def test_altered_caller_binding_and_process_affinity_are_rejected():
    original = build_runtime_contract([[0, 1], [2, 3]])
    changed = deepcopy(original)
    changed["stages"][0]["caller_core"] = 2
    with pytest.raises(ValueError, match="fields or identity"):
        validate_runtime_contract(changed)
    changed = deepcopy(original)
    changed["process_cpu_affinity"] = [0, 1, 2, 3]
    with pytest.raises(ValueError, match="fields or identity"):
        validate_runtime_contract(changed)
