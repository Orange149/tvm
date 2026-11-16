"""Pure-data mapping of supplied CPU groups to the existing TVM runner.

This adapter does not choose thread counts, core groups, or a host core. It
maps an allocation produced elsewhere to caller-inclusive fixed pinning.

The binding reference is official ARM Compute Library v21.02 (MIT), commit
7dcb9fadb98cad05fca72de3273311d570d98b4e:
  arm_compute/runtime/IScheduler.h:51-56,142-153
  src/runtime/CPP/CPPScheduler.cpp:94-105,249-261,311-337
Only its interface/semantics are referenced; no ACL implementation is copied.

ACL's caller binds through callback index 0 but executes the last workload id.
TVM's caller executes task 0 with TVM_EXCLUDE_WORKER0=1. The returned map is
therefore a TVM logical-slot map, not a copy of ACL workload identifiers.
Flexi-BOPI names ACL v21.02 as its framework but does not disclose these author
callsites. This module is not a reproduction of its Bayesian optimizer.
"""

from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json


SOURCE_REFERENCES = {
    "allocation": "supplied_by_caller; no thread or host selection heuristic",
    "acl_commit": "7dcb9fadb98cad05fca72de3273311d570d98b4e",
    "acl_binding_interface":
        "https://github.com/ARM-software/ComputeLibrary/blob/"
        "7dcb9fadb98cad05fca72de3273311d570d98b4e/"
        "src/runtime/CPP/CPPScheduler.cpp#L249",
    "acl_scope": "framework binding capability; Flexi-BOPI author usage not established",
    "tvm_caller_task": "src/runtime/thread_pool.cc:287-317",
    "tvm_native_binding_adapter":
        "vta/experiments/cpu_vta_partitioning/supplements/cpu_strategy/"
        "cpu_strategy/policy/native.py:110-118",
    "project_gate": "vta/apps/native_deploy/vta_cpu_resource_groups.h:18-46",
}


def _mask(value, label, *, optional=False):
    if value is None and optional:
        return None
    if not isinstance(value, (list, tuple, set, frozenset)) or not value:
        raise ValueError(f"{label} must be a nonempty collection of physical core IDs")
    rows = list(value)
    if any(type(n) is not int or n < 0 for n in rows) or len(set(rows)) != len(rows):
        raise ValueError(f"{label} must contain distinct nonnegative integer core IDs")
    return sorted(rows)


def build_runtime_contract(cpu_groups, stage_group_indices=None, *,
                           physical_cores=(0, 1, 2, 3), host_mask=None,
                           control_mask=None, allow_stage_group_reuse=False,
                           allow_host_compute_overlap=False):
    """Map disjoint physical CPU groups to JSON-serializable execution settings.

    One stage per group is the default. Sharing a CPU group among multiple
    stages is allowed only explicitly and uses a whole-stage resource-group
    gate; this is a project extension to a disjoint-stage allocation.

    Each group's membership is sorted before assigning TVM logical slots. Slot
    zero is the caller and counts as one of ``threads=len(group)``. All other
    slots are workers, each assigned one core. No thread is added for the caller.

    ``host_mask`` describes VTA host threads; ``control_mask`` independently
    describes producer/collector threads. Neither sets process-wide affinity.
    A host using a core assigned to an actual CPU stage requires the explicit
    overlap option. ``None`` means the role was not specified by this adapter.

    Returned maps are requested settings. They still require the existing
    native affinity probe before making claims about effective runtime binding.
    Stage indices are ordinals in the supplied CPU-stage sequence; a mixed
    CPU/VTA pipeline adapter must translate them to its full pipeline indices.
    """
    for value, label in ((allow_stage_group_reuse, "allow_stage_group_reuse"),
                         (allow_host_compute_overlap, "allow_host_compute_overlap")):
        if type(value) is not bool:
            raise ValueError(f"{label} must be a boolean")
    physical = _mask(physical_cores, "physical_cores")
    if not isinstance(cpu_groups, (list, tuple)) or not cpu_groups:
        raise ValueError("cpu_groups must contain at least one physical CPU group")
    groups, occupied = [], set()
    for index, value in enumerate(cpu_groups):
        group = _mask(value, f"cpu_groups[{index}]")
        if not set(group) <= set(physical):
            raise ValueError("CPU group exceeds the declared physical core budget")
        if occupied.intersection(group):
            raise ValueError("CPU groups overlap; express stage reuse through one group index")
        occupied.update(group)
        groups.append(group)
    indices = list(range(len(groups))) if stage_group_indices is None else stage_group_indices
    if not isinstance(indices, (list, tuple)) or not indices:
        raise ValueError("stage_group_indices must be a nonempty sequence")
    if any(type(index) is not int or not 0 <= index < len(groups) for index in indices):
        raise ValueError("stage group index is outside cpu_groups")
    indices = list(indices)
    counts = Counter(indices)
    reused = sorted(index for index, count in counts.items() if count > 1)
    if reused and not allow_stage_group_reuse:
        raise ValueError("stage group reuse requires allow_stage_group_reuse=True")
    assigned = sorted({core for index in indices for core in groups[index]})
    host = _mask(host_mask, "host_mask", optional=True)
    control = _mask(control_mask, "control_mask", optional=True)
    for mask, label in ((host, "host_mask"), (control, "control_mask")):
        if mask is not None and not set(mask) <= set(physical):
            raise ValueError(f"{label} exceeds the declared physical core budget")
    overlap = sorted(set(host or ()).intersection(assigned))
    if overlap and not allow_host_compute_overlap:
        raise ValueError("host shares computation cores; set allow_host_compute_overlap=True explicitly")
    extensions = []
    if reused:
        extensions.append("nonadjacent_cpu_stage_group_reuse")
    if host is not None:
        extensions.append("explicit_accelerator_host_affinity")
    if control is not None:
        extensions.append("independent_pipeline_control_affinity")
    stages = []
    for stage_index, group_index in enumerate(indices):
        cores = groups[group_index]
        stages.append({"stage_index": stage_index, "group_index": group_index,
                       "threads": len(cores), "mask": list(cores),
                       "worker_core_map": list(cores), "caller_slot": 0,
                       "caller_core": cores[0], "affinity_mode": "pinned"})
    result = {
        "schema_version": 1,
        "scope": "runtime_mapping_of_supplied_cpu_allocation",
        "stage_index_scope": "cpu_stage_sequence",
        "physical_cores": physical,
        "cpu_groups": groups,
        "allocated_cpu_cores": sorted(occupied),
        "assigned_cpu_cores": assigned,
        "unassigned_group_indices": sorted(set(range(len(groups))) - set(indices)),
        "stage_group_indices": indices,
        "stages": stages,
        "affinity_mode": "pinned",
        "execution_mode": "resource_group",
        "serialization_scope": "complete_cpu_stage_service",
        "same_group_serialization": [
            {"group_index": group_index,
             "stage_indices": [i for i, index in enumerate(indices) if index == group_index]}
            for group_index in sorted(counts)
        ],
        "caller_included_in_threads": True,
        "tvm_exclude_worker0": 1,
        "host_mask": host,
        "control_mask": control,
        "host_compute_overlap": overlap,
        "host_shares_compute": bool(overlap),
        "control_compute_overlap": sorted(set(control or ()).intersection(assigned)),
        "process_cpu_affinity": None,
        "forbid_process_affinity_override": True,
        "requires_runtime_affinity_probe": True,
        "project_extension": bool(extensions),
        "project_extensions": extensions,
    }
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
    result["configuration_id"] = "runtime_" + sha256(encoded.encode()).hexdigest()[:24]
    result["source_references"] = deepcopy(SOURCE_REFERENCES)
    return result


def validate_runtime_contract(contract):
    """Reject altered maps, unsafe role placement, or stale execution identity."""
    if not isinstance(contract, dict):
        raise ValueError("runtime contract must be an object")
    try:
        extensions = contract["project_extensions"]
        expected = build_runtime_contract(
            contract["cpu_groups"], contract["stage_group_indices"],
            physical_cores=contract["physical_cores"],
            host_mask=contract["host_mask"], control_mask=contract["control_mask"],
            allow_stage_group_reuse="nonadjacent_cpu_stage_group_reuse" in extensions,
            allow_host_compute_overlap=contract["host_shares_compute"],
        )
    except (KeyError, TypeError) as error:
        raise ValueError("runtime contract is incomplete") from error
    if expected != contract:
        raise ValueError("runtime contract fields or identity do not match the execution mapping")
    return contract
