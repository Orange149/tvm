"""Explain frozen configurations using component costs, never pipeline FPS."""
from math import isclose, isfinite

from .costs import (Costs, FIELDS, NS_PER_MS, allocate_core_ns, ddr_stage_service,
                    quantize_ns, score_configuration)
from .domain import CORES, context


def _selected_scores(rankings, wanted):
    """Preserve each method's own prediction; do not rescore the prefix method."""
    result = {identifier: {} for identifier in wanted}
    series = {name: item.get("rankings", [])
              for name, item in rankings.get("methods", {}).items()}
    series.update({"enhanced_pure": rankings.get("pure_static20", []),
                   "enhanced_diverse": rankings.get("diverse20", [])})
    for method, rows in series.items():
        for position, row in enumerate(rows, 1):
            identifier = row.get("id")
            if identifier not in result:
                continue
            value = row.get("score_ms")
            if type(value) not in (int, float) or not isfinite(value) or value < 0:
                raise ValueError("invalid frozen method score")
            entry = {"score_ms": value, "selection_position": position,
                     "static_rank": row.get("static_rank"),
                     "selection_reason": row.get("selection_reason", "static_score")}
            if method in result[identifier] and result[identifier][method] != entry:
                raise ValueError("duplicate method ranking identity")
            result[identifier][method] = entry
    return result


def explain_pool(pool, records, ddr, rankings):
    """Return physical resource accounting for an already frozen shortlist.

    Identical-mask service sums are shown for all configurations, but constrain
    the model only when the actual executor's resource-group gate is enabled.
    Shared-host work has no invented fixed per-core placement. Producer and
    collector work remains explicitly unmodelled.
    """
    if isinstance(pool, dict):
        configurations = list(pool.values())
        if any(key != value.get("id") for key, value in pool.items()):
            raise ValueError("pool map key differs from configuration identity")
    else:
        configurations = list(pool)
    identifiers = [value["id"] for value in configurations]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate configuration in explanation pool")
    selected = _selected_scores(rankings, identifiers)
    # This report explains the main balanced model. Session-share sensitivity
    # is a different, separately labelled diagnostic.
    costs = Costs(records.records if isinstance(records, Costs) else records,
                  ddr=ddr, distribution="balanced")
    explanations = {}
    for config in sorted(configurations, key=lambda c: c["id"]):
        model = score_configuration(config, costs, ddr)
        stages, cpu_compute, cpu_boundary = [], [0] * 4, [0] * 4
        isolated_group_sums, group_members, vta_members = {}, {}, []
        for index, row in enumerate(config["stages"]):
            value = costs.get(row, config)
            wall_ns, run_ns, boundary_ns = (quantize_ns(value[k]) for k in
                                           ("service_ms", "run_core_ms", "boundary_core_ms"))
            ddr_ns = quantize_ns(ddr_stage_service(ddr, row)) if ddr is not None else None
            item = {"stage_index": index, "program": row["program"], "device": row["device"],
                    "begin": row["begin"], "end": row["end"], "threads": row["threads"],
                    "mask": list(row["mask"]), "worker_core_map": list(row["worker_core_map"]),
                    "caller_core": row["caller_core"],
                    "cost": {key: value[key] for key in FIELDS if key in value},
                    "component_source": {key: value[key] for key in
                                         ("source_key", "requested_key", "kind", "assumptions")},
                    "quantized_cost_ns": {"service": wall_ns, "run_core": run_ns,
                                          "boundary_core": boundary_ns, "ddr_service": ddr_ns},
                    "ddr_service_ms": ddr_ns / NS_PER_MS if ddr_ns is not None else None}
            if row["device"] == "cpu":
                item["compute_weights"] = list(value["cpu_core_weights"])
                item["compute_weight_scope"] = "uniform_split_of_scored_isolated_total_not_per_thread_measurement"
                item["compute_core_demand_ms"] = [0.] * 4
                item["boundary_core_demand_ms"] = [0.] * 4
                allocated = allocate_core_ns(run_ns, value["cpu_core_weights"], threads=row["threads"])
                item["allocated_compute_ns_by_slot"] = list(allocated)
                for core, work in zip(row["mask"], allocated):
                    cpu_compute[core] += work
                    item["compute_core_demand_ms"][core] += work / NS_PER_MS
                caller = row["caller_core"]
                cpu_boundary[caller] += boundary_ns
                item["boundary_core_demand_ms"][caller] = boundary_ns / NS_PER_MS
                group = ",".join(map(str, row["mask"]))
                isolated_group_sums[group] = isolated_group_sums.get(group, 0) + wall_ns
                group_members.setdefault(group, []).append(row["program"])
            else:
                item["host_core_ms"] = (run_ns + boundary_ns) / NS_PER_MS
                vta_members.append({"program": row["program"], "service_ms": wall_ns / NS_PER_MS,
                                    "host_core_ms": item["host_core_ms"]})
            stages.append(item)
        pinned_host = len(config["host_mask"]) == 1
        host_core = config["host_mask"][0] if pinned_host else None
        core_rows = []
        for core in CORES:
            cpu_total = cpu_compute[core] + cpu_boundary[core]
            host_fixed = model["host_core_ns"] if pinned_host and core == host_core else 0
            core_rows.append({"core": core, "cpu_compute_ns": cpu_compute[core],
                              "cpu_boundary_ns": cpu_boundary[core], "cpu_total_ns": cpu_total,
                              "fixed_host_ns": host_fixed, "known_fixed_demand_ns": cpu_total + host_fixed,
                              "cpu_compute_ms": cpu_compute[core] / NS_PER_MS,
                              "cpu_boundary_ms": cpu_boundary[core] / NS_PER_MS,
                              "cpu_total_ms": cpu_total / NS_PER_MS,
                              "fixed_host_ms": host_fixed / NS_PER_MS,
                              "known_fixed_demand_ms": (cpu_total + host_fixed) / NS_PER_MS,
                              "shared_host_may_use_core": core in config["host_mask"] and not pinned_host,
                              "control_work_ms": None})
        host_bits = sum(1 << c for c in config["host_mask"])
        subsets = []
        for bits in range(1, 16):
            cores = [core for core in CORES if bits & (1 << core)]
            fixed = sum(model["cpu_core_demand_ns"][core] for core in cores)
            host = model["host_core_ns"] if bits & host_bits == host_bits else 0
            subsets.append({"cores": cores, "cpu_core_work_ns": fixed,
                            "confined_host_core_work_ns": host,
                            "cpu_core_work_ms": fixed / NS_PER_MS,
                            "confined_host_core_work_ms": host / NS_PER_MS,
                            "capacity_bound_ns": model["subset_capacity_ns"][bits - 1],
                            "capacity_bound_ms": model["subset_capacity_ms"][bits - 1]})
        terms = [{"term": "maximum_CPU_stage_service", "value_ms": model["cpu_service_max_ms"]},
                 {"term": "all_VTA_islands_service", "value_ms": model["vta_service_sum_ms"]}]
        terms += [{"term": "core_subset_" + ",".join(map(str, r["cores"])),
                   "value_ms": r["capacity_bound_ms"]} for r in subsets]
        terms += [{"term": "enabled_serial_group_" + group, "value_ms": value}
                  for group, value in sorted(model["group_services_ms"].items())]
        if ddr is not None:
            terms.append({"term": "shared_DDR_service", "value_ms": model["ddr_service_ms"]})
        maximum = max(term["value_ms"] for term in terms)
        bottlenecks = [term["term"] for term in terms
                       if isclose(term["value_ms"], maximum, rel_tol=1e-10, abs_tol=1e-8)]
        explanations[config["id"]] = {
            "context_id": config["context_id"], "context": context(config)["label"],
            "execution_mode": config["execution_mode"], "compute_pool": list(config["compute_pool"]),
            "reserved_cores": list(config["reserved_cores"]), "stages": stages,
            "physical_cores": core_rows, "host_allowed_mask": list(config["host_mask"]),
            "host_total_core_ms": model["host_core_ms"],
            "host_placement_model": "fixed_single_core" if pinned_host else "ideal_movable_within_allowed_set",
            "movable_host_core_work_ms": 0. if pinned_host else model["host_core_ms"],
            "control_allowed_mask": list(config["control_mask"]),
            "control_work_ms": None, "control_cost_model_status": "not_separately_profiled_not_zero",
            "subset_capacity_bounds": subsets,
            "all_VTA_islands": vta_members, "vta_service_sum_ms": model["vta_service_sum_ms"],
            "isolated_service_sum_by_mask_ms": {k: v / NS_PER_MS for k, v in sorted(isolated_group_sums.items())},
            "isolated_service_sum_by_mask_ns": dict(sorted(isolated_group_sums.items())),
            "group_members": dict(sorted(group_members.items())),
            "group_sums_are_execution_constraints": config["execution_mode"] == "resource_group",
            "enabled_serial_group_services_ms": model["group_services_ms"],
            "ddr_available": ddr is not None,
            "ddr_service_ms": model["ddr_service_ms"] if ddr is not None else None,
            "diagnostic_enhanced_score_ms": model["score_ms"], "diagnostic_enhanced_score_ns": model["score_ns"],
            "integer_accounting": model["accounting"], "resource_terms": terms,
            "bottleneck_terms": bottlenecks,
            "selected_scores_by_method": selected[config["id"]],
            "diagnostic_score_replaces_method_score": False}
    return {"scope": "frozen_pool_component_only_resource_explanation",
            "uses_pipeline_fps": False, "changes_selection": False,
            "main_compute_distribution": "balanced_not_session_share_sensitivity",
            "configuration_count": len(explanations), "configurations": explanations,
            "method_prediction_note": "Original per-method static predictions are preserved separately; the enhanced diagnostic does not replace the prefix score.",
            "gate_note": "Same-mask service sums are diagnostic only unless the actual resource-group gate is enabled.",
            "control_note": "Producer, collector and result-processing work has not been separately profiled; missing is not zero."}
