#!/usr/bin/env python3
"""Pure protocol, provenance, and scoring for the paired C1 memory ablation.

No board calls or performance labels are needed to freeze a support.  The full
support is retained by every ordering, including configurations that later fail
numerical qualification.  This module intentionally does not import TVM.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

from shared_memory_topk import pareto_layers


INPUT_SCHEMA = "c1_memory_ablation_inventory_v1"
CONTRACT_SCHEMA = "c1_memory_ablation_contract_v1"
BOOT_SCHEMA = "c1_memory_ablation_boot_v1"
FACTORS = {
    "B": "framework_materialization_bytes",
    "C": "additional_slot_bytes",
    "D": "vta_dma_calls",
    "I": "vta_islands",
}
QUOTAS = {1: 3, 2: 3, 3: 2}
SEEDS = tuple(range(58001, 58021))
CAPACITIES = tuple((1 << 20) * n for n in (1, 2, 4, 8))
COMMON_ARTIFACTS = ("canonical_qir", "weights", "tile_history", "runtime", "driver", "bitstream", "runner", "inputs")
QUALIFICATION_KEYS = ("queue_worker_binding", "global_vta_mutex", "fresh_completion",
                      "store_before_finish", "cross_frame_outputs", "slot_generation")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def seal(value):
    result = dict(value)
    result.pop("artifact_sha256", None)
    result["artifact_sha256"] = digest(result)
    return result


def verify_seal(value):
    unsealed = dict(value)
    expected = unsealed.pop("artifact_sha256", None)
    if expected != digest(unsealed):
        raise ValueError("artifact seal mismatch")


def finite(value, label, positive=False, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("{} must be a known number".format(label))
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError("invalid {}".format(label))
    if integer and int(value) != value:
        raise ValueError("{} must be an integer".format(label))
    return int(value) if integer else float(value)


def verify_ref(ref, root=Path(".")):
    if not isinstance(ref, dict) or not isinstance(ref.get("path"), str):
        raise ValueError("artifact requires path and sha256")
    path = Path(ref["path"])
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    expected = ref.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64 or file_hash(path) != expected:
        raise ValueError("artifact hash mismatch: {}".format(path))
    return {"path": str(path), "sha256": expected}


def label_free(value):
    """Reject labels anywhere in an inventory; component profile sources are refs."""
    if isinstance(value, dict):
        for key, item in value.items():
            name = key.lower()
            if key != "new_performance_labels_read" and any(
                word in name for word in ("observed_ii", "measured_ii", "board_latency",
                                          "performance_label", "throughput_label", "oracle_rank", "labels_ii")
            ):
                raise ValueError("forbidden performance label: " + key)
            label_free(item)
    elif isinstance(value, list):
        for item in value:
            label_free(item)


def mask_name(factors):
    return "".join(f for f in FACTORS if f in factors) or "NONE"


def base_key(row):
    return (row["m1_score_ms"], row["topology_id"],
            0 if row["policy"] == "ordinary_copy" else 1, row["configuration_id"])


def ordered_support(rows, factors, first_id):
    baseline = sorted(rows, key=base_key)
    first = next(row for row in baseline if row["configuration_id"] == first_id)
    rest = [row for row in baseline if row is not first]
    if not factors:
        return [first_id] + [r["configuration_id"] for r in rest]
    limit = first["m1_score_ms"] * 1.02
    band = [r for r in rest if r["m1_score_ms"] <= limit + 1e-12]
    outside = [r for r in rest if r["m1_score_ms"] > limit + 1e-12]
    layers = pareto_layers(band, tuple(FACTORS[f] for f in FACTORS if f in factors))
    # Keep the policy tie rule identical to M1. A constant-only feature must
    # never change copy/shared order merely through configuration hash order.
    for layer in layers:
        layer.sort(key=lambda r: tuple(r[FACTORS[f]] for f in FACTORS if f in factors) + base_key(r))
    return [first_id] + [r["configuration_id"] for layer in layers for r in layer] + [
        r["configuration_id"] for r in outside]


def coupling(rows):
    result = {}
    for left, right in itertools.combinations(FACTORS, 2):
        a, b = ([float(r[FACTORS[f]]) for r in rows] for f in (left, right))
        if not a or len(set(a)) < 2 or len(set(b)) < 2:
            result[left + right] = {"pearson": None, "reason": "constant_or_empty"}
            continue
        mean_a, mean_b = statistics.mean(a), statistics.mean(b)
        da, db = [x - mean_a for x in a], [x - mean_b for x in b]
        corr = sum(x * y for x, y in zip(da, db)) / math.sqrt(
            sum(x * x for x in da) * sum(y * y for y in db))
        ratios = [y / x for x, y in zip(a, b) if x]
        proportional = (not any(x == 0 and y != 0 for x, y in zip(a, b)) and
                        bool(ratios) and max(ratios) - min(ratios) <= 1e-10)
        result[left + right] = {"pearson": corr, "strong_coupling": abs(corr) >= 0.95,
                                "constant_ratio": proportional}
    return result


def _validate_base(row):
    if row.get("static_legal") is not True:
        raise ValueError("inventory contains a statically illegal candidate")
    finite(row.get("m1_score_ms"), "M1", positive=True)
    stages, threads = row.get("stages", []), row.get("thread_vector", [])
    if not stages or len(stages) != len(threads):
        raise ValueError("complete stage and thread vectors required")
    cursor, devices = 0, []
    for stage, threads_n in zip(stages, threads):
        if stage.get("begin") != cursor or stage.get("end", 0) <= cursor:
            raise ValueError("stages must cover a contiguous dependency-closed 21-unit chain")
        cursor = stage["end"]
        device = stage.get("device")
        if device not in ("cpu", "vta") or (devices and devices[-1] == device):
            raise ValueError("stage devices must alternate CPU/VTA")
        devices.append(device)
        if threads_n not in (1, 2, 3, 4) or isinstance(threads_n, bool):
            raise ValueError("invalid CPU thread count")
        if device == "vta" and threads_n != 1:
            raise ValueError("VTA runtime thread count must be one")
        if stage.get("cpu_affinity") != (list(range(threads_n)) if device == "cpu" else []):
            raise ValueError("explicit prefix CPU affinity required")
        if not stage.get("input_names") or not stage.get("output_node_ids"):
            raise ValueError("complete live-frontier metadata required")
    if cursor != 21 or devices[0] != "cpu" or devices[-1] != "cpu":
        raise ValueError("support must cover 21 units, starting/ending on CPU")
    if devices.count("vta") != row.get("vta_islands") or row["vta_islands"] not in QUOTAS:
        raise ValueError("island count disagrees with stage vector")


def freeze(payload, root=Path(".")):
    if payload.get("schema") != INPUT_SCHEMA or payload.get("new_performance_labels_read") is not False:
        raise ValueError("a pre-label inventory is required")
    label_free(payload)
    if not isinstance(payload.get("prior_knowledge"), list):
        raise ValueError("prior knowledge disclosure required (an empty list is allowed)")
    common = {name: verify_ref(payload["artifacts"][name], root) for name in COMMON_ARTIFACTS}
    q = payload.get("queue_capacity_bytes", {})
    for name in ("instruction", "uop"):
        finite(q.get(name), name + " capacity", positive=True, integer=True)
        if q[name] % 256 or q[name] > (1 << 25):
            raise ValueError("queue capacity must be 256-byte aligned and at most 32 MiB")
    if payload.get("model") != "pretrained MXNet ResNet18_v1@224" or payload.get("extra_quantize_calls") != 0:
        raise ValueError("current pretrained, once-quantized ResNet18 is required")
    rows = payload.get("candidates", [])
    if not rows or len({r["candidate_id"] for r in rows}) != len(rows):
        raise ValueError("unique static candidate identities required")
    by_topology, topology_names = {}, {}
    for row in rows:
        _validate_base(row)
        partition = tuple((s["begin"], s["end"], s["device"]) for s in row["stages"])
        if row["topology_id"] in topology_names and topology_names[row["topology_id"]] != partition:
            raise ValueError("one topology-id refers to different stage partitions")
        topology_names[row["topology_id"]] = partition
        by_topology.setdefault(partition, []).append(row)
    best = [min(group, key=lambda r: (r["m1_score_ms"], r["candidate_id"]))
            for group in by_topology.values()]
    selected = []
    for islands, count in QUOTAS.items():
        group = sorted((r for r in best if r["vta_islands"] == islands),
                       key=lambda r: (r["m1_score_ms"], r["topology_id"]))
        if len(group) < count:
            raise ValueError("insufficient {}-island topologies: need {}".format(islands, count))
        selected.extend(group[:count])
    support, sources = [], list(common.values())
    for source in selected:
        row = json.loads(json.dumps(source))
        stages = row["stages"]
        for stage in stages:
            for field in ("graph", "lib", "params"):
                stage[field] = verify_ref(stage[field], root)
                sources.append(stage[field])
            if stage["device"] == "vta":
                stage["final_tir"] = verify_ref(stage["final_tir"], root)
                sources.append(stage["final_tir"])
        evidence = {}
        for factor in ("B", "C", "D"):
            refs = row.get("feature_sources", {}).get(factor, [])
            if not refs:
                raise ValueError("missing feature provenance: " + factor)
            evidence[factor] = [verify_ref(ref, root) for ref in refs]
            sources.extend(evidence[factor])
        plan_ref = verify_ref(row["shared_plan"], root)
        sources.append(plan_ref)
        plan = read_json(plan_ref["path"])
        if plan.get("schema") != "shared-buffer-plan-v1" or len(plan.get("edges", [])) != len(stages) - 1:
            raise ValueError("exact shared plan must cover all edges")
        if len(plan.get("stages", [])) != len(stages):
            raise ValueError("shared plan stage coverage mismatch")
        for actual, declared in zip(stages, plan["stages"]):
            if actual["device"] != declared["device"] or actual["input_names"] != declared["input_names"]:
                raise ValueError("shared plan stage interface mismatch")
            for field in ("graph", "lib", "params"):
                if actual[field]["sha256"] != declared[field + "_sha256"]:
                    raise ValueError("shared plan/program identity mismatch")
        for name in ("runtime", "driver"):
            if plan.get(name + "_sha256") != common[name]["sha256"]:
                raise ValueError("shared plan/runtime identity mismatch")
        anchor = False
        for index, edge in enumerate(plan["edges"]):
            if edge.get("producer_stage") != index or edge.get("consumer_stage") != index + 1:
                raise ValueError("shared plan edge identity mismatch")
            if edge.get("source") not in ("external", "pool-anchor"):
                raise ValueError("unsupported shared source")
            anchor |= edge["source"] == "pool-anchor"
        external_bytes = sum(finite(edge.get("external_k2_bytes"), "external K2 bytes", integer=True)
                             for edge in plan["edges"])
        saved_bytes = finite(plan.get("predicted_saved_bytes"), "anchor saved bytes", integer=True)
        if saved_bytes > external_bytes or row.get("additional_slot_bytes") != external_bytes - saved_bytes:
            raise ValueError("slot feature disagrees with exact shared plan")
        for field in ("framework_materialization_bytes", "shared_framework_materialization_bytes",
                      "additional_slot_bytes", "vta_dma_calls"):
            finite(row.get(field), field, integer=True, positive=field in ("additional_slot_bytes", "vta_dma_calls"))
        identity = {"topology_id": row["topology_id"], "thread_vector": row["thread_vector"],
                    "stages": stages, "artifacts": common, "queue_capacity_bytes": q}
        pair_id = digest(identity)
        for policy in ("ordinary_copy", "pool_anchor_k2_shared" if anchor else "external_k2_shared"):
            config_identity = {**identity, "policy": policy,
                               "shared_plan": plan_ref if policy != "ordinary_copy" else None}
            config_id = digest(config_identity)
            support.append({"configuration_id": config_id, "candidate_id": config_id,
                            "base_candidate_id": row["candidate_id"], "pair_id": pair_id,
                            "topology_id": row["topology_id"], "policy": policy,
                            "m1_score_ms": float(row["m1_score_ms"]),
                            "identity": config_identity, "feature_sources": evidence,
                            "vta_islands": row["vta_islands"], "vta_dma_calls": row["vta_dma_calls"],
                            "framework_materialization_bytes": row["framework_materialization_bytes"] if policy == "ordinary_copy" else row["shared_framework_materialization_bytes"],
                            "additional_slot_bytes": row["additional_slot_bytes"] if policy != "ordinary_copy" else 0})
    support.sort(key=base_key)
    first = support[0]["configuration_id"]
    orders = {}
    for size in range(5):
        for factors in itertools.combinations(FACTORS, size):
            orders[mask_name(factors)] = ordered_support(support, factors, first)
    # PCG64 exactly matches the established 58001..58020 random-control protocol.
    import numpy as np
    rest = sorted(r["configuration_id"] for r in support if r["configuration_id"] != first)
    random_orders = {str(seed): [first] + [rest[int(i)] for i in
                     np.random.Generator(np.random.PCG64(seed)).permutation(len(rest))] for seed in SEEDS}
    limit = support[0]["m1_score_ms"] * 1.02
    band = [r for r in support if r["configuration_id"] != first and r["m1_score_ms"] <= limit + 1e-12]
    factors_audit = {}
    for factor, field in FACTORS.items():
        minus = orders[mask_name(set(FACTORS) - {factor})]
        constant = len({r[field] for r in band}) < 2
        factors_audit[factor] = {"field": field, "constant_in_reorder_band": constant,
                                "identifiable": not constant,
                                "full_order_changes_when_removed": minus != orders["BCDI"],
                                "top5_changes_when_removed": minus[:5] != orders["BCDI"][:5],
                                "status": "not_identifiable" if constant else "no_ranking_effect" if minus == orders["BCDI"] else "testable"}
    base = sorted(r["configuration_id"] for r in support)
    measurement = {str(index): base[offset:] + base[:offset] for index, offset in enumerate((0, 5, 10))}
    shared = sorted((r for r in support if r["policy"] != "ordinary_copy"),
                    key=lambda r: (r["additional_slot_bytes"], r["configuration_id"]))
    boundary_ids = [shared[0]["configuration_id"], shared[-1]["configuration_id"]]
    testable = orders["BCDI"][:5] != orders["NONE"][:5]
    return seal({"schema": CONTRACT_SCHEMA, "status": "frozen_before_new_performance_labels",
                 "model": payload["model"], "artifacts": common, "queue_capacity_bytes": q,
                 "prior_knowledge": payload["prior_knowledge"], "support": support,
                 "support_identity_sha256": digest(base), "orders": orders,
                 "random_orders": random_orders, "measurement_orders": measurement,
                 "common_first": first, "band_fraction": 0.02, "factor_fields": FACTORS,
                 "identifiability": {"factors": factors_audit, "band_configuration_ids": [r["configuration_id"] for r in band],
                                      "coupling": coupling(band), "ranking_windows_allowed": testable,
                                      "reason": "different_full_and_M1_top5" if testable else "identical_full_and_M1_top5"},
                 "capacity_bytes": list(CAPACITIES), "capacity_boundary_ids": boundary_ids,
                 "sources": sorted({(r["path"], r["sha256"]) for r in sources}),
                 "protocol_sources": {name: {"path": str(Path(__file__).with_name(name).resolve()),
                                                "sha256": file_hash(Path(__file__).with_name(name))}
                                      for name in ("thesis_c1_memory_ablation.py", "thesis_c1_memory_evidence.py",
                                                   "run_thesis_c1_memory_ablation.py", "analyze_thesis_c1_memory_ablation.py",
                                                   "prepare_thesis_c1_memory_inputs.py", "thesis_e2e_ablation_protocol.py",
                                                   "shared_memory_topk.py", "thesis_completion_dependency_audit.py")},
                 "protocol": {"boot_count": 3, "qualification_real_inputs": 64,
                              "diagnostic_inputs": 3, "warmup": 5, "window_rows": 302,
                              "discard_front": 2, "budgets": [1, 2, 3, 4, 5],
                              "quality_bands": [0.02, 0.05], "udmabuf_bytes": 192 << 20,
                              "performance_trace": False, "performance_profiler": False,
                              "inference_scope": "one_model_one_ranking_task_three_boots",
                              "online_time_to_quality_measured": False}})


def validate_contract(contract, verify_files=True):
    verify_seal(contract)
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise ValueError("unexpected contract schema")
    rows = contract["support"]
    ids = {r["configuration_id"] for r in rows}
    if len(rows) != 16 or len(ids) != 16 or digest(sorted(ids)) != contract["support_identity_sha256"]:
        raise ValueError("paired support identity mismatch")
    groups = {}
    for row in rows:
        if digest(row["identity"]) != row["configuration_id"]:
            raise ValueError("configuration identity drift")
        groups.setdefault(row["pair_id"], []).append(row)
    if len(groups) != 8:
        raise ValueError("eight program pairs required")
    partitions = {tuple((s["begin"], s["end"], s["device"]) for s in pair[0]["identity"]["stages"])
                  for pair in groups.values()}
    if len(partitions) != 8 or any(sum(pair[0]["vta_islands"] == k for pair in groups.values()) != n
                                   for k, n in QUOTAS.items()):
        raise ValueError("distinct topology support or 3/3/2 island quotas differ")
    for pair in groups.values():
        if len(pair) != 2 or sum(r["policy"] == "ordinary_copy" for r in pair) != 1:
            raise ValueError("copy/shared pair incomplete")
        left, right = pair
        a, b = dict(left["identity"]), dict(right["identity"])
        for value in (a, b):
            value.pop("policy")
            value.pop("shared_plan")
        if a != b or digest(a) != left["pair_id"]:
            raise ValueError("copy/shared program, threads, Tile or environment differ")
        if any(left[k] != right[k] for k in ("m1_score_ms", "vta_dma_calls", "vta_islands", "topology_id")):
            raise ValueError("paired baseline or DMA differs")
    masks = {mask_name(c) for n in range(5) for c in itertools.combinations(FACTORS, n)}
    if set(contract["orders"]) != masks or set(contract["random_orders"]) != set(map(str, SEEDS)):
        raise ValueError("all 16 masks and 20 prescribed random controls required")
    baseline = sorted(rows, key=base_key)
    if contract["common_first"] != baseline[0]["configuration_id"] or baseline[0]["policy"] != "ordinary_copy":
        raise ValueError("common first must be the M1-best copy configuration")
    for name, order in contract["orders"].items():
        expected = ordered_support(rows, () if name == "NONE" else name, contract["common_first"])
        if order != expected:
            raise ValueError("ordering does not match frozen feature mask")
    for order in [*contract["orders"].values(), *contract["random_orders"].values(),
                  *contract["measurement_orders"].values()]:
        if len(order) != 16 or set(order) != ids:
            raise ValueError("order/support mismatch")
    for order in [*contract["orders"].values(), *contract["random_orders"].values()]:
        if order[0] != contract["common_first"]:
            raise ValueError("common first configuration changed")
    import numpy as np
    rest = sorted(ids - {contract["common_first"]})
    for seed in SEEDS:
        expected = [contract["common_first"]] + [rest[int(i)] for i in
                    np.random.Generator(np.random.PCG64(seed)).permutation(len(rest))]
        if contract["random_orders"][str(seed)] != expected:
            raise ValueError("random sequence differs from prescribed PCG64 seed")
    base = sorted(ids)
    for index, offset in enumerate((0, 5, 10)):
        if contract["measurement_orders"].get(str(index)) != base[offset:] + base[:offset]:
            raise ValueError("measurement order rotation differs")
    expected_gate = contract["orders"]["BCDI"][:5] != contract["orders"]["NONE"][:5]
    if contract["identifiability"]["ranking_windows_allowed"] is not expected_gate:
        raise ValueError("identifiability stop was changed")
    if verify_files:
        for path, expected in contract["sources"]:
            if file_hash(path) != expected:
                raise ValueError("frozen source drift: " + path)
        for ref in contract["protocol_sources"].values():
            verify_ref(ref)


def score_order(order, labels, quality_band=0.02, capacity=None, rows=None, prefilter=False):
    by_id = {r["configuration_id"]: r for r in (rows or [])}
    feasible = lambda key: capacity is None or by_id[key]["additional_slot_bytes"] <= capacity
    valid = {key: value for key, value in labels.items() if value is not None and feasible(key)}
    if not valid:
        return {"oracle_ii_ms": None, "first_hit": None, "curve": [], "status": "no_qualified_oracle"}
    oracle = min(valid.values())
    sequence = [key for key in order if feasible(key)] if prefilter else list(order)
    best, first, first_qualified, rejected, curve = None, None, None, 0, []
    for index, key in enumerate(sequence, 1):
        if not feasible(key):
            rejected += 1
        elif labels.get(key) is not None:
            best = labels[key] if best is None else min(best, labels[key])
            if first_qualified is None:
                first_qualified = index
        hit = best is not None and best <= oracle * (1.0 + quality_band) + 1e-12
        if hit and first is None:
            first = index
        curve.append({"k": index, "hit": hit, "best_ii_ms": best,
                      "regret": None if best is None else (best - oracle) / oracle,
                      "capacity_rejections": rejected})
    return {"oracle_ii_ms": oracle, "first_hit": first, "first_qualified": first_qualified,
            "first_hit_within_5": first if first and first <= 5 else None,
            "right_censored_at_5": first is None or first > 5, "curve": curve,
            "status": "evaluated", "deployable_configurations": len(valid),
            "capacity_rejections_total": rejected}


def improvement(full, baseline):
    if not full.get("curve") or not baseline.get("curve"):
        return False
    h, before = full["first_hit"], baseline["first_hit"]
    after_r = full["curve"][min(4, len(full["curve"]) - 1)]["regret"]
    before_r = baseline["curve"][min(4, len(baseline["curve"]) - 1)]["regret"]
    return bool(h is not None and h <= 5 and (before is None or h < before) and
                after_r is not None and (before_r is None or after_r <= before_r + 1e-12))


def evaluate_orders(contract, labels):
    if set(labels) != {r["configuration_id"] for r in contract["support"]}:
        raise ValueError("labels must retain the complete frozen support, including failed positions")
    if not any(value is not None for value in labels.values()):
        raise ValueError("no qualified oracle in the frozen support")
    output = {}
    for band in (0.02, 0.05):
        methods = {name: score_order(order, labels, band) for name, order in contract["orders"].items()}
        random = {seed: score_order(order, labels, band) for seed, order in contract["random_orders"].items()}
        marginals = []
        for mask, baseline in methods.items():
            factors = set() if mask == "NONE" else set(mask)
            for factor in FACTORS.keys() - factors:
                extended_mask = mask_name(factors | {factor})
                extended = methods[extended_mask]
                marginals.append({"factor_added": factor, "from_mask": mask, "to_mask": extended_mask,
                                  "first_hit_before": baseline["first_hit"], "first_hit_after": extended["first_hit"],
                                  "regret_at_5_before": baseline["curve"][4]["regret"],
                                  "regret_at_5_after": extended["curve"][4]["regret"],
                                  "improved_by_primary_rule": improvement(extended, baseline)})
        random_curve = []
        for k in range(16):
            regrets = [r["curve"][k]["regret"] for r in random.values()]
            random_curve.append({"k": k + 1,
                                 "hit_rate": statistics.mean(int(r["curve"][k]["hit"]) for r in random.values()),
                                 "mean_regret": None if None in regrets else statistics.mean(regrets)})
        first_hits = [r["first_hit"] for r in random.values()]
        full_curve = methods["BCDI"]["curve"]
        comparable = all(full_curve[k]["regret"] is not None and random_curve[k]["mean_regret"] is not None
                         for k in range(5))
        output[str(band)] = {"methods": methods, "random": random,
                             "marginal_factor_effects": sorted(marginals, key=lambda r: (r["from_mask"], r["factor_added"])),
                             "random_summary": {"first_hit_by_seed": {seed: r["first_hit"] for seed, r in random.items()},
                                                "first_hit_distribution": {str(k): first_hits.count(k) for k in sorted(set(first_hits), key=lambda k: 17 if k is None else k)},
                                                "curve": random_curve,
                                                "full_pointwise_no_worse_than_random_mean_at_K1_to_5": None if not comparable else all(
                                                    int(full_curve[k]["hit"]) >= random_curve[k]["hit_rate"] and
                                                    full_curve[k]["regret"] <= random_curve[k]["mean_regret"] + 1e-12
                                                    for k in range(5))},
                             "full_vs_M1_improved": improvement(methods["BCDI"], methods["NONE"]),
                             "conditional_factor_contribution": {
                                 factor: improvement(methods["BCDI"], methods[mask_name(set(FACTORS) - {factor})])
                                 if contract["identifiability"]["factors"][factor]["identifiable"] else None
                                 for factor in FACTORS}}
    return output
