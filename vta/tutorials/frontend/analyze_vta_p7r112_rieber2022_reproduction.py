#!/usr/bin/env python3
"""Reproduce the Rieber-2022 validity initialization on the frozen C3 pool.

The frozen input is a sparse union of five residency tasks rather than a complete
AutoTVM Cartesian ConfigSpace.  This script therefore reproduces Algorithm 1 on
the *observed* same-mode Manhattan-1 graph, reproduces balanced/max-distance E0
selection on the presampled set, and provides a clearly labelled proxy for the
paper's validity-biased simulated annealing.  It never contacts a board and never
uses latency or FPGA-correctness labels.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import shlex
import statistics
import sys
from collections import defaultdict
from pathlib import Path


SCHEMA = "c3_p7r112_rieber2022_sparse_reproduction_v1"
SEED_BASE = 20250901
SEED_COUNT = 20
EXPECTED_CANDIDATES = 310
EXPECTED_VALID = 197
EXPECTED_WORKLOADS = 10
GROSS_COMPILER_BUDGET = 16
PARALLEL_BATCH = 4
E0_SIZE = 8
FIRST_K_VALID = 4
PREFIX_BUDGETS = (4, 8, 12, 16)
KNOBS = (
    "tile_b",
    "tile_h",
    "tile_w",
    "tile_ci",
    "tile_co",
    "oc_nthread",
    "h_nthread",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_frozen_pool(paths):
    rows = []
    ordinal = 0
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            item["_source_ordinal"] = ordinal
            ordinal += 1
            rows.append(item)
    return rows


def knob_values(row):
    entity = row["identity"]["complete_config_entity"]
    values = {}
    for name, kind, value in entity["entity"]:
        if name in values:
            raise ValueError("duplicate ConfigEntity knob {}".format(name))
        if kind == "sp":
            if not isinstance(value, list) or not value:
                raise ValueError("malformed split entity {}".format(name))
            value = value[-1]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("non-numeric ConfigEntity value for {}".format(name))
        if not math.isfinite(float(value)):
            raise ValueError("non-finite ConfigEntity value for {}".format(name))
        values[name] = value
    if set(values) != set(KNOBS):
        raise ValueError("unexpected ConfigEntity knobs: {}".format(sorted(values)))
    return values


def validate_frozen_pool(rows):
    candidate_ids = [row["candidate_id"] for row in rows]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate IDs must be unique across frozen pools")
    if len(rows) != EXPECTED_CANDIDATES:
        raise ValueError("expected {} frozen candidates, got {}".format(EXPECTED_CANDIDATES, len(rows)))
    valid = sum(row.get("status") == "ok" for row in rows)
    if valid != EXPECTED_VALID:
        raise ValueError("expected {} lowering-valid candidates, got {}".format(EXPECTED_VALID, valid))
    workloads = sorted({row["workload_id"] for row in rows})
    if len(workloads) != EXPECTED_WORKLOADS:
        raise ValueError("expected {} workloads, got {}".format(EXPECTED_WORKLOADS, len(workloads)))
    for row in rows:
        if row.get("status") not in ("ok", "failed"):
            raise ValueError("validity label must be local lowering status ok/failed")
        knob_values(row)
    return workloads


def coordinate_system(rows):
    """Build rank coordinates from observed knob values, separately per workload."""

    by_workload = defaultdict(list)
    for row in rows:
        by_workload[row["workload_id"]].append(row)
    systems = {}
    coordinates = {}
    for workload_id, group in by_workload.items():
        domains = {
            name: sorted({knob_values(row)[name] for row in group})
            for name in KNOBS
        }
        lookup = {}
        for row in group:
            values = knob_values(row)
            coordinate = tuple(domains[name].index(values[name]) for name in KNOBS)
            key = (row["residence_mode"], coordinate)
            if key in lookup:
                raise ValueError("duplicate observed coordinate in one residency task")
            lookup[key] = row["candidate_id"]
            coordinates[row["candidate_id"]] = coordinate
        systems[workload_id] = {"domains": domains, "lookup": lookup}
    return systems, coordinates


def knob_manhattan(left, right, coordinates):
    return sum(
        abs(a - b)
        for a, b in zip(coordinates[left["candidate_id"]], coordinates[right["candidate_id"]])
    )


def same_task_manhattan_one(left, right, coordinates):
    return (
        left["workload_id"] == right["workload_id"]
        and left["residence_mode"] == right["residence_mode"]
        and knob_manhattan(left, right, coordinates) == 1
    )


def selection_distance(left, right, coordinates):
    """Categorical-mode Hamming plus knob-rank Manhattan, only for E0 diversity.

    Residency mode is a task argument in this repository, not an AutoTVM knob.
    Treating a cross-mode difference as one categorical step is therefore a
    declared proxy and is never used to create Algorithm-1 neighbours.
    """

    return knob_manhattan(left, right, coordinates) + int(
        left["residence_mode"] != right["residence_mode"]
    )


def observed_neighbor_map(rows, coordinates):
    result = {row["candidate_id"]: set() for row in rows}
    for index, left in enumerate(rows):
        for right in rows[index + 1:]:
            if same_task_manhattan_one(left, right, coordinates):
                result[left["candidate_id"]].add(right["candidate_id"])
                result[right["candidate_id"]].add(left["candidate_id"])
    return result


def graph_audit(rows, systems, coordinates):
    by_workload = defaultdict(list)
    for row in rows:
        by_workload[row["workload_id"]].append(row)
    result = {}
    for workload_id, group in sorted(by_workload.items()):
        system = systems[workload_id]
        present_directed = 0
        possible_directed = 0
        for row in group:
            coordinate = coordinates[row["candidate_id"]]
            for axis, name in enumerate(KNOBS):
                for delta in (-1, 1):
                    target = list(coordinate)
                    target[axis] += delta
                    if not 0 <= target[axis] < len(system["domains"][name]):
                        continue
                    possible_directed += 1
                    if (row["residence_mode"], tuple(target)) in system["lookup"]:
                        present_directed += 1
        unique_entities = {
            tuple(coordinates[row["candidate_id"]])
            for row in group
        }
        result[workload_id] = {
            "candidate_count": len(group),
            "residency_task_count": len({row["residence_mode"] for row in group}),
            "unique_config_entity_coordinates": len(unique_entities),
            "observed_neighbor_edges": present_directed // 2,
            "observed_domain_possible_directed_neighbors": possible_directed,
            "observed_domain_present_directed_neighbors": present_directed,
            "observed_domain_neighbor_closure_fraction": (
                present_directed / possible_directed if possible_directed else None
            ),
            "observed_axis_cardinalities": {
                name: len(system["domains"][name]) for name in KNOBS
            },
        }
    return result


def _sample(rng, values, count):
    values = sorted(values)
    return rng.sample(values, min(count, len(values)))


def rieber_presample(rows, coordinates, n_samples, n_parallel, seed, initial_ids=None):
    """Algorithm-1 replay on observed same-task Manhattan-1 neighbours."""

    if n_samples <= 0 or n_parallel <= 0:
        raise ValueError("presample sizes must be positive")
    n_samples = min(n_samples, len(rows))
    by_id = {row["candidate_id"]: row for row in rows}
    neighbours = observed_neighbor_map(rows, coordinates)
    rng = random.Random(seed)
    unseen = set(by_id)
    if initial_ids is None:
        pending = _sample(rng, unseen, min(n_parallel, n_samples))
    else:
        pending = list(initial_ids)
        if len(pending) > n_parallel or any(item not in unseen for item in pending):
            raise ValueError("invalid explicit initial batch")
    order = []
    trace = []
    random_fills = 0
    while len(order) < n_samples:
        pending = [item for item in pending if item in unseen]
        if not pending:
            pending = _sample(rng, unseen, min(n_parallel, n_samples - len(order)))
            random_fills += len(pending)
        pending = pending[: n_samples - len(order)]
        candidate_frontier = set()
        round_rows = []
        for candidate_id in pending:
            row = by_id[candidate_id]
            unseen.remove(candidate_id)
            order.append(row)
            if row["status"] == "ok":
                additions = neighbours[candidate_id] & unseen
                candidate_frontier.update(additions)
                source = "valid_observed_manhattan1_neighbours"
            else:
                available = unseen - candidate_frontier
                additions = set(_sample(rng, available, 1))
                candidate_frontier.update(additions)
                source = "invalid_random_exploration"
            round_rows.append(
                {
                    "candidate_id": candidate_id,
                    "status": row["status"],
                    "frontier_source": source,
                    "frontier_additions": sorted(additions),
                }
            )
        next_count = min(n_parallel, n_samples - len(order))
        available_frontier = candidate_frontier & unseen
        if len(available_frontier) < next_count:
            fill = _sample(rng, unseen - available_frontier, next_count - len(available_frontier))
            candidate_frontier.update(fill)
            random_fills += len(fill)
        pending = _sample(rng, candidate_frontier & unseen, next_count)
        trace.append({"evaluated": round_rows, "next_batch": list(pending)})
    if len({row["candidate_id"] for row in order}) != len(order):
        raise AssertionError("presampling queried a candidate twice")
    return {
        "order": order,
        "trace": trace,
        "sparse_frontier_random_fill_count": random_fills,
    }


def _subset_distance_key(subset, coordinates):
    distances = [
        selection_distance(left, right, coordinates)
        for left, right in itertools.combinations(subset, 2)
    ]
    if not distances:
        return 0, 0
    return min(distances), sum(distances)


def exact_maxmin_subset(rows, count, coordinates):
    """Choose the exact max-min-distance subset; use total distance as tie-break."""

    if count <= 0:
        return []
    if count >= len(rows):
        return sorted(rows, key=lambda row: row["candidate_id"])
    best = None
    best_key = None
    for subset in itertools.combinations(rows, count):
        distance_key = _subset_distance_key(subset, coordinates)
        ids = tuple(sorted(row["candidate_id"] for row in subset))
        if best is None or distance_key > best_key or (distance_key == best_key and ids < best[0]):
            best = (ids, subset)
            best_key = distance_key
    return sorted(best[1], key=lambda row: row["candidate_id"])


def _interleave(left, right):
    result = []
    for index in range(max(len(left), len(right))):
        if index < len(left):
            result.append(left[index])
        if index < len(right):
            result.append(right[index])
    return result


def balanced_maxmin_e0(presampled, e0_size, coordinates):
    valid = [row for row in presampled if row["status"] == "ok"]
    invalid = [row for row in presampled if row["status"] != "ok"]
    valid_target = e0_size // 2
    selected_valid = exact_maxmin_subset(valid, min(valid_target, len(valid)), coordinates)
    invalid_target = e0_size - len(selected_valid)
    selected_invalid = exact_maxmin_subset(
        invalid, min(invalid_target, len(invalid)), coordinates
    )
    selected = _interleave(selected_valid, selected_invalid)
    return {
        "order": selected,
        "target_size": e0_size,
        "achieved_size": len(selected),
        "valid": len(selected_valid),
        "invalid": len(selected_invalid),
        "shortfall": e0_size - len(selected),
        "distance_metric": "categorical-mode Hamming + observed-knob-rank Manhattan proxy",
        "valid_distance": _subset_distance_key(selected_valid, coordinates),
        "invalid_distance": _subset_distance_key(selected_invalid, coordinates),
        "intra_epoch_order": "deterministic valid/invalid interleave for prefix reporting only",
    }


def _base_prediction(candidate_id, seed):
    digest = hashlib.sha256("{}:{}".format(seed, candidate_id).encode("utf-8")).digest()
    unit = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    return -7.0 + 14.0 * unit


def validity_biased_proxy_order(rows, seed):
    """Apply the paper's validity biases to deterministic synthetic base scores."""

    scored = []
    for row in rows:
        base = _base_prediction(row["candidate_id"], seed)
        score = base + 1.0 if row["status"] == "ok" else -1_000_000.0
        scored.append((score, row["candidate_id"], row))
    return [row for _, _, row in sorted(scored, key=lambda item: (-item[0], item[1]))]


def deterministic_random_order(rows, seed):
    result = list(rows)
    random.Random(seed).shuffle(result)
    return result


def calls_to_k_valid(order, k):
    valid = 0
    for index, row in enumerate(order, 1):
        valid += row["status"] == "ok"
        if valid >= k:
            return index
    return None


def prefix_metrics(order, coordinates, budgets=PREFIX_BUDGETS):
    result = {}
    for budget in budgets:
        prefix = order[: min(budget, len(order))]
        valid = sum(row["status"] == "ok" for row in prefix)
        distances = [
            selection_distance(left, right, coordinates)
            for left, right in itertools.combinations(prefix, 2)
        ]
        result[str(budget)] = {
            "attempts": len(prefix),
            "valid_yield": valid,
            "invalid": len(prefix) - valid,
            "invalid_ratio": (len(prefix) - valid) / len(prefix) if prefix else None,
            "residency_task_coverage": len({row["residence_mode"] for row in prefix}),
            "unique_config_entity_coverage": len(
                {coordinates[row["candidate_id"]] for row in prefix}
            ),
            "mean_pairwise_selection_distance": (
                statistics.fmean(distances) if distances else 0.0
            ),
            "minimum_pairwise_selection_distance": min(distances) if distances else 0,
        }
    return result


def distribution(values):
    values = list(values)
    quartiles = statistics.quantiles(values, n=4, method="inclusive") if len(values) > 1 else [values[0]] * 3
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "q1": quartiles[0],
        "q3": quartiles[2],
        "iqr": quartiles[2] - quartiles[0],
        "minimum": min(values),
        "maximum": max(values),
    }


def summarize_runs(runs):
    methods = sorted(runs[0]["compiler_metrics"])
    result = {"compiler": {}, "dispatch": {}, "e0": {}}
    for method in methods:
        result["compiler"][method] = {
            "calls_to_first_k_valid_censored": distribution(
                run["compiler_metrics"][method]["calls_to_first_k_valid_censored"]
                for run in runs
            ),
            "first_k_reached_rate": statistics.fmean(
                run["compiler_metrics"][method]["first_k_reached"] for run in runs
            ),
            "prefixes": {},
        }
        for budget in PREFIX_BUDGETS:
            key = str(budget)
            result["compiler"][method]["prefixes"][key] = {
                metric: distribution(
                    run["compiler_metrics"][method]["prefixes"][key][metric]
                    for run in runs
                )
                for metric in (
                    "valid_yield",
                    "invalid_ratio",
                    "residency_task_coverage",
                    "unique_config_entity_coverage",
                )
            }
    dispatch_methods = sorted(runs[0]["dispatch_metrics"])
    for method in dispatch_methods:
        result["dispatch"][method] = {"prefixes": {}}
        for budget in PREFIX_BUDGETS:
            key = str(budget)
            result["dispatch"][method]["prefixes"][key] = {
                metric: distribution(
                    run["dispatch_metrics"][method]["prefixes"][key][metric]
                    for run in runs
                )
                for metric in (
                    "valid_yield",
                    "invalid_ratio",
                    "residency_task_coverage",
                    "unique_config_entity_coverage",
                )
            }
    result["e0"] = {
        metric: distribution(run["e0"][metric] for run in runs)
        for metric in ("achieved_size", "valid", "invalid", "shortfall")
    }
    return result


def run_replay(rows, seed_count=SEED_COUNT):
    systems, coordinates = coordinate_system(rows)
    by_workload = defaultdict(list)
    for row in rows:
        by_workload[row["workload_id"]].append(row)
    runs = []
    for workload_offset, (workload_id, group) in enumerate(sorted(by_workload.items())):
        static_order = sorted(group, key=lambda row: row["_source_ordinal"])
        for replicate in range(seed_count):
            seed = SEED_BASE + replicate
            stream_seed = seed + workload_offset * 100_000
            random_order = deterministic_random_order(group, stream_seed)
            presample = rieber_presample(
                group,
                coordinates,
                GROSS_COMPILER_BUDGET,
                PARALLEL_BATCH,
                stream_seed,
            )
            r1_order = presample["order"]
            e0 = balanced_maxmin_e0(r1_order, E0_SIZE, coordinates)
            e0_ids = {row["candidate_id"] for row in e0["order"]}
            remaining = [row for row in r1_order if row["candidate_id"] not in e0_ids]
            r2_order = e0["order"] + remaining
            r3_order = e0["order"] + validity_biased_proxy_order(remaining, stream_seed)
            compile_orders = {
                "random": random_order[:GROSS_COMPILER_BUDGET],
                "current_static_generation_order": static_order[:GROSS_COMPILER_BUDGET],
                "r1_locality_presample": r1_order,
                # E0 selection and bias only reorder cached presample results, so
                # they cannot honestly improve gross compiler-call discovery.
                "r2_balanced_maxmin_e0": r1_order,
                "r3_validity_biased_proxy": r1_order,
            }
            dispatch_orders = {
                "random": compile_orders["random"],
                "current_static_generation_order": compile_orders[
                    "current_static_generation_order"
                ],
                "r1_locality_presample": r1_order,
                "r2_balanced_maxmin_e0": r2_order,
                "r3_validity_biased_proxy": r3_order,
            }
            for order in compile_orders.values():
                if len(order) != GROSS_COMPILER_BUDGET:
                    raise AssertionError("unequal gross compiler-call budget")
                if len({row["candidate_id"] for row in order}) != len(order):
                    raise AssertionError("gross compiler calls contain duplicates")
            compiler_metrics = {}
            for method, order in compile_orders.items():
                first_k = calls_to_k_valid(order, FIRST_K_VALID)
                compiler_metrics[method] = {
                    "first_k_reached": first_k is not None,
                    "calls_to_first_k_valid": first_k,
                    "calls_to_first_k_valid_censored": (
                        first_k if first_k is not None else GROSS_COMPILER_BUDGET + 1
                    ),
                    "prefixes": prefix_metrics(order, coordinates),
                }
            runs.append(
                {
                    "workload_id": workload_id,
                    "replicate": replicate,
                    "seed": stream_seed,
                    "gross_compiler_call_budget": GROSS_COMPILER_BUDGET,
                    "compiler_order_candidate_ids": {
                        name: [row["candidate_id"] for row in order]
                        for name, order in compile_orders.items()
                    },
                    "dispatch_order_candidate_ids": {
                        name: [row["candidate_id"] for row in order]
                        for name, order in dispatch_orders.items()
                    },
                    "compiler_metrics": compiler_metrics,
                    "dispatch_metrics": {
                        name: {"prefixes": prefix_metrics(order, coordinates)}
                        for name, order in dispatch_orders.items()
                    },
                    "e0": {key: value for key, value in e0.items() if key != "order"},
                    "presample_sparse_frontier_random_fill_count": presample[
                        "sparse_frontier_random_fill_count"
                    ],
                }
            )
    return {
        "systems": systems,
        "coordinates": coordinates,
        "graph_audit": graph_audit(rows, systems, coordinates),
        "runs": runs,
        "aggregate": summarize_runs(runs),
    }


def write_results(path, result):
    aggregate = result["aggregate"]
    lines = [
        "# P7R112 Rieber 2022 严格方法审计与稀疏池降级复现",
        "",
        "> 结论边界：`degraded_sparse_pool_replay`。这不是完整 AutoTVM ConfigSpace 上的论文原样复现，也不是板端实验。",
        "",
        "## 为什么必须降级",
        "",
        "- 310 行由 10 个 workload、5 个独立 residency task 组成；每个 workload 只有 31 行和 7 个独特 ConfigEntity。",
        "- 冻结文件没有完整 ConfigSpace/所有 Manhattan-1 邻居；同 mode 内只能复现 observed-knob-rank Manhattan-1。",
        "- 论文的 E0=50、presample=min(1000,|S|)、750 次硬件测量无法用于每个仅 31 行的稀疏池，本实验缩放为 presample=16、parallel=4、E0=8。",
        "- 310 行没有候选 latency，无法训练论文中的 AutoTVM performance model 或忠实运行 SA；第三级仅对确定性 [-7,7] 基础分数施加论文的 valid +1 / invalid -1e6 bias。",
        "",
        "## 等 gross compiler-call 结果",
        "",
        "所有方法每个 workload/seed 都只查询 16 个不重复 frozen lowering labels；R2/R3 只重排 R1 已缓存结果，因此 compiler discovery 指标与 R1 相同。",
        "",
        "| 方法 | valid@16 median [IQR] | invalid ratio@16 median [IQR] | 首4个valid calls median [IQR] | 达成率 |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = (
        ("random", "Random"),
        ("current_static_generation_order", "冻结静态生成顺序"),
        ("r1_locality_presample", "R1 locality presample"),
        ("r2_balanced_maxmin_e0", "R2（compiler顺序同R1）"),
        ("r3_validity_biased_proxy", "R3（compiler顺序同R1）"),
    )
    for key, label in labels:
        row = aggregate["compiler"][key]
        valid = row["prefixes"][str(GROSS_COMPILER_BUDGET)]["valid_yield"]
        invalid = row["prefixes"][str(GROSS_COMPILER_BUDGET)]["invalid_ratio"]
        first = row["calls_to_first_k_valid_censored"]
        lines.append(
            "| {} | {:.2f} [{:.2f}] | {:.3f} [{:.3f}] | {:.2f} [{:.2f}] | {:.1f}% |".format(
                label,
                valid["median"], valid["iqr"],
                invalid["median"], invalid["iqr"],
                first["median"], first["iqr"],
                100.0 * row["first_k_reached_rate"],
            )
        )
    lines += [
        "",
        "## E0 与 validity-bias dispatch 代理",
        "",
        "| 方法 | valid@8 median | valid@12 median | valid@16 median |",
        "|---|---:|---:|---:|",
    ]
    for key, label in labels:
        row = aggregate["dispatch"][key]["prefixes"]
        lines.append(
            "| {} | {:.2f} | {:.2f} | {:.2f} |".format(
                label,
                row["8"]["valid_yield"]["median"],
                row["12"]["valid_yield"]["median"],
                row["16"]["valid_yield"]["median"],
            )
        )
    e0 = aggregate["e0"]
    lines += [
        "",
        "E0 valid/invalid median 为 {:.1f}/{:.1f}，size shortfall median 为 {:.1f}。".format(
            e0["valid"]["median"], e0["invalid"]["median"], e0["shortfall"]["median"]
        ),
        "",
        "第三级只能说明已知 legality label 如何影响派发优先级；不能说明性能收敛、FPGA correctness 或真实 SA 收敛。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validity-pool", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable P7R112 result")
    pool_paths = [Path(path).resolve() for path in args.validity_pool]
    rows = read_frozen_pool(pool_paths)
    workloads = validate_frozen_pool(rows)
    replay = run_replay(rows)
    result = {
        "schema": SCHEMA,
        "status": "degraded_sparse_pool_replay",
        "board_contacted": False,
        "labels": "frozen local TIR-lowering ok/failed only; not FPGA correctness",
        "paper": {
            "title": "HW-Aware Initialization of DNN Auto-Tuning to Improve Exploration Time and Robustness",
            "year": 2022,
            "url": "https://arxiv.org/abs/2205.15568",
        },
        "frozen_protocol": {
            "seed_base": SEED_BASE,
            "seed_count": SEED_COUNT,
            "seeds": [SEED_BASE + index for index in range(SEED_COUNT)],
            "gross_compiler_call_budget_per_workload_seed": GROSS_COMPILER_BUDGET,
            "parallel_batch": PARALLEL_BATCH,
            "e0_size": E0_SIZE,
            "first_k_valid": FIRST_K_VALID,
            "prefix_budgets": list(PREFIX_BUDGETS),
            "workloads": workloads,
            "level_1": "Algorithm-1 control flow on observed same-mode knob-rank Manhattan-1 graph",
            "level_2": "balanced E0 with exact max-min subset selection within each validity class",
            "level_3": "ordering proxy only: synthetic [-7,7] base + valid 1 / invalid -1e6",
            "equal_budget_rule": (
                "every method queries exactly 16 unique lowering labels; E0 and bias reuse cached labels"
            ),
        },
        "fidelity": {
            "exactly_reproduced": [
                "valid point expands observed same-task Manhattan-1 neighbours",
                "invalid point adds random exploration candidate",
                "balanced valid/invalid E0 selection",
                "known-valid +1 and known-invalid -1e6 bias constants",
                "20 seeded repetitions with median and IQR",
            ],
            "degraded_or_unavailable": [
                "full Cartesian ConfigSpace and its complete Manhattan-1 graph are absent",
                "five residency modes are separate tasks and are never treated as Algorithm-1 neighbours",
                "E0/presample/trial counts are scaled to a 31-row per-workload pool",
                "cross-mode E0 diversity uses categorical Hamming plus knob-rank Manhattan proxy",
                "no latency labels, trained AutoTVM performance model, real simulated annealing, or hardware trials",
                "frozen static generation order is an audit baseline, not a claimed tuning policy",
            ],
        },
        "inputs": [
            {"path": str(path), "sha256": sha256_file(path), "rows": sum(1 for line in path.read_text().splitlines() if line.strip())}
            for path in pool_paths
        ],
        "input_summary": {
            "candidate_count": len(rows),
            "valid": sum(row["status"] == "ok" for row in rows),
            "invalid": sum(row["status"] != "ok" for row in rows),
            "workload_count": len(workloads),
        },
        "graph_audit": replay["graph_audit"],
        "aggregate": replay["aggregate"],
        "run_count": len(replay["runs"]),
        "claim_boundary": (
            "offline sparse-pool method reproduction only; no performance, FPGA correctness, unseen-workload, "
            "or exact-full-space claim"
        ),
    }
    output.mkdir(parents=True)
    analysis_path = output / "analysis.json"
    analysis_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output / "runs.jsonl").open("x", encoding="utf-8") as stream:
        for run in replay["runs"]:
            stream.write(json.dumps(run, sort_keys=True) + "\n")
    write_results(output / "RESULTS.md", result)
    (output / "command.txt").write_text(shlex.join([str(Path(__file__).resolve())] + sys.argv[1:]) + "\n")
    artifacts = {
        path.name: sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    (output / "artifact_hashes.json").write_text(
        json.dumps({"artifacts": artifacts}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": result["status"], "runs": len(replay["runs"]), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
