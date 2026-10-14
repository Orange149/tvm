#!/usr/bin/env python3
"""Same-budget reference evaluation for exact selector-v3 graph pools."""

from __future__ import annotations

import hashlib
import math
import statistics
from typing import Any, Dict, List, Mapping, Sequence, Tuple


SCHEMA = "c3_selector_v3_same_budget_evaluation_v1"
DEFAULT_BUDGETS = (1, 2, 3)
DEFAULT_RANDOM_SEED = 20260924
DEFAULT_RANDOM_REPLICATES = 1000


def _finite(value):
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _unknown_last(value) -> Tuple[int, float]:
    parsed = _finite(value)
    return (1, 0.0) if parsed is None else (0, parsed)


def _metrics(row):
    return row.get("aggregate_metrics") or row.get("candidate_metrics") or {}


def _target(row, incumbent=False):
    if incumbent and row.get("target_log_latency_ratio") is None:
        return 0.0
    explicit = _finite(row.get("target_log_latency_ratio"))
    if explicit is not None:
        return explicit
    candidate = _finite(row.get("candidate_latency_ms"))
    baseline = _finite(row.get("incumbent_latency_ms"))
    if candidate is None or baseline is None or candidate <= 0.0 or baseline <= 0.0:
        return None
    return math.log(candidate / baseline)


def _valid(row):
    if row.get("fpga_valid") is not None:
        return row.get("fpga_valid") is True
    return row.get("fpga_correct") is True


def validate_pool(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("evaluation pool is empty")
    ids = [str(row.get("candidate_set_id", "")) for row in rows]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("candidate_set_id values must be non-empty and unique")
    incumbents = [row for row in rows if bool(row.get("is_incumbent"))]
    if len(incumbents) != 1:
        raise ValueError("evaluation pool requires exactly one incumbent")
    task_ids = {str(row.get("execution_task_id", "")) for row in rows}
    if len(task_ids) != 1 or "" in task_ids:
        raise ValueError("evaluation pool must contain one non-empty execution_task_id")
    for row in rows:
        if _valid(row) and _target(row, incumbent=bool(row.get("is_incumbent"))) is None:
            raise ValueError("FPGA-valid candidate lacks comparable latency target: {}".format(
                row["candidate_set_id"]))
    return {"execution_task_id": next(iter(task_ids)), "incumbent": incumbents[0],
            "new_rows": [row for row in rows if not bool(row.get("is_incumbent"))]}


def deterministic_orders(rows: Sequence[Mapping[str, Any]]) -> Dict[str, List[str]]:
    new_rows = list(rows)
    selector_ready = all(
        _finite(row.get("predicted_graph_log_latency_ratio")) is not None
        and row.get("prediction_source") == "frozen_completion_aware_ridge"
        for row in new_rows
    )
    orders = {
        "bytes_only": sorted(new_rows, key=lambda row: (
            _unknown_last(_metrics(row).get("total_request_bytes")),
            _unknown_last(_metrics(row).get("total_request_calls")),
            str(row["candidate_set_id"]))),
        "calls_only": sorted(new_rows, key=lambda row: (
            _unknown_last(_metrics(row).get("total_request_calls")),
            _unknown_last(_metrics(row).get("total_request_bytes")),
            str(row["candidate_set_id"]))),
        "command_only": sorted(new_rows, key=lambda row: (
            _unknown_last(_metrics(row).get("submissions")),
            _unknown_last(_metrics(row).get("uop_load_bytes")),
            _unknown_last(_metrics(row).get("instruction_bytes")),
            str(row["candidate_set_id"]))),
    }
    if selector_ready:
        orders["selector_v3"] = sorted(new_rows, key=lambda row: (
            float(row["predicted_graph_log_latency_ratio"]),
            str(row["candidate_set_id"])))
    return {method: [str(row["candidate_set_id"]) for row in ordered]
            for method, ordered in orders.items()}


def fixed_random_orders(candidate_ids, seed=DEFAULT_RANDOM_SEED,
                        count=DEFAULT_RANDOM_REPLICATES):
    ids = sorted(str(value) for value in candidate_ids)
    return [sorted(ids, key=lambda candidate_id: hashlib.sha256(
        "{}:{}:{}".format(seed, replicate, candidate_id).encode("utf-8")
    ).hexdigest()) for replicate in range(int(count))]


def evaluate_order(rows: Sequence[Mapping[str, Any]], order: Sequence[str], budget: int):
    pool = validate_pool(rows)
    by_id = {str(row["candidate_set_id"]): row for row in rows}
    incumbent = pool["incumbent"]
    incumbent_id = str(incumbent["candidate_set_id"])
    new_ids = {str(row["candidate_set_id"]) for row in pool["new_rows"]}
    ordered = [str(value) for value in order]
    if len(ordered) != len(set(ordered)) or set(ordered) != new_ids:
        raise ValueError("order must contain every new candidate exactly once")
    attempted = ordered[:max(0, int(budget))]
    eligible = [incumbent_id] + [candidate_id for candidate_id in attempted
                                 if _valid(by_id[candidate_id])]
    winner = min(eligible, key=lambda candidate_id: (
        _target(by_id[candidate_id], incumbent=(candidate_id == incumbent_id)), candidate_id))
    oracle_eligible = [incumbent_id] + [candidate_id for candidate_id in sorted(new_ids)
                                           if _valid(by_id[candidate_id])]
    oracle = min(oracle_eligible, key=lambda candidate_id: (
        _target(by_id[candidate_id], incumbent=(candidate_id == incumbent_id)), candidate_id))
    selected_target = _target(by_id[winner], incumbent=(winner == incumbent_id))
    oracle_target = _target(by_id[oracle], incumbent=(oracle == incumbent_id))
    regret = max(0.0, math.exp(selected_target - oracle_target) - 1.0)
    return {
        "budget_new_candidates": int(budget),
        "attempted_new_candidate_ids": attempted,
        "fpga_invalid_attempt_count": sum(not _valid(by_id[value]) for value in attempted),
        "timing_eligible_candidate_ids": eligible,
        "selected_candidate_set_id": winner,
        "oracle_candidate_set_id": oracle,
        "selected_log_latency_ratio": selected_target,
        "oracle_log_latency_ratio": oracle_target,
        "simple_regret_fraction": regret,
        "improvement_over_incumbent_fraction": 1.0 - math.exp(selected_target),
        "oracle_plus_1pct_hit": selected_target <= oracle_target + math.log(1.01),
        "oracle_plus_2pct_hit": selected_target <= oracle_target + math.log(1.02),
        "invalid_attempts_consume_budget_without_replacement": True,
    }


def evaluate_task_pool(rows: Sequence[Mapping[str, Any]], budgets=DEFAULT_BUDGETS,
                       random_seed=DEFAULT_RANDOM_SEED,
                       random_replicates=DEFAULT_RANDOM_REPLICATES):
    pool = validate_pool(rows)
    orders = deterministic_orders(pool["new_rows"])
    deterministic = {method: {"B{}".format(budget): evaluate_order(rows, order, budget)
                              for budget in budgets}
                     for method, order in orders.items()}
    incumbent_order = [str(row["candidate_set_id"]) for row in pool["new_rows"]]
    deterministic["incumbent_only"] = {
        "B0": evaluate_order(rows, incumbent_order, 0)
    }
    random_results = {}
    random_orders = fixed_random_orders(
        [row["candidate_set_id"] for row in pool["new_rows"]],
        seed=random_seed, count=random_replicates,
    )
    for budget in budgets:
        samples = [evaluate_order(rows, order, budget) for order in random_orders]
        regrets = sorted(sample["simple_regret_fraction"] for sample in samples)
        random_results["B{}".format(budget)] = {
            "replicates": len(samples),
            "mean_simple_regret_fraction": sum(regrets) / len(regrets),
            "median_simple_regret_fraction": statistics.median(regrets),
            "oracle_plus_2pct_hit_rate": sum(sample["oracle_plus_2pct_hit"]
                                               for sample in samples) / len(samples),
            "mean_fpga_invalid_attempts": sum(sample["fpga_invalid_attempt_count"]
                                                for sample in samples) / len(samples),
        }
    return {
        "schema": SCHEMA,
        "execution_task_id": pool["execution_task_id"],
        "pool_size_including_incumbent": len(rows),
        "budgets_are_additional_new_candidates": True,
        "deterministic_methods": deterministic,
        "fixed_random": {"seed": int(random_seed), "results": random_results},
        "oracle_scope": "FPGA-valid candidates in this fully measured declared pool",
    }
