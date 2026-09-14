#!/usr/bin/env python3
"""Analyze frozen P7Q search policies from physically measured FPGA labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


BUDGETS = (4, 8, 16, 24)
DETERMINISTIC_POLICIES = ("B0", "B1", "B3", "B4", "B5", "B6", "B7", "B8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values, fraction):
    values = sorted(float(value) for value in values)
    if not values:
        return None
    location = (len(values) - 1) * fraction
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return values[lower]
    weight = location - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def evaluate_order(order, latency, invalid_ids, oracle_id, oracle_ms, budget):
    prefix = list(order[: min(budget, len(order))])
    observed = [(latency[candidate_id], candidate_id) for candidate_id in prefix if candidate_id in latency]
    if not observed:
        return {
            "gross_dispatches": len(prefix),
            "valid_labels": 0,
            "invalid_dispatches": sum(candidate_id in invalid_ids for candidate_id in prefix),
            "best_candidate_id": None,
            "best_ms": None,
            "regret_percent": None,
            "oracle_hit": False,
        }
    best_ms, best_id = min(observed)
    return {
        "gross_dispatches": len(prefix),
        "valid_labels": len(observed),
        "invalid_dispatches": sum(candidate_id in invalid_ids for candidate_id in prefix),
        "best_candidate_id": best_id,
        "best_ms": best_ms,
        "regret_percent": (best_ms / oracle_ms - 1.0) * 100.0,
        "oracle_hit": oracle_id in prefix,
    }


def random_order(candidate_ids, seed, workload_id, replicate):
    def key(candidate_id):
        value = f"{seed}:{workload_id}:{replicate}:{candidate_id}".encode()
        return hashlib.sha256(value).hexdigest()

    return sorted(candidate_ids, key=key)


def summarize_random(rows):
    best_values = [row["best_ms"] for row in rows if row["best_ms"] is not None]
    regrets = [row["regret_percent"] for row in rows if row["regret_percent"] is not None]
    return {
        "replicates": len(rows),
        "best_ms_median": statistics.median(best_values),
        "best_ms_p05": percentile(best_values, 0.05),
        "best_ms_p95": percentile(best_values, 0.95),
        "regret_percent_mean": statistics.fmean(regrets),
        "regret_percent_median": statistics.median(regrets),
        "regret_percent_p95": percentile(regrets, 0.95),
        "oracle_hit_rate": statistics.fmean(float(row["oracle_hit"]) for row in rows),
        "invalid_dispatches_mean": statistics.fmean(row["invalid_dispatches"] for row in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--b3-replay", required=True)
    parser.add_argument("--timing-summary", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise FileExistsError("refusing to overwrite analysis")
    contract_path = Path(args.contract).resolve()
    b3_path = Path(args.b3_replay).resolve()
    contract = json.loads(contract_path.read_text())
    b3 = json.loads(b3_path.read_text())
    if contract.get("schema") != "c3_p7_qualified_timing_contract_v1":
        raise ValueError("unexpected contract schema")
    if b3.get("schema") != "c3_p7_b3_xgb_replay_v1":
        raise ValueError("unexpected B3 schema")

    summaries = {}
    timing_hashes = {}
    for value in args.timing_summary:
        path = Path(value).resolve()
        summary = json.loads(path.read_text())
        workload_id = summary["workload_id"]
        if summary.get("status") != "complete":
            raise ValueError(f"incomplete timing summary: {workload_id}")
        if workload_id in summaries:
            raise ValueError(f"duplicate timing summary: {workload_id}")
        summaries[workload_id] = summary
        timing_hashes[workload_id] = {"path": str(path), "sha256": sha256(path)}

    result = {
        "schema": "c3_p7q_search_policy_analysis_v1",
        "contract": {"path": str(contract_path), "sha256": sha256(contract_path)},
        "b3_replay": {"path": str(b3_path), "sha256": sha256(b3_path)},
        "timing_summaries": timing_hashes,
        "budgets": list(BUDGETS),
        "random_replicates": int(contract["original_p7_contract"] and 1000),
        "random_replicate_indexing": "0 through 999",
        "failure_accounting": contract["failure_accounting"],
        "workloads": {},
    }
    random_seed = 20260911
    aggregate_deterministic = {
        policy: {str(budget): [] for budget in BUDGETS} for policy in DETERMINISTIC_POLICIES
    }
    aggregate_random = {str(budget): [[] for _ in range(1000)] for budget in BUDGETS}

    for workload_id in contract["workload_order"]:
        workload = contract["workloads"][workload_id]
        entries = {row["candidate_id"]: row for row in workload["gross_candidates"]}
        gross_ids = list(entries)
        invalid_ids = set(workload["invalid_candidate_ids"])
        latency = {
            candidate_id: float(row["median_ms"])
            for candidate_id, row in summaries[workload_id]["candidate_summaries"].items()
        }
        if set(latency) != set(workload["timed_candidate_ids"]):
            raise ValueError(f"timing coverage mismatch: {workload_id}")
        oracle_ms, oracle_id = min((value, candidate_id) for candidate_id, value in latency.items())
        b3_order = [row["candidate_id"] for row in b3["workloads"][workload_id]["dispatches"]]
        policy_orders = dict(workload["policy_orders_gross"])
        policy_orders["B3"] = b3_order
        workload_result = {
            "gross_candidate_count": len(gross_ids),
            "valid_candidate_count": len(latency),
            "invalid_candidate_ids": sorted(invalid_ids),
            "oracle": {
                "candidate_id": oracle_id,
                "median_ms": oracle_ms,
                "residence_mode": entries[oracle_id]["residence_mode"],
                "config_index": entries[oracle_id]["config_index"],
            },
            "deterministic": {},
            "random_B2": {},
        }
        for policy in DETERMINISTIC_POLICIES:
            order = policy_orders[policy]
            oracle_position = next(
                (index + 1 for index, candidate_id in enumerate(order) if candidate_id == oracle_id), None
            )
            policy_result = {"oracle_gross_position": oracle_position, "budgets": {}}
            for budget in BUDGETS:
                metrics = evaluate_order(order, latency, invalid_ids, oracle_id, oracle_ms, budget)
                policy_result["budgets"][str(budget)] = metrics
                aggregate_deterministic[policy][str(budget)].append(metrics)
            workload_result["deterministic"][policy] = policy_result

        random_rows = {str(budget): [] for budget in BUDGETS}
        random_oracle_positions = []
        for replicate in range(1000):
            order = random_order(gross_ids, random_seed, workload_id, replicate)
            random_oracle_positions.append(order.index(oracle_id) + 1)
            for budget in BUDGETS:
                metrics = evaluate_order(order, latency, invalid_ids, oracle_id, oracle_ms, budget)
                random_rows[str(budget)].append(metrics)
                aggregate_random[str(budget)][replicate].append(metrics)
        workload_result["random_B2"] = {
            "oracle_gross_position_median": statistics.median(random_oracle_positions),
            "oracle_gross_position_p05": percentile(random_oracle_positions, 0.05),
            "oracle_gross_position_p95": percentile(random_oracle_positions, 0.95),
            "budgets": {
                str(budget): summarize_random(random_rows[str(budget)]) for budget in BUDGETS
            },
        }
        result["workloads"][workload_id] = workload_result

    result["aggregate"] = {"deterministic": {}, "random_B2": {}}
    for policy in DETERMINISTIC_POLICIES:
        result["aggregate"]["deterministic"][policy] = {"budgets": {}}
        for budget in BUDGETS:
            rows = aggregate_deterministic[policy][str(budget)]
            regrets = [row["regret_percent"] for row in rows if row["regret_percent"] is not None]
            result["aggregate"]["deterministic"][policy]["budgets"][str(budget)] = {
                "mean_regret_percent": statistics.fmean(regrets),
                "max_regret_percent": max(regrets),
                "oracle_workloads": sum(row["oracle_hit"] for row in rows),
                "workload_count": len(rows),
                "invalid_dispatches": sum(row["invalid_dispatches"] for row in rows),
            }
    for budget in BUDGETS:
        replicate_mean_regrets = []
        replicate_oracle_fractions = []
        for rows in aggregate_random[str(budget)]:
            replicate_mean_regrets.append(statistics.fmean(row["regret_percent"] for row in rows))
            replicate_oracle_fractions.append(statistics.fmean(float(row["oracle_hit"]) for row in rows))
        result["aggregate"]["random_B2"][str(budget)] = {
            "mean_regret_percent_mean": statistics.fmean(replicate_mean_regrets),
            "mean_regret_percent_median": statistics.median(replicate_mean_regrets),
            "mean_regret_percent_p95": percentile(replicate_mean_regrets, 0.95),
            "oracle_workload_fraction_mean": statistics.fmean(replicate_oracle_fractions),
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
