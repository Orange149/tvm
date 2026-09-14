"""Offline C3 shortlist replay over the archived 80-point VTA neighborhood.

The archive is development/retrospective evidence.  This tool never contacts a
board and does not claim that outcome labels or runtime profiles were available
before the historical dispatch occurred.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

from c3_candidate_identity import candidate_identity_record, canonical_json_bytes


POOL_SCHEMA = "c3_historical_candidate_pool_v1"
RESULT_SCHEMA = "c3_historical_replay_result_v1"
SUMMARY_SCHEMA = "c3_historical_replay_summary_v1"
METHODS = ("B2", "B4", "B5", "B6")
FAILURE_CATEGORIES = ("lower", "compile", "wrong_answer", "timeout", "rpc", "environment")


def _load_json(path):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def _load_jsonl(path):
    with open(path, "r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _key(workload, config_index):
    return canonical_json_bytes(workload).decode("utf-8"), int(config_index)


def _workload_from_measure_input(measure_input):
    return [measure_input[1]] + measure_input[2]


def _product(values):
    result = 1
    for value in values:
        result *= int(value)
    return result


def _runtime_features(workload, runtime_profile):
    input_unique_bytes = _product(workload[1][1])
    weight_unique_bytes = _product(workload[2][1])
    load_bytes = int(runtime_profile["load_buffer_2d_bytes"])
    store_bytes = int(runtime_profile["store_buffer_2d_bytes"])
    input_bytes = int(runtime_profile.get("load_buffer_2d_inp_bytes", 0))
    weight_bytes = int(runtime_profile.get("load_buffer_2d_wgt_bytes", 0))
    return {
        "total_dma_bytes": load_bytes + store_bytes,
        "load_bytes": load_bytes,
        "store_bytes": store_bytes,
        "load_calls": int(runtime_profile["load_buffer_2d_calls"]),
        "store_calls": int(runtime_profile["store_buffer_2d_calls"]),
        "small_load_calls": int(runtime_profile.get("load_buffer_2d_small_calls", 0)),
        "strided_load_calls": int(runtime_profile.get("load_buffer_2d_strided_calls", 0)),
        "padded_load_calls": int(runtime_profile.get("load_buffer_2d_padded_calls", 0)),
        "input_reload_ratio": input_bytes / input_unique_bytes,
        "weight_reload_ratio": weight_bytes / weight_unique_bytes,
    }


def load_historical_pool(
    selected_tasks_path,
    autotvm_log_path,
    artifacts_dir,
    hardware_fingerprint,
    schedule_version,
):
    """Join selected candidates, AutoTVM outcomes, and saved measurement artifacts."""

    selected_tasks = _load_json(selected_tasks_path)
    log_rows = _load_jsonl(autotvm_log_path)
    log_by_key = {}
    for row in log_rows:
        row_key = _key(_workload_from_measure_input(row["input"]), row["config"]["index"])
        if row_key in log_by_key:
            raise ValueError("duplicate AutoTVM log row: {}".format(row_key[1]))
        log_by_key[row_key] = row

    artifacts = {}
    for path in sorted(Path(artifacts_dir).glob("*/measurement.json")):
        artifact = _load_json(path)
        artifact_key = _key(artifact["workload"], artifact["config"]["index"])
        if artifact_key in artifacts:
            raise ValueError("duplicate measurement artifact: {}".format(artifact_key[1]))
        artifact["_path"] = str(path)
        artifacts[artifact_key] = artifact

    workloads = []
    pool_keys = set()
    for workload_ordinal, task in enumerate(selected_tasks):
        workload = task["workload"]
        incumbent_index = int(task["incumbent_index"])
        candidates = []
        for pool_position, selected in enumerate(task["candidates"]):
            config_index = int(selected["config_index"])
            candidate_key = _key(workload, config_index)
            if candidate_key in pool_keys:
                raise ValueError("duplicate selected candidate: {}".format(config_index))
            pool_keys.add(candidate_key)
            if candidate_key not in log_by_key:
                raise ValueError("selected candidate missing from AutoTVM log: {}".format(config_index))

            artifact = artifacts.get(candidate_key)
            if artifact is None:
                outcome = "compile"
                latency_ms = None
                features = None
                artifact_path = None
            elif bool(artifact.get("correct", False)):
                outcome = "pass"
                costs = [float(value) * 1000.0 for value in artifact["costs_s"]]
                latency_ms = statistics.median(costs)
                features = _runtime_features(workload, artifact["runtime_profile"])
                artifact_path = artifact["_path"]
            else:
                outcome = "wrong_answer"
                latency_ms = None
                features = None
                artifact_path = artifact["_path"]

            identity = candidate_identity_record(
                hardware_fingerprint,
                "conv2d_packed.vta",
                schedule_version,
                workload,
                "original",
                selected["config"],
                config_index=config_index,
            )
            candidates.append(
                {
                    **identity,
                    "pool_position": pool_position,
                    "is_incumbent": config_index == incumbent_index,
                    "changed_axes": selected.get("changed_axes", []),
                    "outcome": outcome,
                    "latency_ms": latency_ms,
                    "request_features": features,
                    "artifact_path": artifact_path,
                }
            )
        if not any(candidate["is_incumbent"] for candidate in candidates):
            raise ValueError("workload {} lost its TopHub incumbent".format(workload_ordinal))
        workloads.append(
            {
                "workload_id": "W{:02d}".format(workload_ordinal),
                "workload": workload,
                "incumbent_index": incumbent_index,
                "candidates": candidates,
            }
        )

    unused_logs = set(log_by_key) - pool_keys
    return {
        "schema": POOL_SCHEMA,
        "workloads": workloads,
        "join_diagnostics": {
            "selected_candidates": len(pool_keys),
            "autotvm_log_rows": len(log_rows),
            "unused_autotvm_log_rows": len(unused_logs),
            "measurement_artifacts_total": len(artifacts),
            "measurement_artifacts_used": sum(
                candidate["artifact_path"] is not None
                for workload in workloads
                for candidate in workload["candidates"]
            ),
        },
    }


def _pareto_dominates(left, right, feature_names):
    left_values = [left["request_features"][name] for name in feature_names]
    right_values = [right["request_features"][name] for name in feature_names]
    return all(a <= b for a, b in zip(left_values, right_values)) and any(
        a < b for a, b in zip(left_values, right_values)
    )


def pareto_fronts(candidates, feature_names):
    """Return candidate ID -> non-dominated front, with zero as the best front."""

    remaining = list(candidates)
    fronts = {}
    front_index = 0
    while remaining:
        front = [
            candidate
            for candidate in remaining
            if not any(
                _pareto_dominates(other, candidate, feature_names)
                for other in remaining
                if other is not candidate
            )
        ]
        if not front:
            raise RuntimeError("Pareto sort made no progress")
        for candidate in front:
            fronts[candidate["candidate_id"]] = front_index
        front_ids = {candidate["candidate_id"] for candidate in front}
        remaining = [candidate for candidate in remaining if candidate["candidate_id"] not in front_ids]
        front_index += 1
    return fronts


def _rank_sum(candidates, feature_names):
    """Unit-free equal-feature rank sum used only inside a Pareto front."""

    scores = {candidate["candidate_id"]: 0.0 for candidate in candidates}
    denominator = max(1, len(candidates) - 1)
    for feature in feature_names:
        ordered = sorted(
            candidates,
            key=lambda candidate: (candidate["request_features"][feature], candidate["candidate_id"]),
        )
        for rank, candidate in enumerate(ordered):
            scores[candidate["candidate_id"]] += rank / denominator
    return scores


def order_candidates(workload_record, method):
    candidates = list(workload_record["candidates"])
    if method == "B4":
        outcome_rank = {"pass": 0, "wrong_answer": 1, "compile": 2}
        return sorted(candidates, key=lambda item: (outcome_rank[item["outcome"]], item["pool_position"]))

    valid = [candidate for candidate in candidates if candidate["outcome"] == "pass"]
    invalid = [candidate for candidate in candidates if candidate["outcome"] != "pass"]
    invalid = sorted(
        invalid,
        key=lambda item: (0 if item["outcome"] == "wrong_answer" else 1, item["pool_position"]),
    )
    if method == "B5":
        valid.sort(
            key=lambda item: (
                item["request_features"]["total_dma_bytes"],
                item["request_features"]["load_calls"],
                item["candidate_id"],
            )
        )
        return valid + invalid
    if method == "B6":
        feature_names = (
            "total_dma_bytes",
            "load_calls",
            "store_calls",
            "small_load_calls",
            "strided_load_calls",
            "padded_load_calls",
            "input_reload_ratio",
            "weight_reload_ratio",
        )
        if valid:
            fronts = pareto_fronts(valid, feature_names)
            rank_sum = _rank_sum(valid, feature_names)
            valid.sort(
                key=lambda item: (
                    fronts[item["candidate_id"]],
                    rank_sum[item["candidate_id"]],
                    item["request_features"]["total_dma_bytes"],
                    item["candidate_id"],
                )
            )
        return valid + invalid
    raise ValueError("deterministic ordering unsupported for {}".format(method))


def replay_prefix(workload_record, ordered_candidates, budget):
    prefix = ordered_candidates[: min(int(budget), len(ordered_candidates))]
    valid_pool = [candidate for candidate in workload_record["candidates"] if candidate["outcome"] == "pass"]
    oracle_ms = min((candidate["latency_ms"] for candidate in valid_pool), default=None)
    observed_valid = [candidate for candidate in prefix if candidate["outcome"] == "pass"]
    best_ms = min((candidate["latency_ms"] for candidate in observed_valid), default=None)
    regret = None if oracle_ms is None or best_ms is None else (best_ms - oracle_ms) / oracle_ms
    outcome_counts = {"pass": 0, **{category: 0 for category in FAILURE_CATEGORIES}}
    for candidate in prefix:
        outcome_counts[candidate["outcome"]] += 1
    return {
        "requested_budget": int(budget),
        "effective_budget": len(prefix),
        "attempted_positions": len(prefix),
        "outcomes": outcome_counts,
        "valid_trial_ratio": outcome_counts["pass"] / len(prefix) if prefix else 0.0,
        "oracle_latency_ms": oracle_ms,
        "best_latency_ms": best_ms,
        "regret": regret,
        "oracle_discovered": regret is not None and abs(regret) <= 1e-12,
        "near_oracle_2pct": regret is not None and regret <= 0.02,
        "incumbent_seen": any(candidate["is_incumbent"] for candidate in prefix),
        "candidate_ids": [candidate["candidate_id"] for candidate in prefix],
    }


def _mean(values):
    return sum(values) / len(values) if values else None


def aggregate_random_replays(replays):
    regrets = [replay["regret"] for replay in replays if replay["regret"] is not None]
    return {
        "requested_budget": replays[0]["requested_budget"],
        "effective_budget": replays[0]["effective_budget"],
        "attempted_positions_per_permutation": replays[0]["attempted_positions"],
        "permutations": len(replays),
        "coverage_rate": len(regrets) / len(replays),
        "oracle_discovery_rate": _mean([float(replay["oracle_discovered"]) for replay in replays]),
        "near_oracle_2pct_rate": _mean([float(replay["near_oracle_2pct"]) for replay in replays]),
        "incumbent_seen_rate": _mean([float(replay["incumbent_seen"]) for replay in replays]),
        "conditional_regret_mean": _mean(regrets),
        "conditional_regret_median": statistics.median(regrets) if regrets else None,
        "mean_outcomes": {
            name: _mean([replay["outcomes"][name] for replay in replays])
            for name in ("pass",) + FAILURE_CATEGORIES
        },
    }


def run_replay(pool, budgets, random_permutations=1000, random_seed=0):
    results = []
    for workload in pool["workloads"]:
        for budget in budgets:
            random_replays = []
            for permutation in range(random_permutations):
                ordered = list(workload["candidates"])
                seed_material = "{}:{}:{}".format(random_seed, workload["workload_id"], permutation)
                seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
                random.Random(seed).shuffle(ordered)
                random_replays.append(replay_prefix(workload, ordered, budget))
            results.append(
                {
                    "schema": RESULT_SCHEMA,
                    "data_scope": "historical_development_retrospective",
                    "method": "B2",
                    "evaluation_status": "evaluable_historical_random_baseline",
                    "workload_id": workload["workload_id"],
                    "workload": workload["workload"],
                    "metrics": aggregate_random_replays(random_replays),
                }
            )
        for method in ("B4", "B5", "B6"):
            ordered = order_candidates(workload, method)
            if len(ordered) != len(workload["candidates"]) or not any(
                candidate["is_incumbent"] for candidate in ordered
            ):
                raise RuntimeError("{} deleted a candidate or incumbent".format(method))
            for budget in budgets:
                results.append(
                    {
                        "schema": RESULT_SCHEMA,
                        "data_scope": "historical_development_retrospective",
                        "method": method,
                        "evaluation_status": (
                            "diagnostic_oracle_upper_bound_leaky"
                            if method == "B4"
                            else "not_evaluable_missing_prefeatures"
                        ),
                        "leakage_warning": (
                            "ordering uses post-dispatch correctness outcomes"
                            if method == "B4"
                            else "ordering uses post-dispatch correctness outcomes and runtime DMA counters"
                        ),
                        "workload_id": workload["workload_id"],
                        "workload": workload["workload"],
                        "diagnostic_metrics_leaky": replay_prefix(workload, ordered, budget),
                    }
                )
    return results


def summarize(pool, results, budgets, random_permutations):
    candidates = [candidate for workload in pool["workloads"] for candidate in workload["candidates"]]
    outcomes = {"pass": 0, **{category: 0 for category in FAILURE_CATEGORIES}}
    for candidate in candidates:
        outcomes[candidate["outcome"]] += 1
    scorable = [
        workload["workload_id"]
        for workload in pool["workloads"]
        if any(candidate["outcome"] == "pass" for candidate in workload["candidates"])
    ]
    unscorable = [
        workload["workload_id"]
        for workload in pool["workloads"]
        if workload["workload_id"] not in scorable
    ]
    runtime_feature_count = sum(candidate["request_features"] is not None for candidate in candidates)
    feature_availability = {
        "full_config_entity": {"available": len(candidates), "total": len(candidates)},
        "historical_outcome_label": {"available": len(candidates), "total": len(candidates)},
        "latency_and_runtime_dma_counters": {
            "available": runtime_feature_count,
            "total": len(candidates),
            "availability_condition": "archived correct=true measurement only",
        },
        "correctness_failure_label": {
            "available": outcomes["wrong_answer"],
            "total": len(candidates),
            "meaning": "positive observations, unavailable before historical dispatch",
        },
        "compile_failure_label": {
            "available": outcomes["compile"],
            "total": len(candidates),
            "meaning": "positive observations, unavailable before historical dispatch",
        },
        "sram_working_set_estimate": {"available": 0, "total": len(candidates)},
        "lowered_tir_hash": {"available": 0, "total": len(candidates)},
        "exact_dma_request_histogram": {"available": 0, "total": len(candidates)},
        "maximum_dma_request_bytes": {"available": 0, "total": len(candidates)},
    }
    by_method_budget = {
        "B2": {
            "evaluation_status": "evaluable_historical_random_baseline",
            "by_budget": {},
        },
        "B4": {
            "evaluation_status": "diagnostic_oracle_upper_bound_leaky",
            "leakage": "post-dispatch correctness outcomes rank pass before wrong-answer/compile",
            "diagnostic_oracle_upper_bound_by_budget": {},
        },
        "B5": {
            "evaluation_status": "not_evaluable_missing_prefeatures",
            "leakage": "post-dispatch correctness and runtime DMA counters exist only for passing candidates",
            "diagnostic_oracle_upper_bound_by_budget": {},
        },
        "B6": {
            "evaluation_status": "not_evaluable_missing_prefeatures",
            "leakage": "post-dispatch correctness and runtime request counters exist only for passing candidates",
            "diagnostic_oracle_upper_bound_by_budget": {},
        },
    }
    for method in METHODS:
        for budget in budgets:
            if method == "B2":
                rows = [
                    row
                    for row in results
                    if row["method"] == method and row["metrics"]["requested_budget"] == budget
                ]
                by_method_budget[method]["by_budget"][str(budget)] = {
                    "workloads": len(rows),
                    "scorable_workloads": len(scorable),
                    "mean_coverage_rate": _mean([row["metrics"]["coverage_rate"] for row in rows]),
                    "mean_near_oracle_2pct_rate": _mean(
                        [row["metrics"]["near_oracle_2pct_rate"] for row in rows]
                    ),
                    "mean_valid_trials_per_workload": _mean(
                        [row["metrics"]["mean_outcomes"]["pass"] for row in rows]
                    ),
                    "total_positions_simulated": sum(
                        row["metrics"]["attempted_positions_per_permutation"]
                        * row["metrics"]["permutations"]
                        for row in rows
                    ),
                }
            else:
                rows = [
                    row
                    for row in results
                    if row["method"] == method
                    and row["diagnostic_metrics_leaky"]["requested_budget"] == budget
                ]
                scored_rows = [row for row in rows if row["workload_id"] in scorable]
                diagnostic = lambda row: row["diagnostic_metrics_leaky"]
                by_method_budget[method]["diagnostic_oracle_upper_bound_by_budget"][str(budget)] = {
                    "interpretation": "leaky diagnostic only; forbidden as a baseline effectiveness result",
                    "workloads": len(rows),
                    "scorable_workloads": len(scored_rows),
                    "total_attempted_positions": sum(diagnostic(row)["attempted_positions"] for row in rows),
                    "total_pass": sum(diagnostic(row)["outcomes"]["pass"] for row in rows),
                    "total_compile": sum(diagnostic(row)["outcomes"]["compile"] for row in rows),
                    "total_wrong_answer": sum(
                        diagnostic(row)["outcomes"]["wrong_answer"] for row in rows
                    ),
                    "near_oracle_2pct_workloads": sum(
                        bool(diagnostic(row)["near_oracle_2pct"]) for row in scored_rows
                    ),
                    "oracle_discovered_workloads": sum(
                        bool(diagnostic(row)["oracle_discovered"]) for row in scored_rows
                    ),
                    "diagnostic_mean_regret_covered": _mean(
                        [
                            diagnostic(row)["regret"]
                            for row in scored_rows
                            if diagnostic(row)["regret"] is not None
                        ]
                    ),
                }
    return {
        "schema": SUMMARY_SCHEMA,
        "status": "completed",
        "data_scope": "historical_development_retrospective_not_holdout",
        "pool": {
            "workloads": len(pool["workloads"]),
            "candidates": len(candidates),
            "candidates_per_workload": sorted(
                {len(workload["candidates"]) for workload in pool["workloads"]}
            ),
            "outcomes": outcomes,
            "scorable_workloads": scorable,
            "unscorable_workloads": unscorable,
            "incumbents_retained": all(
                any(candidate["is_incumbent"] for candidate in workload["candidates"])
                for workload in pool["workloads"]
            ),
            "join_diagnostics": pool["join_diagnostics"],
        },
        "budgets": list(budgets),
        "random_permutations": int(random_permutations),
        "feature_availability": feature_availability,
        "primary_claim_eligible_methods": ["B2"],
        "methods": {
            "B2": "uniform random permutation of all archived positions",
            "B4": "leaky diagnostic upper bound using retrospective correctness; no archived pre-dispatch SRAM estimate",
            "B5": "not evaluable: total DMA bytes are runtime-observed only for passing candidates",
            "B6": "not evaluable: request-shape fields are runtime-observed only for passing candidates",
        },
        "method_results": by_method_budget,
        "limitations": [
            "Outcome labels and runtime profiles are retrospective and cannot establish prospective trial avoidance.",
            "Only B2 is evaluable here; every B4/B5/B6 number is stored under an explicitly leaky diagnostic namespace.",
            "Two projection workloads have no direct-run passing candidate and therefore no latency oracle.",
            "Archived candidates have no SRAM working-set estimate, TIR hash, exact request histogram, or max request size.",
            "B6 is partial request-shape Pareto over available runtime counters, not the complete P4 feature contract.",
            "Budget 16 saturates the eight-candidate-per-workload archive at effective budget 8.",
            "The archive is development evidence, not grouped holdout or prospective validation.",
        ],
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-tasks", required=True)
    parser.add_argument("--autotvm-log", required=True)
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--hardware-fingerprint-json", required=True)
    parser.add_argument("--schedule-version", required=True)
    parser.add_argument("--budgets", default="4,8,16")
    parser.add_argument("--random-permutations", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--results", required=True)
    parser.add_argument("--summary", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    budgets = tuple(int(item) for item in args.budgets.split(",") if item)
    if not budgets or any(budget <= 0 for budget in budgets):
        raise ValueError("budgets must be positive")
    if args.random_permutations < 1000:
        raise ValueError("historical random replay requires at least 1000 permutations")
    pool = load_historical_pool(
        args.selected_tasks,
        args.autotvm_log,
        args.artifacts_dir,
        json.loads(args.hardware_fingerprint_json),
        args.schedule_version,
    )
    results = run_replay(pool, budgets, args.random_permutations, args.random_seed)
    summary = summarize(pool, results, budgets, args.random_permutations)
    summary["sources"] = {
        "selected_tasks": {"path": args.selected_tasks, "sha256": _sha256_file(args.selected_tasks)},
        "autotvm_log": {"path": args.autotvm_log, "sha256": _sha256_file(args.autotvm_log)},
        "artifacts_dir": args.artifacts_dir,
    }

    results_path = Path(args.results)
    summary_path = Path(args.summary)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as stream:
        for record in results:
            stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
    summary["results_sha256"] = _sha256_file(results_path)
    with open(summary_path, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps(summary, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
