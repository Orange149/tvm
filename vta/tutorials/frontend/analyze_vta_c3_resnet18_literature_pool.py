#!/usr/bin/env python3
"""Analyze the completed R18 literature pool with frozen six-policy semantics."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import run_vta_c3_literature_baselines as baseline


SCHEMA = "c3_resnet18_literature_pool_analysis_v1"
SEEDS = tuple(range(57001, 57021))
BUDGETS = (10, 20, 50, 75, 96)
MAIN_POLICIES = (
    "random",
    "stock_mode_aware_xgb",
    "rieber_hw_init_xgb",
    "ml2tuner_pva",
    "cheng_minimum_access",
    "ours_dma_multifidelity",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


def verify(directory):
    directory = Path(directory).resolve()
    artifacts = read_json(directory / "artifact_hashes.json")["artifacts"]
    for name, expected in artifacts.items():
        if isinstance(expected, str) and baseline.sha256_file(directory / name) != expected:
            raise ValueError("artifact hash mismatch: {}".format(directory / name))
    return baseline.sha256_file(directory / "artifact_hashes.json")


def distribution(values):
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return {
        "n": len(clean),
        "median": statistics.median(clean),
        "q1": float(np.percentile(clean, 25)),
        "q3": float(np.percentile(clean, 75)),
        "iqr": float(np.percentile(clean, 75) - np.percentile(clean, 25)),
        "minimum": min(clean),
        "maximum": max(clean),
    }


def development_rows(contracts, payloads, target):
    rows = []
    for workload_id, candidates in contracts.items():
        if workload_id == target:
            continue
        outcomes = payloads[workload_id]["outcomes"]
        for candidate in candidates:
            phase_rows = outcomes[candidate["candidate_id"]]
            rows.append(baseline.outcome_to_observation(candidate, phase_rows))
    return rows


def target_metrics(results, oracle):
    best = math.inf
    best_candidate_id = None
    best_rows = []
    first_2 = None
    first_5 = None
    for row in results:
        latency = row.get("latency_ms")
        if latency is not None and float(latency) < best:
            best = float(latency)
            best_candidate_id = row["candidate_id"]
        best_rows.append(
            {
                "dispatch": row["dispatch"],
                "dispatched_candidate_id": row["candidate_id"],
                "cumulative_wall_seconds": row["cumulative_wall_seconds"],
                "best_latency_ms": None if not math.isfinite(best) else best,
                "best_candidate_id": best_candidate_id,
                "regret_fraction": None
                if not math.isfinite(best)
                else best / oracle["latency_ms"] - 1.0,
            }
        )
        if math.isfinite(best) and first_2 is None and best <= 1.02 * oracle["latency_ms"]:
            first_2 = dict(best_rows[-1])
        if math.isfinite(best) and first_5 is None and best <= 1.05 * oracle["latency_ms"]:
            first_5 = dict(best_rows[-1])
    return {
        "best_so_far": best_rows,
        "first_oracle_2_percent": first_2,
        "first_oracle_5_percent": first_5,
        "success_oracle_2_percent": first_2 is not None,
        "success_oracle_5_percent": first_5 is not None,
        "final_best_latency_ms": None if not math.isfinite(best) else best,
        "final_best_candidate_id": best_candidate_id,
        "final_regret_fraction": None
        if not math.isfinite(best)
        else best / oracle["latency_ms"] - 1.0,
    }


def e0_local_maps(qualification_dir):
    static = {
        row["candidate_id"]: row
        for row in read_jsonl(Path(qualification_dir) / "static_results.jsonl")
    }
    fsim = {
        row["candidate_id"]: row
        for row in read_jsonl(Path(qualification_dir) / "fsim_results.jsonl")
    }
    return static, fsim


def append_phase(timeline, candidate_id, name, row, cumulative):
    cumulative += float(row.get("wall_seconds", 0))
    event = {
        "event_index": len(timeline) + 1,
        "candidate_id": candidate_id,
        "phase": name,
        "status": row.get("status"),
        "phase_wall_seconds": float(row.get("wall_seconds", 0)),
        "cumulative_wall_seconds": cumulative,
    }
    for key in (
        "compiler_attempts",
        "fpga_kernel_invocations",
        "logical_dma_bytes",
        "logical_dma_calls",
    ):
        if key in row:
            event[key] = row[key]
    timeline.append(event)
    return cumulative


def replay_rieber(
    candidates,
    ledger_payload,
    member_ids,
    static,
    fsim,
    presampling_rows,
    budget,
    seed,
    development,
):
    """Paid balanced E0, then validity-biased mode-aware performance XGB."""

    ledger = baseline.OutcomeLedger(ledger_payload)
    by_id = {row["candidate_id"]: row for row in candidates}
    timeline, results, observed = [], [], []
    cumulative = 0.0
    attempted = set()
    prepaid_lower_ids = set()
    compiler_observed = []
    for row in presampling_rows:
        candidate_id = row["candidate_id"]
        prepaid_lower_ids.add(candidate_id)
        compiler_observed.append(row)
        cumulative = append_phase(
            timeline,
            candidate_id,
            "lower",
            {
                "status": "ok" if row["is_valid"] else "invalid",
                "wall_seconds": row["wall_seconds"],
                "compiler_attempts": 1,
            },
            cumulative,
        )
    for candidate_id in member_ids:
        if len(results) >= budget:
            break
        attempted.add(candidate_id)
        if candidate_id in by_id:
            candidate = by_id[candidate_id]
            phases = (
                baseline.TERMINAL_PHASES[1:]
                if candidate_id in prepaid_lower_ids
                else baseline.TERMINAL_PHASES
            )
            phase_rows, cumulative = baseline.execute_candidate(
                candidate, ledger, timeline, cumulative, phases=phases
            )
            if candidate_id in prepaid_lower_ids:
                phase_rows["lower"] = {"status": "ok"}
            observation = baseline.outcome_to_observation(candidate, phase_rows)
            observed.append(observation)
            results.append(
                {
                    "dispatch": len(results) + 1,
                    "candidate_id": candidate_id,
                    "is_valid": observation["is_valid"],
                    "latency_ms": observation.get("latency_ms"),
                    "cumulative_wall_seconds": cumulative,
                    "source": "balanced_valid_e0",
                }
            )
        else:
            # The original passed real lowering but failed local FSim.  It is a
            # paid invalid profiling attempt and must remain in gross/wall.
            fsim_row = fsim[candidate_id]
            cumulative = append_phase(
                timeline,
                candidate_id,
                "fsim",
                {"status": "invalid", "wall_seconds": fsim_row["wall_seconds"]},
                cumulative,
            )
            results.append(
                {
                    "dispatch": len(results) + 1,
                    "candidate_id": candidate_id,
                    "is_valid": False,
                    "latency_ms": None,
                    "cumulative_wall_seconds": cumulative,
                    "source": "balanced_valid_e0_fsim_rejected",
                }
            )
    remaining = [row for row in candidates if row["candidate_id"] not in attempted]
    while remaining and len(results) < budget:
        performance_rank = baseline.select_model_rank(
            remaining,
            development,
            observed,
            seed,
            baseline.visible_vector,
            validity=False,
        )
        validity_rank = baseline.select_model_rank(
            remaining,
            [],
            compiler_observed,
            seed,
            baseline.visible_vector,
            validity=True,
        )
        performance_position = {
            row["candidate_id"]: index
            for index, (row, _) in enumerate(performance_rank)
        }
        validity_position = {
            row["candidate_id"]: index
            for index, (row, _) in enumerate(validity_rank)
        }
        candidate = min(
            remaining,
            key=lambda row: (
                performance_position[row["candidate_id"]]
                + validity_position[row["candidate_id"]],
                performance_position[row["candidate_id"]],
                row["candidate_id"],
            ),
        )
        remaining = [
            row for row in remaining if row["candidate_id"] != candidate["candidate_id"]
        ]
        phases = (
            baseline.TERMINAL_PHASES[1:]
            if candidate["candidate_id"] in prepaid_lower_ids
            else baseline.TERMINAL_PHASES
        )
        phase_rows, cumulative = baseline.execute_candidate(
            candidate, ledger, timeline, cumulative, phases=phases
        )
        if candidate["candidate_id"] in prepaid_lower_ids:
            phase_rows["lower"] = {"status": "ok"}
        observation = baseline.outcome_to_observation(candidate, phase_rows)
        observed.append(observation)
        results.append(
            {
                "dispatch": len(results) + 1,
                "candidate_id": candidate["candidate_id"],
                "is_valid": observation["is_valid"],
                "latency_ms": observation.get("latency_ms"),
                "cumulative_wall_seconds": cumulative,
                "source": "post_e0_mode_aware_xgb",
            }
        )
    return timeline, results, {
        "presampling_compiler_calls": len(presampling_rows),
        "presampling_valid": sum(row["is_valid"] for row in presampling_rows),
        "presampling_invalid": sum(not row["is_valid"] for row in presampling_rows),
        "presampling_wall_seconds": sum(
            float(row["wall_seconds"]) for row in presampling_rows
        ),
        "balanced_valid_e0_requested": len(member_ids),
        "balanced_valid_e0_fsim_rejected": sum(
            candidate_id not in by_id for candidate_id in member_ids[:budget]
        ),
        "post_e0_attempts": sum(
            row["source"] == "post_e0_mode_aware_xgb" for row in results
        ),
        "post_e0_selector": "rank-sum of the common visible-feature performance XGB and target-presampling lowering-validity XGB",
    }


def build_hw_presampling_rows(full_original_dir, scan_dirs, memberships):
    """Reconstruct each paid Algorithm-1-style compiler presampling trace."""

    full_original_dir = Path(full_original_dir).resolve()
    scans = {}
    for directory in scan_dirs:
        directory = Path(directory).resolve()
        summary = read_json(directory / "summary.json")
        workload_id = summary["workload_id"]
        scans[workload_id] = {
            row["candidate_id"]: row
            for row in read_jsonl(directory / "results.jsonl")
        }
    if set(scans) != {"R18-H1", "R18-H2", "R18-H3"}:
        raise ValueError("all three complete original scans are required")
    output = {}
    for workload_id in sorted(scans):
        contract = read_json(
            full_original_dir
            / "{}_complete_original_space.json".format(workload_id.lower())
        )
        candidates = baseline.validate_workload_contract(contract)
        by_id = {row["candidate_id"]: row for row in candidates}
        labels = scans[workload_id]
        if set(by_id) != set(labels):
            raise ValueError("complete original scan identity mismatch for {}".format(workload_id))
        for seed in SEEDS:
            order = baseline.rieber_presampling_order(
                candidates,
                seed,
                lambda candidate_id: bool(labels[candidate_id]["is_valid"]),
                limit=min(1000, len(candidates)),
                parallel=8,
            )
            rows = [
                {
                    **by_id[candidate_id],
                    "is_valid": bool(labels[candidate_id]["is_valid"]),
                    "wall_seconds": float(labels[candidate_id]["wall_seconds"]),
                }
                for candidate_id in order
            ]
            expected = memberships[(workload_id, seed)]
            order_ids = {row["candidate_id"] for row in rows}
            if (
                len(expected) != 25
                or len(set(expected)) != 25
                or any(candidate_id not in order_ids for candidate_id in expected)
                or any(not labels[candidate_id]["is_valid"] for candidate_id in expected)
            ):
                raise ValueError(
                    "frozen balanced E0 is inconsistent with presampling for {} seed {}".format(
                        workload_id, seed
                    )
                )
            output[(workload_id, seed)] = rows
    return output


def cheng_candidates(candidates):
    groups = defaultdict(list)
    for row in candidates:
        groups[row["family_id"]].append(row)
    eligible, excluded = [], []
    for family_id, rows in groups.items():
        originals = [row for row in rows if row["residence_mode"] == "original"]
        if len(originals) == 1:
            eligible.extend(rows)
        else:
            excluded.append(
                {
                    "family_id": family_id,
                    "reason": "no_fpga_pool_original_for_same_tile_fallback",
                    "available_modes": sorted(row["residence_mode"] for row in rows),
                }
            )
    return eligible, excluded


def gain_ndcg(labels, predictions):
    if not labels:
        return None
    worst = max(labels)
    gains = [worst - value + 1e-12 for value in labels]
    order = sorted(range(len(labels)), key=lambda index: (predictions[index], index))
    ideal = sorted(range(len(labels)), key=lambda index: (labels[index], index))

    def dcg(indices):
        return sum(gains[index] / math.log2(rank + 2) for rank, index in enumerate(indices))

    denominator = dcg(ideal)
    return dcg(order) / denominator if denominator else 1.0


def validity_metrics(labels, probabilities, top_k=20):
    """Classification and ranking metrics for final executable validity."""

    if len(labels) != len(probabilities) or not labels:
        raise ValueError("validity metrics require aligned non-empty arrays")
    labels = [bool(value) for value in labels]
    predicted = [float(value) >= 0.5 for value in probabilities]
    tp = sum(actual and guess for actual, guess in zip(labels, predicted))
    fp = sum((not actual) and guess for actual, guess in zip(labels, predicted))
    fn = sum(actual and (not guess) for actual, guess in zip(labels, predicted))
    tn = len(labels) - tp - fp - fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    order = sorted(
        range(len(labels)), key=lambda index: (-float(probabilities[index]), index)
    )
    ideal = sorted(range(len(labels)), key=lambda index: (not labels[index], index))

    def dcg(indices):
        return sum(
            float(labels[index]) / math.log2(rank + 2)
            for rank, index in enumerate(indices)
        )

    denominator = dcg(ideal)
    k = min(int(top_k), len(labels))
    return {
        "support": len(labels),
        "valid": sum(labels),
        "invalid": sum(not value for value in labels),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / len(labels),
        "ndcg": dcg(order) / denominator if denominator else 1.0,
        "top_k": k,
        "top_k_valid_yield": sum(labels[index] for index in order[:k]) / k,
        "base_valid_yield": sum(labels) / len(labels),
    }


def model_v_metrics(contracts, payloads):
    """Leave-one-workload-out Model V against final hardware validity."""

    rows = []
    for target, candidates in contracts.items():
        development = development_rows(contracts, payloads, target)
        test = [
            baseline.outcome_to_observation(
                candidate, payloads[target]["outcomes"][candidate["candidate_id"]]
            )
            for candidate in candidates
        ]
        training = baseline.validity_training_rows(development, [])
        labels = [bool(row["is_valid"]) for row in test]
        for seed in SEEDS:
            probabilities = baseline._xgb_validity(
                [baseline.visible_vector(row) for row in training],
                [int(row["is_valid"]) for row in training],
                [baseline.visible_vector(row) for row in test],
                seed,
            )
            rows.append(
                {
                    "target_workload": target,
                    "seed": seed,
                    "model": "V_final_executable_validity",
                    "training_rows": len(training),
                    **validity_metrics(labels, probabilities, top_k=20),
                }
            )
    return rows


def model_pa_metrics(contracts, payloads):
    rows = []
    for target, candidates in contracts.items():
        development = development_rows(contracts, payloads, target)
        test = []
        for candidate in candidates:
            observation = baseline.outcome_to_observation(
                candidate, payloads[target]["outcomes"][candidate["candidate_id"]]
            )
            if observation.get("latency_ms") is not None:
                test.append(observation)
        labels = [float(row["latency_ms"]) for row in test]
        for seed in SEEDS:
            for model, vector in (
                ("P_visible", baseline.visible_vector),
                ("A_compiler_hidden", lambda row: baseline.advanced_vector(row, False)),
                ("A_plus_DMA", lambda row: baseline.advanced_vector(row, True)),
            ):
                train = baseline.performance_training_rows(development, [])
                predictions = baseline._xgb_regression(
                    [vector(row) for row in train],
                    [float(row["latency_ms"]) for row in train],
                    [vector(row) for row in test],
                    seed,
                )
                rows.append(
                    {
                        "target_workload": target,
                        "seed": seed,
                        "model": model,
                        "training_rows": len(train),
                        "test_rows": len(test),
                        "rmse_ms": math.sqrt(
                            statistics.mean(
                                (prediction - label) ** 2
                                for prediction, label in zip(predictions, labels)
                            )
                        ),
                        "ndcg": gain_ndcg(labels, predictions),
                    }
                )
    return rows


def aggregate_runs(runs):
    output = {}
    for workload_id in sorted({row["workload_id"] for row in runs}):
        output[workload_id] = {}
        for policy in sorted({row["policy"] for row in runs}):
            output[workload_id][policy] = {}
            for budget in BUDGETS:
                selected = [
                    row
                    for row in runs
                    if row["workload_id"] == workload_id
                    and row["policy"] == policy
                    and row["requested_budget"] == budget
                ]
                if not selected:
                    continue
                output[workload_id][policy][str(budget)] = {
                    "effective_budget": distribution(
                        [row["effective_attempts"] for row in selected]
                    ),
                    "success_oracle_2_percent": statistics.mean(
                        row["target"]["success_oracle_2_percent"] for row in selected
                    ),
                    "success_oracle_5_percent": statistics.mean(
                        row["target"]["success_oracle_5_percent"] for row in selected
                    ),
                    "trials_to_2_percent": distribution(
                        [
                            row["target"]["first_oracle_2_percent"]["dispatch"]
                            if row["target"]["first_oracle_2_percent"]
                            else None
                            for row in selected
                        ]
                    ),
                    "time_to_2_percent_seconds": distribution(
                        [
                            row["target"]["first_oracle_2_percent"][
                                "cumulative_wall_seconds"
                            ]
                            if row["target"]["first_oracle_2_percent"]
                            else None
                            for row in selected
                        ]
                    ),
                    "trials_to_5_percent": distribution(
                        [
                            row["target"]["first_oracle_5_percent"]["dispatch"]
                            if row["target"]["first_oracle_5_percent"]
                            else None
                            for row in selected
                        ]
                    ),
                    "time_to_5_percent_seconds": distribution(
                        [
                            row["target"]["first_oracle_5_percent"][
                                "cumulative_wall_seconds"
                            ]
                            if row["target"]["first_oracle_5_percent"]
                            else None
                            for row in selected
                        ]
                    ),
                    "final_regret_fraction": distribution(
                        [row["target"]["final_regret_fraction"] for row in selected]
                    ),
                    "lazy_wall_seconds": distribution(
                        [row["cost"]["wall_seconds"] for row in selected]
                    ),
                    "invalid_candidates": distribution(
                        [row["cost"]["invalid_candidates"] for row in selected]
                    ),
                    "gross_candidates": distribution(
                        [row["cost"]["gross_candidates"] for row in selected]
                    ),
                    "compiler_attempts": distribution(
                        [row["cost"]["compiler_attempts"] for row in selected]
                    ),
                    "fpga_dispatches": distribution(
                        [row["cost"]["fpga_dispatches"] for row in selected]
                    ),
                    "fpga_kernel_invocations": distribution(
                        [row["cost"]["fpga_kernel_invocations"] for row in selected]
                    ),
                    "logical_dma_bytes": distribution(
                        [row["cost"]["logical_dma_bytes"] for row in selected]
                    ),
                    "logical_dma_calls": distribution(
                        [row["cost"]["logical_dma_calls"] for row in selected]
                    ),
                    "phase_wall_seconds": {
                        phase: distribution(
                            [
                                row["cost"]["phase_wall_seconds"][phase]
                                for row in selected
                            ]
                        )
                        for phase in baseline.TERMINAL_PHASES
                    },
                    "qualification_wall_seconds": distribution(
                        [row["cost"]["qualification_wall_seconds"] for row in selected]
                    ),
                    "cross_compile_wall_seconds": distribution(
                        [row["cost"]["cross_compile_wall_seconds"] for row in selected]
                    ),
                    "correctness_wall_seconds": distribution(
                        [row["cost"]["correctness_wall_seconds"] for row in selected]
                    ),
                    "timing_wall_seconds": distribution(
                        [row["cost"]["timing_wall_seconds"] for row in selected]
                    ),
                }
    return output


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))
    adapter_dir = Path(args.adapter_dir).resolve()
    input_hashes = {
        "adapter": verify(adapter_dir),
        "hw_union": verify(args.hw_union_dir),
        "hw_qualification": verify(args.hw_qualification_dir),
        "policy_amendment": verify(args.policy_amendment_dir),
        "full_original_contracts": verify(args.full_original_dir),
    }
    scan_dirs = [Path(value).resolve() for value in args.full_scan_dir]
    input_hashes["full_original_scans"] = {
        str(directory): verify(directory) for directory in scan_dirs
    }
    adapter_summary = read_json(adapter_dir / "summary.json")
    if adapter_summary.get("status") != "complete_post_board_replay_adapter":
        raise RuntimeError("complete post-board adapter is required")
    contracts, payloads = {}, {}
    for workload_id in ("R18-H1", "R18-H2", "R18-H3"):
        contract = read_json(
            adapter_dir / "{}_workload_contract.json".format(workload_id.lower())
        )
        contracts[workload_id] = baseline.validate_workload_contract(contract)
        payload = read_json(
            adapter_dir / "{}_outcome_ledger.json".format(workload_id.lower())
        )
        if payload.get("pool_complete") is not True:
            raise RuntimeError("oracle forbidden before complete pool")
        payloads[workload_id] = payload
    memberships = {
        (row["workload_id"], int(row["seed"])): row["valid_e0_candidate_ids"]
        for row in read_jsonl(Path(args.hw_union_dir) / "seed_e0_membership.jsonl")
    }
    presampling = build_hw_presampling_rows(
        args.full_original_dir, scan_dirs, memberships
    )
    static, fsim = e0_local_maps(args.hw_qualification_dir)
    output.mkdir(parents=True)
    results_path = output / "results.jsonl"
    results_path.write_text("", encoding="utf-8")
    contract = {
        "schema": SCHEMA,
        "status": "frozen_before_policy_replay_and_oracle_join",
        "policies": list(MAIN_POLICIES),
        "ablations": ["ml2tuner_pva_A_plus_DMA"],
        "seeds": list(SEEDS),
        "budgets": list(BUDGETS),
        "target_protocol": "leave one workload for target replay; other two complete workloads are development data",
        "cost_views": [
            "counterfactual_lazy_per_candidate",
            "actual_complete_pool_common_cost_reported_separately",
        ],
        "input_hashes": input_hashes,
    }
    write_json(output / "contract.json", contract)
    started = time.monotonic()
    runs = []
    for workload_id, candidates in contracts.items():
        development = development_rows(contracts, payloads, workload_id)
        oracle = baseline.OutcomeLedger(payloads[workload_id]).oracle_after_completion(
            [row["candidate_id"] for row in candidates]
        )
        cheng_pool, cheng_excluded = cheng_candidates(candidates)
        for seed in SEEDS:
            for requested_budget in BUDGETS:
                for policy in MAIN_POLICIES:
                    budget = min(requested_budget, len(candidates))
                    diagnostic = {}
                    if policy == "rieber_hw_init_xgb":
                        timeline, policy_results, diagnostic = replay_rieber(
                            candidates,
                            payloads[workload_id],
                            memberships[(workload_id, seed)],
                            static,
                            fsim,
                            presampling[(workload_id, seed)],
                            budget,
                            seed,
                            development,
                        )
                    elif policy == "ml2tuner_pva":
                        ledger = baseline.OutcomeLedger(payloads[workload_id])
                        timeline, policy_results, diagnostic = baseline.replay_ml2tuner(
                            candidates, budget, seed, development, ledger, include_dma=False
                        )
                    else:
                        selected_candidates = (
                            cheng_pool if policy == "cheng_minimum_access" else candidates
                        )
                        ledger = baseline.OutcomeLedger(payloads[workload_id])
                        timeline, policy_results, fallback, diagnostic = baseline.replay_standard(
                            policy,
                            selected_candidates,
                            min(budget, len(selected_candidates)),
                            seed,
                            development,
                            ledger,
                        )
                        if policy == "cheng_minimum_access":
                            diagnostic = {
                                **diagnostic,
                                "families_without_original": cheng_excluded,
                                "resource_fallbacks": fallback,
                            }
                    row = {
                        "schema": SCHEMA,
                        "workload_id": workload_id,
                        "policy": policy,
                        "seed": seed,
                        "requested_budget": requested_budget,
                        "effective_attempts": len(policy_results),
                        "oracle": oracle,
                        "target": target_metrics(policy_results, oracle),
                        "cost": baseline.summarize_cost(timeline, policy_results),
                        "diagnostic": diagnostic,
                    }
                    runs.append(row)
                    append_jsonl(results_path, row)
                # A+DMA is kept as a named ML2Tuner ablation, not a seventh
                # headline policy.
                budget = min(requested_budget, len(candidates))
                ledger = baseline.OutcomeLedger(payloads[workload_id])
                timeline, policy_results, diagnostic = baseline.replay_ml2tuner(
                    candidates, budget, seed, development, ledger, include_dma=True
                )
                row = {
                    "schema": SCHEMA,
                    "workload_id": workload_id,
                    "policy": "ml2tuner_pva_A_plus_DMA",
                    "seed": seed,
                    "requested_budget": requested_budget,
                    "effective_attempts": len(policy_results),
                    "oracle": oracle,
                    "target": target_metrics(policy_results, oracle),
                    "cost": baseline.summarize_cost(timeline, policy_results),
                    "diagnostic": diagnostic,
                }
                runs.append(row)
                append_jsonl(results_path, row)
            print("{} seed {} complete".format(workload_id, seed), flush=True)
    pa_rows = model_pa_metrics(contracts, payloads)
    (output / "model_pa_metrics.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in pa_rows),
        encoding="utf-8",
    )
    pa_summary = {}
    for workload_id in contracts:
        pa_summary[workload_id] = {}
        for model in ("P_visible", "A_compiler_hidden", "A_plus_DMA"):
            selected = [
                row
                for row in pa_rows
                if row["target_workload"] == workload_id and row["model"] == model
            ]
            pa_summary[workload_id][model] = {
                "rmse_ms": distribution([row["rmse_ms"] for row in selected]),
                "ndcg": distribution([row["ndcg"] for row in selected]),
                "training_rows": selected[0]["training_rows"] if selected else 0,
                "test_rows": selected[0]["test_rows"] if selected else 0,
            }
    v_rows = model_v_metrics(contracts, payloads)
    (output / "model_v_metrics.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in v_rows),
        encoding="utf-8",
    )
    v_summary = {}
    for workload_id in contracts:
        selected = [
            row for row in v_rows if row["target_workload"] == workload_id
        ]
        v_summary[workload_id] = {
            name: distribution([row[name] for row in selected])
            for name in (
                "precision",
                "recall",
                "f1",
                "accuracy",
                "ndcg",
                "top_k_valid_yield",
                "base_valid_yield",
            )
        }
        v_summary[workload_id].update(
            {
                "training_rows": selected[0]["training_rows"] if selected else 0,
                "test_rows": selected[0]["support"] if selected else 0,
                "valid": selected[0]["valid"] if selected else 0,
                "invalid": selected[0]["invalid"] if selected else 0,
            }
        )
    summary = {
        "schema": SCHEMA,
        "status": "complete_six_policy_twenty_seed_pool_analysis",
        "aggregate": aggregate_runs(runs),
        "model_pa": pa_summary,
        "model_v": v_summary,
        "actual_complete_pool_common_cost": adapter_summary["common_actual_cost"],
        "analysis_wall_seconds": time.monotonic() - started,
        "claim_boundary": "offline replay of one complete non-spliced FPGA pool; actual all-pool cost and counterfactual lazy cost remain separate; logical DMA is not physical AXI",
    }
    write_json(output / "summary.json", summary)
    (output / "timeline.jsonl").write_text("", encoding="utf-8")
    (output / "command.txt").write_text(
        " ".join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]) + "\n",
        encoding="utf-8",
    )
    write_json(
        output / "artifact_hashes.json",
        {
            "artifacts": {
                path.name: baseline.sha256_file(path)
                for path in sorted(output.iterdir())
                if path.is_file() and path.name != "artifact_hashes.json"
            },
            "source_hashes": {
                str(Path(__file__).resolve()): baseline.sha256_file(Path(__file__).resolve()),
                str(Path(baseline.__file__).resolve()): baseline.sha256_file(Path(baseline.__file__).resolve()),
            },
        },
    )
    print(json.dumps({"status": summary["status"], "analysis_wall_seconds": summary["analysis_wall_seconds"]}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--hw-union-dir", required=True)
    parser.add_argument("--hw-qualification-dir", required=True)
    parser.add_argument("--policy-amendment-dir", required=True)
    parser.add_argument("--full-original-dir", required=True)
    parser.add_argument("--full-scan-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
