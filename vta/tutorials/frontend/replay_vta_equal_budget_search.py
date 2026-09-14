#!/usr/bin/env python3
"""No-leak, equal-gross-budget VTA search replay and prospective dispatcher.

The canonical pool keeps label-free, frozen candidate features in ``predispatch``
and outcomes in ``oracle``.  Selectors only receive a projected public candidate;
an outcome is revealed after a gross dispatch.  Offline mode evaluates five
policies on full labels.  Prospective mode recomputes one next dispatch from an
immutable history and can consume separate, previously exposed training pools.

A sealed reference is external to the candidate list, is never dispatched or
used as a model label, and is read only after a run to calculate trials/success
to its 2% and 5% bands.  An exact TopHub identity is one allowed reference;
long-budget stock XGB and a post-run full-pool oracle are the other two.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


POOL_SCHEMA = "c3_no_leak_search_pool_v1"
RESULT_SCHEMA = "c3_equal_gross_budget_search_replay_v2"
DISPATCH_SCHEMA = "c3_prospective_search_dispatch_v1"
BASE_POLICY_ALIASES = (
    "random", "stock_knob_xgb", "rules_only", "paper_minimum_access",
    "shared_memory_lexicographic", "shared_memory_service_proxy",
    "validity_v", "v_plus_delta_t"
)
DELTA_POLICY_FEATURE_SET = {
    # Backward-compatible fifth policy: the strongest P7R113 feature boundary.
    "v_plus_delta_t": "bytes_calls",
    "v_plus_delta_t_request": "request",
    "v_plus_delta_t_command": "command",
}
POLICIES = BASE_POLICY_ALIASES + ("v_plus_delta_t_request", "v_plus_delta_t_command")
PHASES = ("lower", "fsim", "compile", "fpga", "measure")
RESOURCE_KEYS = (
    "logical_vta_load_bytes",
    "logical_vta_store_bytes",
    "logical_vta_dma_calls",
    "fpga_kernel_invocations",
)
OUTCOME_CLASSES = tuple(f"{phase}_invalid" for phase in PHASES) + ("measured_ok", "incomplete")
THRESHOLDS = (2, 5)
DEFAULT_BUDGETS = (4, 8, 12)
DEFAULT_SEEDS = 20
XGB_WARMUP_GROSS_DISPATCHES = 4
# Development-frozen acquisition weights.  These are byte-equivalent search
# penalties for request launch and extra submission service, not physical AXI
# traffic and not direct measurements of bandwidth or latency.
SERVICE_REQUEST_EQUIVALENT_BYTES = 64 * 1024
SERVICE_EXTRA_SUBMISSION_EQUIVALENT_BYTES = 128 * 1024


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_hash(seed, workload_id, candidate_id, namespace):
    value = f"{seed}:{namespace}:{workload_id}:{candidate_id}".encode()
    return hashlib.sha256(value).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def numeric_map(value, name):
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{name} must be a non-empty numeric object")
    result = {}
    for key, item in value.items():
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
            raise ValueError(f"{name}.{key} must be finite numeric")
        result[str(key)] = float(item)
    return result


def validate_phase_outcome(outcome, require_oracle=True):
    if outcome is None:
        if require_oracle:
            raise ValueError("offline replay requires oracle outcomes")
        return
    phases = outcome.get("phases")
    if not isinstance(phases, dict) or set(phases) != set(PHASES):
        raise ValueError("outcome phases must contain lower/fsim/compile/fpga/measure exactly")
    stopped = False
    for phase in PHASES:
        record = phases[phase]
        status = record.get("status")
        if status not in ("ok", "invalid", "not_run"):
            raise ValueError(f"invalid {phase} status")
        wall = record.get("wall_ms")
        if wall is not None and (isinstance(wall, bool) or not isinstance(wall, (int, float))
                                 or wall < 0 or not math.isfinite(wall)):
            raise ValueError(f"invalid {phase} wall_ms")
        resource = record.get("resource_cost")
        if resource is not None:
            if not isinstance(resource, dict) or set(resource) != set(RESOURCE_KEYS):
                raise ValueError(f"invalid {phase} resource_cost keys")
            for name, value in resource.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)) \
                        or value < 0 or not math.isfinite(value):
                    raise ValueError(f"invalid {phase} resource_cost.{name}")
        if stopped and status != "not_run":
            raise ValueError("downstream phase ran after invalid/not_run phase")
        if status in ("invalid", "not_run"):
            stopped = True
    measure_ok = phases["measure"]["status"] == "ok"
    latency = outcome.get("latency_ms")
    if measure_ok != (isinstance(latency, (int, float)) and not isinstance(latency, bool)
                      and latency > 0 and math.isfinite(latency)):
        raise ValueError("positive latency_ms is required exactly when measure status is ok")
    delta = outcome.get("delta_t_ms")
    if delta is not None and (isinstance(delta, bool) or not isinstance(delta, (int, float))
                              or not math.isfinite(delta)):
        raise ValueError("delta_t_ms must be finite or null")


def validate_pool(pool, require_oracle=True):
    if pool.get("schema") != POOL_SCHEMA:
        raise ValueError(f"expected {POOL_SCHEMA}")
    if pool.get("frozen_before_target_labels") is not True:
        raise ValueError("pool must attest frozen_before_target_labels=true")
    workloads = pool.get("workloads")
    if not isinstance(workloads, dict) or not workloads:
        raise ValueError("pool workloads must be a non-empty object")
    globally_seen = set()
    delta_feature_sets = pool.get("delta_feature_sets")
    if not isinstance(delta_feature_sets, dict) or set(delta_feature_sets) != {
        "bytes_calls", "request", "command"
    }:
        raise ValueError("pool must freeze bytes_calls/request/command delta_feature_sets")
    for name, fields in delta_feature_sets.items():
        if not isinstance(fields, list) or not fields or len(fields) != len(set(fields)):
            raise ValueError(f"invalid frozen delta feature set: {name}")
    if not set(delta_feature_sets["bytes_calls"]).issubset(delta_feature_sets["request"]):
        raise ValueError("request delta feature set must include bytes_calls")
    if not set(delta_feature_sets["request"]).issubset(delta_feature_sets["command"]):
        raise ValueError("command delta feature set must include request")
    feature_keys = {"knobs": None, "validity": None, "delta_t": None}
    for workload_id, workload in workloads.items():
        reference = workload.get("sealed_reference")
        if not isinstance(reference, dict) or reference.get("reference_kind") not in (
            "tophub", "long_budget_stock_xgb", "full_pool_oracle"
        ):
            raise ValueError(f"{workload_id}: valid sealed_reference missing")
        reference_id = reference.get("reference_id")
        if reference["reference_kind"] == "tophub" and (
            not reference_id or reference.get("exact_identity") is not True
        ):
            raise ValueError("tophub reference requires an exact sealed identity")
        reference_latency = reference.get("latency_ms")
        if require_oracle and reference["reference_kind"] != "full_pool_oracle" and not (
            isinstance(reference_latency, (int, float)) and not isinstance(reference_latency, bool)
            and reference_latency > 0 and math.isfinite(reference_latency)
        ):
            raise ValueError(f"{workload_id}: sealed reference latency missing")
        if reference["reference_kind"] == "full_pool_oracle" and reference_latency is not None:
            raise ValueError("full_pool_oracle latency must be computed after the order, not stored")
        candidates = workload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"{workload_id}: empty candidate list")
        local_seen = set()
        for candidate in candidates:
            candidate_id = candidate.get("candidate_id")
            if not candidate_id or candidate_id in local_seen or candidate_id in globally_seen:
                raise ValueError("missing or duplicate candidate_id")
            if candidate_id == reference_id or candidate.get("is_sealed_reference") is True \
                    or candidate.get("is_tophub") is True:
                raise ValueError("sealed reference must not appear in candidate list")
            if candidate.get("workload_id") != workload_id:
                raise ValueError("candidate workload_id mismatch")
            pre = candidate.get("predispatch", {})
            if pre.get("feature_provenance") not in (
                "frozen_before_target_labels", "p7q_development_pre_timing"
            ):
                raise ValueError("candidate feature provenance is not no-leak eligible")
            rule_score = pre.get("rule_score")
            if isinstance(rule_score, bool) or not isinstance(rule_score, (int, float)) \
                    or not math.isfinite(rule_score):
                raise ValueError("rule_score must be finite numeric")
            for name in feature_keys:
                values = numeric_map(pre.get(name), f"{candidate_id}.{name}")
                keys = tuple(sorted(values))
                if feature_keys[name] is None:
                    feature_keys[name] = keys
                elif feature_keys[name] != keys:
                    raise ValueError(f"pool-wide {name} feature keys differ")
            delta_values = pre["delta_t"]
            for feature_set, fields in delta_feature_sets.items():
                if not set(fields).issubset(delta_values):
                    raise ValueError(f"candidate lacks {feature_set} delta features")
            validate_phase_outcome(candidate.get("oracle"), require_oracle=require_oracle)
            local_seen.add(candidate_id)
            globally_seen.add(candidate_id)
    return pool


def public_candidate(candidate):
    """Project away every outcome field before calling a selector."""
    pre = candidate["predispatch"]
    return {
        "candidate_id": candidate["candidate_id"],
        "workload_id": candidate["workload_id"],
        "residence_mode": candidate.get("residence_mode", "unknown"),
        "same_tile_control_id": candidate.get("same_tile_control_id"),
        "predispatch": {
            "feature_provenance": pre["feature_provenance"],
            "rule_score": float(pre["rule_score"]),
            "knobs": numeric_map(pre["knobs"], "knobs"),
            "validity": numeric_map(pre["validity"], "validity"),
            "delta_t": numeric_map(pre["delta_t"], "delta_t"),
        },
    }


def outcome_class(outcome):
    for phase in PHASES:
        if outcome["phases"][phase]["status"] == "invalid":
            return f"{phase}_invalid"
    if outcome["phases"]["measure"]["status"] == "ok":
        return "measured_ok"
    return "incomplete"


def vector(candidate, family):
    values = candidate["predispatch"][family]
    return [values[name] for name in sorted(values)]


def training_record(candidate, outcome):
    return {
        "candidate": public_candidate(candidate),
        "outcome": outcome,
        "valid": outcome_class(outcome) == "measured_ok",
    }


def held_out_training_records(pools, target_workload_id):
    records = []
    for pool in pools:
        validate_pool(pool, require_oracle=True)
        for workload_id, workload in pool["workloads"].items():
            if workload_id == target_workload_id:
                continue
            for candidate in workload["candidates"]:
                records.append(training_record(candidate, candidate["oracle"]))
    return records


def current_training_records(public_by_id, history):
    records = []
    for item in history:
        candidate_id = item["candidate_id"]
        if candidate_id not in public_by_id:
            raise ValueError("history candidate absent from target pool")
        validate_phase_outcome(item["outcome"], require_oracle=True)
        records.append({
            "candidate": public_by_id[candidate_id],
            "outcome": item["outcome"],
            "valid": outcome_class(item["outcome"]) == "measured_ok",
        })
    return records


def static_metric(candidate, name):
    key = "static_" + name
    values = candidate["predispatch"]["validity"]
    if key not in values:
        raise ValueError(f"candidate lacks required shared-memory metric {key}")
    return float(values[key])


def same_tile_group(candidate):
    return candidate.get("same_tile_control_id") or candidate["candidate_id"]


def minimum_access_winners(candidates):
    """Return one label-free minimum-total-DMA mode per same-tile family."""
    groups = defaultdict(list)
    for candidate in candidates:
        groups[same_tile_group(candidate)].append(candidate)
    winners = set()
    for members in groups.values():
        chosen = min(members, key=lambda candidate: (
            static_metric(candidate, "dma_total_bytes"),
            static_metric(candidate, "dma_total_calls"),
            candidate["candidate_id"],
        ))
        winners.add(chosen["candidate_id"])
    return winners


def fit_validity(records):
    if not records:
        return {"kind": "constant", "probability": 0.5, "training_count": 0}
    labels = np.asarray([int(record["valid"]) for record in records])
    probability = float(np.mean(labels))
    if len(set(labels.tolist())) < 2:
        return {"kind": "constant", "probability": probability,
                "training_count": len(records)}
    estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", random_state=0, max_iter=2000),
    )
    estimator.fit(np.asarray([vector(record["candidate"], "validity") for record in records]), labels)
    return {"kind": "logistic", "estimator": estimator, "training_count": len(records)}


def predict_validity(model, candidates):
    if model["kind"] == "constant":
        return [model["probability"]] * len(candidates)
    return model["estimator"].predict_proba(
        np.asarray([vector(candidate, "validity") for candidate in candidates])
    )[:, 1].tolist()


def fit_delta(records, feature_names):
    labeled = [record for record in records if record["valid"]
               and record["outcome"].get("delta_t_ms") is not None]
    if not labeled:
        return {"kind": "constant", "value_ms": 0.0, "training_count": 0}
    modes = sorted({record["candidate"]["residence_mode"] for record in labeled})
    feature_names = list(feature_names)
    scales = {}
    for name in feature_names:
        nonzero = [abs(record["candidate"]["predispatch"]["delta_t"][name])
                   for record in labeled
                   if record["candidate"]["predispatch"]["delta_t"][name] != 0]
        scales[name] = statistics.median(nonzero) if nonzero else 1.0
    design = []
    target = []
    for record in labeled:
        candidate = record["candidate"]
        design.append(
            [float(candidate["residence_mode"] == mode) for mode in modes]
            + [candidate["predispatch"]["delta_t"][name] / scales[name]
               for name in feature_names]
        )
        target.append(float(record["outcome"]["delta_t_ms"]))
    matrix = np.asarray(design, dtype="float64")
    values = np.asarray(target, dtype="float64")
    lower = np.asarray([-np.inf] * len(modes) + [0.0] * len(feature_names))
    # A small deterministic ridge stabilizes the collinear cumulative ablations;
    # it does not relax non-negativity of the physical feature coefficients.
    penalty = np.zeros((len(feature_names), matrix.shape[1]), dtype="float64")
    penalty[:, len(modes):] = np.eye(len(feature_names)) * math.sqrt(1e-6)
    matrix = np.vstack([matrix, penalty])
    values = np.concatenate([values, np.zeros(len(feature_names))])
    solution = lsq_linear(
        matrix, values, bounds=(lower, np.full(matrix.shape[1], np.inf)),
        method="trf", lsmr_tol="auto", max_iter=500,
    )
    if not solution.success:
        raise RuntimeError(f"delta proxy fit failed: {solution.message}")
    return {
        "kind": "mode_intercept_nonnegative_linear",
        "modes": modes,
        "features": feature_names,
        "scales": scales,
        "coefficients": solution.x.tolist(),
        "fallback_mode_intercept_ms": statistics.fmean(target),
        "training_count": len(labeled),
    }


def predict_delta(model, candidate):
    if model["kind"] == "constant":
        return model["value_ms"]
    mode = candidate["residence_mode"]
    value = model["fallback_mode_intercept_ms"]
    if mode in model["modes"]:
        value = model["coefficients"][model["modes"].index(mode)]
    base = len(model["modes"])
    for index, name in enumerate(model["features"]):
        value += (model["coefficients"][base + index]
                  * candidate["predispatch"]["delta_t"][name] / model["scales"][name])
    return float(value)


def rank_fraction(values, smaller_is_better):
    order = sorted(range(len(values)), key=lambda i: ((values[i] if smaller_is_better else -values[i]), i))
    denominator = max(1, len(values) - 1)
    result = [0.0] * len(values)
    for rank, index in enumerate(order):
        result[index] = rank / denominator
    return result


def fit_stock_xgb(records):
    measured = [record for record in records if record["valid"]]
    if len(measured) < 2:
        return None
    import xgboost as xgb
    estimator = xgb.XGBRegressor(
        n_estimators=64, max_depth=3, learning_rate=0.1, min_child_weight=1,
        subsample=1.0, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=1.0,
        objective="reg:squarederror", tree_method="hist", random_state=20250901,
        n_jobs=1, verbosity=0,
    )
    estimator.fit(
        np.asarray([vector(record["candidate"], "knobs") for record in measured]),
        np.asarray([record["outcome"]["latency_ms"] for record in measured]),
    )
    return estimator


def choose_next(policy, candidates, history, prior_training, seed, delta_feature_sets):
    public = [public_candidate(candidate) for candidate in candidates]
    public_by_id = {candidate["candidate_id"]: candidate for candidate in public}
    observed_ids = [item["candidate_id"] for item in history]
    if len(observed_ids) != len(set(observed_ids)):
        raise ValueError("history contains duplicate dispatch")
    remaining = [candidate for candidate in public if candidate["candidate_id"] not in observed_ids]
    if not remaining:
        return None, {"reason": "exhausted"}
    workload_id = remaining[0]["workload_id"]
    random_key = lambda candidate, namespace: stable_hash(
        seed, workload_id, candidate["candidate_id"], namespace
    )
    current = current_training_records(public_by_id, history)

    if policy == "random":
        chosen = min(remaining, key=lambda candidate: random_key(candidate, "random"))
        return chosen, {"selection": "seeded_random_permutation"}

    if policy == "rules_only":
        chosen = min(remaining, key=lambda candidate: (
            candidate["predispatch"]["rule_score"], random_key(candidate, "rules_tie")
        ))
        return chosen, {
            "selection": "minimum_frozen_rule_score",
            "rule_score": chosen["predispatch"]["rule_score"],
        }

    if policy == "paper_minimum_access":
        # Reproduce the mechanism-level rule only: choose the lowest-access
        # residence mode inside each same-tile family, without using absolute
        # cross-tile memory cost to decide which family should be tried first.
        winners = minimum_access_winners(public)
        eligible = [candidate for candidate in remaining if candidate["candidate_id"] in winners]
        tier = "same_tile_minimum_access_winner"
        if not eligible:
            eligible = remaining
            tier = "remaining_mode_after_family_winners"
        chosen = min(eligible, key=lambda candidate: random_key(candidate, "paper_minimum_access"))
        return chosen, {"selection": tier, "same_tile_family_winner_count": len(winners)}

    if policy == "shared_memory_lexicographic":
        chosen = min(remaining, key=lambda candidate: (
            static_metric(candidate, "dma_total_bytes"),
            static_metric(candidate, "dma_total_calls"),
            static_metric(candidate, "padded_dma_calls"),
            static_metric(candidate, "submissions"),
            random_key(candidate, "shared_memory_lexicographic_tie"),
        ))
        return chosen, {
            "selection": "absolute_shared_memory_lexicographic",
            "dma_total_bytes": static_metric(chosen, "dma_total_bytes"),
            "dma_total_calls": static_metric(chosen, "dma_total_calls"),
            "padded_dma_calls": static_metric(chosen, "padded_dma_calls"),
            "submissions": static_metric(chosen, "submissions"),
        }

    if policy == "shared_memory_service_proxy":
        def service_score(candidate):
            submissions = static_metric(candidate, "submissions")
            return (
                static_metric(candidate, "dma_total_bytes")
                + SERVICE_REQUEST_EQUIVALENT_BYTES
                * static_metric(candidate, "dma_total_calls")
                + SERVICE_EXTRA_SUBMISSION_EQUIVALENT_BYTES
                * max(submissions - 1.0, 0.0)
            )

        chosen = min(remaining, key=lambda candidate: (
            service_score(candidate),
            random_key(candidate, "shared_memory_service_proxy_tie"),
        ))
        return chosen, {
            "selection": "shared_memory_service_cost_proxy",
            "service_score_byte_equivalent": service_score(chosen),
            "request_equivalent_bytes": SERVICE_REQUEST_EQUIVALENT_BYTES,
            "extra_submission_equivalent_bytes": SERVICE_EXTRA_SUBMISSION_EQUIVALENT_BYTES,
            "dma_total_bytes": static_metric(chosen, "dma_total_bytes"),
            "dma_total_calls": static_metric(chosen, "dma_total_calls"),
            "submissions": static_metric(chosen, "submissions"),
        }

    if policy == "stock_knob_xgb":
        if len(history) < XGB_WARMUP_GROSS_DISPATCHES:
            chosen = min(remaining, key=lambda candidate: random_key(candidate, "xgb_warmup"))
            return chosen, {"selection": "seeded_random_gross_warmup", "training_count": 0}
        estimator = fit_stock_xgb(current)
        if estimator is None:
            chosen = min(remaining, key=lambda candidate: random_key(candidate, "xgb_fallback"))
            return chosen, {"selection": "seeded_random_insufficient_valid_labels",
                            "training_count": sum(record["valid"] for record in current)}
        predictions = estimator.predict(
            np.asarray([vector(candidate, "knobs") for candidate in remaining])
        ).tolist()
        _, chosen = min(zip(predictions, remaining), key=lambda item: (
            item[0], random_key(item[1], "xgb_tie")
        ))
        return chosen, {
            "selection": "stock_knob_xgb_min_prediction",
            "training_count": sum(record["valid"] for record in current),
            "predicted_latency_ms": float(predictions[remaining.index(chosen)]),
        }

    records = list(prior_training) + current
    validity = fit_validity(records)
    probabilities = predict_validity(validity, remaining)
    if policy == "validity_v":
        _, chosen = min(zip(probabilities, remaining), key=lambda item: (
            -item[0], random_key(item[1], "validity_tie")
        ))
        return chosen, {
            "selection": "maximum_leave-target-out_validity_probability",
            "validity_training_count": validity["training_count"],
            "predicted_valid_probability": float(probabilities[remaining.index(chosen)]),
        }

    if policy in DELTA_POLICY_FEATURE_SET:
        feature_set = DELTA_POLICY_FEATURE_SET[policy]
        delta = fit_delta(records, delta_feature_sets[feature_set])
        delta_predictions = [predict_delta(delta, candidate) for candidate in remaining]
        risk_ranks = rank_fraction(probabilities, smaller_is_better=False)
        delta_ranks = rank_fraction(delta_predictions, smaller_is_better=True)
        acquisitions = [(risk_ranks[i] + delta_ranks[i]) / 2.0 for i in range(len(remaining))]
        _, chosen = min(zip(acquisitions, remaining), key=lambda item: (
            item[0], random_key(item[1], "v_delta_tie")
        ))
        index = remaining.index(chosen)
        return chosen, {
            "selection": "equal_rank_fusion_validity_risk_and_same_tile_delta_t",
            "delta_feature_set": feature_set,
            "validity_training_count": validity["training_count"],
            "delta_training_count": delta["training_count"],
            "predicted_valid_probability": float(probabilities[index]),
            "predicted_delta_t_ms": float(delta_predictions[index]),
            "acquisition_rank_score": float(acquisitions[index]),
        }

    raise ValueError(f"unknown policy: {policy}")


def reveal(candidate):
    """Offline oracle boundary: called only after choose_next returns."""
    return json.loads(json.dumps(candidate["oracle"]))


def run_offline_workload(pool, workload_id, policy, seed, training_pools=None):
    workload = pool["workloads"][workload_id]
    candidates = workload["candidates"]
    sources = [pool] if training_pools is None else training_pools
    prior = held_out_training_records(sources, workload_id)
    history = []
    dispatches = []
    while len(history) < len(candidates):
        chosen, diagnostics = choose_next(
            policy, candidates, history, prior, seed, pool["delta_feature_sets"]
        )
        if chosen is None:
            break
        original = next(item for item in candidates if item["candidate_id"] == chosen["candidate_id"])
        outcome = reveal(original)
        item = {
            "gross_dispatch": len(history) + 1,
            "candidate_id": chosen["candidate_id"],
            "selection_diagnostics": diagnostics,
            "outcome": outcome,
            "outcome_class": outcome_class(outcome),
        }
        history.append({"candidate_id": chosen["candidate_id"], "outcome": outcome})
        dispatches.append(item)
    return {
        "workload_id": workload_id,
        "policy": policy,
        "seed": seed,
        "candidate_count_excluding_sealed_reference": len(candidates),
        "held_out_training_count": len(prior),
        "dispatches": dispatches,
    }


def prefix_metrics(run, workload, budget):
    prefix = run["dispatches"][:budget]
    valid_latencies = [item["outcome"]["latency_ms"] for item in prefix
                       if item["outcome_class"] == "measured_ok"]
    all_valid = [candidate["oracle"]["latency_ms"] for candidate in workload["candidates"]
                 if outcome_class(candidate["oracle"]) == "measured_ok"]
    if not all_valid:
        raise ValueError("offline workload has no valid non-reference candidate")
    oracle = min(all_valid)
    best = min(valid_latencies) if valid_latencies else None
    components = {}
    complete = True
    resource_complete = True
    resource_totals = {name: 0.0 for name in RESOURCE_KEYS}
    for phase in PHASES:
        known = []
        missing = 0
        phase_resource_totals = {name: 0.0 for name in RESOURCE_KEYS}
        resource_missing = 0
        for item in prefix:
            record = item["outcome"]["phases"][phase]
            if record["status"] == "not_run":
                continue
            if record.get("wall_ms") is None:
                missing += 1
            else:
                known.append(float(record["wall_ms"]))
            resource = record.get("resource_cost")
            if resource is None:
                resource_missing += 1
            else:
                for name in RESOURCE_KEYS:
                    value = float(resource[name])
                    phase_resource_totals[name] += value
                    resource_totals[name] += value
        components[phase] = {
            "known_sum_ms": sum(known),
            "missing_observation_count": missing,
            "resource_known_sums": phase_resource_totals,
            "resource_missing_observation_count": resource_missing,
        }
        complete = complete and missing == 0
        resource_complete = resource_complete and resource_missing == 0
    counts = {name: sum(item["outcome_class"] == name for item in prefix)
              for name in OUTCOME_CLASSES}
    result = {
        "requested_budget": budget,
        "gross_dispatches": len(prefix),
        "outcome_counts": counts,
        "wall_clock_components": components,
        "wall_clock_complete": complete,
        "wall_clock_known_total_ms": sum(value["known_sum_ms"] for value in components.values()),
        "resource_cost_complete": resource_complete,
        "resource_cost_known_totals": resource_totals,
        "pool_oracle_latency_ms": oracle,
        "best_observed_latency_ms": best,
        "regret_ms": None if best is None else best - oracle,
        "regret_percent": None if best is None else (best / oracle - 1.0) * 100.0,
    }
    for threshold in THRESHOLDS:
        cutoff = oracle * (1.0 + threshold / 100.0)
        result[f"pool_success_at_{threshold}_percent"] = any(
            value <= cutoff for value in valid_latencies
        )
    reference = sealed_reference_latency(workload, all_valid)
    result["sealed_reference_kind"] = workload["sealed_reference"]["reference_kind"]
    result["sealed_reference_latency_ms"] = reference
    for threshold in THRESHOLDS:
        cutoff = reference * (1.0 + threshold / 100.0)
        result[f"success_at_{threshold}_percent"] = any(value <= cutoff for value in valid_latencies)
    return result


def trials_to_threshold(run, workload, threshold):
    all_valid = [candidate["oracle"]["latency_ms"] for candidate in workload["candidates"]
                 if outcome_class(candidate["oracle"]) == "measured_ok"]
    cutoff = sealed_reference_latency(workload, all_valid) * (1.0 + threshold / 100.0)
    for item in run["dispatches"]:
        if item["outcome_class"] == "measured_ok" and item["outcome"]["latency_ms"] <= cutoff:
            return item["gross_dispatch"]
    return None


def trials_to_pool_threshold(run, workload, threshold):
    all_valid = [candidate["oracle"]["latency_ms"] for candidate in workload["candidates"]
                 if outcome_class(candidate["oracle"]) == "measured_ok"]
    if not all_valid:
        raise ValueError("offline workload has no valid non-reference candidate")
    cutoff = min(all_valid) * (1.0 + threshold / 100.0)
    for item in run["dispatches"]:
        if item["outcome_class"] == "measured_ok" and item["outcome"]["latency_ms"] <= cutoff:
            return item["gross_dispatch"]
    return None


def sealed_reference_latency(workload, full_pool_valid_latencies):
    """Called only by post-order metrics, never by candidate selection."""
    reference = workload["sealed_reference"]
    if reference["reference_kind"] == "full_pool_oracle":
        if not full_pool_valid_latencies:
            raise ValueError("cannot derive empty full_pool_oracle")
        return float(min(full_pool_valid_latencies))
    return float(reference["latency_ms"])


def quantile_summary(values):
    values = [float(value) for value in values]
    if not values:
        return {"count": 0, "median": None, "q1": None, "q3": None, "iqr": None}
    q1, median, q3 = np.quantile(np.asarray(values), [0.25, 0.5, 0.75], method="linear")
    return {"count": len(values), "median": float(median), "q1": float(q1),
            "q3": float(q3), "iqr": float(q3 - q1)}


def summarize_run_group(group_runs, pool, budgets):
    if not group_runs:
        raise ValueError("cannot summarize an empty run group")
    result = {"run_count": len(group_runs), "budgets": {}}
    for budget in budgets:
        metrics = [prefix_metrics(run, pool["workloads"][run["workload_id"]], budget)
                   for run in group_runs]
        budget_result = {
            "gross_dispatches": quantile_summary(item["gross_dispatches"] for item in metrics),
            "regret_ms_defined": quantile_summary(
                item["regret_ms"] for item in metrics if item["regret_ms"] is not None
            ),
            "regret_percent_defined": quantile_summary(
                item["regret_percent"] for item in metrics if item["regret_percent"] is not None
            ),
            "regret_defined_rate": statistics.fmean(item["regret_ms"] is not None for item in metrics),
            "wall_clock_complete_rate": statistics.fmean(item["wall_clock_complete"] for item in metrics),
            "wall_clock_known_total_ms": quantile_summary(
                item["wall_clock_known_total_ms"] for item in metrics
            ),
            "resource_cost_complete_rate": statistics.fmean(
                item["resource_cost_complete"] for item in metrics
            ),
            "resource_cost_known_totals": {
                name: quantile_summary(
                    item["resource_cost_known_totals"][name] for item in metrics
                )
                for name in RESOURCE_KEYS
            },
            "outcome_counts": {
                outcome: quantile_summary(item["outcome_counts"][outcome] for item in metrics)
                for outcome in metrics[0]["outcome_counts"]
            },
        }
        for phase in PHASES:
            budget_result[f"{phase}_known_wall_ms"] = quantile_summary(
                item["wall_clock_components"][phase]["known_sum_ms"] for item in metrics
            )
            budget_result[f"{phase}_wall_complete_rate"] = statistics.fmean(
                item["wall_clock_components"][phase]["missing_observation_count"] == 0
                for item in metrics
            )
        for threshold in THRESHOLDS:
            budget_result[f"success_at_{threshold}_percent_rate"] = statistics.fmean(
                item[f"success_at_{threshold}_percent"] for item in metrics
            )
            budget_result[f"pool_success_at_{threshold}_percent_rate"] = statistics.fmean(
                item[f"pool_success_at_{threshold}_percent"] for item in metrics
            )
        result["budgets"][str(budget)] = budget_result
    for threshold in THRESHOLDS:
        trials = [trials_to_threshold(run, pool["workloads"][run["workload_id"]], threshold)
                  for run in group_runs]
        successes = [value for value in trials if value is not None]
        censored = [value if value is not None else len(run["dispatches"]) + 1
                    for value, run in zip(trials, group_runs)]
        result[f"trials_to_{threshold}_percent"] = {
            "success_rate": len(successes) / len(trials),
            "successful_only": quantile_summary(successes),
            "failure_censored_at_pool_size_plus_one": quantile_summary(censored),
        }
        pool_trials = [trials_to_pool_threshold(
            run, pool["workloads"][run["workload_id"]], threshold
        ) for run in group_runs]
        pool_successes = [value for value in pool_trials if value is not None]
        pool_censored = [value if value is not None else len(run["dispatches"]) + 1
                         for value, run in zip(pool_trials, group_runs)]
        result[f"trials_to_pool_{threshold}_percent"] = {
            "success_rate": len(pool_successes) / len(pool_trials),
            "successful_only": quantile_summary(pool_successes),
            "failure_censored_at_pool_size_plus_one": quantile_summary(pool_censored),
        }
    return result


def summarize_runs(runs, pool, budgets):
    result = {}
    for policy in POLICIES:
        policy_runs = [run for run in runs if run["policy"] == policy]
        policy_result = summarize_run_group(policy_runs, pool, budgets)
        policy_result["by_workload"] = {
            workload_id: summarize_run_group(
                [run for run in policy_runs if run["workload_id"] == workload_id], pool, budgets
            )
            for workload_id in sorted(pool["workloads"])
        }
        result[policy] = policy_result
    return result


def run_offline(pool, budgets, seed_count):
    runs = []
    for seed in range(seed_count):
        for workload_id in sorted(pool["workloads"]):
            for policy in POLICIES:
                runs.append(run_offline_workload(pool, workload_id, policy, seed))
    return runs, summarize_runs(runs, pool, budgets)


def knob_map(entry):
    result = {}
    for name, kind, value in entry["complete_config_entity"]["entity"]:
        result[name] = float(value[-1] if kind == "sp" else value)
    return result


def mode_map(mode):
    modes = ("original", "input_stationary", "weight_stationary",
             "weight_stationary_barrier", "paper_inspired_hybrid")
    return {f"mode_{name}": float(mode == name) for name in modes}


def p7q_delta_features(entry, control):
    names = (
        "input_dma_bytes", "weight_dma_bytes", "output_dma_bytes", "input_dma_calls",
        "weight_dma_calls", "output_dma_calls", "small_dma_calls", "strided_dma_calls",
        "padded_dma_calls", "input_reload", "weight_reload", "output_reload",
        "total_insn_bytes", "total_uop_bytes", "submissions", "source_derived_finish_count",
    )
    return {name: float(entry["static_metrics"][name] - control["static_metrics"][name])
            for name in names}


def adapt_p7q(contract_path: Path, summary_paths):
    contract = load_json(contract_path)
    if contract.get("schema") != "c3_p7_qualified_timing_contract_v1":
        raise ValueError("unexpected P7Q contract schema")
    summaries = {summary["workload_id"]: summary
                 for summary in (load_json(path) for path in summary_paths)}
    workloads = {}
    for workload_id in contract["workload_order"]:
        source = contract["workloads"][workload_id]
        if workload_id not in summaries:
            raise ValueError(f"missing P7Q timing summary for {workload_id}")
        labels = summaries[workload_id]["candidate_summaries"]
        entries = {entry["candidate_id"]: entry for entry in source["gross_candidates"]}
        incumbent_id = source["incumbent_candidate_id"]
        if incumbent_id not in labels:
            raise ValueError("P7Q incumbent timing missing")
        originals = {}
        for entry in entries.values():
            if entry["residence_mode"] == "original":
                signature = json.dumps(entry["complete_config_entity"]["entity"], sort_keys=True)
                originals[signature] = entry
        candidates = []
        for candidate_id, entry in entries.items():
            if candidate_id == incumbent_id:
                continue
            signature = json.dumps(entry["complete_config_entity"]["entity"], sort_keys=True)
            control = originals.get(signature, entry)
            static = entry["static_metrics"]
            knobs = knob_map(entry)
            validity = dict(knobs)
            validity.update(mode_map(entry["residence_mode"]))
            validity.update({f"static_{name}": float(value) for name, value in static.items()
                             if isinstance(value, (int, float))})
            if candidate_id in labels:
                phases = {phase: {"status": "ok", "wall_ms": None} for phase in PHASES}
                latency = float(labels[candidate_id]["median_ms"])
            else:
                phases = {
                    "lower": {"status": "ok", "wall_ms": None},
                    "fsim": {"status": "ok", "wall_ms": None},
                    "compile": {"status": "ok", "wall_ms": None},
                    "fpga": {"status": "invalid", "wall_ms": None},
                    "measure": {"status": "not_run", "wall_ms": None},
                }
                latency = None
            control_id = control["candidate_id"]
            delta_label = None
            if latency is not None and control_id != incumbent_id and control_id in labels:
                delta_label = latency - float(labels[control_id]["median_ms"])
            candidates.append({
                "candidate_id": candidate_id,
                "workload_id": workload_id,
                "residence_mode": entry["residence_mode"],
                "same_tile_control_id": control_id,
                "predispatch": {
                    "feature_provenance": "p7q_development_pre_timing",
                    "rule_score": float(static["dma_total_bytes"]),
                    "knobs": knobs,
                    "validity": validity,
                    "delta_t": p7q_delta_features(entry, control),
                },
                "oracle": {"phases": phases, "latency_ms": latency,
                           "delta_t_ms": delta_label},
            })
        workloads[workload_id] = {
            "sealed_reference": {
                "reference_kind": "tophub",
                "reference_id": incumbent_id,
                "exact_identity": True,
                "latency_ms": float(labels[incumbent_id]["median_ms"]),
                "role": "sealed_metric_only",
            },
            "candidates": candidates,
        }
    bytes_calls = [
        "input_dma_bytes", "weight_dma_bytes", "output_dma_bytes", "input_dma_calls",
        "weight_dma_calls", "output_dma_calls",
    ]
    request = bytes_calls + [
        "small_dma_calls", "strided_dma_calls", "padded_dma_calls", "input_reload",
        "weight_reload", "output_reload",
    ]
    command = request + [
        "total_insn_bytes", "total_uop_bytes", "submissions", "source_derived_finish_count",
    ]
    return {
        "schema": POOL_SCHEMA,
        "pool_id": "p7q_development_adapter_" + sha256(contract_path)[:16],
        "frozen_before_target_labels": True,
        "delta_feature_sets": {
            "bytes_calls": bytes_calls,
            "request": request,
            "command": command,
        },
        "claim_status": "development_only_not_confirmation",
        "feature_availability": (
            "P7Q static metrics were frozen before FPGA timing but after lowering/cross-compile; "
            "phase wall-clock costs are unavailable and remain null"
        ),
        "source": {
            "contract": {"path": str(contract_path.resolve()), "sha256": sha256(contract_path)},
            "timing_summaries": [{"path": str(path.resolve()), "sha256": sha256(path)}
                                 for path in summary_paths],
        },
        "workloads": workloads,
    }


def protocol(budgets, seed_count):
    return {
        "schema": "c3_equal_gross_budget_search_protocol_v2",
        "policies": list(POLICIES),
        "gross_budgets": list(budgets),
        "seeds": seed_count,
        "xgb": {
            "features": "seven ConfigEntity knob leaves only",
            "warmup": f"{XGB_WARMUP_GROSS_DISPATCHES} seeded gross dispatches",
            "training": "successful current-target observations revealed before the next dispatch only",
            "model": "64-tree depth-3 XGBRegressor, seed fixed; dispatch seed affects warmup/ties",
        },
        "rules_only": "ascending frozen label-free rule_score",
        "paper_minimum_access": (
            "minimum total DMA mode within each same-tile family; family order is seeded random"
        ),
        "shared_memory_lexicographic": (
            "absolute total DMA bytes, calls, padded calls and submissions across tile families"
        ),
        "shared_memory_service_proxy": {
            "score": "DMA bytes + request weight * DMA calls + submission weight * max(submissions-1, 0)",
            "request_equivalent_bytes": SERVICE_REQUEST_EQUIVALENT_BYTES,
            "extra_submission_equivalent_bytes": SERVICE_EXTRA_SUBMISSION_EQUIVALENT_BYTES,
            "interpretation": (
                "development-frozen byte-equivalent acquisition penalties; not physical AXI bytes "
                "or literal bandwidth/latency constants"
            ),
        },
        "validity_v": (
            "logistic probability from other-workload exposed labels plus already revealed target outcomes"
        ),
        "v_plus_delta_t": {
            "acquisition": "equal Borda-rank fusion of predicted invalidity risk and same-tile delta-T",
            "frozen_ablations": dict(DELTA_POLICY_FEATURE_SET),
            "compatibility_alias": "v_plus_delta_t means the bytes_calls feature boundary",
            "constraint": "delta physical coefficients are non-negative",
        },
        "no_leak": [
            "selector receives public_candidate projection; oracle is revealed only after selection",
            "target workload is excluded from prior training records",
            "invalid gross dispatches consume budget and are not latency/XGB/delta labels",
            "sealed reference identity is forbidden in candidate lists and all training records",
            "sealed reference latency is read only by post-run 2%/5% threshold metrics",
        ],
        "metrics": [
            "gross dispatches and lower/FSim/compile/FPGA/measure-invalid counts at every budget",
            "known lower/FSim/compile/FPGA/measure wall-ms and logical VTA DMA resource sums",
            "non-reference full-pool oracle simple regret at every budget",
            "success and trials-to-2%/5% against the completed non-reference pool oracle",
            "success@2%/5% and trials-to-2%/5% against the post-run sealed reference",
            "median, Q1, Q3, and IQR over workload-seed runs",
        ],
        "claim_boundary": (
            "offline full-label replay is development evidence, not prospective confirmation; only an "
            "immutable sequential prospective history can support a future confirmation claim"
        ),
    }


def render_results(pool, summary, proto):
    candidates = [candidate for workload in pool["workloads"].values()
                  for candidate in workload["candidates"]]
    outcome_counts = defaultdict(int)
    for candidate in candidates:
        outcome_counts[outcome_class(candidate["oracle"])] += 1
    budgets = proto["gross_budgets"]
    labels = {
        "random": "Random",
        "stock_knob_xgb": "stock-knob XGB",
        "rules_only": "rules-only",
        "paper_minimum_access": "paper minimum-access mode only",
        "shared_memory_lexicographic": "shared-memory lexicographic",
        "shared_memory_service_proxy": "shared-memory service proxy",
        "validity_v": "validity V",
        "v_plus_delta_t": "V + ΔT bytes/calls",
        "v_plus_delta_t_request": "V + ΔT +request",
        "v_plus_delta_t_command": "V + ΔT +command",
    }
    header = "| policy | " + " | ".join(f"regret@{budget} median [IQR] %" for budget in budgets) + " |"
    separator = "|---|" + "---:|" * len(budgets)
    rows = []
    success_rows = []
    pool_success_rows = []
    cost_rows = []

    def fmt_quantile(metric, scale=1.0, digits=3):
        if metric["median"] is None:
            return "n/a"
        return f"{metric['median'] / scale:.{digits}f}"

    for policy in POLICIES:
        values = summary[policy]
        cells = []
        for budget in budgets:
            metric = values["budgets"][str(budget)]["regret_percent_defined"]
            if metric["median"] is None:
                cells.append("n/a")
            else:
                cells.append(f"{metric['median']:.6f} [{metric['iqr']:.6f}]")
        rows.append(f"| {labels[policy]} | " + " | ".join(cells) + " |")
        final = values["budgets"][str(budgets[-1])]
        success_rows.append(
            f"| {labels[policy]} | {final['success_at_2_percent_rate']:.3f} | "
            f"{final['success_at_5_percent_rate']:.3f} | "
            f"{values['trials_to_2_percent']['success_rate']:.3f} | "
            f"{values['trials_to_5_percent']['success_rate']:.3f} |"
        )
        pool_success_rows.append(
            f"| {labels[policy]} | {final['pool_success_at_2_percent_rate']:.3f} | "
            f"{final['pool_success_at_5_percent_rate']:.3f} | "
            f"{values['trials_to_pool_2_percent']['failure_censored_at_pool_size_plus_one']['median']:.3f} | "
            f"{values['trials_to_pool_5_percent']['failure_censored_at_pool_size_plus_one']['median']:.3f} |"
        )
        resources = final["resource_cost_known_totals"]
        if final["resource_cost_complete_rate"] == 0.0:
            cost_rows.append(
                f"| {labels[policy]} | 0.000 | n/a | n/a | n/a | n/a | n/a |"
            )
        else:
            cost_rows.append(
                f"| {labels[policy]} | {final['resource_cost_complete_rate']:.3f} | "
                f"{fmt_quantile(resources['logical_vta_load_bytes'], 1024 * 1024)} | "
                f"{fmt_quantile(resources['logical_vta_store_bytes'], 1024 * 1024)} | "
                f"{fmt_quantile(resources['logical_vta_dma_calls'], digits=1)} | "
                f"{fmt_quantile(resources['fpga_kernel_invocations'], digits=1)} | "
                f"{fmt_quantile(final['wall_clock_known_total_ms'])} |"
            )
    claim_status = str(pool.get("claim_status", ""))
    prospective = claim_status.startswith("prospective_") and claim_status.endswith(
        "_full_pool_labels_complete"
    )
    recovery = claim_status.startswith("recovery_") and claim_status.endswith(
        "_latency_confirmation_labels_complete"
    )
    workload_label = ",".join(sorted(pool["workloads"]))
    title = (
        "# Prospective {} equal-budget search confirmation".format(workload_label)
        if prospective else
        "# Latency-unseen {} recovery confirmation".format(workload_label)
        if recovery else "# Equal-budget P7Q development replay"
    )
    boundary = (
        "> Prospective board labels collected only after the candidate pool and search rules were frozen."
        if prospective else
        "> Candidate latency and service-proxy weights were frozen before this recovery run; historical correctness was already exposed."
        if recovery else
        "> Development-only replay on previously exposed P7Q labels; not prospective confirmation."
    )
    limitations = [
        "- Logical VTA LOAD/STORE bytes and calls come from runtime profiles; they are not physical AXI burst counts.",
        "- TopHub is a sealed post-order metric reference only. It is absent from every dispatch and training label.",
        "- Workload×seed median/IQR and per-workload 20-seed summaries are stored in `summary.json`.",
        "- Full dispatch histories and revealed phase outcomes are stored in `runs.jsonl`.",
    ]
    if not prospective and not recovery:
        limitations = [
            "- P7Q contains 75 measured candidates and only one FPGA-invalid candidate; it has no lower- or "
            "compile-invalid examples for a balanced validity test.",
            "- P7Q phase wall-clock and resource costs are unavailable. Known-sum zero must not be interpreted "
            "as zero tuning cost.",
        ] + limitations
    return "\n".join([
        title, "", boundary, "",
        "## Pool and protocol", "",
        f"- Workloads: {len(pool['workloads'])}",
        f"- Candidates excluding sealed references: {len(candidates)}",
        f"- Measured-ok: {outcome_counts['measured_ok']}",
        f"- Lower-invalid: {outcome_counts['lower_invalid']}",
        f"- Compile-invalid: {outcome_counts['compile_invalid']}",
        f"- FPGA-invalid: {outcome_counts['fpga_invalid']}",
        f"- Policies: {len(POLICIES)} (including paper-mode and cross-tile shared-memory ablations)",
        f"- Seeds: {proto['seeds']}; gross budgets: {budgets}",
        f"- Total workload-policy-seed runs: {len(pool['workloads']) * len(POLICIES) * proto['seeds']}", "",
        "## Simple regret versus the non-reference full-pool oracle", "",
        header, separator, *rows, "",
        f"## Pool-oracle quality at budget {budgets[-1]}", "",
        "| policy | success@2% | success@5% | median trials-to-2% | median trials-to-5% |",
        "|---|---:|---:|---:|---:|", *pool_success_rows, "",
        f"## Sealed-reference success at budget {budgets[-1]}", "",
        "| policy | success@2% | success@5% | eventual trials-to-2% success | eventual trials-to-5% success |",
        "|---|---:|---:|---:|---:|", *success_rows, "",
        "No P7Q non-reference candidate enters either sealed TopHub equivalence band. This is an exposed-data "
        "property, not a prospective search conclusion." if not prospective and not recovery else
        "The sealed reference is the completed FPGA-correct {} pool oracle; TopHub is not used.".format(
            workload_label
        ), "",
        f"## Search resource cost at budget {budgets[-1]}", "",
        "| policy | complete rate | logical LOAD MiB | logical STORE MiB | DMA calls | FPGA invocations | wall ms |",
        "|---|---:|---:|---:|---:|---:|---:|", *cost_rows, "",
        "## Limitations", "",
        *limitations, "",
    ])


def write_offline(output_dir: Path, pool, runs, summary, proto, command):
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable output: {output_dir}")
    output_dir.mkdir(parents=True)
    claim_status = str(pool.get("claim_status", ""))
    result_status = (
        "latency_unseen_recovery_posthoc_equal_budget_replay"
        if claim_status.startswith("recovery_") else
        "development_only_not_prospective_confirmation"
    )
    files = {
        "pool_adapter_snapshot.json": json.dumps(pool, indent=2, sort_keys=True) + "\n",
        "protocol.json": json.dumps(proto, indent=2, sort_keys=True) + "\n",
        "summary.json": json.dumps({
            "schema": RESULT_SCHEMA,
            "status": result_status,
            "pool_id": pool["pool_id"],
            "run_count": len(runs),
            "summary": summary,
        }, indent=2, sort_keys=True) + "\n",
        "runs.jsonl": "".join(json.dumps(run, sort_keys=True) + "\n" for run in runs),
        "RESULTS.md": render_results(pool, summary, proto),
        "command.txt": command.rstrip() + "\n",
    }
    for name, value in files.items():
        (output_dir / name).write_text(value, encoding="utf-8")
    hashes = {name: sha256(output_dir / name) for name in sorted(files)}
    (output_dir / "artifact_hashes.json").write_text(
        json.dumps({"schema": "c3_immutable_artifact_hashes_v1", "files": hashes},
                   indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def prospective_dispatch(pool, workload_id, policy, seed, history, training_pools):
    validate_pool(pool, require_oracle=False)
    if workload_id not in pool["workloads"]:
        raise ValueError("unknown target workload")
    workload = pool["workloads"][workload_id]
    reference_id = workload["sealed_reference"].get("reference_id")
    if reference_id and any(item.get("candidate_id") == reference_id for item in history):
        raise ValueError("sealed reference appeared in prospective history")
    prior = held_out_training_records(training_pools, workload_id)
    chosen, diagnostics = choose_next(
        policy, workload["candidates"], history, prior, seed, pool["delta_feature_sets"]
    )
    return {
        "schema": DISPATCH_SCHEMA,
        "mode": "prospective_next_only",
        "pool_id": pool["pool_id"],
        "workload_id": workload_id,
        "policy": policy,
        "seed": seed,
        "history_length": len(history),
        "gross_dispatch": len(history) + 1 if chosen else None,
        "candidate_id": chosen["candidate_id"] if chosen else None,
        "selection_diagnostics": diagnostics,
        "oracle_fields_exposed_to_selector": False,
        "sealed_reference_used_by_selector": False,
        "status": "dispatch_ready" if chosen else "pool_exhausted",
    }


def parse_budgets(value):
    budgets = tuple(sorted(set(int(item) for item in value.split(","))))
    if not budgets or budgets[0] <= 0:
        raise ValueError("budgets must be positive")
    return budgets


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    offline = sub.add_parser("offline-replay")
    source = offline.add_mutually_exclusive_group(required=True)
    source.add_argument("--pool", type=Path)
    source.add_argument("--p7q-contract", type=Path)
    offline.add_argument("--p7q-timing-summary", type=Path, action="append", default=[])
    offline.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    offline.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    offline.add_argument("--output-dir", type=Path, required=True)

    prospective = sub.add_parser("prospective")
    prospective.add_argument("--pool", type=Path, required=True)
    prospective.add_argument("--workload-id", required=True)
    prospective.add_argument("--policy", choices=POLICIES, required=True)
    prospective.add_argument("--seed", type=int, required=True)
    prospective.add_argument("--history", type=Path)
    prospective.add_argument("--training-pool", type=Path, action="append", default=[])
    prospective.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    command = " ".join([Path(sys.executable).name, Path(__file__).name] + sys.argv[1:])
    if args.mode == "offline-replay":
        if args.p7q_contract:
            if not args.p7q_timing_summary:
                raise ValueError("P7Q adapter requires timing summaries")
            pool = adapt_p7q(args.p7q_contract, args.p7q_timing_summary)
        else:
            if args.p7q_timing_summary:
                raise ValueError("P7Q summaries require --p7q-contract")
            pool = load_json(args.pool)
        validate_pool(pool, require_oracle=True)
        budgets = parse_budgets(args.budgets)
        if args.seeds != DEFAULT_SEEDS:
            raise ValueError(f"comparison protocol requires exactly {DEFAULT_SEEDS} seeds")
        runs, summary = run_offline(pool, budgets, args.seeds)
        write_offline(args.output_dir, pool, runs, summary,
                      protocol(budgets, args.seeds), command)
        replay_status = (
            "latency_unseen_recovery_posthoc"
            if str(pool.get("claim_status", "")).startswith("recovery_") else "development_only"
        )
        print(json.dumps({"status": replay_status, "runs": len(runs),
                          "output_dir": str(args.output_dir.resolve())}, sort_keys=True))
        return

    pool = load_json(args.pool)
    history = [] if args.history is None else load_json(args.history)
    if not isinstance(history, list):
        raise ValueError("history must be a JSON array")
    training_pools = [load_json(path) for path in args.training_pool]
    result = prospective_dispatch(
        pool, args.workload_id, args.policy, args.seed, history, training_pools
    )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite immutable dispatch: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
