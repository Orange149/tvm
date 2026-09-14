#!/usr/bin/env python3
"""Leakage-safe action-level replay for VTA multi-fidelity search.

Unlike candidate-level replay, one dispatch here is exactly one action:
``lower -> fsim -> compile -> fpga -> measure``.  Exact DMA features do not
exist for the selector until the candidate's lowering action has completed;
latency does not exist until measurement.  This permits a fair comparison of
the old "qualify the whole pool first" workflow with bounded-frontier online
qualification.

The validity model follows the ML2Tuner separation of concerns: a per-phase
binary model predicts whether an action will pass, while a deterministic shared
memory service proxy ranks already-lowered candidates for performance.  The
development replay freezes each model leave-target-workload-out; a future live
executor may update it only from append-only target events.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import build_vta_multifidelity_pool as pool_schema


SCHEMA = "c3_vta_multifidelity_replay_v1"
PHASES = pool_schema.PHASES
RESOURCE_KEYS = pool_schema.RESOURCE_KEYS
REQUEST_EQUIVALENT_BYTES = 64 * 1024
EXTRA_SUBMISSION_EQUIVALENT_BYTES = 128 * 1024
POLICIES = (
    "exhaustive_then_service",
    "random_frontier_service",
    "validity_frontier_service",
    "hardware_diverse_frontier_service",
    "hardware_diverse_validity_frontier_service",
)


def uses_validity(policy):
    return policy in ("validity_frontier_service",
                      "hardware_diverse_validity_frontier_service")


def stable_hash(seed, workload_id, candidate_id, namespace):
    return hashlib.sha256(
        f"{seed}:{workload_id}:{candidate_id}:{namespace}".encode()
    ).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def candidate_public(candidate):
    return {
        "candidate_id": candidate["candidate_id"],
        "workload_id": candidate["workload_id"],
        "family_id": candidate["family_id"],
        "residence_mode": candidate["residence_mode"],
        "features": dict(candidate["prelower"]["features"]),
        "geometry_signature_sha256": candidate["prelower"]["geometry_signature_sha256"],
        "hardware_fingerprint_sha256": candidate["prelower"]["hardware_fingerprint_sha256"],
        "schedule_version": candidate["prelower"]["schedule_version"],
    }


def state_for(candidate):
    return {
        "candidate": candidate_public(candidate),
        "next_phase": "lower",
        "revealed": {},
        "terminated": False,
        "measured_latency_ms": None,
    }


def visible_features(state, phase):
    result = dict(state["candidate"]["features"])
    if phase in ("fsim", "compile", "fpga", "measure"):
        result.update({"lower_" + key: value for key, value in
                       state["revealed"].get("lower", {}).get("features", {}).items()})
    if phase in ("compile", "fpga", "measure"):
        result.update({"fsim_" + key: value for key, value in
                       state["revealed"].get("fsim", {}).get("features", {}).items()})
    return result


def training_examples(pool, target_workload_id, phase, observed_states):
    examples = []
    target_geometry = {
        candidate["prelower"]["geometry_signature_sha256"]
        for candidate in pool["workloads"][target_workload_id]["candidates"]
    }
    if len(target_geometry) != 1:
        raise ValueError("one workload_id must map to exactly one geometry")
    for workload_id, workload in pool["workloads"].items():
        workload_geometry = {
            candidate["prelower"]["geometry_signature_sha256"]
            for candidate in workload["candidates"]
        }
        if workload_id == target_workload_id or workload_geometry & target_geometry:
            continue
        for candidate in workload["candidates"]:
            oracle = candidate["oracle"]["phases"]
            if oracle[phase]["status"] == "not_run":
                continue
            synthetic = state_for(candidate)
            for earlier in PHASES[:PHASES.index(phase)]:
                record = oracle[earlier]
                if record["status"] != "ok":
                    break
                if record.get("reveal") is not None:
                    synthetic["revealed"][earlier] = record["reveal"]
            examples.append((visible_features(synthetic, phase),
                             int(oracle[phase]["status"] == "ok")))
    # Target observations become legal labels only after their actions ran.
    for state in observed_states.values():
        observation = state["revealed"].get("_outcome_" + phase)
        if observation in ("ok", "invalid"):
            examples.append((visible_features(state, phase), int(observation == "ok")))
    return examples


def fit_validity(examples):
    if not examples:
        return {"kind": "constant", "probability": 0.5, "training_count": 0}
    keys = sorted({key for values, _ in examples for key in values})
    labels = np.asarray([label for _, label in examples], dtype=np.int64)
    probability = float(np.mean(labels))
    if len(set(labels.tolist())) < 2:
        return {"kind": "constant", "probability": probability,
                "training_count": len(examples), "feature_keys": keys}
    matrix = np.asarray([[float(values.get(key, 0.0)) for key in keys]
                         for values, _ in examples], dtype=np.float64)
    estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000,
                           random_state=0),
    )
    estimator.fit(matrix, labels)
    return {"kind": "logistic", "estimator": estimator, "feature_keys": keys,
            "training_count": len(examples), "positive_rate": probability}


def predict(model, features):
    if model["kind"] == "constant":
        return float(model["probability"])
    row = np.asarray([[float(features.get(key, 0.0)) for key in model["feature_keys"]]],
                     dtype=np.float64)
    return float(model["estimator"].predict_proba(row)[0, 1])


def lower_metric(state, key, default=math.inf):
    values = state["revealed"].get("lower", {}).get("features", {})
    return float(values.get(key, default))


def service_score(state):
    total_bytes = lower_metric(state, "load_dma_bytes") + lower_metric(state, "store_dma_bytes")
    calls = lower_metric(state, "load_dma_calls") + lower_metric(state, "store_dma_calls")
    submissions = lower_metric(state, "submissions", 1.0)
    # Older local rows keep submissions in the FSim command reveal.
    fsim = state["revealed"].get("fsim", {}).get("features", {})
    submissions = float(fsim.get("submissions", submissions))
    return (total_bytes + REQUEST_EQUIVALENT_BYTES * calls
            + EXTRA_SUBMISSION_EQUIVALENT_BYTES * max(submissions - 1.0, 0.0))


def choose_by_service(states, seed, namespace):
    return min(states, key=lambda state: (
        service_score(state),
        stable_hash(seed, state["candidate"]["workload_id"],
                    state["candidate"]["candidate_id"], namespace),
    ))


def choose_phase(states, pool, workload_id, phase, policy, seed, all_states, phase_models,
                 validity_exponent):
    if phase in ("compile", "measure") or not uses_validity(policy):
        chosen = choose_by_service(states, seed, phase + "-service")
        return chosen, {"selector": "shared_memory_service_proxy",
                        "service_score_byte_equivalent": service_score(chosen)}
    model = phase_models[phase]
    scored = []
    for state in states:
        probability = predict(model, visible_features(state, phase))
        score = service_score(state)
        acquisition = score / max(probability, 0.05) ** validity_exponent
        scored.append((acquisition, probability, score, state))
    acquisition, probability, score, chosen = min(scored, key=lambda item: (
        item[0], item[2], stable_hash(seed, workload_id,
                                      item[3]["candidate"]["candidate_id"],
                                      phase + "-validity-service-tie")
    ))
    return chosen, {"selector": phase + "_validity_then_service",
                    "predicted_pass_probability": probability,
                    "service_score_byte_equivalent": score,
                    "validity_adjusted_service_score": acquisition,
                    "training_count": model["training_count"]}


def family_distance_features(state):
    return {key: value for key, value in state["candidate"]["features"].items()
            if key.startswith("knob_") or key.startswith("required_") or
            key.startswith("opportunity_")}


def normalized_family_vectors(states):
    keys = sorted({key for state in states.values() for key in family_distance_features(state)})
    rows = {cid: family_distance_features(state) for cid, state in states.items()}
    ranges = {}
    for key in keys:
        values = [float(row.get(key, 0.0)) for row in rows.values()]
        ranges[key] = (min(values), max(values))
    vectors = {}
    for cid, row in rows.items():
        vectors[cid] = np.asarray([
            0.0 if ranges[key][1] == ranges[key][0] else
            (float(row.get(key, 0.0)) - ranges[key][0]) / (ranges[key][1] - ranges[key][0])
            for key in keys
        ], dtype=np.float64)
    return vectors


def choose_hardware_diverse_unseen(unseen, states, pool, workload_id, policy, seed,
                                   phase_models):
    started_families = {
        state["candidate"]["family_id"] for state in states.values()
        if "_outcome_lower" in state["revealed"]
    }
    partial = [state for state in unseen
               if state["candidate"]["family_id"] in started_families]
    model = phase_models["lower"]
    if partial:
        eligible = partial
        family_score = None
        selection = "complete_started_same_tile_family"
    else:
        by_family = {}
        for state in unseen:
            by_family.setdefault(state["candidate"]["family_id"], []).append(state)
        vectors = normalized_family_vectors(states)
        explored = [state for state in states.values()
                    if "_outcome_lower" in state["revealed"]]
        family_scores = {}
        for family_id, members in by_family.items():
            vector = vectors[members[0]["candidate"]["candidate_id"]]
            if explored:
                diversity = min(float(np.linalg.norm(
                    vector - vectors[item["candidate"]["candidate_id"]]
                )) for item in explored)
            else:
                diversity = float(np.linalg.norm(vector - 0.5))
            probability = statistics.mean(
                predict(model, visible_features(item, "lower")) for item in members
            ) if uses_validity(policy) else 1.0
            family_scores[family_id] = diversity * math.sqrt(max(probability, 0.05))
        selected_family, family_score = min(family_scores.items(), key=lambda item: (
            -item[1], stable_hash(seed, workload_id, item[0], "hardware-family-tie")
        ))
        eligible = by_family[selected_family]
        selection = "new_maximin_hardware_family"
    scored = [(predict(model, visible_features(state, "lower")), state) for state in eligible]
    probability, chosen = min(scored, key=lambda pair: (
        -pair[0] if uses_validity(policy) else 0.0,
        stable_hash(seed, workload_id, pair[1]["candidate"]["candidate_id"],
                    "hardware-mode-tie")
    ))
    return chosen, {
        "selector": selection,
        "family_id": chosen["candidate"]["family_id"],
        "hardware_diversity_validity_score": family_score,
        "predicted_lower_pass_probability": probability if uses_validity(policy) else None,
        "validity_training_count": model["training_count"] if uses_validity(policy) else 0,
    }


def next_action_frontier(pool, workload_id, states, policy, seed, frontier_width,
                         promotion_width, phase_models, validity_exponent):
    available = {phase: [state for state in states.values()
                         if not state["terminated"] and state["next_phase"] == phase]
                 for phase in PHASES}
    # Once hardware correctness is known, measure immediately.  Before the
    # FPGA gate, compile a small promotion wave so V_fpga and the full service
    # proxy (including FSim-revealed submissions) can compare peers.
    if available["measure"]:
        chosen, diagnostic = choose_phase(
            available["measure"], pool, workload_id, "measure", policy, seed, states,
            phase_models, validity_exponent
        )
        return chosen, "measure", diagnostic
    unseen = available["lower"]
    lower_ready = available["fsim"]
    compile_ready = available["compile"]
    fpga_ready = available["fpga"]
    if fpga_ready and (len(fpga_ready) >= promotion_width or not compile_ready):
        chosen, diagnostic = choose_phase(
            fpga_ready, pool, workload_id, "fpga", policy, seed, states, phase_models,
            validity_exponent
        )
        return chosen, "fpga", diagnostic
    if fpga_ready and compile_ready:
        chosen, diagnostic = choose_phase(
            compile_ready, pool, workload_id, "compile", policy, seed, states, phase_models,
            validity_exponent
        )
        return chosen, "compile", diagnostic
    if compile_ready and (
        len(compile_ready) >= promotion_width or (not unseen and not lower_ready)
    ):
        chosen, diagnostic = choose_phase(
            compile_ready, pool, workload_id, "compile", policy, seed, states, phase_models,
            validity_exponent
        )
        return chosen, "compile", diagnostic
    if lower_ready and (len(lower_ready) >= frontier_width or not unseen):
        chosen, diagnostic = choose_phase(
            lower_ready, pool, workload_id, "fsim", policy, seed, states, phase_models,
            validity_exponent
        )
        return chosen, "fsim", diagnostic
    if unseen:
        if policy.startswith("hardware_diverse_"):
            chosen, diagnostic = choose_hardware_diverse_unseen(
                unseen, states, pool, workload_id, policy, seed, phase_models
            )
            return chosen, "lower", diagnostic
        # The development policy uses a frozen leave-target-workload-out Model V.
        if uses_validity(policy):
            model = phase_models["lower"]
            scored = [(predict(model, visible_features(state, "lower")), state)
                      for state in unseen]
            probability, chosen = min(scored, key=lambda pair: (
                -pair[0], stable_hash(seed, workload_id,
                                      pair[1]["candidate"]["candidate_id"],
                                      "lower-validity-tie")
            ))
            return chosen, "lower", {
                "selector": "leave_target_out_plus_observed_lower_validity",
                "predicted_pass_probability": probability,
                "training_count": model["training_count"],
            }
        chosen = min(unseen, key=lambda state: stable_hash(
            seed, workload_id, state["candidate"]["candidate_id"], "unseen-random"
        ))
        return chosen, "lower", {"selector": "seeded_random"}
    if lower_ready:
        chosen, diagnostic = choose_phase(
            lower_ready, pool, workload_id, "fsim", policy, seed, states, phase_models,
            validity_exponent
        )
        return chosen, "fsim", diagnostic
    if compile_ready:
        chosen, diagnostic = choose_phase(
            compile_ready, pool, workload_id, "compile", policy, seed, states, phase_models,
            validity_exponent
        )
        return chosen, "compile", diagnostic
    return None, None, None


def next_action_exhaustive(pool, workload_id, states, seed):
    for phase in ("lower", "fsim"):
        available = [state for state in states.values()
                     if not state["terminated"] and state["next_phase"] == phase]
        if available:
            chosen = min(available, key=lambda state: stable_hash(
                seed, workload_id, state["candidate"]["candidate_id"],
                "exhaustive-" + phase
            ))
            return chosen, phase, {"selector": "qualify_entire_pool_" + phase}
    for phase in ("measure", "fpga", "compile"):
        available = [state for state in states.values()
                     if not state["terminated"] and state["next_phase"] == phase]
        if available:
            chosen = choose_by_service(available, seed, "exhaustive-" + phase)
            return chosen, phase, {"selector": "postqualification_service_proxy",
                                   "service_score_byte_equivalent": service_score(chosen)}
    return None, None, None


def reveal_action(candidate, phase):
    # This is the sole oracle boundary.  Selection has already returned.
    return json.loads(json.dumps(candidate["oracle"]["phases"][phase]))


def execute_action(state, candidate, phase, action_index, diagnostic):
    if state["next_phase"] != phase or state["terminated"]:
        raise ValueError("illegal state transition")
    record = reveal_action(candidate, phase)
    state["revealed"]["_outcome_" + phase] = record["status"]
    if record.get("reveal") is not None:
        state["revealed"][phase] = record["reveal"]
    if record["status"] == "ok":
        index = PHASES.index(phase)
        if phase == "measure":
            state["terminated"] = True
            state["next_phase"] = None
            state["measured_latency_ms"] = float(candidate["oracle"]["latency_ms"])
        else:
            state["next_phase"] = PHASES[index + 1]
    else:
        state["terminated"] = True
        state["next_phase"] = None
    return {
        "action_index": action_index,
        "candidate_id": candidate["candidate_id"],
        "phase": phase,
        "selection_diagnostics": diagnostic,
        "outcome": {
            "status": record["status"],
            "wall_ms": record.get("wall_ms"),
            "resource_cost": record.get("resource_cost"),
            # Keep newly revealed features in the audit log, but never pass the
            # full candidate oracle into a selector.
            "reveal": record.get("reveal"),
            "latency_ms": state["measured_latency_ms"] if phase == "measure" else None,
        },
    }


def heldout_phase_models(pool, workload_id):
    return {
        phase: fit_validity(training_examples(pool, workload_id, phase, {}))
        for phase in ("lower", "fsim", "fpga")
    }


def replay(pool, workload_id, policy, seed, frontier_width, promotion_width=2,
           phase_models=None, validity_exponent=0.25):
    workload = pool["workloads"][workload_id]
    candidates = {row["candidate_id"]: row for row in workload["candidates"]}
    states = {cid: state_for(row) for cid, row in candidates.items()}
    phase_models = phase_models or heldout_phase_models(pool, workload_id)
    actions = []
    while True:
        if policy == "exhaustive_then_service":
            state, phase, diagnostic = next_action_exhaustive(pool, workload_id, states, seed)
        else:
            state, phase, diagnostic = next_action_frontier(
                pool, workload_id, states, policy, seed, frontier_width, promotion_width,
                phase_models, validity_exponent
            )
        if state is None:
            break
        cid = state["candidate"]["candidate_id"]
        actions.append(execute_action(
            state, candidates[cid], phase, len(actions) + 1, diagnostic
        ))
    return {
        "workload_id": workload_id,
        "policy": policy,
        "seed": seed,
        "frontier_width": None if policy == "exhaustive_then_service" else frontier_width,
        "promotion_width": None if policy == "exhaustive_then_service" else promotion_width,
        "oracle_fields_exposed_before_action": False,
        "actions": actions,
    }


def prefix_to_target(run, oracle_latency, tolerance_percent=2.0):
    threshold = oracle_latency * (1.0 + tolerance_percent / 100.0)
    stop = len(run["actions"])
    found = False
    for index, action in enumerate(run["actions"]):
        latency = action["outcome"].get("latency_ms")
        if latency is not None and latency <= threshold:
            stop = index + 1
            found = True
            break
    actions = run["actions"][:stop]
    walls = [action["outcome"].get("wall_ms") for action in actions]
    resources = {key: 0.0 for key in RESOURCE_KEYS}
    resource_complete = True
    for action in actions:
        values = action["outcome"].get("resource_cost")
        if values is None:
            resource_complete = False
            continue
        for key in RESOURCE_KEYS:
            resources[key] += float(values[key])
    phase_counts = Counter(action["phase"] for action in actions)
    phase_invalid = Counter(action["phase"] for action in actions
                            if action["outcome"]["status"] == "invalid")
    return {
        "target_tolerance_percent": tolerance_percent,
        "target_reached": found,
        "actions_to_target": stop if found else None,
        "actions_executed_until_stop": len(actions),
        "gross_candidates_considered": len({action["candidate_id"] for action in actions}),
        "phase_action_counts": {phase: phase_counts[phase] for phase in PHASES},
        "phase_invalid_counts": {phase: phase_invalid[phase] for phase in PHASES},
        "known_wall_ms": None if any(value is None for value in walls) else sum(walls),
        "resource_cost_complete": resource_complete,
        "resource_cost": resources,
    }


def summarize(runs, pool, tolerance_percent):
    rows = []
    for run in runs:
        oracle = float(pool["workloads"][run["workload_id"]]["pool_oracle_latency_ms"])
        metric = prefix_to_target(run, oracle, tolerance_percent)
        rows.append({**{key: run[key] for key in ("workload_id", "policy", "seed",
                                                  "frontier_width", "promotion_width")},
                     **metric})
    grouped = {}
    for policy in POLICIES:
        selected = [row for row in rows if row["policy"] == policy]
        if not selected:
            continue
        reached = [row for row in selected if row["target_reached"]]
        grouped[policy] = {
            "runs": len(selected),
            "target_reached_rate": len(reached) / len(selected),
            "median_actions_to_target": (statistics.median(
                row["actions_to_target"] for row in reached) if reached else None),
            "median_gross_candidates_to_target": (statistics.median(
                row["gross_candidates_considered"] for row in reached) if reached else None),
            "median_known_wall_ms_to_target": (statistics.median(
                row["known_wall_ms"] for row in reached if row["known_wall_ms"] is not None)
                if any(row["known_wall_ms"] is not None for row in reached) else None),
            "median_phase_actions_to_target": {
                phase: (statistics.median(row["phase_action_counts"][phase] for row in reached)
                        if reached else None) for phase in PHASES
            },
            "median_resources_to_target": {
                key: (statistics.median(row["resource_cost"][key] for row in reached
                                        if row["resource_cost_complete"])
                      if any(row["resource_cost_complete"] for row in reached) else None)
                for key in RESOURCE_KEYS
            },
        }
    return rows, grouped


def distribution(values):
    values = [float(value) for value in values]
    if not values:
        return None
    return {
        "min": min(values),
        "q1": float(np.percentile(values, 25)),
        "median": statistics.median(values),
        "q3": float(np.percentile(values, 75)),
        "max": max(values),
    }


def structured_summary(metrics):
    by_workload = {}
    for workload_id in sorted({row["workload_id"] for row in metrics}):
        by_workload[workload_id] = {}
        for policy in POLICIES:
            selected = [row for row in metrics
                        if row["workload_id"] == workload_id and row["policy"] == policy]
            by_workload[workload_id][policy] = {
                "runs": len(selected),
                "target_reached_rate": (sum(row["target_reached"] for row in selected) /
                                        len(selected)),
                "known_wall_ms_to_target": distribution(
                    row["known_wall_ms"] for row in selected if row["known_wall_ms"] is not None
                ),
                "gross_candidates_to_target": distribution(
                    row["gross_candidates_considered"] for row in selected
                ),
                "phase_actions_to_target": {
                    phase: distribution(row["phase_action_counts"][phase] for row in selected)
                    for phase in PHASES
                },
                "resources_to_target": {
                    key: distribution(row["resource_cost"][key] for row in selected
                                      if row["resource_cost_complete"])
                    for key in RESOURCE_KEYS
                },
            }
    aggregate = {}
    for policy in POLICIES:
        by_seed = []
        seeds = sorted({row["seed"] for row in metrics if row["policy"] == policy})
        for seed in seeds:
            selected = [row for row in metrics if row["policy"] == policy and row["seed"] == seed]
            by_seed.append({
                "all_workloads_reached_target": all(row["target_reached"] for row in selected),
                "known_wall_ms": (sum(row["known_wall_ms"] for row in selected)
                                  if all(row["known_wall_ms"] is not None for row in selected)
                                  else None),
                "gross_candidates": sum(row["gross_candidates_considered"] for row in selected),
                "phase_actions": {phase: sum(row["phase_action_counts"][phase]
                                              for row in selected) for phase in PHASES},
                "resources": {key: sum(row["resource_cost"][key] for row in selected)
                              for key in RESOURCE_KEYS},
            })
        aggregate[policy] = {
            "seeds": len(by_seed),
            "all_workloads_target_success_rate": (
                sum(row["all_workloads_reached_target"] for row in by_seed) / len(by_seed)
            ),
            "known_wall_ms_to_all_targets": distribution(
                row["known_wall_ms"] for row in by_seed if row["known_wall_ms"] is not None
            ),
            "gross_candidates_to_all_targets": distribution(
                row["gross_candidates"] for row in by_seed
            ),
            "phase_actions_to_all_targets": {
                phase: distribution(row["phase_actions"][phase] for row in by_seed)
                for phase in PHASES
            },
            "resources_to_all_targets": {
                key: distribution(row["resources"][key] for row in by_seed)
                for key in RESOURCE_KEYS
            },
        }
    return {"by_workload": by_workload, "aggregate_by_seed": aggregate}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--frontier-width", type=int, default=4)
    parser.add_argument("--promotion-width", type=int, default=2)
    parser.add_argument("--tolerance-percent", type=float, default=2.0)
    parser.add_argument("--validity-exponent", type=float, default=0.25)
    args = parser.parse_args()
    if (args.seeds < 1 or args.frontier_width < 1 or args.promotion_width < 1 or
            args.promotion_width > args.frontier_width or args.validity_exponent < 0):
        raise ValueError("invalid seeds/frontier/promotion/validity exponent")
    pool_path = Path(args.pool).resolve()
    pool = pool_schema.validate(read_json(pool_path))
    runs = []
    for workload_id in pool["workloads"]:
        phase_models = heldout_phase_models(pool, workload_id)
        for seed in range(args.seeds):
            for policy in POLICIES:
                runs.append(replay(
                    pool, workload_id, policy, seed, args.frontier_width,
                    args.promotion_width, phase_models, args.validity_exponent
                ))
    metrics, summary = summarize(runs, pool, args.tolerance_percent)
    structured = structured_summary(metrics)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": SCHEMA,
        "pool_binding": {"path": str(pool_path), "sha256": sha256(pool_path)},
        "frontier_width": args.frontier_width,
        "promotion_width": args.promotion_width,
        "validity_exponent": args.validity_exponent,
        "seeds": args.seeds,
        "target_tolerance_percent": args.tolerance_percent,
        "policies": list(POLICIES),
        "leakage_contract": {
            "selector_time_zero": "prelower only",
            "exact_dma_visible_after": "lower action",
            "command_visible_after": "fsim action",
            "latency_visible_after": "measure action",
            "validity_training": "frozen leave-target-workload-out phase-specific models",
        },
        "summary": summary,
        "structured_summary": structured,
        "metrics": metrics,
        "runs": runs,
    }
    result_path = output / "multifidelity_replay.json"
    write_json(result_path, result)
    write_json(output / "artifact_hashes.json", {
        "artifacts": {"multifidelity_replay.json": sha256(result_path)},
        "source_sha256": {
            str(Path(__file__).resolve()): sha256(__file__),
            str(Path(pool_schema.__file__).resolve()): sha256(pool_schema.__file__),
        },
    })
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
