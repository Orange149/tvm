#!/usr/bin/env python3
"""Family-wave multi-fidelity search over completed staged VTA pools.

One wave lowers every residence mode of one same-tile family, then immediately
promotes the cheapest revealed mode through FSim, compile, FPGA correctness and
measurement.  This removes the fixed-frontier requirement that is wasteful
when valid candidates are sparse.  Remaining modes are revisited only after
every family has received one wave, so the replay eventually remains complete.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np

import build_vta_multifidelity_pool as pool_builder
import replay_vta_multifidelity_search as base
from run_vta_p7r119_yolo_barrier_pilot import sha256_file, write_json


SCHEMA = "c3_vta_family_wave_replay_v1"
POLICIES = ("family_wave_service", "family_wave_validity_service")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def merge_pools(training, additions):
    merged = json.loads(json.dumps(training))
    for addition in additions:
        for workload_id, workload in addition["workloads"].items():
            if workload_id in merged["workloads"]:
                raise ValueError("duplicate workload " + workload_id)
            merged["workloads"][workload_id] = workload
    return merged


def hardware_family_order(pool, workload_id, seed, use_validity, models):
    candidates = pool["workloads"][workload_id]["candidates"]
    states = {row["candidate_id"]: base.state_for(row) for row in candidates}
    vectors = base.normalized_family_vectors(states)
    families = {}
    for state in states.values():
        families.setdefault(state["candidate"]["family_id"], []).append(state)
    selected = []
    while len(selected) < len(families):
        scores = {}
        for family_id, members in families.items():
            if family_id in selected:
                continue
            vector = vectors[members[0]["candidate"]["candidate_id"]]
            if selected:
                diversity = min(float(np.linalg.norm(
                    vector - vectors[families[chosen][0]["candidate"]["candidate_id"]]
                )) for chosen in selected)
            else:
                diversity = float(np.linalg.norm(vector - 0.5))
            probability = (statistics.mean(base.predict(
                models["lower"], base.visible_features(member, "lower")
            ) for member in members) if use_validity else 1.0)
            scores[family_id] = diversity * math.sqrt(max(probability, 0.05))
        chosen = min(scores, key=lambda family_id: (
            -scores[family_id],
            base.stable_hash(seed, workload_id, family_id, "family-wave-order"),
        ))
        selected.append(chosen)
    return selected


def execute(states, candidates, state, phase, actions, diagnostic):
    cid = state["candidate"]["candidate_id"]
    actions.append(base.execute_action(
        state, candidates[cid], phase, len(actions) + 1, diagnostic
    ))
    return actions[-1]["outcome"]["status"]


def ranked_ready(states, family_id, phase, policy, models, seed):
    ready = [state for state in states.values()
             if not state["terminated"] and state["next_phase"] == phase and
             state["candidate"]["family_id"] == family_id]
    scored = []
    for state in ready:
        service = base.service_score(state)
        probability = (base.predict(models[phase], base.visible_features(state, phase))
                       if policy == "family_wave_validity_service" and
                       phase in models else 1.0)
        score = service / max(probability, 0.05) ** 0.25
        scored.append((score, service, probability, state))
    return [item[3] for item in sorted(scored, key=lambda item: (
        item[0], item[1], base.stable_hash(
            seed, item[3]["candidate"]["workload_id"],
            item[3]["candidate"]["candidate_id"], phase + "-family-wave-tie"
        )
    ))]


def replay(pool, workload_id, policy, seed, models=None):
    if policy not in POLICIES:
        raise ValueError("unknown family-wave policy")
    workload = pool["workloads"][workload_id]
    candidates = {row["candidate_id"]: row for row in workload["candidates"]}
    states = {cid: base.state_for(row) for cid, row in candidates.items()}
    models = models or base.heldout_phase_models(pool, workload_id)
    family_order = hardware_family_order(
        pool, workload_id, seed, policy == "family_wave_validity_service", models
    )
    actions = []
    # Every pass measures at most one new valid candidate per family.  Invalid
    # modes fail over immediately; untried valid modes remain for later passes.
    while True:
        progress = False
        for family_id in family_order:
            unseen = [state for state in states.values()
                      if state["candidate"]["family_id"] == family_id and
                      not state["terminated"] and state["next_phase"] == "lower"]
            for state in sorted(unseen, key=lambda item: base.stable_hash(
                    seed, workload_id, item["candidate"]["candidate_id"], "family-wave-lower")):
                execute(states, candidates, state, "lower", actions,
                        {"selector": "complete_same_tile_family", "family_id": family_id})
                progress = True
            measured_this_pass = False
            while not measured_this_pass:
                ready = ranked_ready(states, family_id, "fsim", policy, models, seed)
                if not ready:
                    break
                state = ready[0]
                status = execute(states, candidates, state, "fsim", actions, {
                    "selector": "family_min_validity_adjusted_service",
                    "family_id": family_id,
                    "service_score_byte_equivalent": base.service_score(state),
                })
                progress = True
                if status != "ok":
                    continue
                if execute(states, candidates, state, "compile", actions,
                           {"selector": "immediate_family_wave_promotion"}) != "ok":
                    continue
                if execute(states, candidates, state, "fpga", actions,
                           {"selector": "immediate_family_wave_promotion"}) != "ok":
                    continue
                if execute(states, candidates, state, "measure", actions,
                           {"selector": "immediate_after_fpga_correct"}) == "ok":
                    measured_this_pass = True
        if not progress:
            break
    return {
        "workload_id": workload_id, "policy": policy, "seed": seed,
        "family_order": family_order, "oracle_fields_exposed_before_action": False,
        "actions": actions,
    }


def summarize(metrics):
    result = {}
    for policy in POLICIES:
        selected = [row for row in metrics if row["policy"] == policy]
        result[policy] = {
            "runs": len(selected),
            "success_rate": sum(row["target_reached"] for row in selected) / len(selected),
            "wall_ms_to_target": base.distribution(
                row["known_wall_ms"] for row in selected if row["known_wall_ms"] is not None
            ),
            "gross_candidates_to_target": base.distribution(
                row["gross_candidates_considered"] for row in selected
            ),
            "phase_actions_to_target": {
                phase: base.distribution(row["phase_action_counts"][phase] for row in selected)
                for phase in base.PHASES
            },
            "resources_to_target": {
                key: base.distribution(row["resource_cost"][key] for row in selected)
                for key in base.RESOURCE_KEYS
            },
        }
    return result


def run(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    training_path = Path(args.training_pool).resolve()
    training = pool_builder.validate(read_json(training_path))
    additions = [pool_builder.validate(read_json(Path(path).resolve()))
                 for path in args.addition_pool]
    pool = merge_pools(training, additions)
    runs, metrics = [], []
    for workload_id in pool["workloads"]:
        models = base.heldout_phase_models(pool, workload_id)
        oracle = float(pool["workloads"][workload_id]["pool_oracle_latency_ms"])
        for seed in range(args.seeds):
            for policy in POLICIES:
                item = replay(pool, workload_id, policy, seed, models)
                runs.append(item)
                metrics.append({
                    "workload_id": workload_id, "policy": policy, "seed": seed,
                    **base.prefix_to_target(item, oracle, 2.0),
                })
    result = {
        "schema": SCHEMA,
        "claim_status": "development_only_all_included_workload_labels_exposed",
        "training_pool": {"path": str(training_path), "sha256": sha256_file(training_path)},
        "addition_pools": [{"path": str(Path(path).resolve()),
                            "sha256": sha256_file(Path(path).resolve())}
                           for path in args.addition_pool],
        "seeds": args.seeds,
        "summary": summarize(metrics),
        "metrics": metrics,
        "runs": runs,
    }
    output.mkdir(parents=True)
    write_json(output / "family_wave_replay.json", result)
    write_json(output / "artifact_hashes.json", {
        "artifacts": {"family_wave_replay.json": sha256_file(
            output / "family_wave_replay.json"
        )},
        "source_sha256": {str(Path(__file__).resolve()): sha256_file(__file__),
                          str(Path(base.__file__).resolve()): sha256_file(base.__file__)},
    })
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-pool", required=True)
    parser.add_argument("--addition-pool", action="append", default=[])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
