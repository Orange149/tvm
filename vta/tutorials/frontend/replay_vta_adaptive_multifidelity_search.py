#!/usr/bin/env python3
"""Survival-adaptive shared-memory multi-fidelity VTA search.

The policy probes all three modes of the first hardware-diverse same-tile
family.  If at most one lowers successfully, it switches from the fixed 4->2
frontier to immediate family-wave promotion.  Otherwise it retains the frozen
4->2 policy.  The threshold is motivated by the Y05 sparse-space failure of the
fixed frontier and must be frozen before any next holdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import build_vta_multifidelity_pool as pool_builder
import replay_vta_family_wave_search as family_wave
import replay_vta_multifidelity_search as base
from run_vta_p7r119_yolo_barrier_pilot import sha256_file, write_json


SCHEMA = "c3_vta_adaptive_multifidelity_replay_v1"
POLICY = "adaptive_survival_frontier_service"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def execute(states, candidates, state, phase, actions, diagnostic):
    actions.append(base.execute_action(
        state, candidates[state["candidate"]["candidate_id"]], phase,
        len(actions) + 1, diagnostic,
    ))
    return actions[-1]["outcome"]["status"]


def finish_family_wave(states, candidates, family_order, actions, models, seed, workload_id):
    while True:
        progress = False
        for family_id in family_order:
            unseen = [state for state in states.values()
                      if state["candidate"]["family_id"] == family_id and
                      not state["terminated"] and state["next_phase"] == "lower"]
            for state in sorted(unseen, key=lambda item: base.stable_hash(
                    seed, workload_id, item["candidate"]["candidate_id"],
                    "adaptive-family-lower")):
                execute(states, candidates, state, "lower", actions, {
                    "selector": "sparse_space_complete_same_tile_family",
                    "family_id": family_id,
                })
                progress = True
            measured = False
            while not measured:
                ready = family_wave.ranked_ready(
                    states, family_id, "fsim", "family_wave_service", models, seed
                )
                if not ready:
                    break
                state = ready[0]
                if execute(states, candidates, state, "fsim", actions, {
                        "selector": "sparse_space_family_min_service",
                        "service_score_byte_equivalent": base.service_score(state),
                }) != "ok":
                    progress = True
                    continue
                progress = True
                if execute(states, candidates, state, "compile", actions,
                           {"selector": "sparse_space_immediate_promotion"}) != "ok":
                    continue
                if execute(states, candidates, state, "fpga", actions,
                           {"selector": "sparse_space_immediate_promotion"}) != "ok":
                    continue
                if execute(states, candidates, state, "measure", actions,
                           {"selector": "immediate_after_fpga_correct"}) == "ok":
                    measured = True
        if not progress:
            return


def replay(pool, workload_id, seed, models=None):
    workload = pool["workloads"][workload_id]
    candidates = {row["candidate_id"]: row for row in workload["candidates"]}
    states = {cid: base.state_for(row) for cid, row in candidates.items()}
    models = models or base.heldout_phase_models(pool, workload_id)
    family_order = family_wave.hardware_family_order(
        pool, workload_id, seed, False, models
    )
    first_family = family_order[0]
    actions = []
    members = [state for state in states.values()
               if state["candidate"]["family_id"] == first_family]
    for state in sorted(members, key=lambda item: base.stable_hash(
            seed, workload_id, item["candidate"]["candidate_id"],
            "adaptive-first-family-lower")):
        execute(states, candidates, state, "lower", actions, {
            "selector": "three_mode_survival_probe", "family_id": first_family
        })
    first_passes = sum(action["outcome"]["status"] == "ok" for action in actions)
    if first_passes <= 1:
        path = "sparse_family_wave"
        finish_family_wave(
            states, candidates, family_order, actions, models, seed, workload_id
        )
    else:
        path = "dense_fixed_4_to_2"
        while True:
            state, phase, diagnostic = base.next_action_frontier(
                pool, workload_id, states, "hardware_diverse_frontier_service", seed,
                frontier_width=4, promotion_width=2, phase_models=models,
                validity_exponent=0.25,
            )
            if state is None:
                break
            execute(states, candidates, state, phase, actions, diagnostic)
    return {
        "workload_id": workload_id, "policy": POLICY, "seed": seed,
        "first_family": first_family, "first_family_lower_passes": first_passes,
        "adaptive_path": path, "oracle_fields_exposed_before_action": False,
        "actions": actions,
    }


def run(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    training_path = Path(args.training_pool).resolve()
    training = pool_builder.validate(read_json(training_path))
    additions = [pool_builder.validate(read_json(Path(path).resolve()))
                 for path in args.addition_pool]
    pool = family_wave.merge_pools(training, additions)
    runs, metrics = [], []
    for workload_id in pool["workloads"]:
        models = base.heldout_phase_models(pool, workload_id)
        oracle = float(pool["workloads"][workload_id]["pool_oracle_latency_ms"])
        for seed in range(args.seeds):
            item = replay(pool, workload_id, seed, models)
            runs.append(item)
            metrics.append({
                "workload_id": workload_id, "policy": POLICY, "seed": seed,
                "adaptive_path": item["adaptive_path"],
                **base.prefix_to_target(item, oracle, 2.0),
            })
    summary = {
        "runs": len(metrics),
        "success_rate": sum(row["target_reached"] for row in metrics) / len(metrics),
        "path_counts": {
            path: sum(row["adaptive_path"] == path for row in metrics)
            for path in ("sparse_family_wave", "dense_fixed_4_to_2")
        },
        "wall_ms_to_target": base.distribution(row["known_wall_ms"] for row in metrics),
        "gross_candidates_to_target": base.distribution(
            row["gross_candidates_considered"] for row in metrics
        ),
        "phase_actions_to_target": {
            phase: base.distribution(row["phase_action_counts"][phase] for row in metrics)
            for phase in base.PHASES
        },
        "resources_to_target": {
            key: base.distribution(row["resource_cost"][key] for row in metrics)
            for key in base.RESOURCE_KEYS
        },
    }
    result = {
        "schema": SCHEMA,
        "claim_status": "development_only_y05_trigger_and_all_labels_exposed",
        "policy": {
            "name": POLICY, "first_family_modes": 3,
            "sparse_threshold_lower_passes": 1,
            "dense_path": "hardware-diverse 4-to-2 service frontier",
            "sparse_path": "immediate family-wave service promotion",
        },
        "summary": summary, "metrics": metrics, "runs": runs,
    }
    output.mkdir(parents=True)
    write_json(output / "adaptive_replay.json", result)
    write_json(output / "artifact_hashes.json", {
        "artifacts": {"adaptive_replay.json": sha256_file(output / "adaptive_replay.json")},
        "source_sha256": {str(Path(__file__).resolve()): sha256_file(__file__),
                          str(Path(base.__file__).resolve()): sha256_file(base.__file__),
                          str(Path(family_wave.__file__).resolve()): sha256_file(family_wave.__file__)},
    })
    print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-pool", required=True)
    parser.add_argument("--addition-pool", action="append", default=[])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
