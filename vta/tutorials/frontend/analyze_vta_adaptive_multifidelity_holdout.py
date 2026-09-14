#!/usr/bin/env python3
"""Audit a frozen adaptive multi-fidelity holdout against its completed oracle."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import build_vta_multifidelity_pool as pool_builder
import replay_vta_adaptive_multifidelity_search as adaptive
import replay_vta_family_wave_search as family_wave
import replay_vta_multifidelity_search as base
import run_vta_p7r132_y00_search_confirmation as board_cost
from run_vta_p7r119_yolo_barrier_pilot import sha256_file, write_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def verify(directory):
    directory = Path(directory)
    ledger = read_json(directory / "artifact_hashes.json")
    for name, expected in ledger["artifacts"].items():
        if sha256_file(directory / name) != expected:
            raise ValueError("artifact hash mismatch: " + str(directory / name))
    return sha256_file(directory / "artifact_hashes.json")


def metric(item, oracle):
    return base.prefix_to_target(item, oracle, 2.0)


def summarize(rows):
    result = {}
    for policy in sorted({row["policy"] for row in rows}):
        selected = [row for row in rows if row["policy"] == policy]
        result[policy] = {
            "runs": len(selected),
            "success_rate": sum(row["target_reached"] for row in selected) / len(selected),
            "wall_ms_to_target": base.distribution(row["known_wall_ms"] for row in selected),
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
    paths = {
        "registry": Path(args.registry_dir),
        "live_local_prefix": Path(args.local_prefix_dir),
        "prospective_first_board_wave": Path(args.first_wave_dir),
        "full_board_oracle_completion": Path(args.completion_dir),
        "target_multifidelity_pool": Path(args.target_pool_dir),
        "development_training_pool": Path(args.training_pool_dir),
    }
    addition_paths = [Path(path) for path in args.training_addition_pool_dir]
    bindings = {name: {"path": str(path.resolve()), "ledger_sha256": verify(path)}
                for name, path in paths.items()}
    for index, path in enumerate(addition_paths):
        bindings["development_addition_pool_{}".format(index)] = {
            "path": str(path.resolve()), "ledger_sha256": verify(path)
        }
    registry = read_json(paths["registry"] / "registry_contract.json")
    for source in (adaptive, base):
        source_path = str(Path(source.__file__).resolve())
        if sha256_file(source.__file__) != registry["source_sha256"][source_path]:
            raise ValueError("frozen policy source changed: " + source_path)

    training = pool_builder.validate(read_json(
        paths["development_training_pool"] / "multifidelity_pool.json"
    ))
    additions = [pool_builder.validate(read_json(path / "multifidelity_pool.json"))
                 for path in addition_paths]
    target = pool_builder.validate(read_json(
        paths["target_multifidelity_pool"] / "multifidelity_pool.json"
    ))
    combined = family_wave.merge_pools(training, additions + [target])
    workload_id = registry["workload_id"]
    models = base.heldout_phase_models(combined, workload_id)
    oracle = float(target["workloads"][workload_id]["pool_oracle_latency_ms"])

    runs = []
    for seed in range(args.seeds):
        for policy in base.POLICIES:
            runs.append(base.replay(
                combined, workload_id, policy, seed, 4, 2, models,
                validity_exponent=0.25,
            ))
        runs.append(family_wave.replay(
            combined, workload_id, "family_wave_service", seed, models
        ))
        runs.append(adaptive.replay(combined, workload_id, seed, models))
    metrics = [{"policy": item["policy"], "seed": item["seed"], **metric(item, oracle)}
               for item in runs]

    primary = next(item for item in runs
                   if item["policy"] == adaptive.POLICY and item["seed"] == 0)
    live_history = read_json(paths["live_local_prefix"] / "history.json")["events"]
    live_pairs = [(row["candidate_id"], row["phase"], row["status"])
                  for row in live_history]
    replay_pairs = [(row["candidate_id"], row["phase"], row["outcome"]["status"])
                    for row in primary["actions"][:len(live_history)]]
    if live_pairs != replay_pairs:
        raise ValueError("live local chain diverges from frozen adaptive replay")
    first_fpga = next(row for row in primary["actions"] if row["phase"] == "fpga")
    first_measure = next(row for row in primary["actions"] if row["phase"] == "measure")
    first_id = first_measure["candidate_id"]

    first_wave_summary = read_json(paths["prospective_first_board_wave"] /
                                   "timing_summary.json")
    completion = read_json(paths["full_board_oracle_completion"] / "timing_summary.json")
    first_correctness = read_jsonl(paths["prospective_first_board_wave"] /
                                   "correctness.jsonl")
    first_timing = read_jsonl(paths["prospective_first_board_wave"] / "timing.jsonl")
    certificates = read_json(paths["prospective_first_board_wave"] /
                             "cross_certificates.json")
    selected_correctness = next(row for row in first_correctness
                                if row["candidate_id"] == first_id)
    selected_timing = [row for row in first_timing if row["candidate_id"] == first_id]
    local_static = read_jsonl(paths["live_local_prefix"] / "static_results.jsonl")
    local_fsim = read_jsonl(paths["live_local_prefix"] / "fsim_results.jsonl")
    first_wave_contract = read_json(paths["prospective_first_board_wave"] /
                                    "execution_contract.json")
    promoted_ids = first_wave_contract["candidate_order"]
    canonical_local_wall = 1000.0 * sum(
        float(row["diagnostic_wall_seconds"]) for row in local_static + local_fsim
    )
    observed_local_wall = sum(float(row["wall_ms"]) for row in live_history)
    # The dense 4->2 policy cross-compiles both promoted candidates before its
    # first FPGA action.  Charge both compiles even when the first measurement
    # later turns out to be the oracle.
    compile_wall = sum(float(certificates[cid]["cross_compile_wall_ms"])
                       for cid in promoted_ids)
    correctness_wall = sum(float(row["host_wall_ms"])
                           for row in selected_correctness["seeds"])
    measure_wall = statistics.median(float(row["host_wall_ms"]) for row in selected_timing)
    correctness_resources = board_cost.aggregate_row_resources(selected_correctness["seeds"])
    timing_resources = board_cost.representative_timing_cost(selected_timing)["resource_cost"]
    actual_resources = {key: correctness_resources[key] + timing_resources[key]
                        for key in base.RESOURCE_KEYS}

    target_candidates = target["workloads"][workload_id]["candidates"]
    confirmation = {
        "workload_id": workload_id,
        "registered_candidates": len(target_candidates),
        "complete_outcomes": {
            "lower_invalid": sum(row["oracle"]["phases"]["lower"]["status"] == "invalid"
                                 for row in target_candidates),
            "fsim_invalid": sum(row["oracle"]["phases"]["fsim"]["status"] == "invalid"
                                for row in target_candidates),
            "fpga_invalid": sum(row["oracle"]["phases"]["fpga"]["status"] == "invalid"
                                for row in target_candidates),
            "measured_ok": sum(row["oracle"]["phases"]["measure"]["status"] == "ok"
                               for row in target_candidates),
        },
        "adaptive_path": primary["adaptive_path"],
        "first_family_lower_passes": primary["first_family_lower_passes"],
        "pool_oracle_candidate_id": completion["pool_oracle_candidate_id"],
        "pool_oracle_latency_ms": oracle,
        "prospective_first_fpga_candidate_id": first_fpga["candidate_id"],
        "prospective_first_measured_candidate_id": first_id,
        "first_candidate_is_pool_oracle": first_id == completion["pool_oracle_candidate_id"],
        "first_wave_latency_ms": first_wave_summary["candidate_stats"][first_id]["median_ms"],
        "independent_completion_latency_ms": completion["candidate_stats"][first_id]["median_ms"],
        "live_local_actions_match_frozen_replay": True,
        "live_local_action_count": len(live_history),
        "actual_prospective_time_to_oracle_cost": {
            "observed_executor_wall_ms": (
                observed_local_wall + compile_wall + correctness_wall + measure_wall
            ),
            "canonical_phase_wall_ms": (
                canonical_local_wall + compile_wall + correctness_wall + measure_wall
            ),
            "local_orchestration_overhead_ms": observed_local_wall - canonical_local_wall,
            "phase_actions": {
                              "lower": sum(row["phase"] == "lower" for row in live_history),
                              "fsim": sum(row["phase"] == "fsim" for row in live_history),
                              "compile": len(promoted_ids),
                              "fpga": 1, "measure": 1},
            "resource_cost": actual_resources,
        },
        "first_wave_collection_deviation": (
            None if len(promoted_ids) == 1 else
            "collector certified and timed every promoted candidate; time-to-oracle charges all "
            "policy-required pre-FPGA compiles but only the first candidate's correctness and "
            "representative measurement"
        ),
    }
    output.mkdir(parents=True)
    write_json(output / "confirmation.json", confirmation)
    write_json(output / "policy_summary.json", summarize(metrics))
    (output / "metrics.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in metrics),
        encoding="utf-8",
    )
    write_json(output / "bindings.json", bindings)
    write_json(output / "claim_boundary.json", {
        "supported": (
            "on strict latency-unseen {}, the frozen survival-adaptive policy followed its "
            "{} path; whether its first FPGA measurement was the completed-pool oracle is "
            "reported explicitly in confirmation.json"
        ).format(workload_id, primary["adaptive_path"]),
        "not_supported": [
            "physical AXI traffic reduction", "whole-network FPS improvement",
            "universal generalization from one adaptive holdout",
        ],
    })
    artifacts = {path.name: sha256_file(path) for path in sorted(output.iterdir())
                 if path.is_file() and path.name != "artifact_hashes.json"}
    write_json(output / "artifact_hashes.json", {
        "artifacts": artifacts,
        "source_sha256": {str(Path(__file__).resolve()): sha256_file(__file__),
                          str(Path(adaptive.__file__).resolve()): sha256_file(adaptive.__file__),
                          str(Path(base.__file__).resolve()): sha256_file(base.__file__)},
    })
    print(json.dumps(confirmation, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-dir", required=True)
    parser.add_argument("--local-prefix-dir", required=True)
    parser.add_argument("--first-wave-dir", required=True)
    parser.add_argument("--completion-dir", required=True)
    parser.add_argument("--target-pool-dir", required=True)
    parser.add_argument("--training-pool-dir", required=True)
    parser.add_argument("--training-addition-pool-dir", action="append", default=[])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
