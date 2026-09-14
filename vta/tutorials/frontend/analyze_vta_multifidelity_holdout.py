#!/usr/bin/env python3
"""Audit a frozen multi-fidelity holdout against its completed oracle."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import build_vta_multifidelity_pool as pool_builder
import replay_vta_multifidelity_search as search
import run_vta_p7r132_y00_search_confirmation as board_cost
from run_vta_p7r119_yolo_barrier_pilot import sha256_file, write_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify(directory):
    directory = Path(directory)
    ledger = read_json(directory / "artifact_hashes.json")
    for name, expected in ledger["artifacts"].items():
        if sha256_file(directory / name) != expected:
            raise ValueError("artifact hash mismatch: " + str(directory / name))
    return sha256_file(directory / "artifact_hashes.json")


def distribution(values):
    return search.distribution(list(values))


def run(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    registry_dir = Path(args.registry_dir)
    local_prefix_dir = Path(args.local_prefix_dir)
    first_wave_dir = Path(args.first_wave_dir)
    completion_dir = Path(args.completion_dir)
    target_dir = Path(args.target_pool_dir)
    training_dir = Path(args.training_pool_dir)
    bindings = {name: {"path": str(path.resolve()), "ledger_sha256": verify(path)} for name, path in {
        "registry": registry_dir,
        "live_local_prefix": local_prefix_dir,
        "prospective_first_board_wave": first_wave_dir,
        "full_board_oracle_completion": completion_dir,
        "target_multifidelity_pool": target_dir,
        "development_training_pool": training_dir,
    }.items()}
    registry = read_json(registry_dir / "registry_contract.json")
    expected_policy_hash = registry["source_sha256"][str(Path(search.__file__).resolve())]
    if sha256_file(search.__file__) != expected_policy_hash:
        raise ValueError("frozen policy source changed after holdout registration")
    target = pool_builder.validate(read_json(target_dir / "multifidelity_pool.json"))
    training = pool_builder.validate(read_json(training_dir / "multifidelity_pool.json"))
    workload_id = registry["workload_id"]
    combined = json.loads(json.dumps(training))
    combined["workloads"][workload_id] = target["workloads"][workload_id]
    models = search.heldout_phase_models(combined, workload_id)
    policy = registry["policy"]
    runs = []
    for seed in range(args.seeds):
        for name in search.POLICIES:
            runs.append(search.replay(
                combined, workload_id, name, seed,
                policy["frontier_width"], policy["promotion_width"], models,
                validity_exponent=0.25,
            ))
    oracle = float(target["workloads"][workload_id]["pool_oracle_latency_ms"])
    metrics = []
    for item in runs:
        metrics.append({
            "policy": item["policy"], "seed": item["seed"],
            **search.prefix_to_target(item, oracle, 2.0),
        })
    summary = {}
    for name in search.POLICIES:
        selected = [row for row in metrics if row["policy"] == name]
        summary[name] = {
            "runs": len(selected),
            "success_rate": sum(row["target_reached"] for row in selected) / len(selected),
            "wall_ms_to_target": distribution(row["known_wall_ms"] for row in selected),
            "gross_candidates_to_target": distribution(
                row["gross_candidates_considered"] for row in selected
            ),
            "phase_actions_to_target": {
                phase: distribution(row["phase_action_counts"][phase] for row in selected)
                for phase in search.PHASES
            },
            "resources_to_target": {
                key: distribution(row["resource_cost"][key] for row in selected)
                for key in search.RESOURCE_KEYS
            },
        }

    primary = next(item for item in runs if item["policy"] == policy["name"] and item["seed"] == 0)
    live_history = read_json(local_prefix_dir / "history.json")["events"]
    replay_prefix = primary["actions"][:len(live_history)]
    live_pairs = [(row["candidate_id"], row["phase"], row["status"]) for row in live_history]
    replay_pairs = [(row["candidate_id"], row["phase"], row["outcome"]["status"])
                    for row in replay_prefix]
    if live_pairs != replay_pairs:
        raise ValueError("live local event chain diverges from frozen completed-oracle replay")
    first_fpga = next(row for row in primary["actions"] if row["phase"] == "fpga")
    first_measure = next(row for row in primary["actions"] if row["phase"] == "measure")
    completion = read_json(completion_dir / "timing_summary.json")
    first_wave = read_json(first_wave_dir / "timing_summary.json")
    first_wave_correctness = [
        json.loads(line) for line in (first_wave_dir / "correctness.jsonl").read_text(
            encoding="utf-8"
        ).splitlines() if line.strip()
    ]
    first_wave_timing = [
        json.loads(line) for line in (first_wave_dir / "timing.jsonl").read_text(
            encoding="utf-8"
        ).splitlines() if line.strip()
    ]
    certificates = read_json(first_wave_dir / "cross_certificates.json")
    selected_id = first_measure["candidate_id"]
    selected_correctness = next(
        row for row in first_wave_correctness if row["candidate_id"] == selected_id
    )
    selected_timing = [row for row in first_wave_timing if row["candidate_id"] == selected_id]
    correctness_wall = sum(float(row["host_wall_ms"])
                           for row in selected_correctness["seeds"])
    measure_wall = statistics.median(float(row["host_wall_ms"]) for row in selected_timing)
    compile_wall = sum(float(certificates[cid]["cross_compile_wall_ms"])
                       for cid in read_json(first_wave_dir / "execution_contract.json")[
                           "candidate_order"
                       ])
    local_wall = sum(float(row["wall_ms"]) for row in live_history)
    live_static_rows = [json.loads(line) for line in (
        local_prefix_dir / "static_results.jsonl"
    ).read_text(encoding="utf-8").splitlines() if line.strip()]
    live_fsim_rows = [json.loads(line) for line in (
        local_prefix_dir / "fsim_results.jsonl"
    ).read_text(encoding="utf-8").splitlines() if line.strip()]
    canonical_local_wall = 1000.0 * sum(
        float(row["diagnostic_wall_seconds"]) for row in live_static_rows + live_fsim_rows
    )
    correctness_resources = board_cost.aggregate_row_resources(selected_correctness["seeds"])
    timing_resources = board_cost.representative_timing_cost(selected_timing)["resource_cost"]
    actual_resources = {
        key: correctness_resources[key] + timing_resources[key]
        for key in search.RESOURCE_KEYS
    }
    actual_prospective_cost = {
        "observed_executor_wall_ms": local_wall + compile_wall + correctness_wall + measure_wall,
        "canonical_phase_wall_ms": (
            canonical_local_wall + compile_wall + correctness_wall + measure_wall
        ),
        "local_orchestration_overhead_ms": local_wall - canonical_local_wall,
        "phase_actions": {
            "lower": sum(row["phase"] == "lower" for row in live_history),
            "fsim": sum(row["phase"] == "fsim" for row in live_history),
            "compile": len(read_json(first_wave_dir / "execution_contract.json")[
                "candidate_order"
            ]),
            "fpga": 1,
            "measure": 1,
        },
        "resource_cost": actual_resources,
        "semantics": (
            "observed executor wall includes local worker orchestration; canonical phase wall uses "
            "the same diagnostic cost semantics as every baseline, then adds both frozen promotion "
            "compiles, first-candidate three-seed correctness, and representative measurement"
        ),
    }
    confirmation = {
        "workload_id": workload_id,
        "registered_candidates": len(target["workloads"][workload_id]["candidates"]),
        "complete_outcomes": {
            "lower_invalid": sum(row["oracle"]["phases"]["lower"]["status"] == "invalid"
                                 for row in target["workloads"][workload_id]["candidates"]),
            "fsim_invalid": sum(row["oracle"]["phases"]["fsim"]["status"] == "invalid"
                                for row in target["workloads"][workload_id]["candidates"]),
            "fpga_invalid": sum(row["oracle"]["phases"]["fpga"]["status"] == "invalid"
                                for row in target["workloads"][workload_id]["candidates"]),
            "measured_ok": sum(row["oracle"]["phases"]["measure"]["status"] == "ok"
                               for row in target["workloads"][workload_id]["candidates"]),
        },
        "pool_oracle_candidate_id": completion["pool_oracle_candidate_id"],
        "pool_oracle_latency_ms": oracle,
        "prospective_first_fpga_candidate_id": first_fpga["candidate_id"],
        "prospective_first_measured_candidate_id": first_measure["candidate_id"],
        "first_candidate_is_pool_oracle": first_measure["candidate_id"] == completion[
            "pool_oracle_candidate_id"
        ],
        "first_wave_latency_ms": first_wave["candidate_stats"][first_measure["candidate_id"]][
            "median_ms"
        ],
        "independent_completion_latency_ms": completion["candidate_stats"][
            first_measure["candidate_id"]
        ]["median_ms"],
        "live_local_actions_match_frozen_replay": True,
        "live_local_action_count": len(live_history),
        "actual_prospective_time_to_oracle_cost": actual_prospective_cost,
        "actual_first_wave_collection_deviation": (
            "collector certified and timed both promoted candidates before oracle was known; "
            "prospective identity/order remains valid, but its extra collection work is not "
            "claimed as an online early-stop saving"
        ),
    }
    output.mkdir(parents=True)
    write_json(output / "confirmation.json", confirmation)
    write_json(output / "policy_summary.json", summary)
    (output / "metrics.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in metrics), encoding="utf-8"
    )
    write_json(output / "bindings.json", bindings)
    write_json(output / "claim_boundary.json", {
        "supported": (
            "on a latency-isolated Y05 pool, the frozen hardware-diverse 4-to-2 service policy's "
            "first FPGA/measure identity equals the completed pool oracle"
        ),
        "not_supported": [
            "physical AXI traffic reduction",
            "whole-network FPS improvement",
            "literal online early stop of the first-wave collector",
            "universal generalization from one holdout",
        ],
    })
    artifacts = {path.name: sha256_file(path) for path in sorted(output.iterdir())
                 if path.is_file() and path.name != "artifact_hashes.json"}
    write_json(output / "artifact_hashes.json", {
        "artifacts": artifacts,
        "source_sha256": {str(Path(__file__).resolve()): sha256_file(__file__),
                          str(Path(search.__file__).resolve()): sha256_file(search.__file__)},
    })
    print(json.dumps(confirmation, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-dir", type=Path, required=True)
    parser.add_argument("--local-prefix-dir", type=Path, required=True)
    parser.add_argument("--first-wave-dir", type=Path, required=True)
    parser.add_argument("--completion-dir", type=Path, required=True)
    parser.add_argument("--target-pool-dir", type=Path, required=True)
    parser.add_argument("--training-pool-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
