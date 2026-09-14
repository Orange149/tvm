#!/usr/bin/env python3
"""Execute the frozen survival probe and local promotion for a fresh target."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import replay_vta_adaptive_multifidelity_search as adaptive
import replay_vta_family_wave_search as family_wave
import replay_vta_multifidelity_search as search
import build_vta_multifidelity_pool as pool_builder
import run_vta_multifidelity_local_prefix as live
from run_vta_p7r117_yolo_no_leak_local import sram_features, static_feature_vector
from run_vta_p7r119_yolo_barrier_pilot import run_fsim, sha256_file, static_result, write_json


SCHEMA = "c3_vta_adaptive_multifidelity_live_local_prefix_v1"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def run(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    registry_dir = Path(args.registry_dir)
    registry_ledger = live.verify_ledger(registry_dir)
    registry = read_json(registry_dir / "registry_contract.json")
    if registry["status"] != "frozen_before_target_lower_fsim_fpga_or_latency":
        raise ValueError("adaptive registry is not pristine")
    if sha256_file(adaptive.__file__) != registry["source_sha256"][
            str(Path(adaptive.__file__).resolve())]:
        raise ValueError("adaptive policy source changed after registry freeze")
    raw = read_jsonl(registry_dir / "candidates_v2.jsonl")
    if live.canonical_sha256(raw) != registry["candidate_commitment_sha256"]:
        raise ValueError("candidate commitment mismatch")
    training_path = Path(args.training_pool).resolve()
    training = pool_builder.validate(read_json(training_path))
    target = [live.target_candidate(row) for row in raw]
    combined = json.loads(json.dumps(training))
    combined["workloads"][args.workload_id] = {
        "candidate_count": len(target), "pool_oracle_latency_ms": None,
        "candidates": target,
    }
    states = {row["candidate_id"]: search.state_for(row) for row in target}
    target_by_id = {row["candidate_id"]: row for row in target}
    raw_by_id = {row["candidate_id"]: row for row in raw}
    models = search.heldout_phase_models(combined, args.workload_id)
    family_order = family_wave.hardware_family_order(
        combined, args.workload_id, args.seed, False, models
    )
    first_family = family_order[0]
    execution = {
        "schema": SCHEMA,
        "status": "frozen_before_target_local_actions",
        "registry": {"path": str(registry_dir.resolve()), "ledger_sha256": registry_ledger},
        "training_pool": {"path": str(training_path), "sha256": sha256_file(training_path)},
        "policy": registry["policy"], "first_family": first_family,
        "source_sha256": {str(Path(__file__).resolve()): sha256_file(__file__),
                          str(Path(adaptive.__file__).resolve()): sha256_file(adaptive.__file__)},
        "board_contact": "forbidden", "performance_labels": "forbidden",
    }
    output.mkdir(parents=True)
    write_json(output / "execution_contract.json", execution)
    write_json(output / "pre_observation_hashes.json", {"artifacts": {
        "execution_contract.json": sha256_file(output / "execution_contract.json")
    }})
    events, static_rows, fsim_rows = [], [], []
    parent = None

    def perform(state, phase, diagnostic):
        nonlocal parent
        cid = state["candidate"]["candidate_id"]
        visible_hash = live.canonical_sha256(search.visible_features(state, phase))
        started = time.perf_counter()
        if phase == "lower":
            row = static_result(raw_by_id[cid])
            row["sram_features"] = sram_features(raw_by_id[cid])
            if row["status"] == "ok":
                row["static_feature_vector"] = static_feature_vector(
                    row["transfer_signature"], row["sram_features"]
                )
            status = "ok" if row["status"] == "ok" else "invalid"
            reveal = pool_builder.lower_reveal(row)
            static_rows.append(row)
        else:
            lower = next(row for row in static_rows if row["candidate_id"] == cid)
            row, stderr = run_fsim(
                raw_by_id[cid], lower["tir_sha256"], lower["sync"]["residency_drains"],
                args.fsim_timeout, args.fsim_hw_path,
            )
            if (row.get("failure") or {}).get("retryable"):
                raise RuntimeError("retryable FSim infrastructure failure: " + str(row["failure"]))
            status = "ok" if row["status"] == "passed" else "invalid"
            reveal = pool_builder.fsim_reveal(row)
            row["stderr_sha256"] = hashlib.sha256(stderr.encode()).hexdigest()
            fsim_rows.append(row)
        wall_ms = (time.perf_counter() - started) * 1000
        live.apply_result(state, phase, status, reveal)
        item = live.event(len(events) + 1, parent, state, phase, visible_hash,
                          diagnostic, status, wall_ms, reveal)
        parent = item["event_sha256"]
        events.append(item)
        print("adaptive-live {} {} {}".format(phase, cid[:12], status), flush=True)
        return status

    first_members = [state for state in states.values()
                     if state["candidate"]["family_id"] == first_family]
    for state in sorted(first_members, key=lambda item: search.stable_hash(
            args.seed, args.workload_id, item["candidate"]["candidate_id"],
            "adaptive-first-family-lower")):
        perform(state, "lower", {"selector": "three_mode_survival_probe",
                                  "family_id": first_family})
    first_passes = sum(row["phase"] == "lower" and row["status"] == "ok" for row in events)
    if first_passes <= registry["policy"]["sparse_threshold_lower_passes"]:
        path = "sparse_family_wave"
        board_candidate_found = False
        for family_id in family_order:
            if family_id != first_family:
                members = [state for state in states.values()
                           if state["candidate"]["family_id"] == family_id and
                           not state["terminated"] and state["next_phase"] == "lower"]
                for state in sorted(members, key=lambda item: search.stable_hash(
                        args.seed, args.workload_id, item["candidate"]["candidate_id"],
                        "adaptive-family-lower")):
                    perform(state, "lower", {
                        "selector": "sparse_space_complete_next_hardware_family",
                        "family_id": family_id,
                    })
            while True:
                ready = family_wave.ranked_ready(
                    states, family_id, "fsim", "family_wave_service", models, args.seed
                )
                if not ready:
                    break
                state = ready[0]
                if perform(state, "fsim", {"selector": "sparse_family_min_service",
                                            "family_id": family_id,
                                            "service_score": search.service_score(state)}) == "ok":
                    board_candidate_found = True
                    break
            if board_candidate_found:
                break
        if not board_candidate_found:
            raise RuntimeError("sparse path exhausted without a FSim-pass board candidate")
    else:
        path = "dense_fixed_4_to_2"
        dense = registry["policy"]["dense_path"]
        while True:
            state, phase, diagnostic = search.next_action_frontier(
                combined, args.workload_id, states, dense["policy"], args.seed,
                dense["frontier_width"], dense["promotion_width"], models, 0.25,
            )
            if phase == "compile":
                break
            if phase not in ("lower", "fsim"):
                raise RuntimeError("unexpected pre-board phase " + str(phase))
            perform(state, phase, diagnostic)
    compile_ready = [state for state in states.values()
                     if not state["terminated"] and state["next_phase"] == "compile"]
    ordered = sorted(compile_ready, key=lambda state: (
        search.service_score(state), state["candidate"]["candidate_id"]
    ))
    expected_wave = 1 if path == "sparse_family_wave" else registry["policy"][
        "dense_path"
    ]["promotion_width"]
    if len(ordered) < expected_wave:
        raise RuntimeError("undersized board wave")
    ordered = ordered[:expected_wave]
    write_json(output / "history.json", {"schema": "c3_vta_multifidelity_event_chain_v1",
                                          "head_sha256": parent, "events": events})
    (output / "static_results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in static_rows), encoding="utf-8"
    )
    (output / "fsim_results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in fsim_rows), encoding="utf-8"
    )
    write_json(output / "board_wave.json", {
        "status": "frozen_after_adaptive_live_local_prefix_before_cross_or_board",
        "adaptive_path": path, "first_family": first_family,
        "first_family_lower_passes": first_passes,
        "candidate_order": [state["candidate"]["candidate_id"] for state in ordered],
        "service_scores": {state["candidate"]["candidate_id"]: search.service_score(state)
                           for state in ordered},
        "candidate_records": [raw_by_id[state["candidate"]["candidate_id"]]
                              for state in ordered],
        "history_head_sha256": parent, "performance_labels_used": False,
        "board_contacted": False,
    })
    write_json(output / "summary.json", {
        "status": "adaptive_live_local_prefix_complete_board_wave_frozen",
        "adaptive_path": path, "first_family_lower_passes": first_passes,
        "lower_actions": sum(row["phase"] == "lower" for row in events),
        "lower_invalid": sum(row["phase"] == "lower" and row["status"] == "invalid"
                             for row in events),
        "fsim_actions": sum(row["phase"] == "fsim" for row in events),
        "fsim_invalid": sum(row["phase"] == "fsim" and row["status"] == "invalid"
                            for row in events),
        "board_wave_size": len(ordered), "board_contacted": False,
        "performance_labels_collected": False,
    })
    (output / "command.txt").write_text(
        " ".join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]) + "\n",
        encoding="utf-8",
    )
    artifacts = {path.name: sha256_file(path) for path in sorted(output.iterdir())
                 if path.is_file() and path.name != "artifact_hashes.json"}
    write_json(output / "artifact_hashes.json", {
        "artifacts": artifacts,
        "source_sha256": {str(Path(__file__).resolve()): sha256_file(__file__)},
    })
    print(json.dumps(read_json(output / "summary.json"), indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-dir", type=Path, required=True)
    parser.add_argument("--training-pool", type=Path, required=True)
    parser.add_argument("--workload-id", required=True)
    parser.add_argument("--fsim-hw-path", type=Path, default=Path("/tmp/vta-hw-fsim"))
    parser.add_argument("--fsim-timeout", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
