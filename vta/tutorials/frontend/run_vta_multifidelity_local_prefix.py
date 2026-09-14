#!/usr/bin/env python3
"""Execute the frozen online LOWER/FSim prefix until a board wave is ready."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from run_vta_p7r117_yolo_no_leak_local import sram_features, static_feature_vector
from run_vta_p7r119_yolo_barrier_pilot import run_fsim, sha256_file, static_result, write_json
import build_vta_multifidelity_pool as pool_builder
import replay_vta_multifidelity_search as search


SCHEMA = "c3_vta_multifidelity_live_local_prefix_v1"
SEEDS = (0, 20250901, 20260910)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def verify_ledger(directory):
    directory = Path(directory)
    ledger = read_json(directory / "artifact_hashes.json")
    for name, expected in ledger["artifacts"].items():
        if sha256_file(directory / name) != expected:
            raise ValueError("artifact hash mismatch: " + str(directory / name))
    return sha256_file(directory / "artifact_hashes.json")


def target_candidate(row):
    return {
        "candidate_id": row["candidate_id"],
        "workload_id": row["workload_id"],
        "family_id": row["family_id"],
        "residence_mode": row["public_mode"],
        "prelower": {
            "feature_provenance": "frozen_before_target_labels",
            "features": pool_builder.cheap_features(row),
            "geometry_signature_sha256": pool_builder.canonical_sha256(
                row["identity"]["workload"]
            ),
            "hardware_fingerprint_sha256": pool_builder.canonical_sha256(
                row["identity"]["hardware_fingerprint"]
            ),
            "schedule_version": row["identity"]["schedule_version"],
        },
    }


def apply_result(state, phase, status, reveal):
    state["revealed"]["_outcome_" + phase] = status
    if reveal is not None:
        state["revealed"][phase] = reveal
    if status == "ok":
        state["next_phase"] = search.PHASES[search.PHASES.index(phase) + 1]
    else:
        state["next_phase"] = None
        state["terminated"] = True


def event(seq, parent, state, phase, visible_hash, diagnostic, status, wall_ms, reveal):
    body = {
        "seq": seq,
        "parent_event_sha256": parent,
        "candidate_id": state["candidate"]["candidate_id"],
        "phase": phase,
        "visible_feature_sha256": visible_hash,
        "selection_diagnostics": diagnostic,
        "status": status,
        "wall_ms": wall_ms,
        "feature_patch": reveal,
        "performance_label": None,
        "board_contacted": False,
    }
    body["event_sha256"] = canonical_sha256(body)
    return body


def run(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable live prefix " + str(output))
    registry_dir = Path(args.registry_dir)
    registry_ledger = verify_ledger(registry_dir)
    registry = read_json(registry_dir / "registry_contract.json")
    if registry["status"] != "frozen_before_y05_lower_fsim_fpga_or_latency":
        raise ValueError("registry is not a pristine holdout")
    raw = read_jsonl(registry_dir / "candidates_v2.jsonl")
    if canonical_sha256(raw) != registry["candidate_commitment_sha256"]:
        raise ValueError("candidate registry commitment mismatch")
    training_path = Path(args.training_pool).resolve()
    training = pool_builder.validate(read_json(training_path))
    target = [target_candidate(row) for row in raw]
    target_by_id = {row["candidate_id"]: row for row in target}
    raw_by_id = {row["candidate_id"]: row for row in raw}
    combined = json.loads(json.dumps(training))
    combined["workloads"][args.workload_id] = {
        "candidate_count": len(target), "pool_oracle_latency_ms": None,
        "candidates": target,
    }
    states = {row["candidate_id"]: search.state_for(row) for row in target}
    phase_models = search.heldout_phase_models(combined, args.workload_id)
    policy = registry["policy"]
    execution_contract = {
        "schema": SCHEMA,
        "status": "frozen_before_live_local_actions",
        "registry": {"path": str(registry_dir.resolve()), "ledger_sha256": registry_ledger},
        "training_pool": {"path": str(training_path), "sha256": sha256_file(training_path)},
        "policy": policy,
        "fsim": {"hardware_path": str(Path(args.fsim_hw_path).resolve()),
                 "timeout_seconds": args.fsim_timeout, "seeds": list(SEEDS)},
        "source_sha256": {str(Path(__file__).resolve()): sha256_file(__file__),
                          str(Path(search.__file__).resolve()): sha256_file(search.__file__)},
        "board_contact": "forbidden in this executor",
        "performance_labels": "forbidden",
    }
    output.mkdir(parents=True)
    write_json(output / "execution_contract.json", execution_contract)
    write_json(output / "pre_observation_hashes.json", {"artifacts": {
        "execution_contract.json": sha256_file(output / "execution_contract.json")
    }})
    events = []
    static_rows = []
    fsim_rows = []
    parent = None
    while True:
        state, phase, diagnostic = search.next_action_frontier(
            combined, args.workload_id, states, policy["name"], args.seed,
            policy["frontier_width"], policy["promotion_width"], phase_models,
            validity_exponent=0.25,
        )
        if state is None:
            raise RuntimeError("search exhausted without a board-ready wave")
        if phase == "compile":
            break
        if phase not in ("lower", "fsim"):
            raise RuntimeError("local prefix reached unexpected phase " + phase)
        cid = state["candidate"]["candidate_id"]
        visible_hash = canonical_sha256(search.visible_features(state, phase))
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
            lower_event = next(item for item in reversed(static_rows)
                               if item["candidate_id"] == cid)
            row, stderr = run_fsim(
                raw_by_id[cid], lower_event["tir_sha256"],
                lower_event["sync"]["residency_drains"],
                args.fsim_timeout, args.fsim_hw_path,
            )
            if (row.get("failure") or {}).get("retryable"):
                raise RuntimeError("retryable FSim infrastructure failure: " + str(row["failure"]))
            status = "ok" if row["status"] == "passed" else "invalid"
            reveal = pool_builder.fsim_reveal(row)
            row["stderr_sha256"] = hashlib.sha256(stderr.encode()).hexdigest()
            fsim_rows.append(row)
        wall_ms = (time.perf_counter() - started) * 1000.0
        apply_result(state, phase, status, reveal)
        item = event(len(events) + 1, parent, state, phase, visible_hash, diagnostic,
                     status, wall_ms, reveal)
        parent = item["event_sha256"]
        events.append(item)
        print("live {} {} {}".format(phase, cid[:12], status), flush=True)
    compile_ready = [state for state in states.values()
                     if not state["terminated"] and state["next_phase"] == "compile"]
    ordered = sorted(compile_ready, key=lambda item: (
        search.service_score(item), item["candidate"]["candidate_id"]
    ))
    if len(ordered) < policy["promotion_width"]:
        raise RuntimeError("board wave is smaller than frozen promotion width")
    write_json(output / "history.json", {
        "schema": "c3_vta_multifidelity_event_chain_v1",
        "head_sha256": parent,
        "event_count": len(events),
        "events": events,
    })
    (output / "static_results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in static_rows), encoding="utf-8"
    )
    (output / "fsim_results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in fsim_rows), encoding="utf-8"
    )
    write_json(output / "board_wave.json", {
        "status": "frozen_after_live_local_prefix_before_cross_or_board",
        "candidate_order": [state["candidate"]["candidate_id"] for state in ordered],
        "service_scores": {state["candidate"]["candidate_id"]: search.service_score(state)
                           for state in ordered},
        "candidate_records": [raw_by_id[state["candidate"]["candidate_id"]]
                              for state in ordered],
        "history_head_sha256": parent,
        "performance_labels_used": False,
        "board_contacted": False,
    })
    write_json(output / "summary.json", {
        "status": "live_local_prefix_complete_board_wave_frozen",
        "lower_actions": sum(item["phase"] == "lower" for item in events),
        "lower_invalid": sum(item["phase"] == "lower" and item["status"] == "invalid"
                             for item in events),
        "fsim_actions": sum(item["phase"] == "fsim" for item in events),
        "fsim_invalid": sum(item["phase"] == "fsim" and item["status"] == "invalid"
                            for item in events),
        "board_wave_size": len(ordered),
        "history_head_sha256": parent,
        "board_contacted": False,
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
    parser.add_argument("--workload-id", default="Y05")
    parser.add_argument("--fsim-hw-path", type=Path, default=Path("/tmp/vta-hw-fsim"))
    parser.add_argument("--fsim-timeout", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
