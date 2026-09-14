#!/usr/bin/env python3
"""Execute the frozen P7R120 Y02 contract through direct TVM RPC only.

The script never invokes SSH and never restarts or reconfigures the board.  Each
locally rebuilt module is uploaded to the RPC temporary area, loaded, and then
immediately removed by the RPC file API.  The sealed TopHub identity is used
only as a correctness/latency canary and protected deployment fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import tvm
from tvm import rpc
import vta

import vta.top.vta_conv2d_residency  # pylint: disable=unused-import
from c3_candidate_identity import canonical_json_bytes
from prepare_vta_p7r120_yolo_rpc_contract import (
    GUARD_PATHS,
    REPO,
    SCHEMA as CONTRACT_SCHEMA,
    build_and_export,
    load_json,
    sha256_file,
    write_json,
)
from qualify_vta_residency_fsim import array_sha256, reference_data_from_workload


RESULT_SCHEMA = "c3_p7r120_y02_rpc_board_result_v2"
HERE = Path(__file__).resolve().parent
C3 = HERE / "report_out" / "stage_tile_cotuning" / "c3_dma_residency_autotune"
DEFAULT_CONTRACT = (
    C3
    / "07_grouped_holdout"
    / "20260911_p7r120_y02_rpc_contract_v2_run01"
    / "contract.json"
)
DEFAULT_OUTPUT = (
    C3
    / "07_grouped_holdout"
    / "20260911_p7r120_y02_rpc_board_v2_run01"
)
PROFILE_FIELDS = (
    "load_buffer_2d_calls",
    "load_buffer_2d_bytes",
    "load_buffer_2d_inp_bytes",
    "load_buffer_2d_wgt_bytes",
    "store_buffer_2d_calls",
    "store_buffer_2d_bytes",
    "driver_run_insns",
    "synchronize_insns",
)


def verify_contract(path):
    path = Path(path).resolve()
    ledger_path = path.parent / "artifact_hashes.json"
    ledger = load_json(ledger_path)
    expected = ledger["artifacts"].get(path.name)
    if expected != sha256_file(path):
        raise ValueError("contract artifact hash mismatch")
    contract = load_json(path)
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise ValueError("unexpected P7R120 contract schema")
    if contract.get("status") != "frozen_after_local_cross_qualification_before_rpc_labels":
        raise ValueError("contract is not pre-label frozen")
    guards = contract["fresh_hardware_certificate_v2"]["source_guards_sha256"]
    observed = {str(source.relative_to(REPO)): sha256_file(source) for source in GUARD_PATHS}
    if guards != observed:
        raise ValueError("fresh v2 source/hardware guards changed")
    candidates = contract["candidate_pool"]["candidates"]
    if len(candidates) != 6 or len({row["candidate_id"] for row in candidates}) != 6:
        raise ValueError("contract must contain six unique candidates")
    if contract["candidate_pool"]["pool_commitment_sha256"] != hashlib.sha256(
        canonical_json_bytes(candidates)
    ).hexdigest():
        raise ValueError("candidate pool commitment mismatch")
    orders = contract["timing"]["candidate_orders"]
    ids = {row["candidate_id"] for row in candidates}
    if len(orders) != 7 or any(len(order) != 6 or set(order) != ids for order in orders):
        raise ValueError("timing order is not seven complete equal-budget rounds")
    canary_id = contract["sealed_reference"]["candidate_id"]
    if canary_id in ids or any(canary_id in order for order in orders):
        raise ValueError("sealed TopHub leaked into candidate dispatch")
    return contract, sha256_file(ledger_path)


def runtime_inventory(remote):
    inventory = {
        "transport": "direct_tvm_rpc",
        "ssh_used": False,
        "boot_id": "unknown_not_exposed_by_rpc",
        "system_lib": "unavailable",
        "functions": {},
    }
    try:
        remote.system_lib()
        inventory["system_lib"] = "available"
    except Exception as error:  # Old RPC builds may not expose a usable system lib.
        inventory["system_lib_error"] = (str(error) or type(error).__name__)[:1000]
    functions = {}
    for name in (
        "vta.runtime.profiler_status",
        "vta.runtime.profiler_clear",
        "vta.runtime.queue_capacity_status",
        "vta.runtime.replay_status",
    ):
        try:
            functions[name] = remote.get_function(name)
            inventory["functions"][name] = "available"
        except Exception as error:
            inventory["functions"][name] = "unavailable: {}".format(type(error).__name__)
    if "vta.runtime.profiler_status" not in functions:
        raise RuntimeError("required RPC profiler_status function is unavailable")
    inventory["profiler_initial"] = json.loads(functions["vta.runtime.profiler_status"]())
    if "vta.runtime.replay_status" in functions:
        inventory["replay_initial"] = json.loads(functions["vta.runtime.replay_status"]())
    if "vta.runtime.queue_capacity_status" in functions:
        inventory["queue_capacity_initial"] = json.loads(
            functions["vta.runtime.queue_capacity_status"]()
        )
    return inventory, functions


def load_ephemeral(remote, binary):
    remote.upload(str(binary))
    module = remote.load_module(binary.name)
    remote.remove(binary.name)
    return module


def exact_run(function, device, workload, seed):
    data, weight, expected = reference_data_from_workload(workload, int(seed))
    buffers = [
        tvm.nd.array(data, device),
        tvm.nd.array(weight, device),
        tvm.nd.empty(expected.shape, "int8", device),
    ]
    function(*buffers)
    actual = buffers[-1].numpy()
    mismatch = int(np.count_nonzero(actual != expected))
    return {
        "seed": int(seed),
        "correct": mismatch == 0,
        "mismatch_count": mismatch,
        "expected_sha256": array_sha256(expected),
        "actual_sha256": array_sha256(actual),
    }


def profile_snapshot(functions):
    status = json.loads(functions["vta.runtime.profiler_status"]())
    return {name: status.get(name) for name in PROFILE_FIELDS}


def clear_profile(functions):
    clear = functions.get("vta.runtime.profiler_clear")
    if clear is not None:
        clear()


def time_once(item, device, functions):
    clear_profile(functions)
    timer = item["module"].time_evaluator("main", device, number=1, repeat=1)
    result = timer(*item["buffers"])
    latency = float(result.mean * 1000.0)
    if not np.isfinite(latency) or latency <= 0:
        raise RuntimeError("invalid RPC timer result")
    actual = item["buffers"][-1].numpy()
    mismatch = int(np.count_nonzero(actual != item["expected"]))
    if mismatch:
        raise RuntimeError("timed invocation wrong answer: {} mismatches".format(mismatch))
    return latency, profile_snapshot(functions)


def promotion_decision(candidate_stats, canary_samples):
    canary_median = statistics.median(float(row["latency_ms"]) for row in canary_samples)
    midpoints = {}
    by_round = defaultdict(dict)
    for row in canary_samples:
        by_round[int(row["round"])][row["bracket"]] = float(row["latency_ms"])
    for round_index, values in by_round.items():
        if set(values) != {"before", "after"}:
            raise ValueError("incomplete sealed canary bracket")
        midpoints[round_index] = (values["before"] + values["after"]) / 2.0
    eligible = []
    enriched = {}
    for candidate_id, stats in candidate_stats.items():
        wins = sum(
            float(value) < midpoints[round_index]
            for round_index, value in enumerate(stats["values_ms"])
        )
        improvement = (canary_median - stats["median_ms"]) / canary_median
        promote = improvement >= 0.02 and wins >= 5
        enriched[candidate_id] = {
            **stats,
            "versus_tophub_fraction": improvement,
            "round_midpoint_wins": wins,
            "promotion_gate_passed": promote,
        }
        if promote:
            eligible.append(candidate_id)
    if eligible:
        choice = min(eligible, key=lambda candidate_id: enriched[candidate_id]["median_ms"])
        return {
            "deployment_choice": choice,
            "fallback_retained": False,
            "sealed_tophub_median_ms": canary_median,
            "candidate_stats": enriched,
            "rule": "at least 2% faster and at least 5/7 bracket-midpoint wins",
        }
    return {
        "deployment_choice": "sealed_tophub_canary",
        "fallback_retained": True,
        "sealed_tophub_median_ms": canary_median,
        "candidate_stats": enriched,
        "rule": "at least 2% faster and at least 5/7 bracket-midpoint wins",
    }


def candidate_summary(samples, candidates):
    by_id = defaultdict(list)
    for row in samples:
        by_id[row["candidate_id"]].append(float(row["latency_ms"]))
    result = {}
    for entry in candidates:
        values = by_id[entry["candidate_id"]]
        if len(values) != 7:
            raise ValueError("candidate does not have exactly seven timing samples")
        result[entry["candidate_id"]] = {
            "family_id": entry["family_id"],
            "public_mode": entry["public_mode"],
            "values_ms": values,
            "median_ms": statistics.median(values),
            "q1_ms": float(np.percentile(values, 25)),
            "q3_ms": float(np.percentile(values, 75)),
        }
    return result


def same_tile_comparisons(stats):
    grouped = defaultdict(dict)
    for candidate_id, row in stats.items():
        grouped[row["family_id"]][row["public_mode"]] = (candidate_id, row)
    output = {}
    for family_id, modes in grouped.items():
        original = modes["original"][1]["median_ms"]
        output[family_id] = {}
        for mode in ("input_stationary", "weight_resident_barrier"):
            candidate_id, row = modes[mode]
            output[family_id][mode] = {
                "candidate_id": candidate_id,
                "original_median_ms": original,
                "mechanism_median_ms": row["median_ms"],
                "same_tile_improvement_fraction": (original - row["median_ms"]) / original,
            }
    return output


def run(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite board observations {}".format(output))
    contract, contract_ledger_hash = verify_contract(args.contract)
    if args.host != contract["rpc_contract"]["host"] or args.port != contract["rpc_contract"]["port"]:
        raise ValueError("RPC endpoint differs from frozen contract")
    output.mkdir(parents=True)
    preregistered = {
        "schema": "c3_p7r120_rpc_execution_preregistered_v2",
        "contract_path": str(Path(args.contract).resolve()),
        "contract_sha256": sha256_file(args.contract),
        "contract_artifact_ledger_sha256": contract_ledger_hash,
        "runner_sha256": sha256_file(__file__),
        "ssh_forbidden": True,
        "boot_id": "unknown_not_exposed_by_rpc",
        "candidate_correctness_gate": "6 identities x 3 seeds exact; all 18 required",
        "timing_gate": "global correctness gate only",
        "timing_orders": contract["timing"]["candidate_orders"],
        "sealed_canary_excluded_from_search_and_training": True,
    }
    write_json(output / "preregistered.json", preregistered)
    write_json(
        output / "pre_rpc_hashes.json",
        {"artifacts": {"preregistered.json": sha256_file(output / "preregistered.json")}},
    )

    env = vta.get_env()
    if env.TARGET != "axu5evb":
        raise RuntimeError("board runner requires VTA TARGET=axu5evb")
    candidates = contract["candidate_pool"]["candidates"]
    canary = contract["sealed_reference"]
    entries = candidates + [canary]
    built = {}
    start = time.monotonic()
    try:
        # Rebuild and bind every binary before opening the RPC session.
        temporary = tempfile.TemporaryDirectory(prefix="c3_p7r120_rpc_")
        directory = Path(temporary.name)
        for entry in entries:
            expected = entry["fresh_axu_cross_certificate"]
            certificate = build_and_export(entry, env, directory, expected["tir_sha256"])
            if certificate["binary_sha256"] != expected["binary_sha256"]:
                raise RuntimeError("cross-compiled binary hash changed")
            built[entry["candidate_id"]] = {"binary": directory / (entry["candidate_id"] + ".so"), "certificate": certificate}

        remote = rpc.connect(args.host, args.port, session_timeout=args.session_timeout)
        inventory, runtime_functions = runtime_inventory(remote)
        write_json(output / "rpc_inventory.json", inventory)
        device = remote.ext_dev(0)
        for entry in entries:
            candidate_id = entry["candidate_id"]
            module = load_ephemeral(remote, built[candidate_id]["binary"])
            built[candidate_id].update(entry=entry, module=module, function=module["main"])

        correctness_rows = []
        canary_correctness = []
        canary_item = built[canary["candidate_id"]]
        for phase in ("before_candidates",):
            for seed in contract["correctness"]["seeds"]:
                result = exact_run(
                    canary_item["function"], device, canary["identity"]["workload"], seed
                )
                canary_correctness.append({"phase": phase, **result})
                if not result["correct"]:
                    raise RuntimeError("sealed TopHub pre-canary wrong answer")

        with (output / "correctness.jsonl").open("x", encoding="utf-8") as stream:
            for position, candidate_id in enumerate(contract["correctness"]["candidate_order"]):
                item = built[candidate_id]
                seed_rows = []
                failure = None
                for seed in contract["correctness"]["seeds"]:
                    try:
                        seed_row = exact_run(
                            item["function"], device, item["entry"]["identity"]["workload"], seed
                        )
                    except Exception as error:
                        seed_row = {
                            "seed": int(seed),
                            "correct": False,
                            "failure": {
                                "exception_type": type(error).__name__,
                                "message": (str(error) or type(error).__name__)[:2000],
                            },
                        }
                    seed_rows.append(seed_row)
                    if not seed_row.get("correct"):
                        failure = "failed seed {}; remaining seeds for this identity not run".format(seed)
                        break
                row = {
                    "schema": RESULT_SCHEMA,
                    "correctness_position": position,
                    "candidate_id": candidate_id,
                    "family_id": item["entry"]["family_id"],
                    "public_mode": item["entry"]["public_mode"],
                    "seeds": seed_rows,
                    "status": "passed" if len(seed_rows) == 3 and all(seed["correct"] for seed in seed_rows) else "failed",
                    "failure": failure,
                    "performance_measurement": "not_collected",
                }
                correctness_rows.append(row)
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
                print(
                    "correctness {}/6 {} {}".format(position + 1, row["public_mode"], row["status"]),
                    flush=True,
                )

        global_gate = len(correctness_rows) == 6 and all(
            row["status"] == "passed" for row in correctness_rows
        )
        candidate_timing = []
        canary_timing = []
        if global_gate:
            for entry in entries:
                item = built[entry["candidate_id"]]
                data, weight, expected = reference_data_from_workload(entry["identity"]["workload"], 0)
                item["buffers"] = [
                    tvm.nd.array(data, device),
                    tvm.nd.array(weight, device),
                    tvm.nd.empty(expected.shape, "int8", device),
                ]
                item["expected"] = expected
            by_id = {entry["candidate_id"]: entry for entry in candidates}
            with (output / "timing.jsonl").open("x", encoding="utf-8") as timing_stream, (
                output / "sealed_canary_timing.jsonl"
            ).open("x", encoding="utf-8") as canary_stream:
                sequence = 0
                for round_index, order in enumerate(contract["timing"]["candidate_orders"]):
                    latency, profile = time_once(canary_item, device, runtime_functions)
                    canary_row = {
                        "round": round_index,
                        "bracket": "before",
                        "candidate_id": canary["candidate_id"],
                        "latency_ms": latency,
                        "runtime_profile": profile,
                        "selection_or_training_label": False,
                    }
                    canary_timing.append(canary_row)
                    canary_stream.write(json.dumps(canary_row, sort_keys=True) + "\n")
                    for position, candidate_id in enumerate(order):
                        latency, profile = time_once(built[candidate_id], device, runtime_functions)
                        entry = by_id[candidate_id]
                        row = {
                            "schema": RESULT_SCHEMA,
                            "sequence": sequence,
                            "round": round_index,
                            "position": position,
                            "candidate_id": candidate_id,
                            "family_id": entry["family_id"],
                            "public_mode": entry["public_mode"],
                            "number": 1,
                            "latency_ms": latency,
                            "runtime_profile": profile,
                        }
                        candidate_timing.append(row)
                        timing_stream.write(json.dumps(row, sort_keys=True) + "\n")
                        timing_stream.flush()
                        sequence += 1
                    latency, profile = time_once(canary_item, device, runtime_functions)
                    canary_row = {
                        "round": round_index,
                        "bracket": "after",
                        "candidate_id": canary["candidate_id"],
                        "latency_ms": latency,
                        "runtime_profile": profile,
                        "selection_or_training_label": False,
                    }
                    canary_timing.append(canary_row)
                    canary_stream.write(json.dumps(canary_row, sort_keys=True) + "\n")
                    canary_stream.flush()
                    print("timing round {}/7 complete".format(round_index + 1), flush=True)

        for seed in contract["correctness"]["seeds"]:
            result = exact_run(
                canary_item["function"], device, canary["identity"]["workload"], seed
            )
            canary_correctness.append({"phase": "after_experiment", **result})
            if not result["correct"]:
                raise RuntimeError("sealed TopHub post-canary wrong answer")
        (output / "sealed_canary_correctness.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in canary_correctness),
            encoding="utf-8",
        )
        final_profile = profile_snapshot(runtime_functions)
        temporary.cleanup()
    except Exception as error:
        write_json(
            output / "failure.json",
            {
                "status": "failed",
                "exception_type": type(error).__name__,
                "message": (str(error) or type(error).__name__)[:4000],
                "traceback": traceback.format_exc()[-12000:],
                "ssh_used": False,
                "boot_id": "unknown_not_exposed_by_rpc",
            },
        )
        raise

    summary = {
        "schema": "c3_p7r120_y02_rpc_board_summary_v2",
        "status": "timing_completed" if global_gate else "correctness_gate_failed_no_timing",
        "rpc_endpoint": "{}:{}".format(args.host, args.port),
        "ssh_used": False,
        "board_restart_or_reconfiguration": False,
        "persistent_board_write": False,
        "boot_id": "unknown_not_exposed_by_rpc",
        "candidate_correctness": {
            "identities": len(correctness_rows),
            "passed": sum(row["status"] == "passed" for row in correctness_rows),
            "seed_checks_completed": sum(len(row["seeds"]) for row in correctness_rows),
            "global_timing_gate": global_gate,
        },
        "sealed_canary": {
            "identity": canary["candidate_id"],
            "correctness_checks": len(canary_correctness),
            "correct": all(row["correct"] for row in canary_correctness),
            "excluded_from_candidate_pool_and_training": True,
        },
        "timing": {
            "candidate_samples": len(candidate_timing),
            "canary_samples": len(canary_timing),
            "number_per_sample": 1,
        },
        "final_runtime_profile": final_profile,
        "elapsed_seconds": time.monotonic() - start,
        "claim_boundary": "single Y02 geometry/current unknown-boot RPC session; no stage/FPS or cross-boot claim",
    }
    if global_gate:
        stats = candidate_summary(candidate_timing, candidates)
        promotion = promotion_decision(stats, canary_timing)
        summary["candidate_stats"] = stats
        summary["same_tile_comparisons"] = same_tile_comparisons(stats)
        summary["incumbent_protection"] = promotion
    write_json(output / "summary.json", summary)
    (output / "STATUS.md").write_text(
        "# P7R120 Y02 RPC-only board pilot\n\n"
        "- Status: `{}`\n"
        "- Candidate correctness: {}/6 identities; {} seed checks\n"
        "- Candidate timing samples: {}; sealed canary samples: {}\n"
        "- SSH/restart/persistent board write: no\n"
        "- Boot ID: `unknown_not_exposed_by_rpc`\n"
        "- Claim: single-geometry/current-session pilot only\n".format(
            summary["status"],
            summary["candidate_correctness"]["passed"],
            summary["candidate_correctness"]["seed_checks_completed"],
            summary["timing"]["candidate_samples"],
            summary["timing"]["canary_samples"],
        ),
        encoding="utf-8",
    )
    (output / "command.txt").write_text(
        " ".join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]) + "\n",
        encoding="utf-8",
    )
    hashes = {
        path.name: sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    write_json(
        output / "artifact_hashes.json",
        {"artifacts": hashes, "source_sha256": {str(Path(__file__).resolve()): sha256_file(__file__)}},
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--session-timeout", type=int, default=120)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
