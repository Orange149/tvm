#!/usr/bin/env python3
"""P7R123: health-bracketed same-tile timing for two correct Y02B00 identities."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import tvm
from tvm import rpc
import vta

import vta.top.vta_conv2d_residency  # pylint: disable=unused-import
from prepare_vta_p7r120_yolo_rpc_contract import build_and_export, load_json, sha256_file, write_json
from qualify_vta_residency_fsim import array_sha256, reference_data_from_workload
from run_vta_p7r120_yolo_rpc_board import load_ephemeral, runtime_inventory, verify_contract
from run_vta_p7r121_tophub_adapter_diagnostic import finalize
from run_vta_p7r122_y02_health_gated_correctness import (
    DEFAULT_P7R120,
    DEFAULT_OUTPUT as DEFAULT_P7R122,
    verify_run,
)


SCHEMA = "c3_p7r123_y02_b00_paired_timing_v1"
SEED = 0
ROUNDS = 7
HERE = Path(__file__).resolve().parent
C3 = HERE / "report_out" / "stage_tile_cotuning" / "c3_dma_residency_autotune"
P7 = C3 / "07_grouped_holdout"
DEFAULT_OUTPUT = P7 / "20260912_p7r123_y02_b00_paired_timing_run01"


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def select_entries(p7r122_contract, p7r122_rows):
    correct = {
        row["candidate_id"]
        for row in p7r122_rows
        if row["status"] == "passed"
        and len(row["seeds"]) == 3
        and all(seed["correct"] for seed in row["seeds"])
    }
    selected = [
        entry
        for entry in p7r122_contract["candidates"]
        if entry["candidate_id"] in correct
        and entry["family_id"] == "Y02B00"
        and entry["public_mode"] in {"original", "weight_resident_barrier"}
    ]
    modes = {entry["public_mode"] for entry in selected}
    if len(selected) != 2 or modes != {"original", "weight_resident_barrier"}:
        raise ValueError("requires exactly the two P7R122-passed Y02B00 identities")
    return sorted(selected, key=lambda row: row["public_mode"] != "original")


def timing_orders(entries):
    original = next(row["candidate_id"] for row in entries if row["public_mode"] == "original")
    barrier = next(
        row["candidate_id"] for row in entries if row["public_mode"] == "weight_resident_barrier"
    )
    return [
        [original, barrier] if round_index % 2 == 0 else [barrier, original]
        for round_index in range(ROUNDS)
    ]


def validate_orders(orders, entries):
    ids = {row["candidate_id"] for row in entries}
    if len(orders) != 7 or any(len(order) != 2 or set(order) != ids for order in orders):
        raise ValueError("invalid paired timing orders")
    first = {candidate_id: sum(order[0] == candidate_id for order in orders) for candidate_id in ids}
    if sorted(first.values()) != [3, 4]:
        raise ValueError("seven rounds are not maximally position-balanced")
    return first


def timed_checked_call(module, device, workload, seed, functions):
    data, weight, expected = reference_data_from_workload(workload, int(seed))
    output = tvm.nd.empty(expected.shape, "int8", device)
    buffers = [
        tvm.nd.array(data, device),
        tvm.nd.array(weight, device),
        output,
    ]
    clear = functions.get("vta.runtime.profiler_clear")
    if clear is None:
        raise RuntimeError("profiler_clear is required for per-call complete profiles")
    clear()
    timer = module.time_evaluator("main", device, number=1, repeat=1)
    measured = timer(*buffers)
    latency_ms = float(measured.mean * 1000.0)
    actual = output.numpy()
    mismatch = int(np.count_nonzero(actual != expected))
    profile = json.loads(functions["vta.runtime.profiler_status"]())
    row = {
        "seed": int(seed),
        "number": 1,
        "repeat": 1,
        "latency_ms": latency_ms,
        "correct": mismatch == 0,
        "mismatch_count": mismatch,
        "expected_sha256": array_sha256(expected),
        "actual_sha256": array_sha256(actual),
        "runtime_profile_complete": profile,
    }
    if not np.isfinite(latency_ms) or latency_ms <= 0:
        raise RuntimeError("invalid timed result")
    if mismatch:
        raise RuntimeError("timed invocation wrong answer: {} mismatches".format(mismatch))
    return row


def quartiles(values):
    return {
        "values_ms": values,
        "median_ms": statistics.median(values),
        "q1_ms": float(np.percentile(values, 25)),
        "q3_ms": float(np.percentile(values, 75)),
        "iqr_ms": float(np.percentile(values, 75) - np.percentile(values, 25)),
    }


def summarize(candidate_rows, canary_rows, entries):
    by_id = {entry["candidate_id"]: [] for entry in entries}
    profiles = {entry["candidate_id"]: [] for entry in entries}
    for row in candidate_rows:
        by_id[row["candidate_id"]].append(float(row["latency_ms"]))
        profiles[row["candidate_id"]].append(row["runtime_profile_complete"])
    if any(len(values) != ROUNDS for values in by_id.values()):
        raise ValueError("each identity requires exactly seven samples")
    stats = {candidate_id: quartiles(values) for candidate_id, values in by_id.items()}
    by_mode = {entry["public_mode"]: entry["candidate_id"] for entry in entries}
    original_id = by_mode["original"]
    barrier_id = by_mode["weight_resident_barrier"]
    original = stats[original_id]["median_ms"]
    barrier = stats[barrier_id]["median_ms"]
    paired = []
    for round_index in range(ROUNDS):
        values = {
            row["candidate_id"]: row["latency_ms"]
            for row in candidate_rows
            if row["round"] == round_index
        }
        paired.append(
            {
                "round": round_index,
                "original_ms": values[original_id],
                "barrier_ms": values[barrier_id],
                "barrier_improvement_fraction": (
                    values[original_id] - values[barrier_id]
                )
                / values[original_id],
                "barrier_wins": values[barrier_id] < values[original_id],
            }
        )
    def invariant_profile(candidate_id, field):
        values = [profile.get(field) for profile in profiles[candidate_id]]
        return {"values": values, "constant": len(set(values)) == 1, "median": statistics.median(values)}

    dma = {}
    for mode, candidate_id in by_mode.items():
        dma[mode] = {
            field: invariant_profile(candidate_id, field)
            for field in (
                "load_buffer_2d_bytes",
                "load_buffer_2d_calls",
                "load_buffer_2d_inp_bytes",
                "load_buffer_2d_wgt_bytes",
                "store_buffer_2d_bytes",
                "store_buffer_2d_calls",
                "driver_run_insns",
                "synchronize_insns",
                "synchronize_calls",
            )
        }
    canary_values = [float(row["latency_ms"]) for row in canary_rows]
    bracket_drift = []
    for round_index in range(ROUNDS):
        rows = [row for row in canary_rows if row["round"] == round_index]
        phases = {row["bracket"]: row for row in rows}
        if set(phases) != {"before", "after"}:
            raise ValueError("incomplete W05 canary bracket")
        before = phases["before"]["latency_ms"]
        after = phases["after"]["latency_ms"]
        bracket_drift.append(
            {"round": round_index, "before_ms": before, "after_ms": after, "fraction": (after - before) / before}
        )
    return {
        "candidate_stats": stats,
        "same_tile": {
            "original_id": original_id,
            "weight_resident_barrier_id": barrier_id,
            "original_median_ms": original,
            "barrier_median_ms": barrier,
            "barrier_improvement_fraction": (original - barrier) / original,
            "barrier_round_wins": sum(row["barrier_wins"] for row in paired),
            "paired_rounds": paired,
        },
        "dma_and_sync": dma,
        "w05_health_canary": {
            **quartiles(canary_values),
            "samples": len(canary_rows),
            "bracket_drift": bracket_drift,
            "candidate_or_training_role": False,
            "different_geometry_not_a_performance_baseline": True,
        },
    }


def run(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable P7R123 output {}".format(output))
    p7r122_ledger = verify_run(args.p7r122_dir)
    p7r122_contract = load_json(Path(args.p7r122_dir) / "contract.json")
    p7r122_summary = load_json(Path(args.p7r122_dir) / "summary.json")
    if not p7r122_summary["w05_health_gate"]["passed"]:
        raise ValueError("P7R122 health gate did not pass")
    correctness = load_jsonl(Path(args.p7r122_dir) / "candidate_correctness.jsonl")
    entries = select_entries(p7r122_contract, correctness)
    orders = timing_orders(entries)
    first_counts = validate_orders(orders, entries)
    p7r120, p7r120_ledger = verify_contract(args.p7r120_contract)
    health = p7r122_contract["health_canary"]
    cross = load_json(Path(args.p7r122_dir) / "fresh_cross_certificates.json")
    contract = {
        "schema": SCHEMA,
        "status": "frozen_before_rpc_timing",
        "inputs": {
            "p7r122_dir": str(Path(args.p7r122_dir).resolve()),
            "p7r122_artifact_ledger_sha256": p7r122_ledger,
            "p7r120_contract_artifact_ledger_sha256": p7r120_ledger,
        },
        "source_guards_sha256": p7r120["fresh_hardware_certificate_v2"]["source_guards_sha256"],
        "health_canary": {
            "entry": health,
            "fresh_p7r122_certificate": cross["health_canary"],
            "role": "independent before/after health canary only",
            "candidate_or_training_role": False,
            "performance_baseline_role": False,
        },
        "candidates": [
            {
                **entry,
                "fresh_p7r122_certificate": cross["candidates"][entry["candidate_id"]],
            }
            for entry in entries
        ],
        "candidate_orders": orders,
        "first_position_counts": first_counts,
        "protocol": {
            "rounds": ROUNDS,
            "seed": SEED,
            "number": 1,
            "repeat": 1,
            "each_round": "W05 before, two candidates in frozen AB/BA order, W05 after",
            "each_call": "new input/weight/output, profiler_clear, timed call, exact output check, complete profiler snapshot",
            "failure_policy": "stop immediately on any error or mismatch and retain partial records",
            "latency_claim": "same-tile Y02B00 pair only",
            "tophub_equivalence_claim": False,
            "ssh_restart_reconfiguration_persistent_write": "forbidden",
            "transport": "direct RPC ephemeral upload/load/remove only",
            "boot_id": "unknown_not_exposed_by_rpc",
        },
        "rpc_endpoint": {"host": args.host, "port": args.port},
    }
    if args.host != p7r120["rpc_contract"]["host"] or args.port != p7r120["rpc_contract"]["port"]:
        raise ValueError("RPC endpoint differs from frozen contract")
    output.mkdir(parents=True)
    write_json(output / "contract.json", contract)
    write_json(output / "pre_observation_hashes.json", {"artifacts": {"contract.json": sha256_file(output / "contract.json")}})

    temporary = None
    candidate_rows = []
    canary_rows = []
    try:
        env = vta.get_env()
        if env.TARGET != "axu5evb":
            raise RuntimeError("P7R123 requires TARGET=axu5evb")
        temporary = tempfile.TemporaryDirectory(prefix="c3_p7r123_cross_")
        directory = Path(temporary.name)
        all_entries = [health] + entries
        expected = {health["candidate_id"]: cross["health_canary"]}
        expected.update({entry["candidate_id"]: cross["candidates"][entry["candidate_id"]] for entry in entries})
        for entry in all_entries:
            certificate = build_and_export(entry, env, directory, expected[entry["candidate_id"]]["tir_sha256"])
            if certificate["binary_sha256"] != expected[entry["candidate_id"]]["binary_sha256"]:
                raise RuntimeError("fresh binary differs from P7R122 certificate")

        remote = rpc.connect(args.host, args.port, session_timeout=args.session_timeout)
        inventory, functions = runtime_inventory(remote)
        if "vta.runtime.profiler_clear" not in functions:
            raise RuntimeError("profiler_clear is required")
        write_json(output / "rpc_inventory.json", inventory)
        device = remote.ext_dev(0)
        loaded = {}
        for entry in all_entries:
            loaded[entry["candidate_id"]] = load_ephemeral(
                remote, directory / (entry["candidate_id"] + ".so")
            )
        by_id = {entry["candidate_id"]: entry for entry in entries}
        with (output / "candidate_timing.jsonl").open("x", encoding="utf-8") as cstream, (
            output / "w05_health_canary_timing.jsonl"
        ).open("x", encoding="utf-8") as hstream:
            sequence = 0
            for round_index, order in enumerate(orders):
                for bracket in ("before",):
                    row = timed_checked_call(
                        loaded[health["candidate_id"]], device, health["identity"]["workload"], SEED, functions
                    )
                    row.update(round=round_index, bracket=bracket, role="health_canary", candidate_or_training_label=False)
                    canary_rows.append(row)
                    hstream.write(json.dumps(row, sort_keys=True) + "\n"); hstream.flush()
                for position, candidate_id in enumerate(order):
                    entry = by_id[candidate_id]
                    row = timed_checked_call(
                        loaded[candidate_id], device, entry["identity"]["workload"], SEED, functions
                    )
                    row.update(
                        schema=SCHEMA,
                        sequence=sequence,
                        round=round_index,
                        position=position,
                        candidate_id=candidate_id,
                        family_id=entry["family_id"],
                        public_mode=entry["public_mode"],
                    )
                    candidate_rows.append(row)
                    cstream.write(json.dumps(row, sort_keys=True) + "\n"); cstream.flush()
                    sequence += 1
                row = timed_checked_call(
                    loaded[health["candidate_id"]], device, health["identity"]["workload"], SEED, functions
                )
                row.update(round=round_index, bracket="after", role="health_canary", candidate_or_training_label=False)
                canary_rows.append(row)
                hstream.write(json.dumps(row, sort_keys=True) + "\n"); hstream.flush()
                print("round {}/7 complete".format(round_index + 1), flush=True)

        analysis = summarize(candidate_rows, canary_rows, entries)
        summary = {
            "schema": SCHEMA,
            "status": "completed_correctness_preserving_paired_timing",
            "candidate_samples": len(candidate_rows),
            "w05_health_canary_samples": len(canary_rows),
            "all_timed_calls_correct": True,
            "analysis": analysis,
            "ssh_used": False,
            "board_restart_or_reconfiguration": False,
            "persistent_board_write": False,
            "boot_id": "unknown_not_exposed_by_rpc",
            "claim_boundary": "same-tile Y02B00 pair only; W05 is health canary, not TopHub-equivalence baseline",
        }
        write_json(output / "summary.json", summary)
        (output / "command.txt").write_text(
            " ".join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]) + "\n",
            encoding="utf-8",
        )
        finalize(output)
        print(json.dumps(summary, indent=2, sort_keys=True))
    except Exception as error:
        write_json(
            output / "failure.json",
            {
                "status": "failed_closed",
                "exception_type": type(error).__name__,
                "message": (str(error) or type(error).__name__)[:4000],
                "traceback": traceback.format_exc()[-12000:],
                "candidate_samples_completed": len(candidate_rows),
                "w05_canary_samples_completed": len(canary_rows),
                "ssh_used": False,
            },
        )
        finalize(output)
        raise
    finally:
        if temporary is not None:
            temporary.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p7r122-dir", type=Path, default=DEFAULT_P7R122)
    parser.add_argument("--p7r120-contract", type=Path, default=DEFAULT_P7R120)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--session-timeout", type=int, default=120)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
