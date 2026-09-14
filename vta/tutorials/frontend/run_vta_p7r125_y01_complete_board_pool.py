#!/usr/bin/env python3
"""P7R125: health-gated six-point Y01 correctness and complete-pool timing."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import tempfile
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
from tvm import rpc
import vta

import vta.top.vta_conv2d_residency  # pylint: disable=unused-import
from prepare_vta_p7r120_yolo_rpc_contract import build_and_export, load_json, sha256_file, write_json
from run_vta_p7r120_yolo_rpc_board import load_ephemeral, runtime_inventory
from run_vta_p7r121_tophub_adapter_diagnostic import board_seed, finalize
from run_vta_p7r122_y02_health_gated_correctness import (
    DEFAULT_OUTPUT as DEFAULT_P7R122,
    verify_run,
)
from run_vta_p7r123_y02_b00_paired_timing import timed_checked_call
from run_vta_p7r124_y01_v2_local_pool import DEFAULT_OUTPUT as DEFAULT_P7R124


SCHEMA = "c3_p7r125_y01_complete_board_pool_v1"
SEEDS = (0, 20250901, 20260910)
TIMING_SEED = 0
ORDER_SEED = 20260912
ROUNDS = 7
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
C3 = HERE / "report_out" / "stage_tile_cotuning" / "c3_dma_residency_autotune"
P7 = C3 / "07_grouped_holdout"
DEFAULT_OUTPUT = P7 / "20260912_p7r125_y01_complete_board_pool_run01"


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def select_six(p7r124_dir):
    selection = load_json(Path(p7r124_dir) / "board_family_selection.json")
    selected = selection["selected_for_future_complete_board_pool"]
    if selected is None or selected["pre_registered_rank"] != 0:
        raise ValueError("P7R124 did not freeze a rank0 structural pair")
    families = set(selected["families"])
    candidates = [
        row
        for row in load_jsonl(Path(p7r124_dir) / "candidates_v2.jsonl")
        if row["family_id"] in families
    ]
    static = {
        row["candidate_id"]: row
        for row in load_jsonl(Path(p7r124_dir) / "static_results.jsonl")
    }
    fsim = {
        row["candidate_id"]: row
        for row in load_jsonl(Path(p7r124_dir) / "fsim_results.jsonl")
    }
    expected_modes = {"original", "input_stationary", "weight_resident_barrier"}
    if len(candidates) != 6:
        raise ValueError("requires exactly two complete three-mode families")
    for family in families:
        if {row["public_mode"] for row in candidates if row["family_id"] == family} != expected_modes:
            raise ValueError("selected family is not a complete three-mode family")
    for row in candidates:
        if static[row["candidate_id"]]["status"] != "ok":
            raise ValueError("candidate lacks P7R124 static pass")
        qualification = fsim[row["candidate_id"]]
        if qualification["status"] != "passed" or len(qualification["seeds"]) != 3 or not all(
            seed["correct"] for seed in qualification["seeds"]
        ):
            raise ValueError("candidate lacks P7R124 three-seed FSim pass")
    mode_order = {mode: index for index, mode in enumerate(expected_modes)}
    # Do not depend on set iteration for dispatch ordering.
    fixed_modes = {"original": 0, "input_stationary": 1, "weight_resident_barrier": 2}
    del mode_order
    return sorted(candidates, key=lambda row: (row["family_id"], fixed_modes[row["public_mode"]]))


def candidate_orders(candidates):
    base = [row["candidate_id"] for row in candidates]
    random.Random(ORDER_SEED).shuffle(base)
    return [base[offset:] + base[:offset] for offset in range(6)] + [list(base)]


def verify_orders(orders, candidates):
    ids = {row["candidate_id"] for row in candidates}
    if len(orders) != 7 or any(len(order) != 6 or set(order) != ids for order in orders):
        raise ValueError("invalid equal-budget seven-round order")
    for position in range(6):
        if {orders[round_index][position] for round_index in range(6)} != ids:
            raise ValueError("first six rounds are not Latin-position balanced")
    return {
        candidate_id: {
            "samples": sum(candidate_id in order for order in orders),
            "first_positions": sum(order[0] == candidate_id for order in orders),
        }
        for candidate_id in ids
    }


def stats(values):
    return {
        "values_ms": values,
        "median_ms": statistics.median(values),
        "q1_ms": float(np.percentile(values, 25)),
        "q3_ms": float(np.percentile(values, 75)),
        "iqr_ms": float(np.percentile(values, 75) - np.percentile(values, 25)),
    }


def summarize(timing_rows, canary_rows, candidates):
    by_id = defaultdict(list)
    profiles = defaultdict(list)
    for row in timing_rows:
        by_id[row["candidate_id"]].append(float(row["latency_ms"]))
        profiles[row["candidate_id"]].append(row["runtime_profile_complete"])
    if len(by_id) != 6 or any(len(values) != 7 for values in by_id.values()):
        raise ValueError("incomplete six-point timing pool")
    candidate_stats = {candidate_id: stats(values) for candidate_id, values in by_id.items()}
    oracle_id = min(candidate_stats, key=lambda candidate_id: candidate_stats[candidate_id]["median_ms"])
    oracle = candidate_stats[oracle_id]["median_ms"]
    entries = {row["candidate_id"]: row for row in candidates}
    pool = {}
    for candidate_id, row in candidate_stats.items():
        pool[candidate_id] = {
            **row,
            "family_id": entries[candidate_id]["family_id"],
            "public_mode": entries[candidate_id]["public_mode"],
            "regret_fraction_vs_measured_pool_oracle": (row["median_ms"] - oracle) / oracle,
        }
    same_tile = {}
    for family_id in sorted({row["family_id"] for row in candidates}):
        modes = {
            row["public_mode"]: row["candidate_id"]
            for row in candidates
            if row["family_id"] == family_id
        }
        original = candidate_stats[modes["original"]]["median_ms"]
        same_tile[family_id] = {}
        for mode in ("input_stationary", "weight_resident_barrier"):
            median = candidate_stats[modes[mode]]["median_ms"]
            wins = 0
            paired = []
            for round_index in range(7):
                values = {
                    row["candidate_id"]: row["latency_ms"]
                    for row in timing_rows
                    if row["round"] == round_index
                }
                won = values[modes[mode]] < values[modes["original"]]
                wins += won
                paired.append(
                    {
                        "round": round_index,
                        "original_ms": values[modes["original"]],
                        "mechanism_ms": values[modes[mode]],
                        "mechanism_wins": won,
                    }
                )
            same_tile[family_id][mode] = {
                "candidate_id": modes[mode],
                "original_median_ms": original,
                "mechanism_median_ms": median,
                "improvement_fraction": (original - median) / original,
                "round_wins": wins,
                "paired": paired,
            }
    dma_sync = {}
    fields = (
        "load_buffer_2d_bytes",
        "load_buffer_2d_calls",
        "load_buffer_2d_inp_bytes",
        "load_buffer_2d_wgt_bytes",
        "store_buffer_2d_bytes",
        "store_buffer_2d_calls",
        "synchronize_calls",
        "driver_run_insns",
    )
    for candidate_id, values in profiles.items():
        dma_sync[candidate_id] = {
            "raw_profile_invocations_per_sample": 2,
            "reason": "TVM time_evaluator performs one discarded warmup plus number=1 timed invocation",
            "raw": {
                field: {
                    "values": [profile.get(field) for profile in values],
                    "constant": len({profile.get(field) for profile in values}) == 1,
                    "median": statistics.median(profile.get(field) for profile in values),
                }
                for field in fields
            },
            "normalized_additive_per_inference": {
                field: statistics.median(profile.get(field) for profile in values) / 2.0
                for field in fields
            },
        }
    canary_values = [float(row["latency_ms"]) for row in canary_rows]
    return {
        "measured_six_point_correct_pool_oracle": {
            "candidate_id": oracle_id,
            "family_id": entries[oracle_id]["family_id"],
            "public_mode": entries[oracle_id]["public_mode"],
            "median_ms": oracle,
            "scope": "exact frozen six-point correct pool only; not full ConfigSpace or TopHub oracle",
        },
        "six_point_pool": pool,
        "same_tile": same_tile,
        "dma_sync": dma_sync,
        "w05_health_canary": {
            **stats(canary_values),
            "samples": len(canary_values),
            "candidate_or_training_role": False,
            "different_geometry_not_a_performance_baseline": True,
        },
    }


def run(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable P7R125 output {}".format(output))
    p7r124_ledger = verify_run(args.p7r124_dir)
    p7r124_contract = load_json(Path(args.p7r124_dir) / "contract.json")
    if p7r124_contract["local_protocol"]["board_contact"] != "forbidden":
        raise ValueError("unexpected P7R124 local contract")
    observed_guards = {
        name: sha256_file(REPO / name)
        for name in p7r124_contract["source_guards_sha256"]
    }
    if observed_guards != p7r124_contract["source_guards_sha256"]:
        raise ValueError("P7R124 source guards changed")
    candidates = select_six(args.p7r124_dir)
    orders = candidate_orders(candidates)
    budgets = verify_orders(orders, candidates)
    p7r122_ledger = verify_run(args.p7r122_dir)
    p7r122_contract = load_json(Path(args.p7r122_dir) / "contract.json")
    health = p7r122_contract["health_canary"]
    static = {
        row["candidate_id"]: row
        for row in load_jsonl(Path(args.p7r124_dir) / "static_results.jsonl")
    }
    contract = {
        "schema": SCHEMA,
        "status": "frozen_before_cross_compile_or_rpc_observations",
        "inputs": {
            "p7r124_dir": str(Path(args.p7r124_dir).resolve()),
            "p7r124_artifact_ledger_sha256": p7r124_ledger,
            "p7r122_health_identity_ledger_sha256": p7r122_ledger,
        },
        "source_guards_sha256": p7r124_contract["source_guards_sha256"],
        "health_canary": {
            "entry": health,
            "role": "independent current-session health gate and timing bracket only",
            "candidate_or_training_role": False,
            "performance_baseline_role": False,
        },
        "candidate_pool": {
            "families": ["Y01F02", "Y01F07"],
            "modes": ["original", "input_stationary", "weight_resident_barrier"],
            "gross_identities": 6,
            "candidates": candidates,
            "no_replacement": True,
            "p7r124_tir_sha256": {row["candidate_id"]: static[row["candidate_id"]]["tir_sha256"] for row in candidates},
        },
        "correctness": {
            "seeds": list(SEEDS),
            "gross_checks": 18,
            "gate": "all six identities x all three seeds exact before any timing",
        },
        "timing": {
            "rounds": 7,
            "number": 1,
            "repeat": 1,
            "seed": TIMING_SEED,
            "orders": orders,
            "budgets": budgets,
            "first_six_rounds": "each identity appears in every position exactly once",
            "w05_bracket": "one before and one after candidates in every round",
            "output_check": "exact after every timed call",
            "profiler": "complete status; raw covers time_evaluator warmup plus one timed invocation",
        },
        "prohibitions": {
            "Y01_TopHub_for_selection_or_canary": True,
            "replacement_after_failure": True,
            "ssh_restart_reconfiguration_persistent_write": True,
        },
        "rpc": {"host": args.host, "port": args.port, "boot_id": "unknown_not_exposed_by_rpc"},
    }
    output.mkdir(parents=True)
    write_json(output / "contract.json", contract)
    write_json(output / "pre_cross_hashes.json", {"artifacts": {"contract.json": sha256_file(output / "contract.json")}})

    temporary = None
    correctness_rows = []
    timing_rows = []
    canary_rows = []
    try:
        env = vta.get_env()
        if env.TARGET != "axu5evb":
            raise RuntimeError("P7R125 requires TARGET=axu5evb")
        temporary = tempfile.TemporaryDirectory(prefix="c3_p7r125_cross_")
        directory = Path(temporary.name)
        certificates = {"health_canary": build_and_export(health, env, directory), "candidates": {}}
        for entry in candidates:
            certificates["candidates"][entry["candidate_id"]] = build_and_export(
                entry, env, directory, static[entry["candidate_id"]]["tir_sha256"]
            )
        write_json(output / "fresh_cross_certificates.json", certificates)
        write_json(
            output / "pre_rpc_hashes.json",
            {"artifacts": {
                "contract.json": sha256_file(output / "contract.json"),
                "fresh_cross_certificates.json": sha256_file(output / "fresh_cross_certificates.json"),
            }},
        )

        remote = rpc.connect(args.host, args.port, session_timeout=args.session_timeout)
        inventory, functions = runtime_inventory(remote)
        if "vta.runtime.profiler_clear" not in functions:
            raise RuntimeError("profiler_clear is required")
        write_json(output / "rpc_inventory.json", inventory)
        device = remote.ext_dev(0)
        health_module = load_ephemeral(remote, directory / (health["candidate_id"] + ".so"))
        health_checks = []
        for seed in SEEDS:
            row = board_seed(health_module["main"], device, health["identity"]["workload"], seed, functions)
            health_checks.append(row)
            if not row.get("correct"):
                break
        health_pass = len(health_checks) == 3 and all(row["correct"] for row in health_checks)
        write_json(output / "health_gate.json", {"status": "passed" if health_pass else "failed", "seeds": health_checks})
        if not health_pass:
            raise RuntimeError("W05 health canary failed; stop before Y01 candidate dispatch")

        loaded = {}
        for entry in candidates:
            loaded[entry["candidate_id"]] = load_ephemeral(remote, directory / (entry["candidate_id"] + ".so"))
        with (output / "correctness.jsonl").open("x", encoding="utf-8") as stream:
            for entry in candidates:
                seeds = [
                    board_seed(
                        loaded[entry["candidate_id"]]["main"], device, entry["identity"]["workload"], seed, functions
                    )
                    for seed in SEEDS
                ]
                row = {
                    "candidate_id": entry["candidate_id"],
                    "family_id": entry["family_id"],
                    "public_mode": entry["public_mode"],
                    "status": "passed" if all(seed.get("correct") for seed in seeds) else "failed",
                    "seeds": seeds,
                    "performance_measurement": "not_collected",
                }
                correctness_rows.append(row)
                stream.write(json.dumps(row, sort_keys=True) + "\n"); stream.flush()
                print("correctness {} {} {}".format(entry["family_id"], entry["public_mode"], row["status"]), flush=True)
        global_gate = len(correctness_rows) == 6 and all(row["status"] == "passed" for row in correctness_rows)
        if global_gate:
            by_id = {entry["candidate_id"]: entry for entry in candidates}
            with (output / "timing.jsonl").open("x", encoding="utf-8") as stream, (
                output / "w05_health_canary_timing.jsonl"
            ).open("x", encoding="utf-8") as hstream:
                sequence = 0
                for round_index, order in enumerate(orders):
                    row = timed_checked_call(
                        health_module, device, health["identity"]["workload"], TIMING_SEED, functions
                    )
                    row.update(round=round_index, bracket="before", candidate_or_training_label=False)
                    canary_rows.append(row); hstream.write(json.dumps(row, sort_keys=True) + "\n"); hstream.flush()
                    for position, candidate_id in enumerate(order):
                        entry = by_id[candidate_id]
                        row = timed_checked_call(
                            loaded[candidate_id], device, entry["identity"]["workload"], TIMING_SEED, functions
                        )
                        row.update(
                            schema=SCHEMA, sequence=sequence, round=round_index, position=position,
                            candidate_id=candidate_id, family_id=entry["family_id"], public_mode=entry["public_mode"],
                            raw_profile_invocations=2,
                        )
                        timing_rows.append(row); stream.write(json.dumps(row, sort_keys=True) + "\n"); stream.flush(); sequence += 1
                    row = timed_checked_call(
                        health_module, device, health["identity"]["workload"], TIMING_SEED, functions
                    )
                    row.update(round=round_index, bracket="after", candidate_or_training_label=False)
                    canary_rows.append(row); hstream.write(json.dumps(row, sort_keys=True) + "\n"); hstream.flush()
                    print("timing round {}/7 complete".format(round_index + 1), flush=True)
        status = "timing_completed" if global_gate else "correctness_gate_failed_no_timing"
        summary = {
            "schema": SCHEMA,
            "status": status,
            "w05_health_gate": {"passed": True, "seed_checks": 3},
            "correctness": {
                "identities": len(correctness_rows),
                "passed": sum(row["status"] == "passed" for row in correctness_rows),
                "seed_checks": sum(len(row["seeds"]) for row in correctness_rows),
                "global_timing_gate": global_gate,
            },
            "candidate_timing_samples": len(timing_rows),
            "w05_canary_timing_samples": len(canary_rows),
            "analysis": summarize(timing_rows, canary_rows, candidates) if global_gate else None,
            "ssh_used": False,
            "board_restart_or_reconfiguration": False,
            "persistent_board_write": False,
            "boot_id": "unknown_not_exposed_by_rpc",
            "claim_boundary": "exact Y01 six-point pool only; no full-ConfigSpace or TopHub-equivalence claim",
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
                "status": "failed_closed", "exception_type": type(error).__name__,
                "message": (str(error) or type(error).__name__)[:4000],
                "traceback": traceback.format_exc()[-12000:],
                "correctness_identities_completed": len(correctness_rows),
                "timing_samples_completed": len(timing_rows), "ssh_used": False,
            },
        )
        finalize(output)
        raise
    finally:
        if temporary is not None:
            temporary.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p7r124-dir", type=Path, default=DEFAULT_P7R124)
    parser.add_argument("--p7r122-dir", type=Path, default=DEFAULT_P7R122)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--session-timeout", type=int, default=120)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
