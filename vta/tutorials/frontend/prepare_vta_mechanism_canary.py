#!/usr/bin/env python3
"""Freeze an offline VTA mechanism-canary contract from qualified local evidence.

This planner selects exactly three mechanisms per workload: the protected original
incumbent, the best qualified input-stationary candidate, and the best qualified
mode-4 weight-barrier candidate.  It contains no RPC, board, or timing execution.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import sys
from pathlib import Path

from c3_candidate_identity import canonical_json_bytes


SCHEMA = "c3_p6b_mechanism_canary_manifest_v1"
PLAN_SCHEMA = "c3_p6b_mechanism_canary_dispatch_plan_v1"
WORKLOADS = ("W00", "W02", "W09")
SEEDS = (0, 20250901, 20260910)
ORDER_SEED = 20260911
WARMUP_RUNS = 3
TIMED_REPEATS = 30


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _ledger_files(ledger):
    if "output_sha256" in ledger:
        return ledger["output_sha256"]
    if "artifacts" in ledger:
        return ledger["artifacts"]
    return ledger


def load_frozen_run(run_dir):
    run_dir = Path(run_dir)
    ledger_path = run_dir / "artifact_hashes.json"
    ledger = load_json(ledger_path)
    hashes = _ledger_files(ledger)
    for name in ("results.jsonl", "summary.json", "manifest.json"):
        if name not in hashes or file_sha256(run_dir / name) != hashes[name]:
            raise ValueError("frozen run hash mismatch for {} {}".format(run_dir, name))
    return {
        "run_dir": run_dir,
        "results": load_jsonl(run_dir / "results.jsonl"),
        "summary": load_json(run_dir / "summary.json"),
        "manifest": load_json(run_dir / "manifest.json"),
        "provenance": {
            "run_dir": str(run_dir),
            "results_sha256": file_sha256(run_dir / "results.jsonl"),
            "summary_sha256": file_sha256(run_dir / "summary.json"),
            "manifest_sha256": file_sha256(run_dir / "manifest.json"),
            "artifact_hashes_sha256": file_sha256(ledger_path),
        },
    }


def _by_id(rows, label):
    result = {row["candidate_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("{} contains duplicate candidate IDs".format(label))
    return result


def _verify_identity(record):
    expected = hashlib.sha256(canonical_json_bytes(record["identity"])).hexdigest()
    if expected != record["candidate_id"]:
        raise ValueError("stable candidate identity mismatch: {}".format(record["candidate_id"]))


def _selection_key(record, memory):
    change = record.get("relative_to_same_tile_original")
    if not change or change[memory]["reduction_fraction"] is None:
        raise ValueError("candidate lacks target DMA reduction")
    calls_key = "load_buffer_2d_{}_calls".format(memory)
    calls = record["transfer_signature"]["totals"][calls_key]
    return (-float(change[memory]["reduction_fraction"]), int(calls), record["candidate_id"])


def choose_best(candidates, memory):
    if not candidates:
        raise ValueError("no qualified {} mechanism candidate".format(memory))
    return min(candidates, key=lambda record: _selection_key(record, memory))


def _full_config(record):
    config = dict(record["identity"]["complete_config_entity"])
    config["index"] = int(record["debug"]["config_index"])
    if set(config) != {"index", "code_hash", "entity"} or not config["entity"]:
        raise ValueError("candidate lacks complete ConfigEntity")
    return config


def validate_and_select(p4b, p4e, p4f, p4g, p4i=None):
    p4b_by_id = _by_id(p4b["results"], "P4b")
    p4e_by_id = _by_id(p4e["results"], "P4e")
    p4f_by_id = _by_id(p4f["results"], "P4f")
    p4g_by_id = _by_id(p4g["results"], "P4g")
    p4i_by_id = _by_id(p4i["results"], "P4i") if p4i else {}

    if p4e["summary"].get("unqualified_lower_success") != 0:
        raise ValueError("P4e has unqualified lower-success candidates")
    if p4f["summary"].get("passed") != 165 or p4f["summary"].get("failed") != 0:
        raise ValueError("P4f production candidate qualification is incomplete")
    if p4g["summary"].get("lower_and_dependency_ok") != 32:
        raise ValueError("P4g static/dependency qualification is incomplete")
    if p4g["summary"].get("fsim_status_counts", {}).get("passed") != 32:
        raise ValueError("P4g FSim qualification is incomplete")
    if p4i and (p4i["summary"].get("passed") != 32 or p4i["summary"].get("failed") != 0):
        raise ValueError("P4i barrier AXU qualification is incomplete")

    selected = {}
    for workload_id in WORKLOADS:
        incumbents = [
            row
            for row in p4b["results"]
            if row["workload_id"] == workload_id
            and row["candidate_role"] == "protected_original_incumbent"
            and row["status"] == "ok"
        ]
        if len(incumbents) != 1:
            raise ValueError("expected one successful incumbent for {}".format(workload_id))
        incumbent = incumbents[0]
        input_candidates = [
            row
            for row in p4b["results"]
            if row["workload_id"] == workload_id
            and row["residence_mode"] == "input_stationary"
            and row["status"] == "ok"
            and p4e_by_id.get(row["candidate_id"], {}).get("local_status") == "fsim_passed"
            and p4f_by_id.get(row["candidate_id"], {}).get("overall_status") == "passed"
        ]
        barrier_candidates = [
            row
            for row in p4g["results"]
            if row["workload_id"] == workload_id
            and row["residence_mode"] == "weight_stationary_barrier"
            and row["status"] == "ok"
            and row["dependency_audit"]["valid"]
            and row["fsim_qualification"]["overall_status"] == "passed"
        ]
        input_best = choose_best(input_candidates, "inp")
        barrier_best = choose_best(barrier_candidates, "wgt")
        for record in (incumbent, input_best, barrier_best):
            _verify_identity(record)
        for record in (incumbent, input_best):
            candidate_id = record["candidate_id"]
            qualification = p4e_by_id.get(candidate_id)
            cross_compile = p4f_by_id.get(candidate_id)
            if (
                not qualification
                or qualification.get("local_status") != "fsim_passed"
                or tuple(qualification.get("correctness_seeds", ())) != SEEDS
                or qualification.get("tir_sha256") != record["tir_sha256"]
                or not cross_compile
                or cross_compile.get("overall_status") != "passed"
                or cross_compile.get("tir_sha256") != record["tir_sha256"]
            ):
                raise ValueError("P4e/P4f qualification mismatch for {}".format(candidate_id))

        entries = []
        for role, record, memory in (
            ("original_incumbent", incumbent, None),
            ("input_stationary_mechanism", input_best, "inp"),
            ("weight_barrier_mechanism", barrier_best, "wgt"),
        ):
            candidate_id = record["candidate_id"]
            entry = {
                "mechanism_role": role,
                "candidate_id": candidate_id,
                "workload_id": workload_id,
                "residence_mode": record["residence_mode"],
                "template_name": record["identity"]["template_name"],
                "schedule_version": record["identity"]["schedule_version"],
                "schedule_source_sha256": record["identity"].get("schedule_source_sha256"),
                "qualification_source_guards_sha256": (
                    p4i["manifest"].get("source_guards_after")
                    if role == "weight_barrier_mechanism" and p4i
                    else p4f["manifest"].get("source_guards_after")
                ),
                "config_index": record["debug"]["config_index"],
                "complete_config_entity": _full_config(record),
                "tir_sha256": record["tir_sha256"],
                "selection_metric": None,
                "local_fsim_evidence": None,
                "axu_cross_compile_evidence": None,
            }
            if memory:
                entry["selection_metric"] = {
                    "target_memory": "input" if memory == "inp" else "weight",
                    "dma_reduction_fraction": record["relative_to_same_tile_original"][memory][
                        "reduction_fraction"
                    ],
                    "candidate_bytes": record["relative_to_same_tile_original"][memory][
                        "candidate_bytes"
                    ],
                    "original_bytes": record["relative_to_same_tile_original"][memory][
                        "original_bytes"
                    ],
                    "target_load_calls": record["transfer_signature"]["totals"][
                        "load_buffer_2d_{}_calls".format(memory)
                    ],
                    "tie_break": "max reduction, fewer target LOAD calls, stable candidate_id",
                }
            if role == "weight_barrier_mechanism":
                qualification = record["fsim_qualification"]
                entry["local_fsim_evidence"] = {
                    "source": "P4g results.jsonl",
                    "status": qualification["overall_status"],
                    "seeds": [seed["seed"] for seed in qualification["seeds"]],
                    "tir_hash_matches": qualification["tir_hash_matches_static"],
                }
                if p4i is None:
                    entry["axu_cross_compile_evidence"] = {
                        "status": "pending",
                        "reason": "P4i barrier AXU qualification absent at contract freeze",
                    }
                else:
                    p4i_record = p4i_by_id.get(candidate_id)
                    if (
                        not p4i_record
                        or p4i_record.get("overall_status") != "passed"
                        or p4i_record.get("p4g_tir_sha256") != record["tir_sha256"]
                        or p4i_record.get("relowered_tir_sha256") != record["tir_sha256"]
                        or p4i_record.get("tir_hash_check") != "passed"
                        or p4i_record.get("stable_id_check") != "passed"
                        or p4i_record.get("source_hash_check") != "passed"
                    ):
                        raise ValueError("selected barrier lacks passing P4i qualification")
                    entry["axu_cross_compile_evidence"] = {
                        "source": "P4i results.jsonl",
                        "status": "passed",
                        "build_status": p4i_record["build"]["status"],
                        "export_status": p4i_record["export"]["status"],
                        "binary_sha256": p4i_record["export"]["binary_sha256"],
                        "tir_sha256": p4i_record["p4g_tir_sha256"],
                    }
            else:
                q = p4e_by_id[candidate_id]
                cc = p4f_by_id[candidate_id]
                entry["local_fsim_evidence"] = {
                    "source": "P4e results.jsonl",
                    "status": q["local_status"],
                    "seeds": q["correctness_seeds"],
                    "tir_sha256": q["tir_sha256"],
                }
                entry["axu_cross_compile_evidence"] = {
                    "source": "P4f results.jsonl",
                    "status": cc["overall_status"],
                    "qualification_source": cc["qualification_source"],
                    "build_status": cc["build"]["status"],
                    "export_status": cc["export"]["status"],
                    "binary_sha256": cc["export"]["binary_sha256"],
                    "tir_sha256": cc["tir_sha256"],
                }
            entries.append(entry)
        ids = [entry["candidate_id"] for entry in entries]
        if len(ids) != 3 or len(set(ids)) != 3:
            raise ValueError("mechanism canary requires exactly three unique candidates")
        selected[workload_id] = entries
    return selected


def _balanced_orders(items, repeats, seed_material):
    permutations = list(itertools.permutations(items))
    if repeats % len(permutations):
        raise ValueError("repeats must be divisible by complete permutation count")
    orders = permutations * (repeats // len(permutations))
    seed = int(hashlib.sha256(seed_material.encode()).hexdigest()[:16], 16)
    random.Random(seed).shuffle(orders)
    return [list(order) for order in orders]


def build_dispatch_plan(selected):
    candidate_orders = {
        workload_id: _balanced_orders(
            [entry["candidate_id"] for entry in selected[workload_id]],
            TIMED_REPEATS,
            "{}:{}:candidates".format(ORDER_SEED, workload_id),
        )
        for workload_id in WORKLOADS
    }
    workload_orders = _balanced_orders(
        list(WORKLOADS), TIMED_REPEATS, "{}:workloads".format(ORDER_SEED)
    )
    rounds = [
        {
            "round": index,
            "workload_order": workload_orders[index],
            "candidate_order_by_workload": {
                workload_id: candidate_orders[workload_id][index] for workload_id in WORKLOADS
            },
        }
        for index in range(TIMED_REPEATS)
    ]
    return {
        "schema": PLAN_SCHEMA,
        "board_executed": False,
        "scope": "future_board_protocol_only",
        "correctness": {
            "seeds": list(SEEDS),
            "oracle": "independent NumPy reference",
            "requirement": "exact output equality for every seed before timing",
            "failed_candidate_policy": "record failure and do not use latency as valid",
        },
        "ordering": {
            "strategy": "seeded randomized complete crossover",
            "seed": ORDER_SEED,
            "timed_rounds": TIMED_REPEATS,
            "balance": "all six candidate permutations and all six workload permutations repeated five times",
            "rounds": rounds,
        },
        "timing": {
            "warmup_runs_per_candidate": WARMUP_RUNS,
            "timed_repeats_per_candidate": TIMED_REPEATS,
            "timer_number": 1,
            "record_all_samples": True,
            "primary_statistic": "median latency_ms",
            "compile_upload_setup_excluded": True,
        },
        "runtime_counters": {
            "collection": "snapshot deltas normalized per inference for every timed round",
            "required": [
                "load_buffer_2d_calls",
                "load_buffer_2d_bytes",
                "load_buffer_2d_inp_bytes",
                "load_buffer_2d_wgt_bytes",
                "load_buffer_2d_small_calls",
                "load_buffer_2d_strided_calls",
                "load_buffer_2d_padded_calls",
                "store_buffer_2d_calls",
                "store_buffer_2d_bytes",
                "driver_run_insns",
                "synchronize_insns",
            ],
        },
        "decision_rule": {
            "comparison": "within-workload mechanism candidate versus original incumbent median latency",
            "relative_improvement": "(incumbent_median - candidate_median) / incumbent_median",
            "equivalent": "absolute relative latency difference <= 0.02",
            "improved": "relative improvement >= 0.05",
            "regressed": "relative improvement <= -0.05",
            "indeterminate_band": "all remaining values; do not round into a claim",
            "correctness_precondition": "all three board seeds pass",
        },
        "preflight": {
            "actual_board_fingerprint": "must be recorded and matched before execution",
            "barrier_axu_qualification": "must be passed before dispatch; pending contracts are not executable",
        },
    }


def prepare_contract(p4b_dir, p4e_dir, p4f_dir, p4g_dir, p4i_dir=None):
    runs = {
        "P4b": load_frozen_run(p4b_dir),
        "P4e": load_frozen_run(p4e_dir),
        "P4f": load_frozen_run(p4f_dir),
        "P4g": load_frozen_run(p4g_dir),
    }
    p4i = load_frozen_run(p4i_dir) if p4i_dir else None
    selected = validate_and_select(runs["P4b"], runs["P4e"], runs["P4f"], runs["P4g"], p4i)
    manifest = {
        "schema": SCHEMA,
        "status": "frozen_offline_mechanism_canary",
        "board_executed": False,
        "contract_kind": "mechanism_canary_not_generic_B7_exploration",
        "selection_rule": {
            "roles": [
                "protected original incumbent",
                "qualified input-stationary with maximum static input-byte reduction",
                "qualified mode-4 barrier with maximum static weight-byte reduction",
            ],
            "tie_break": "fewer target LOAD calls, then stable candidate_id",
        },
        "workloads": selected,
        "source_runs": {name: run["provenance"] for name, run in runs.items()},
        "p4i_barrier_axu": (
            p4i["provenance"] if p4i else {"status": "pending", "reason": "not present at freeze"}
        ),
        "claim_boundary": {
            "board_performance_measured": False,
            "board_correctness_executed": False,
            "generic_B7_effectiveness": False,
            "G6_passed": False,
        },
    }
    return manifest, build_dispatch_plan(selected)


def _write_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError("refusing to overwrite frozen artifact: {}".format(path))
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ensure_empty_output_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError("refusing to write into non-empty frozen output directory")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p4b-run-dir", required=True)
    parser.add_argument("--p4e-run-dir", required=True)
    parser.add_argument("--p4f-run-dir", required=True)
    parser.add_argument("--p4g-run-dir", required=True)
    parser.add_argument("--p4i-run-dir")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = ensure_empty_output_dir(args.output_dir)
    manifest, plan = prepare_contract(
        args.p4b_run_dir,
        args.p4e_run_dir,
        args.p4f_run_dir,
        args.p4g_run_dir,
        args.p4i_run_dir,
    )
    _write_new(output / "mechanism_canary_manifest.json", manifest)
    _write_new(output / "dispatch_plan.json", plan)
    (output / "command.txt").write_text(" ".join([sys.executable] + sys.argv) + "\n")
    stdout = "workloads=3 candidates=9 p4i={} board=false timing_executed=false\n".format(
        "validated" if args.p4i_run_dir else "pending"
    )
    (output / "stdout.log").write_text(stdout)
    (output / "stderr.log").write_text("")
    preregistered = {
        "schema": "c3_p6b_preregistered_v1",
        "contract_kind": "mechanism canary, explicitly not generic B7 exploration",
        "selection_rule": manifest["selection_rule"],
        "future_protocol": {
            key: plan[key] for key in ("correctness", "ordering", "timing", "runtime_counters", "decision_rule", "preflight")
        },
        "claims_forbidden": [
            "board execution by this run",
            "latency measured by this run",
            "P4g static DMA proves speed",
            "pending P4i AXU qualification treated as passed",
            "generic B7 effectiveness",
            "G6 passed",
        ],
    }
    _write_new(output / "preregistered.json", preregistered)
    run_manifest = {
        "schema": "c3_p6b_offline_run_manifest_v1",
        "date": "2026-09-11",
        "tool": str(Path(__file__).resolve()),
        "tool_sha256": file_sha256(Path(__file__)),
        "inputs": manifest["source_runs"],
        "p4i_barrier_axu": manifest["p4i_barrier_axu"],
        "outputs": ["mechanism_canary_manifest.json", "dispatch_plan.json"],
        "board_executed": False,
        "performance_timing_collected": False,
    }
    _write_new(output / "manifest.json", run_manifest)
    (output / "STATUS.md").write_text(
        "# C3-P6b offline mechanism canary\n\nStatus: **frozen_offline_mechanism_canary**. "
        "W00/W02/W09 each contain exactly original, best qualified input-stationary, and "
        "best qualified weight-barrier candidates. P4i barrier AXU qualification is {}. "
        "No SSH/RPC/board access or timing occurred. This is not generic B7 exploration and "
        "does not pass G6.\n".format("validated" if args.p4i_run_dir else "pending")
    )
    (output / "HANDOFF.md").write_text(
        "# HANDOFF\n\n`mechanism_canary_manifest.json` freezes identities and evidence; "
        "`dispatch_plan.json` freezes future correctness, balanced crossover order, warmup, "
        "timing repeats, counters, and decision thresholds. Do not execute barrier candidates "
        "until pending P4i AXU qualification is replaced by a separately frozen passing "
        "certificate. Never edit these artifacts after observing board data.\n"
    )
    artifacts = {
        path.name: file_sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    test = Path(__file__).with_name("test_prepare_vta_mechanism_canary.py")
    _write_new(
        output / "artifact_hashes.json",
        {
            "output_sha256": artifacts,
            "source_sha256": {
                str(Path(__file__).resolve()): file_sha256(Path(__file__)),
                str(test.resolve()): file_sha256(test),
            },
        },
    )
    print(stdout, end="")


if __name__ == "__main__":
    main()
