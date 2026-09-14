#!/usr/bin/env python3
"""Convert a complete R18 board pool into immutable literature replay inputs.

This is deliberately a post-board adapter.  It refuses incomplete/interrupted
sessions and joins identities only after the board runner has completed every
correctness decision and every timing row.  Policies consume the resulting
one-way outcome ledger; they never receive the raw board directory.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import run_vta_c3_literature_baselines as baseline


SCHEMA = "c3_resnet18_literature_replay_adapter_v1"
WORKLOAD_SCHEMA = "c3_literature_workload_contract_v1"
OUTCOME_SCHEMA = "c3_literature_outcome_ledger_v1"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def verify_flat(directory):
    directory = Path(directory).resolve()
    ledger = read_json(directory / "artifact_hashes.json")
    artifacts = ledger.get("artifacts", ledger.get("files"))
    if not isinstance(artifacts, dict):
        raise ValueError("unknown artifact ledger schema: {}".format(directory))
    for name, expected in artifacts.items():
        if isinstance(expected, str) and baseline.sha256_file(directory / name) != expected:
            raise ValueError("artifact hash mismatch: {}".format(directory / name))
    return baseline.sha256_file(directory / "artifact_hashes.json")


def normalize_hidden(value):
    value = dict(value or {})
    return {
        "loop_count": int(value.get("loop_count", 0)),
        "loop_extent_sum": float(value.get("loop_extent_sum", 0)),
        "loop_extent_product_log2": float(
            value.get("loop_extent_product_log2", value.get("loop_extent_log2_product", 0))
        ),
        "branch_count": int(value.get("branch_count", 0)),
        "partial_tile_count": int(value.get("partial_tile_count", 0)),
        "allocation_bytes": int(value.get("allocation_bytes", 0)),
        "tensorize_count": int(
            value.get("tensorize_count", value.get("tensorize_uop_sites", 0))
        ),
    }


def normalize_static(candidate):
    dma = candidate["dma_features"]
    totals = dma["totals"]
    command = candidate["command_features"]["values"]
    return {
        "total_dma_bytes": int(totals.get("load_buffer_2d_bytes", 0))
        + int(totals.get("store_buffer_2d_bytes", 0)),
        "total_dma_calls": int(totals.get("load_buffer_2d_calls", 0))
        + int(totals.get("store_buffer_2d_calls", 0)),
        "padding_dma_calls": int(dma.get("padded_calls", 0)),
        "submissions": int(command.get("submissions", 0)),
        "input_dma_bytes": int(totals.get("load_buffer_2d_inp_bytes", 0)),
        "weight_dma_bytes": int(totals.get("load_buffer_2d_wgt_bytes", 0)),
        "acc_dma_bytes": int(totals.get("load_buffer_2d_acc_bytes", 0))
        + int(totals.get("store_buffer_2d_acc_bytes", 0)),
        "output_dma_bytes": int(totals.get("store_buffer_2d_out_bytes", 0)),
        "uop_dma_bytes": int(totals.get("load_buffer_2d_uop_bytes", 0)),
        "instruction_peak_bytes": int(command["peaks"]["insn_bytes"]),
        "uop_peak_bytes": int(command["peaks"]["uop_bytes"]),
        "explicit_residency_drains": int(
            candidate.get("sync", {}).get("explicit_residency_drains", 0)
        ),
        "scope": "logical lowered VTA LOAD/STORE; not physical AXI traffic",
    }


def profile_cost(rows):
    total = Counter()
    for row in rows:
        profile = row.get("runtime_profile_complete") or {}
        total["logical_dma_bytes"] += int(profile.get("load_buffer_2d_bytes", 0))
        total["logical_dma_bytes"] += int(profile.get("store_buffer_2d_bytes", 0))
        total["logical_dma_calls"] += int(profile.get("load_buffer_2d_calls", 0))
        total["logical_dma_calls"] += int(profile.get("store_buffer_2d_calls", 0))
        total["fpga_kernel_invocations"] += int(profile.get("driver_run_calls", 0))
    return dict(total)


def pure_instruction_ms_per_operator(row):
    """Normalize runtime driver time by evaluator-level operator invocations.

    ``driver_run_calls`` cannot be the divisor because one VTA operator may
    intentionally contain many barrier-separated submissions.  The board
    runner records how many complete ``main`` invocations the time evaluator
    executed; driver time is summed across all submissions of those invocations.
    """

    profile = row.get("runtime_profile_complete") or {}
    if "driver_run_total_us" not in profile:
        raise ValueError("runtime profile lacks driver_run_total_us")
    total_us = float(profile["driver_run_total_us"])
    operator_invocations = int(
        row.get("time_evaluator_total_kernel_invocations", 0)
    )
    if total_us < 0 or operator_invocations <= 0:
        raise ValueError("pure-instruction normalization metadata is invalid")
    return total_us / 1000.0 / operator_invocations


def phase(status, wall_seconds, **extra):
    return {"status": status, "wall_seconds": float(wall_seconds), **extra}


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))
    board_pool = Path(args.board_pool_dir).resolve()
    cross_dir = Path(args.cross_compile_dir).resolve()
    board_dir = Path(args.board_run_dir).resolve()
    qualification_dirs = [Path(value).resolve() for value in args.qualification_dir]
    input_hashes = {
        "board_pool": verify_flat(board_pool),
        "cross_compile": verify_flat(cross_dir),
        "board_run": verify_flat(board_dir),
        "qualifications": {str(path): verify_flat(path) for path in qualification_dirs},
    }
    board_summary = read_json(board_dir / "summary.json")
    if board_summary.get("status") != "complete_non_spliced_board_pool":
        raise RuntimeError("interrupted or incomplete board session cannot be adapted")
    if board_summary.get("session_spliced") is not False:
        raise RuntimeError("spliced board session is forbidden")

    candidates = read_jsonl(board_pool / "candidates.jsonl")
    by_id = {row["candidate_id"]: row for row in candidates}
    if len(by_id) != len(candidates):
        raise ValueError("duplicate board-pool candidate identity")
    static_rows, fsim_rows = {}, {}
    for directory in qualification_dirs:
        for row in read_jsonl(directory / "static_results.jsonl"):
            static_rows[row["candidate_id"]] = row
        for row in read_jsonl(directory / "fsim_results.jsonl"):
            fsim_rows[row["candidate_id"]] = row
    compile_rows = {
        row["candidate_id"]: row for row in read_jsonl(cross_dir / "results.jsonl")
    }
    correctness_rows = {
        row["candidate_id"]: row for row in read_jsonl(board_dir / "correctness.jsonl")
    }
    timing_rows = read_jsonl(board_dir / "timing.jsonl")
    timing_by_id = defaultdict(list)
    for row in timing_rows:
        timing_by_id[row["candidate_id"]].append(row)
    if set(correctness_rows) != set(by_id):
        raise ValueError("board correctness pool is incomplete")

    output.mkdir(parents=True)
    outcomes_by_workload = defaultdict(dict)
    contracts = defaultdict(list)
    cheng_rows = []
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        static = static_rows[candidate_id]
        fsim = fsim_rows[candidate_id]
        compile_row = compile_rows[candidate_id]
        correctness = correctness_rows[candidate_id]
        if static["status"] != "ok" or fsim["status"] != "passed":
            raise ValueError("board pool contains a locally invalid candidate")
        if compile_row["status"] != "passed":
            raise ValueError("board pool contains a cross-compile failure")
        normalized = {
            "candidate_id": candidate_id,
            "workload_id": candidate["workload_id"],
            "family_id": candidate["family_id"],
            "model_hash": candidate["model_hash"],
            "hardware_fingerprint": candidate["hardware_fingerprint"],
            "complete_config_entity": candidate["complete_config_entity"],
            "residence_mode": candidate["residence_mode"],
            "lowered_tir_sha256": candidate["tir_sha256"],
            "visible_features": candidate["visible_features"],
            "workload_features": candidate["workload_features"],
            "static_features": normalize_static(candidate),
            "applicability": candidate["applicability"],
        }
        contracts[candidate["workload_id"]].append(normalized)
        correct_seeds = correctness["seeds"]
        correct_cost = profile_cost(correct_seeds)
        is_correct = correctness["status"] == "passed"
        outcome = {
            "lower": phase(
                "ok",
                static["wall_seconds"],
                compiler_attempts=1,
                hidden_features=normalize_hidden(static["hidden_compiler_features"]),
            ),
            "fsim": phase("ok", fsim["wall_seconds"]),
            "compile": phase(
                "ok", compile_row["wall_seconds"], compiler_attempts=1
            ),
            "fpga_correctness": phase(
                "ok" if is_correct else "invalid",
                sum(float(row.get("host_wall_seconds", 0)) for row in correct_seeds),
                **correct_cost
            ),
        }
        candidate_timings = timing_by_id.get(candidate_id, [])
        if is_correct:
            if len(candidate_timings) != 5:
                raise ValueError("correct candidate lacks five timing rounds")
            timing_cost = profile_cost(candidate_timings)
            latency_values = [float(row["latency_ms"]) for row in candidate_timings]
            driver_values = [
                pure_instruction_ms_per_operator(row)
                for row in candidate_timings
            ]
            outcome["timing"] = phase(
                "ok",
                sum(float(row.get("host_wall_seconds", 0)) for row in candidate_timings),
                latency_ms=statistics.median(latency_values),
                latency_values_ms=latency_values,
                pure_instruction_ms=statistics.median(driver_values),
                **timing_cost
            )
        elif candidate_timings:
            raise ValueError("FPGA-invalid candidate must not have timing labels")
        outcomes_by_workload[candidate["workload_id"]][candidate_id] = outcome
        cheng_rows.append(
            {
                "candidate_id": candidate_id,
                "workload_id": candidate["workload_id"],
                "family_id": candidate["family_id"],
                "residence_mode": candidate["residence_mode"],
                **normalized["static_features"],
                "fpga_correct": is_correct,
                "operator_latency_ms": outcome.get("timing", {}).get("latency_ms"),
                "pure_instruction_ms": outcome.get("timing", {}).get("pure_instruction_ms"),
            }
        )

    artifacts = []
    for workload_id in sorted(contracts):
        rows = sorted(contracts[workload_id], key=lambda row: row["candidate_id"])
        workload_contract = {
            "schema": WORKLOAD_SCHEMA,
            "role": "complete_fpga_observed_locally_qualified_pool",
            "workload_id": workload_id,
            "candidates": rows,
            "candidate_count": len(rows),
            "target_labels_present": False,
        }
        # Reuse the runner's recursive leak scanner as an adapter assertion.
        baseline.validate_workload_contract(workload_contract)
        workload_name = "{}_workload_contract.json".format(workload_id.lower())
        outcome_name = "{}_outcome_ledger.json".format(workload_id.lower())
        write_json(output / workload_name, workload_contract)
        write_json(
            output / outcome_name,
            {
                "schema": OUTCOME_SCHEMA,
                "session_status": "complete",
                "pool_complete": True,
                "workload_id": workload_id,
                "outcomes": outcomes_by_workload[workload_id],
                "source_board_boot_id": board_summary["boot_id"],
                "session_spliced": False,
            },
        )
        artifacts.extend((workload_name, outcome_name))
    (output / "cheng_rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in cheng_rows),
        encoding="utf-8",
    )
    common = {
        "local_qualification_wall_seconds": sum(
            float(read_json(path / "summary.json")["wall_seconds"])
            for path in qualification_dirs
        ),
        "cross_compile_wall_seconds": float(
            read_json(cross_dir / "summary.json")["wall_seconds"]
        ),
        "board_W0_to_T1_seconds": float(board_summary["cost"]["W0_to_T1_seconds"]),
        "board_T0_to_T1_seconds": float(board_summary["cost"]["T0_to_T1_seconds"]),
        "interpretation": "actual all-pool costs; policy replay also reports counterfactual lazy per-candidate costs and must not silently mix the two",
    }
    summary = {
        "schema": SCHEMA,
        "status": "complete_post_board_replay_adapter",
        "workload_candidate_counts": {
            workload_id: len(rows) for workload_id, rows in contracts.items()
        },
        "fpga_correct_counts": {
            workload_id: sum(
                outcome["fpga_correctness"]["status"] == "ok"
                for outcome in outcomes.values()
            )
            for workload_id, outcomes in outcomes_by_workload.items()
        },
        "common_actual_cost": common,
        "input_hashes": input_hashes,
        "logical_dma_scope": "VTA runtime logical LOAD/STORE; not physical AXI traffic",
        "oracle_connection": "only after complete non-spliced board pool",
    }
    write_json(output / "summary.json", summary)
    write_json(
        output / "contract.json",
        {
            "schema": SCHEMA,
            "status": "post_board_only",
            "input_hashes": input_hashes,
            "source_board_status": board_summary["status"],
            "future_label_separation": "workload contracts omit outcomes; labels exist only in one-way outcome ledgers",
        },
    )
    (output / "timeline.jsonl").write_text("", encoding="utf-8")
    (output / "results.jsonl").write_text("", encoding="utf-8")
    (output / "command.txt").write_text(
        " ".join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]) + "\n",
        encoding="utf-8",
    )
    write_json(
        output / "artifact_hashes.json",
        {
            "artifacts": {
                path.name: baseline.sha256_file(path)
                for path in sorted(output.iterdir())
                if path.is_file() and path.name != "artifact_hashes.json"
            },
            "source_hashes": {
                str(Path(__file__).resolve()): baseline.sha256_file(Path(__file__).resolve()),
                str(Path(baseline.__file__).resolve()): baseline.sha256_file(Path(baseline.__file__).resolve()),
            },
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-pool-dir", required=True)
    parser.add_argument("--cross-compile-dir", required=True)
    parser.add_argument("--board-run-dir", required=True)
    parser.add_argument("--qualification-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
