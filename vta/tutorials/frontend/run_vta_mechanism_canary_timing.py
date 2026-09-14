#!/usr/bin/env python3
"""Run the frozen P6b balanced timing protocol after P6c correctness passes."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import tvm
from tvm import rpc
import vta

from qualify_vta_residency_fsim import reference_data_from_workload
from run_vta_mechanism_canary_board import (
    REQUIRED_COUNTERS,
    build_candidate,
    file_sha256,
    load_json,
    load_jsonl,
    ssh_preflight,
    validate_contract,
)


SAMPLE_SCHEMA = "c3_p6d_balanced_timing_sample_v1"
SUMMARY_SCHEMA = "c3_p6d_balanced_timing_summary_v1"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_correctness(results_path, summary_path, manifest_path, plan_path):
    results = load_jsonl(results_path)
    summary = load_json(summary_path)
    if summary.get("schema") != "c3_p6c_ram_board_correctness_summary_v1":
        raise ValueError("unexpected P6c correctness summary schema")
    if (
        summary.get("status") != "passed"
        or summary.get("selected_candidates") != 9
        or summary.get("executed_candidates") != 9
        or summary.get("passed_all_three_seeds") != 9
        or summary.get("failed") != 0
    ):
        raise ValueError("P6c did not pass all nine candidates")
    if summary.get("source_manifest_sha256") != file_sha256(manifest_path):
        raise ValueError("P6c manifest binding changed")
    if summary.get("source_dispatch_plan_sha256") != file_sha256(plan_path):
        raise ValueError("P6c dispatch-plan binding changed")
    if len(results) != 9:
        raise ValueError("P6c result count is not nine")
    for row in results:
        if row.get("overall_status") != "passed" or len(row.get("seeds", ())) != 3:
            raise ValueError("P6c contains a non-passing candidate")
        if not all(seed.get("correct") is True for seed in row["seeds"]):
            raise ValueError("P6c contains a failed seed")
    return {row["candidate_id"]: row for row in results}


def classify(relative_improvement):
    if abs(relative_improvement) <= 0.02:
        return "equivalent"
    if relative_improvement >= 0.05:
        return "improved"
    if relative_improvement <= -0.05:
        return "regressed"
    return "indeterminate"


def summarize(samples, selected):
    grouped = defaultdict(list)
    entry_by_id = {}
    for entry, _ in selected:
        entry_by_id[entry["candidate_id"]] = entry
    for sample in samples:
        grouped[sample["candidate_id"]].append(float(sample["latency_ms"]))
    candidates = {}
    incumbents = {}
    for candidate_id, values in grouped.items():
        entry = entry_by_id[candidate_id]
        array = np.asarray(values, dtype="float64")
        stats = {
            "candidate_id": candidate_id,
            "workload_id": entry["workload_id"],
            "mechanism_role": entry["mechanism_role"],
            "residence_mode": entry["residence_mode"],
            "sample_count": int(array.size),
            "median_latency_ms": float(np.median(array)),
            "mean_latency_ms": float(np.mean(array)),
            "std_latency_ms": float(np.std(array, ddof=1)),
            "min_latency_ms": float(np.min(array)),
            "p25_latency_ms": float(np.percentile(array, 25)),
            "p75_latency_ms": float(np.percentile(array, 75)),
            "max_latency_ms": float(np.max(array)),
        }
        candidates[candidate_id] = stats
        if entry["mechanism_role"] == "original_incumbent":
            incumbents[entry["workload_id"]] = stats
    decisions = []
    for candidate_id, stats in sorted(
        candidates.items(), key=lambda item: (item[1]["workload_id"], item[1]["mechanism_role"])
    ):
        incumbent = incumbents[stats["workload_id"]]
        improvement = (
            incumbent["median_latency_ms"] - stats["median_latency_ms"]
        ) / incumbent["median_latency_ms"]
        decisions.append(
            {
                "candidate_id": candidate_id,
                "workload_id": stats["workload_id"],
                "mechanism_role": stats["mechanism_role"],
                "residence_mode": stats["residence_mode"],
                "median_latency_ms": stats["median_latency_ms"],
                "incumbent_median_latency_ms": incumbent["median_latency_ms"],
                "relative_improvement": float(improvement),
                "decision": "reference" if stats["mechanism_role"] == "original_incumbent" else classify(improvement),
            }
        )
    return candidates, decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dispatch-plan", required=True)
    parser.add_argument("--correctness-results", required=True)
    parser.add_argument("--correctness-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--expected-boot-id", required=True)
    parser.add_argument("--dmesg-after", type=int, default=467)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    samples_path = output / "samples.jsonl"
    if samples_path.exists():
        raise FileExistsError("refusing to overwrite existing timing samples")

    repo_root = Path(__file__).resolve().parents[3]
    _, plan, _, selected = validate_contract(
        args.manifest, args.dispatch_plan, repo_root
    )
    correctness = validate_correctness(
        args.correctness_results,
        args.correctness_summary,
        args.manifest,
        args.dispatch_plan,
    )
    plan_rounds = plan["ordering"]["rounds"]
    warmup_runs = int(plan["timing"]["warmup_runs_per_candidate"])
    timed_repeats = int(plan["timing"]["timed_repeats_per_candidate"])
    timer_number = int(plan["timing"]["timer_number"])
    if len(plan_rounds) != timed_repeats or (warmup_runs, timed_repeats, timer_number) != (3, 30, 1):
        raise ValueError("unexpected P6b timing protocol")

    env = vta.get_env()
    if env.TARGET != "axu5evb":
        raise RuntimeError("VTA TARGET must be axu5evb")
    preflight = ssh_preflight(args.host, args.expected_boot_id, args.dmesg_after)
    remote = rpc.connect(args.host, args.port, session_timeout=120)
    clear = remote.get_function("vta.runtime.profiler_clear")
    status = remote.get_function("vta.runtime.profiler_status")
    device = remote.ext_dev(0)

    loaded = {}
    setup_started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="c3_p6d_board_") as temporary:
            binary_dir = Path(temporary)
            for entry, record in selected:
                binary, _ = build_candidate(entry, record, env, binary_dir)
                remote.upload(str(binary))
                module = remote.load_module(binary.name)
                data, weight, expected = reference_data_from_workload(record["identity"]["workload"], 0)
                buffers = [
                    tvm.nd.array(data, device),
                    tvm.nd.array(weight, device),
                    tvm.nd.empty(expected.shape, "int8", device),
                ]
                loaded[entry["candidate_id"]] = {
                    "entry": entry,
                    "module": module,
                    "buffers": buffers,
                    "expected": expected,
                }

            # Candidate-local setup and warmup are explicitly excluded from timing.
            for entry, _ in selected:
                item = loaded[entry["candidate_id"]]
                function = item["module"]["main"]
                for _ in range(warmup_runs):
                    function(*item["buffers"])
                actual = item["buffers"][-1].numpy()
                if not np.array_equal(actual, item["expected"]):
                    raise AssertionError("post-warmup correctness failed: {}".format(entry["candidate_id"]))

            setup_seconds = time.monotonic() - setup_started
            samples = []
            sequence = 0
            with samples_path.open("x", encoding="utf-8") as stream:
                for round_spec in plan_rounds:
                    ssh_preflight(args.host, args.expected_boot_id, args.dmesg_after)
                    round_index = int(round_spec["round"])
                    for workload_id in round_spec["workload_order"]:
                        for candidate_id in round_spec["candidate_order_by_workload"][workload_id]:
                            item = loaded[candidate_id]
                            clear()
                            timer = item["module"].time_evaluator(
                                "main", device, number=timer_number, repeat=1
                            )
                            values = timer(*item["buffers"]).results
                            if len(values) != 1 or not np.isfinite(values[0]) or values[0] <= 0:
                                raise RuntimeError("invalid timer result: {}".format(values))
                            profile = json.loads(status())
                            missing = [name for name in REQUIRED_COUNTERS if name not in profile]
                            if missing:
                                raise RuntimeError("runtime profile lacks counters: {}".format(missing))
                            sample = {
                                "schema": SAMPLE_SCHEMA,
                                "sequence": sequence,
                                "round": round_index,
                                "workload_id": workload_id,
                                "candidate_id": candidate_id,
                                "mechanism_role": item["entry"]["mechanism_role"],
                                "residence_mode": item["entry"]["residence_mode"],
                                "latency_ms": float(values[0]) * 1000.0,
                                "timer_number": timer_number,
                                # TVM time_evaluator performs one untimed invocation
                                # followed by number * repeat timed invocations.
                                "profiler_invocations": 1 + timer_number,
                                "runtime_profile_raw": {
                                    name: profile[name] for name in REQUIRED_COUNTERS
                                },
                                "runtime_profile": {
                                    name: profile[name] / (1 + timer_number)
                                    for name in REQUIRED_COUNTERS
                                },
                            }
                            stream.write(json.dumps(sample, sort_keys=True) + "\n")
                            stream.flush()
                            samples.append(sample)
                            sequence += 1
                    print("round={}/{} complete".format(round_index + 1, timed_repeats), flush=True)

            # Recheck output after timing before accepting samples.
            for entry, _ in selected:
                item = loaded[entry["candidate_id"]]
                actual = item["buffers"][-1].numpy()
                if not np.array_equal(actual, item["expected"]):
                    raise AssertionError("post-timing correctness failed: {}".format(entry["candidate_id"]))
    except Exception as error:
        write_json(
            output / "failure.json",
            {
                "status": "failed",
                "exception_type": type(error).__name__,
                "message": (str(error) or type(error).__name__)[:4000],
                "traceback": traceback.format_exc()[-12000:],
                "performance_claim_allowed": False,
            },
        )
        raise

    final_preflight = ssh_preflight(args.host, args.expected_boot_id, args.dmesg_after)
    candidates, decisions = summarize(samples, selected)
    expected_samples = 9 * timed_repeats
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "passed" if len(samples) == expected_samples else "incomplete",
        "sample_count": len(samples),
        "expected_sample_count": expected_samples,
        "warmup_runs_per_candidate": warmup_runs,
        "timed_repeats_per_candidate": timed_repeats,
        "timer_number": timer_number,
        "balanced_complete_crossover": True,
        "compile_upload_setup_excluded": True,
        "setup_seconds": setup_seconds,
        "board_host": args.host,
        "board_boot_id": args.expected_boot_id,
        "board_preflight_before": preflight,
        "board_preflight_after": final_preflight,
        "correctness_precondition": {
            "status": "passed_9_of_9_candidates_27_of_27_seed_checks",
            "results_sha256": file_sha256(args.correctness_results),
            "summary_sha256": file_sha256(args.correctness_summary),
            "candidate_count": len(correctness),
        },
        "candidates": candidates,
        "decisions": decisions,
        "decision_thresholds": plan["decision_rule"],
        "samples_sha256": file_sha256(samples_path),
    }
    write_json(output / "summary.json", summary)
    print("timing_status={} samples={}/{}".format(summary["status"], len(samples), expected_samples))
    for decision in decisions:
        print(
            "{} {} median_ms={:.6f} improvement={:.3%} decision={}".format(
                decision["workload_id"],
                decision["residence_mode"],
                decision["median_latency_ms"],
                decision["relative_improvement"],
                decision["decision"],
            )
        )
    if summary["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
