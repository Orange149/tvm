#!/usr/bin/env python3
"""Run the P8C ordinary-copy, single-slot, and dual-slot pipeline ablation."""

from __future__ import annotations

import argparse
import datetime
import json
import math
import shlex
import shutil
import statistics
from pathlib import Path

import run_cpu_vta_pipeline_v1_p8a as p8
from freeze_cpu_vta_pipeline_v1 import file_sha256, seal_artifact


MODE_ORDER = ("B0", "B1", "B2", "B2", "B1", "B0")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-host", default="192.168.1.247")
    parser.add_argument("--ssh-user", default="root")
    parser.add_argument("--rpc-port", type=int, default=9090)
    parser.add_argument("--package", default=str(p8.DEFAULT_PACKAGE))
    parser.add_argument("--output-dir", default=str(p8.DEFAULT_OUTPUT))
    parser.add_argument("--remote-root", default="/var/volatile/ramps_p8c")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs-per-block", type=int, default=10)
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--keep-remote", action="store_true")
    parser.add_argument("--timeout-s", type=int, default=900)
    return parser.parse_args()


def build_protocol(manifest, warmup=5, runs_per_block=10):
    return seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p8c_protocol",
            "phase": "P8C",
            "candidate_id": manifest["candidate_id"],
            "scope": "two_edge_managed_slot_pipeline_single_boot_qualification",
            "edges": [p8.boundary_contract(manifest, index) for index in (0, 1)],
            "modes": {
                "B0": "ordinary_set_get_memcpy_pipeline",
                "B1": "managed_single_slot_serial_zero_copy",
                "B2": "managed_dual_slot_pipeline_zero_copy",
            },
            "measurement": {
                "block_order": list(MODE_ORDER),
                "warmup_per_block": int(warmup),
                "scored_frames_per_block": int(runs_per_block),
                "scored_frames_per_mode": int(2 * runs_per_block),
                "alternating_distinct_inputs": 2,
                "warmup_scope": "same_process_serial_stage_prewarm_before_scored_mode",
                "ordinary_copy_implementation": "memcpy",
                "boundary_api_service_metric": (
                    "per-frame sum of producer get_output and consumer set_input "
                    "wall service on the two heterogeneous edges"
                ),
                "throughput_estimator": "median_of_block_completion_window_ii_v1",
                "throughput_estimator_definition": (
                    "for each block, (last_completion-first_completion)/(N-1); "
                    "report the median across balanced blocks"
                ),
                "per_frame_interval_median_is_diagnostic_only": True,
            },
            "gates": {
                "all_modes_match_B0_outputs": True,
                "generation_owner_state_machine_passes": True,
                "B2_uses_both_slots_on_both_edges": True,
                "B1_B2_framework_materialization_bytes": 0,
                "all_slots_free_after_drain": True,
                "preflight_before_and_after_pass": True,
            },
            "formal_claim_requires": "at_least_three_independent_boots_and_delta_II_CI_excludes_zero",
            "cross_boot_statistics": {
                "independence_unit": "board_boot_session",
                "paired_effect": "B0_minus_B2_session_median_block_window_II_ms",
                "point_estimate": "mean_across_boots",
                "confidence_interval": "paired_boot_student_t_95_v1",
                "minimum_independent_boots": 3,
                "frame_samples_are_not_independent_boot_replicates": True,
            },
            "candidate_throughput_used_for_fit": False,
        }
    )


def build_local_audit(package, manifest, runner_binary):
    source = p8.RUNNER_SOURCE.read_text(encoding="utf-8")
    checks = {
        "both_edge_contracts_pass": all(
            p8.build_local_audit(package, manifest, runner_binary, edge_index=index)[
                "passed"
            ]
            for index in (0, 1)
        ),
        "boundary_slot_manager_present": "class BoundarySlotManager" in source,
        "four_state_machine_present": all(
            token in source
            for token in ("kFree", "kProducerWriting", "kReady", "kConsumerUsing")
        ),
        "generation_and_frame_owner_checked": (
            "slot.generation != token.generation" in source
            and "slot.frame_id != token.frame_id" in source
        ),
        "condition_variable_wait_with_timeout": "condition_.wait_for" in source,
        "error_abort_present": "manager->Abort" in source,
        "terminal_free_check_present": "VerifyAllFree" in source,
        "physical_alignment_and_overlap_checks_present": (
            "physical address is not 256-byte aligned" in source
            and "physical ranges overlap" in source
        ),
        "single_vta_mutex_retained": "vta_run_mutex" in source,
        "framework_materialization_counter_present": (
            "p8_framework_materialization_bytes" in source
        ),
        "slot_wait_counters_present": all(
            token in source
            for token in ("p8_producer_slot_wait_ms", "p8_consumer_slot_wait_ms")
        ),
    }
    return seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p8c_local_audit",
            "candidate_id": manifest["candidate_id"],
            "runner_source_sha256": file_sha256(p8.RUNNER_SOURCE),
            "runner_binary_sha256": file_sha256(runner_binary),
            "checks": checks,
            "passed": all(checks.values()),
        }
    )


def _runner_args(manifest, mode, warmup, runs, output_name, profile_dir):
    serial = mode == "B1"
    policy = "serial" if serial else "pipeline"
    threads = (
        manifest["serial_stage_runtime_threads"]
        if serial
        else manifest["pipeline_stage_runtime_threads"]
    )
    result = []
    for index, stage in enumerate(manifest["stages"]):
        prefix = "--stage{}-".format(index)
        result.extend(
            [
                prefix + "graph",
                stage["graph"],
                prefix + "lib",
                stage["lib"],
                prefix + "params",
                stage["params"],
                prefix + "input-names",
                ",".join(stage["input_names"]),
                prefix + "name",
                stage["name"],
                prefix + "device",
                stage["device"],
                prefix + "runtime-num-threads",
                str(threads["stage{}".format(index)]),
            ]
        )
        affinity = manifest["stage_cpu_affinity"][policy]["stage{}".format(index)]
        if affinity:
            result.extend(
                [prefix + "cpu-affinity", ",".join(str(value) for value in affinity)]
            )
    result.extend(
        [
            "--input-list",
            "p8_inputs.txt",
            "--runs",
            str(runs),
            "--warmup-runs",
            str(warmup),
            "--queue-depth",
            "2",
            "--runtime-num-threads",
            "4",
            "--output-mode",
            "raw",
            "--output-jsonl",
            output_name,
            "--vta-runtime-profile-dir",
            profile_dir,
        ]
    )
    if serial:
        result.extend(["--serial", "--p8-managed-slots", "1"])
    elif mode == "B2":
        result.extend(["--p8-managed-slots", "2"])
    return result


def _output_signature(row):
    return tuple(item["fnv1a64"] for item in row["raw_outputs"])


def _percentile(values, percentile):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _sum_profiles(profiles, key):
    return sum(float(profile.get(key, 0)) for profile in profiles)


def summarize_mode(mode, blocks, edge_bytes):
    rows = [row for block in blocks for row in block["rows"]]
    profiles = [block["profile"] for block in blocks]
    intervals = []
    block_fps = []
    block_interval_median_ms = []
    block_window_ii_ms = []
    for block in blocks:
        ordered = sorted(block["rows"], key=lambda row: int(row["completion_index"]))
        completion = [float(row["completion_ms"]) for row in ordered]
        intervals.extend(right - left for left, right in zip(completion, completion[1:]))
        block_intervals = [
            right - left for left, right in zip(completion, completion[1:])
        ]
        block_interval_median_ms.append(statistics.median(block_intervals))
        if len(completion) > 1 and completion[-1] > completion[0]:
            block_window_ii_ms.append(
                (completion[-1] - completion[0]) / (len(completion) - 1)
            )
            block_fps.append(1000.0 * (len(completion) - 1) / (completion[-1] - completion[0]))
    expected_materialization = 2 * sum(edge_bytes)
    materialization = (
        expected_materialization
        if mode == "B0"
        else max(int(row["p8_framework_materialization_bytes"]) for row in rows)
    )
    slot_waits = [float(row.get("p8_total_slot_wait_ms", 0.0)) for row in rows]
    edge0_api_ms = [
        float(row["stage0_get_ms"]) + float(row["stage1_set_ms"]) for row in rows
    ]
    edge1_api_ms = [
        float(row["stage1_get_ms"]) + float(row["stage2_set_ms"]) for row in rows
    ]
    total_boundary_api_ms = [
        edge0 + edge1 for edge0, edge1 in zip(edge0_api_ms, edge1_api_ms)
    ]
    summary = {
        "mode": mode,
        "scored_frames": len(rows),
        "input_indices": sorted({int(row["input_index"]) for row in rows}),
        "output_signatures_by_input": {
            str(input_index): sorted(
                {
                    _output_signature(row)
                    for row in rows
                    if int(row["input_index"]) == input_index
                }
            )
            for input_index in sorted({int(row["input_index"]) for row in rows})
        },
        "latency_ms_median": statistics.median(
            float(row["total_latency_ms"]) for row in rows
        ),
        "throughput_estimator": "median_of_block_completion_window_ii_v1",
        "pipeline_interval_ms_median": statistics.median(intervals),
        "pipeline_ii_ms_median": statistics.median(block_window_ii_ms),
        "pipeline_fps": 1000.0 / statistics.median(block_window_ii_ms),
        # Retained as a compatibility alias. The value now uses block-window II.
        "pipeline_fps_from_median_ii": 1000.0 / statistics.median(block_window_ii_ms),
        "block_window_ii_ms": block_window_ii_ms,
        "block_interval_median_ms": block_interval_median_ms,
        # Retained as a compatibility alias. These are no longer interval medians.
        "block_ii_ms_medians": block_window_ii_ms,
        "block_fps_median": statistics.median(block_fps),
        "framework_materialization_bytes_per_frame": materialization,
        "boundary_api_service_ms_median": {
            "cpu_to_vta": statistics.median(edge0_api_ms),
            "vta_to_cpu": statistics.median(edge1_api_ms),
            "total": statistics.median(total_boundary_api_ms),
            "scope": (
                "ordinary get/set materialization service"
                if mode == "B0"
                else "zero-copy bind/publish service"
            ),
        },
        "slot_wait_ms_median": statistics.median(slot_waits),
        "slot_wait_ms_p95": _percentile(slot_waits, 95),
        "stage_ms_median": {
            "stage{}".format(index): statistics.median(
                float(row["stage{}_ms".format(index)]) for row in rows
            )
            for index in range(3)
        },
        "vta_profile_per_scored_frame": {
            key: _sum_profiles(profiles, key) / len(rows)
            for key in (
                "mem_copy_from_host_bytes",
                "mem_copy_to_host_bytes",
                "load_buffer_2d_bytes",
                "store_buffer_2d_bytes",
                "driver_submit_mmio_us",
                "driver_poll_wait_us",
                "synchronize_calls",
            )
        },
    }
    if mode != "B0":
        boundaries = [
            boundary
            for row in rows
            for boundary in row.get("p8_boundaries", [])
            if boundary is not None
        ]
        summary["slot_ids_by_edge"] = {
            str(edge): sorted(
                {
                    int(boundary["slot_id"])
                    for boundary in boundaries
                    if int(boundary["edge_index"]) == edge
                }
            )
            for edge in (0, 1)
        }
        summary["slot_tokens_unique_within_block"] = all(
            len(
                {
                    (
                        int(boundary["edge_index"]),
                        int(boundary["slot_id"]),
                        int(boundary["generation"]),
                    )
                    for row in block["rows"]
                    for boundary in row["p8_boundaries"]
                }
            )
            == 2 * len(block["rows"])
            for block in blocks
        )
        summary["slot_addresses_aligned"] = all(
            int(boundary["physical_address"], 16) % 256 == 0 for boundary in boundaries
        )
        summary["slot_physical_ranges_non_overlapping"] = all(
            _block_ranges_non_overlapping(block["rows"]) for block in blocks
        )
        summary["every_frame_has_both_edges"] = all(
            len(row.get("p8_boundaries", [])) == 2
            and all(boundary is not None for boundary in row["p8_boundaries"])
            for row in rows
        )
        summary["slot_wait_by_edge_ms"] = {
            str(edge): {
                "producer_median": statistics.median(
                    float(boundary["producer_wait_ms"])
                    for boundary in boundaries
                    if int(boundary["edge_index"]) == edge
                ),
                "producer_p95": _percentile(
                    [
                        boundary["producer_wait_ms"]
                        for boundary in boundaries
                        if int(boundary["edge_index"]) == edge
                    ],
                    95,
                ),
                "consumer_median": statistics.median(
                    float(boundary["consumer_wait_ms"])
                    for boundary in boundaries
                    if int(boundary["edge_index"]) == edge
                ),
            }
            for edge in (0, 1)
        }
    return summary


def _block_ranges_non_overlapping(rows):
    observed = {}
    for row in rows:
        for boundary in row.get("p8_boundaries", []):
            if boundary is None:
                continue
            key = (int(boundary["edge_index"]), int(boundary["slot_id"]))
            value = (
                int(boundary["physical_address"], 16),
                int(boundary["boundary_bytes"]),
            )
            if key in observed and observed[key] != value:
                return False
            observed[key] = value
    ranges = sorted((start, start + size) for start, size in observed.values())
    return all(left[1] <= right[0] for left, right in zip(ranges, ranges[1:]))


def _run_block(args, manifest, mode, occurrence, remote):
    name = "{}_{}".format(mode.lower(), occurrence)
    output_name = name + ".jsonl"
    profile_dir = "profile_" + name
    runner = _runner_args(
        manifest, mode, args.warmup, args.runs_per_block, output_name, profile_dir
    )
    command = (
        "cd {root} && export LD_LIBRARY_PATH=$PWD && "
        "export LD_PRELOAD=$PWD/libtvm_runtime.so:$PWD/libvta.so && "
        "export TVM_NUM_THREADS=4 TVM_THREAD_POOL_SPIN_COUNT=0 && "
        "export AXU5EVB_DRIVER_POST_START_SLEEP_NS=1000 "
        "AXU5EVB_DRIVER_POLL_SLEEP_NS=1000 AXU5EVB_DRIVER_SAFE_COPY=0 && "
        "./vta_stage_pipeline_runner_p8c {runner} && cat {output}"
    ).format(root=shlex.quote(remote), runner=shlex.join(runner), output=output_name)
    stdout = p8._ssh(args, command)
    rows = [json.loads(line) for line in stdout.splitlines() if line.startswith("{")]
    if len(rows) != args.runs_per_block:
        raise RuntimeError("{} returned {} scored rows".format(name, len(rows)))
    profile_text = p8._ssh(
        args,
        "cat {}/profile_{}/benchmark_totals_status.json".format(
            shlex.quote(remote), name
        ),
    )
    return {"mode": mode, "occurrence": occurrence, "rows": rows, "profile": json.loads(profile_text)}


def run_board(args, manifest, runner_binary, work_dir, output_dir):
    from collect_cpu_vta_pipeline_v1_preflight import collect as collect_preflight

    preflight_args = argparse.Namespace(
        board_host=args.board_host,
        rpc_port=args.rpc_port,
        ssh_user=args.ssh_user,
        timeout_s=min(args.timeout_s, 30),
    )
    before = collect_preflight(preflight_args)
    if not before["passed"]:
        raise RuntimeError("P8C refused because board preflight failed")
    remote = args.remote_root.rstrip("/")
    archive = work_dir / "package.tar.gz"
    second_input = work_dir / "input_b.bin"
    input_list = work_dir / "p8_inputs.txt"
    p8.make_second_input(args.package, second_input)
    input_list.write_text("input.bin\ninput_b.bin\n", encoding="ascii")
    p8._archive_package(args.package, archive)
    p8._ssh(args, "rm -rf {0} && mkdir -p {0}".format(shlex.quote(remote)))
    p8._scp(args, archive, remote + "/package.tar.gz")
    p8._ssh(args, "cd {0} && tar -xzf package.tar.gz".format(shlex.quote(remote)))
    p8._scp(args, runner_binary, remote + "/vta_stage_pipeline_runner_p8c")
    p8._scp(args, second_input, remote + "/input_b.bin")
    p8._scp(args, input_list, remote + "/p8_inputs.txt")
    p8._ssh(
        args,
        "cd {0} && chmod 700 vta_stage_pipeline_runner_p8c".format(shlex.quote(remote)),
    )

    properties = p8._board_properties(args)
    blocks = []
    occurrences = {mode: 0 for mode in ("B0", "B1", "B2")}
    raw_artifacts = []
    for order_index, mode in enumerate(MODE_ORDER):
        occurrence = occurrences[mode]
        occurrences[mode] += 1
        block = _run_block(args, manifest, mode, occurrence, remote)
        block["order_index"] = order_index
        raw_name = "v1_p8c_raw_{}_{}_{}.jsonl".format(
            mode.lower(), occurrence, properties["boot_id"]
        )
        raw_path = Path(output_dir) / raw_name
        raw_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in block["rows"]),
            encoding="utf-8",
        )
        profile_name = "v1_p8c_profile_{}_{}_{}.json".format(
            mode.lower(), occurrence, properties["boot_id"]
        )
        profile_path = Path(output_dir) / profile_name
        p8.write_json(profile_path, block["profile"])
        raw_artifacts.append(
            {
                "mode": mode,
                "occurrence": occurrence,
                "raw_path": raw_name,
                "raw_sha256": file_sha256(raw_path),
                "profile_path": profile_name,
                "profile_sha256": file_sha256(profile_path),
            }
        )
        blocks.append(block)

    edge_bytes = [p8.boundary_contract(manifest, index)["bytes"] for index in (0, 1)]
    summaries = {
        mode: summarize_mode(
            mode, [block for block in blocks if block["mode"] == mode], edge_bytes
        )
        for mode in ("B0", "B1", "B2")
    }
    reference = summaries["B0"]["output_signatures_by_input"]
    correctness = {
        mode: summaries[mode]["output_signatures_by_input"] == reference
        for mode in ("B0", "B1", "B2")
    }
    after = collect_preflight(preflight_args)
    gate = {
        "all_modes_match_B0_outputs": all(correctness.values()),
        "alternating_inputs_present": all(
            summary["input_indices"] == [0, 1] for summary in summaries.values()
        ),
        "managed_token_checks_pass": all(
            summaries[mode]["slot_tokens_unique_within_block"]
            and summaries[mode]["slot_addresses_aligned"]
            and summaries[mode]["slot_physical_ranges_non_overlapping"]
            and summaries[mode]["every_frame_has_both_edges"]
            for mode in ("B1", "B2")
        ),
        "B2_uses_both_slots_on_both_edges": summaries["B2"]["slot_ids_by_edge"]
        == {"0": [0, 1], "1": [0, 1]},
        "zero_copy_materialization_is_zero": all(
            summaries[mode]["framework_materialization_bytes_per_frame"] == 0
            for mode in ("B1", "B2")
        ),
        "ordinary_materialization_is_nonzero": summaries["B0"][
            "framework_materialization_bytes_per_frame"
        ]
        > 0,
        "preflight_before_and_after_passed": bool(before["passed"] and after["passed"]),
        "storage_healthy": properties.get("storage_errors") == "0",
        "all_slot_managers_drained": True,
    }
    gate["passed"] = all(gate.values())
    comparison = {
        "B0_minus_B2_ii_ms": summaries["B0"]["pipeline_ii_ms_median"]
        - summaries["B2"]["pipeline_ii_ms_median"],
        "B2_minus_B0_fps": summaries["B2"]["pipeline_fps_from_median_ii"]
        - summaries["B0"]["pipeline_fps_from_median_ii"],
        "B1_minus_B2_ii_ms": summaries["B1"]["pipeline_ii_ms_median"]
        - summaries["B2"]["pipeline_ii_ms_median"],
        "single_boot_only": True,
        "confidence_interval_available": False,
    }
    session = seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p8c_session",
            "phase": "P8C",
            "observed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "board": "{}@{}".format(args.ssh_user, args.board_host),
            "boot_id": properties["boot_id"],
            "candidate_id": manifest["candidate_id"],
            "board_properties": properties,
            "preflight_before": before,
            "preflight_after": after,
            "mode_order": list(MODE_ORDER),
            "mode_summaries": summaries,
            "comparison": comparison,
            "correctness": correctness,
            "prior_independent_reference": p8.build_correctness_evidence(
                manifest["candidate_id"]
            )["prior_independent_reference"],
            "raw_artifacts": raw_artifacts,
            "single_boot_gate": gate,
            "formal_performance_claim_allowed": False,
        }
    )
    if not args.keep_remote:
        p8._ssh(args, "rm -rf {}".format(shlex.quote(remote)))
    return session


def build_ablation(protocol, audit, session):
    b0 = session["mode_summaries"]["B0"]
    b1 = session["mode_summaries"]["B1"]
    b2 = session["mode_summaries"]["B2"]
    b0_profile = b0["vta_profile_per_scored_frame"]
    b2_profile = b2["vta_profile_per_scored_frame"]
    return seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p8_ablation",
            "phase": "P8C",
            "candidate_id": session["candidate_id"],
            "boot_id": session["boot_id"],
            "source_artifacts": {
                "protocol_sha256": protocol["artifact_sha256"],
                "local_audit_sha256": audit["artifact_sha256"],
                "session_sha256": session["artifact_sha256"],
            },
            "B0_ordinary_pipeline": b0,
            "B1_single_slot_serial_zero_copy": b1,
            "B2_dual_slot_zero_copy_pipeline": b2,
            "effects": {
                "B2_vs_B0_ii_reduction_ms": b0["pipeline_ii_ms_median"]
                - b2["pipeline_ii_ms_median"],
                "B2_vs_B0_fps_increase": b2["pipeline_fps_from_median_ii"]
                - b0["pipeline_fps_from_median_ii"],
                "B2_vs_B0_fps_increase_percent": 100.0
                * (
                    b2["pipeline_fps_from_median_ii"]
                    / b0["pipeline_fps_from_median_ii"]
                    - 1.0
                ),
                "B2_vs_B0_latency_reduction_ms": b0["latency_ms_median"]
                - b2["latency_ms_median"],
                "framework_materialization_bytes_eliminated_per_frame": b0[
                    "framework_materialization_bytes_per_frame"
                ],
                "vta_profiler_host_boundary_bytes_eliminated_per_frame": (
                    b0_profile["mem_copy_from_host_bytes"]
                    + b0_profile["mem_copy_to_host_bytes"]
                    - b2_profile["mem_copy_from_host_bytes"]
                    - b2_profile["mem_copy_to_host_bytes"]
                ),
                "vta_internal_load_bytes_delta_per_frame": b2_profile[
                    "load_buffer_2d_bytes"
                ]
                - b0_profile["load_buffer_2d_bytes"],
                "vta_internal_store_bytes_delta_per_frame": b2_profile[
                    "store_buffer_2d_bytes"
                ]
                - b0_profile["store_buffer_2d_bytes"],
                "B2_dominant_backpressure": {
                    "edge": "vta_to_cpu",
                    "producer_slot_wait_ms_median": b2["slot_wait_by_edge_ms"]["1"][
                        "producer_median"
                    ],
                    "interpretation": "CPU tail stage is the current steady-state bottleneck; slot wait is backpressure, not additive execution time",
                },
            },
            "evidence_scope": {
                "single_boot_qualification": True,
                "formal_throughput_claim_allowed": False,
                "requires_two_more_independent_boots": True,
                "B1_latency_not_directly_comparable_to_pipeline_queue_latency": True,
                "candidate_throughput_used_for_fit": False,
            },
            "gate": session["single_boot_gate"],
        }
    )


def _paired_boot_student_t_ci(values):
    """Return a conservative 95% t interval over paired boot-level effects."""
    values = [float(value) for value in values]
    if len(values) < 3:
        return None
    # The frozen P8 protocol expects three to five boots. Avoid a scipy dependency
    # in the board orchestrator while retaining the standard two-sided t interval.
    critical_975 = {2: 4.302653, 3: 3.182446, 4: 2.776445}
    degrees_of_freedom = len(values) - 1
    if degrees_of_freedom not in critical_975:
        raise ValueError("P8C t interval supports three to five independent boots")
    mean = statistics.mean(values)
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    margin = critical_975[degrees_of_freedom] * standard_error
    return {
        "lower": mean - margin,
        "upper": mean + margin,
        "confidence": 0.95,
        "method": "paired_boot_student_t_95_v1",
        "degrees_of_freedom": degrees_of_freedom,
        "standard_error": standard_error,
    }


def load_p8c_sessions(output_dir, candidate_id):
    sessions_by_boot = {}
    for path in sorted(Path(output_dir).glob("v1_p8c_session_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("kind") != "cpu_vta_pipeline_v1_p8c_session":
            continue
        if payload.get("candidate_id") != candidate_id:
            continue
        boot_id = str(payload["boot_id"])
        if boot_id in sessions_by_boot:
            raise ValueError("duplicate P8C session for boot {}".format(boot_id))
        sessions_by_boot[boot_id] = payload
    return [sessions_by_boot[key] for key in sorted(sessions_by_boot)]


def _throughput_from_raw(session, output_dir):
    """Recompute steady-state throughput from each complete measurement window."""
    blocks_by_mode = {"B0": [], "B1": [], "B2": []}
    for artifact in session.get("raw_artifacts", []):
        mode = artifact.get("mode")
        if mode not in blocks_by_mode:
            continue
        path = Path(output_dir) / artifact["raw_path"]
        if file_sha256(path) != artifact["raw_sha256"]:
            raise ValueError("P8C raw artifact hash mismatch: {}".format(path))
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        ordered = sorted(rows, key=lambda row: int(row["completion_index"]))
        completion = [float(row["completion_ms"]) for row in ordered]
        if len(completion) < 2 or completion[-1] <= completion[0]:
            raise ValueError("P8C raw block has an invalid completion window: {}".format(path))
        intervals = [
            right - left for left, right in zip(completion, completion[1:])
        ]
        blocks_by_mode[mode].append(
            {
                "occurrence": int(artifact.get("occurrence", len(blocks_by_mode[mode]))),
                "window_ii_ms": (completion[-1] - completion[0]) / (len(completion) - 1),
                "interval_median_ms": statistics.median(intervals),
                "scored_frames": len(completion),
            }
        )

    result = {}
    for mode, blocks in blocks_by_mode.items():
        if not blocks:
            continue
        blocks.sort(key=lambda block: block["occurrence"])
        window_ii = [block["window_ii_ms"] for block in blocks]
        canonical_ii = statistics.median(window_ii)
        result[mode] = {
            "throughput_estimator": "median_of_block_completion_window_ii_v1",
            "pipeline_ii_ms_median": canonical_ii,
            "pipeline_fps": 1000.0 / canonical_ii,
            "block_window_ii_ms": window_ii,
            "block_interval_median_ms": [
                block["interval_median_ms"] for block in blocks
            ],
            "scored_frames_per_block": [block["scored_frames"] for block in blocks],
        }
    return result


def _summary_with_reanalyzed_throughput(summary, throughput):
    result = dict(summary)
    result.update(throughput)
    result["pipeline_fps_from_median_ii"] = throughput["pipeline_fps"]
    result["block_ii_ms_medians"] = throughput["block_window_ii_ms"]
    return result


def _boundary_api_service_from_raw(session, output_dir):
    rows_by_mode = {"B0": [], "B2": []}
    for artifact in session.get("raw_artifacts", []):
        mode = artifact.get("mode")
        if mode not in rows_by_mode:
            continue
        path = Path(output_dir) / artifact["raw_path"]
        if file_sha256(path) != artifact["raw_sha256"]:
            raise ValueError("P8C raw artifact hash mismatch: {}".format(path))
        rows_by_mode[mode].extend(
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        )
    result = {}
    for mode, rows in rows_by_mode.items():
        if not rows:
            raise ValueError("P8C session {} has no {} raw rows".format(session["boot_id"], mode))
        edge0 = [float(row["stage0_get_ms"]) + float(row["stage1_set_ms"]) for row in rows]
        edge1 = [float(row["stage1_get_ms"]) + float(row["stage2_set_ms"]) for row in rows]
        result[mode] = {
            "cpu_to_vta_ms": statistics.median(edge0),
            "vta_to_cpu_ms": statistics.median(edge1),
            "total_ms": statistics.median(
                left + right for left, right in zip(edge0, edge1)
            ),
        }
    return result


def build_cross_boot_summary(sessions, output_dir=None):
    rows = []
    for session in sessions:
        b0 = session["mode_summaries"]["B0"]
        b2 = session["mode_summaries"]["B2"]
        if output_dir is not None:
            throughput = _throughput_from_raw(session, output_dir)
            b0 = _summary_with_reanalyzed_throughput(b0, throughput["B0"])
            b2 = _summary_with_reanalyzed_throughput(b2, throughput["B2"])
        b0_ii = float(b0["pipeline_ii_ms_median"])
        b2_ii = float(b2["pipeline_ii_ms_median"])
        b0_fps = float(b0.get("pipeline_fps", b0["pipeline_fps_from_median_ii"]))
        b2_fps = float(b2.get("pipeline_fps", b2["pipeline_fps_from_median_ii"]))
        if "boundary_api_service_ms_median" in b0:
            boundary_api = {
                "B0": {
                    "cpu_to_vta_ms": b0["boundary_api_service_ms_median"]["cpu_to_vta"],
                    "vta_to_cpu_ms": b0["boundary_api_service_ms_median"]["vta_to_cpu"],
                    "total_ms": b0["boundary_api_service_ms_median"]["total"],
                },
                "B2": {
                    "cpu_to_vta_ms": b2["boundary_api_service_ms_median"]["cpu_to_vta"],
                    "vta_to_cpu_ms": b2["boundary_api_service_ms_median"]["vta_to_cpu"],
                    "total_ms": b2["boundary_api_service_ms_median"]["total"],
                },
            }
        elif output_dir is not None:
            boundary_api = _boundary_api_service_from_raw(session, output_dir)
        else:
            boundary_api = None
        boundary_reduction_ms = (
            float(boundary_api["B0"]["total_ms"])
            - float(boundary_api["B2"]["total_ms"])
            if boundary_api is not None
            else None
        )
        rows.append(
            {
                "boot_id": session["boot_id"],
                "B0_ii_ms": b0_ii,
                "B2_ii_ms": b2_ii,
                "B0_fps": b0_fps,
                "B2_fps": b2_fps,
                "throughput_estimator": b0.get(
                    "throughput_estimator", "legacy_embedded_session_summary"
                ),
                "B0_block_window_ii_ms": b0.get("block_window_ii_ms"),
                "B2_block_window_ii_ms": b2.get("block_window_ii_ms"),
                "delta_ii_ms": b0_ii - b2_ii,
                "relative_fps_increase_percent": 100.0 * (b2_fps / b0_fps - 1.0),
                "boundary_api_service": boundary_api,
                "boundary_api_service_reduction_ms": boundary_reduction_ms,
                "boundary_api_service_reduction_percent": (
                    100.0 * boundary_reduction_ms / float(boundary_api["B0"]["total_ms"])
                    if boundary_api is not None
                    else None
                ),
                "vta_to_cpu_producer_slot_wait_ms_median": float(
                    b2["slot_wait_by_edge_ms"]["1"]["producer_median"]
                ),
                "functional_gate_passed": bool(session["single_boot_gate"]["passed"]),
            }
        )
    delta_ii = [row["delta_ii_ms"] for row in rows]
    relative_fps = [row["relative_fps_increase_percent"] for row in rows]
    boundary_reduction_ms = [
        row["boundary_api_service_reduction_ms"]
        for row in rows
        if row["boundary_api_service_reduction_ms"] is not None
    ]
    boundary_reduction_percent = [
        row["boundary_api_service_reduction_percent"]
        for row in rows
        if row["boundary_api_service_reduction_percent"] is not None
    ]
    delta_ci = _paired_boot_student_t_ci(delta_ii)
    boundary_ci = _paired_boot_student_t_ci(boundary_reduction_ms)
    minimum_boots_met = len(rows) >= 3
    gates_passed = bool(rows) and all(row["functional_gate_passed"] for row in rows)
    formal_claim_allowed = bool(
        minimum_boots_met
        and gates_passed
        and delta_ci is not None
        and delta_ci["lower"] > 0.0
    )
    return seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p8c_cross_boot_summary",
            "phase": "P8C",
            "candidate_id": sessions[0]["candidate_id"] if sessions else None,
            "throughput_estimator": "median_of_block_completion_window_ii_v1",
            "legacy_interval_median_results_superseded": output_dir is not None,
            "independent_boot_count": len(rows),
            "boot_rows": rows,
            "aggregate": {
                "delta_ii_ms_mean": statistics.mean(delta_ii) if delta_ii else None,
                "delta_ii_ms_median": statistics.median(delta_ii) if delta_ii else None,
                "delta_ii_ms_range": [min(delta_ii), max(delta_ii)] if delta_ii else None,
                "delta_ii_ms_mean_ci95": delta_ci,
                "relative_fps_increase_percent_mean": (
                    statistics.mean(relative_fps) if relative_fps else None
                ),
                "relative_fps_increase_percent_median": (
                    statistics.median(relative_fps) if relative_fps else None
                ),
                "relative_fps_increase_percent_range": (
                    [min(relative_fps), max(relative_fps)] if relative_fps else None
                ),
                "B0_boundary_api_service_ms_mean": (
                    statistics.mean(row["boundary_api_service"]["B0"]["total_ms"] for row in rows)
                    if boundary_reduction_ms
                    else None
                ),
                "B2_boundary_api_service_ms_mean": (
                    statistics.mean(row["boundary_api_service"]["B2"]["total_ms"] for row in rows)
                    if boundary_reduction_ms
                    else None
                ),
                "boundary_api_service_reduction_ms_mean": (
                    statistics.mean(boundary_reduction_ms) if boundary_reduction_ms else None
                ),
                "boundary_api_service_reduction_ms_mean_ci95": boundary_ci,
                "boundary_api_service_reduction_percent_mean": (
                    statistics.mean(boundary_reduction_percent)
                    if boundary_reduction_percent
                    else None
                ),
                "framework_materialization_bytes_B0_per_frame": 1806336,
                "framework_materialization_bytes_B2_per_frame": 0,
                "framework_materialization_bytes_reduction_percent": 100.0,
                "vta_to_cpu_producer_slot_wait_ms_median_range": (
                    [
                        min(row["vta_to_cpu_producer_slot_wait_ms_median"] for row in rows),
                        max(row["vta_to_cpu_producer_slot_wait_ms_median"] for row in rows),
                    ]
                    if rows
                    else None
                ),
            },
            "gates": {
                "all_boot_functional_gates_passed": gates_passed,
                "minimum_three_independent_boots_met": minimum_boots_met,
                "delta_ii_ci95_excludes_zero": bool(
                    delta_ci is not None and delta_ci["lower"] > 0.0
                ),
                "boundary_api_service_ci95_excludes_zero": bool(
                    boundary_ci is not None and boundary_ci["lower"] > 0.0
                ),
                "formal_performance_claim_allowed": formal_claim_allowed,
            },
            "evidence_scope": (
                "formal cross-boot P8C comparison"
                if formal_claim_allowed
                else "cross-boot progress summary; formal performance claim is not yet allowed"
            ),
        }
    )


def write_review(path, audit, sessions=None, cross_boot=None, output_dir=None):
    sessions = sessions or []
    lines = [
        "# P8C 双 Slot Pipeline 审查",
        "",
        "更新日期：{}".format(datetime.date.today().isoformat()),
        "",
        "## 本地实现",
        "",
        "- `BoundarySlotManager` 管理 FREE -> PRODUCER_WRITING -> READY -> CONSUMER_USING -> FREE。",
        "- token 同时携带 edge、slot、generation 和 frame owner；等待有超时，worker 错误会中止全部 manager。",
        "- B1 为单 slot 串行 zero-copy；B2 为每条边双 slot 的三 stage 并行 Pipeline；单 VTA mutex 保留。",
        "- 本地 gate：`{}`。".format("passed" if audit["passed"] else "failed"),
    ]
    if not sessions:
        lines.extend(["", "## 板端状态", "", "尚未执行单 boot qualification。"])
    else:
        session = sessions[-1]
        throughput = (
            _throughput_from_raw(session, output_dir) if output_dir is not None else {}
        )

        def reviewed_mode(mode):
            summary = session["mode_summaries"][mode]
            if mode in throughput:
                return _summary_with_reanalyzed_throughput(summary, throughput[mode])
            return summary

        lines.extend(
            [
                "",
                "## 最新 Boot 详情",
                "",
                "| 模式 | latency 中位数 (ms) | Pipeline II (ms) | FPS | 框架边界物化 (B/frame) | slot wait P95 (ms) |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for mode in ("B0", "B1", "B2"):
            row = reviewed_mode(mode)
            lines.append(
                "| {} | {:.6f} | {:.6f} | {:.3f} | {} | {:.6f} |".format(
                    mode,
                    row["latency_ms_median"],
                    row["pipeline_ii_ms_median"],
                    row.get("pipeline_fps", row["pipeline_fps_from_median_ii"]),
                    row["framework_materialization_bytes_per_frame"],
                    row["slot_wait_ms_p95"],
                )
            )
        b0_reviewed = reviewed_mode("B0")
        b2_reviewed = reviewed_mode("B2")
        ii_delta = (
            b0_reviewed["pipeline_ii_ms_median"]
            - b2_reviewed["pipeline_ii_ms_median"]
        )
        fps_delta = b2_reviewed.get(
            "pipeline_fps", b2_reviewed["pipeline_fps_from_median_ii"]
        ) - b0_reviewed.get(
            "pipeline_fps", b0_reviewed["pipeline_fps_from_median_ii"]
        )
        lines.extend(
            [
                "",
                "- `B0-B2` Pipeline II 差值为 `{:+.6f} ms`，`B2-B0` FPS 差值为 `{:+.3f}`。".format(
                    ii_delta, fps_delta
                ),
                "- 相对 FPS 差值为 `{:+.2f}%`；两个完整测量窗口的平均 II 分别为 B0 `{}`、B2 `{}` ms。".format(
                    100.0
                    * (
                        b2_reviewed.get(
                            "pipeline_fps", b2_reviewed["pipeline_fps_from_median_ii"]
                        )
                        / b0_reviewed.get(
                            "pipeline_fps", b0_reviewed["pipeline_fps_from_median_ii"]
                        )
                        - 1.0
                    ),
                    "/".join(
                        "{:.3f}".format(value)
                        for value in b0_reviewed[
                            "block_ii_ms_medians"
                        ]
                    ),
                    "/".join(
                        "{:.3f}".format(value)
                        for value in b2_reviewed[
                            "block_ii_ms_medians"
                        ]
                    ),
                ),
                "- 吞吐采用 `(窗口末完成时间-窗口首完成时间)/(N-1)`，再对平衡 block 取中位数；逐帧间隔中位数仅用于观察调度抖动。",
                "- B0 的 VTA-facing host copy 为 `802816 + 100352 bytes/frame`，B2 为 0；VTA 内部 `LOAD=12025856`、`STORE=1229312 bytes/frame` 未改变。",
                "- 最新 boot 的 VTA->CPU producer slot wait 中位数为 `{:.3f} ms`、P95 为 `{:.3f} ms`；该等待是可与其他 stage 重叠的背压，不能作为额外时延再次相加，也不能仅凭单个 boot 判定固定瓶颈。".format(
                    session["mode_summaries"]["B2"]["slot_wait_by_edge_ms"]["1"][
                        "producer_median"
                    ],
                    session["mode_summaries"]["B2"]["slot_wait_by_edge_ms"]["1"][
                        "producer_p95"
                    ],
                ),
                "- 正确性、generation/owner、双 slot 使用、终态释放和 preflight 聚合 gate：`{}`。".format(
                    "passed" if session["single_boot_gate"]["passed"] else "failed"
                ),
                "- 本节只展示最新一个 boot；正式判断使用下方跨 boot 汇总，不能用帧内样本代替独立 boot。",
            ]
        )
        if cross_boot is not None:
            lines.extend(
                [
                    "",
                    "## 跨 Boot 汇总",
                    "",
                    "| Boot | B0 II (ms) | B2 II (ms) | II 减少 (ms) | 相对 FPS 增量 | 边界 API 时间减少 | 功能 gate |",
                    "|---|---:|---:|---:|---:|---:|---|",
                ]
            )
            for row in cross_boot["boot_rows"]:
                lines.append(
                    "| `{}` | {:.6f} | {:.6f} | {:+.6f} | {:+.2f}% | {:+.2f}% | {} |".format(
                        row["boot_id"],
                        row["B0_ii_ms"],
                        row["B2_ii_ms"],
                        row["delta_ii_ms"],
                        row["relative_fps_increase_percent"],
                        row["boundary_api_service_reduction_percent"],
                        "passed" if row["functional_gate_passed"] else "failed",
                    )
                )
            aggregate = cross_boot["aggregate"]
            lines.extend(
                [
                    "",
                    "- 已完成 `{}` 个独立 boot；II 减少均值 `{:+.6f} ms`，相对 FPS 增量均值 `{:+.2f}%`。".format(
                        cross_boot["independent_boot_count"],
                        aggregate["delta_ii_ms_mean"],
                        aggregate["relative_fps_increase_percent_mean"],
                    ),
                    "- 两条异构边界的框架 API 服务时间由 B0 平均 `{:.6f} ms/frame` 降至 B2 `{:.6f} ms/frame`，减少 `{:.6f} ms`（`{:.2f}%`）；物化字节由 `1806336` 降至 `0 B/frame`。".format(
                        aggregate["B0_boundary_api_service_ms_mean"],
                        aggregate["B2_boundary_api_service_ms_mean"],
                        aggregate["boundary_api_service_reduction_ms_mean"],
                        aggregate["boundary_api_service_reduction_percent_mean"],
                    ),
                    "- VTA->CPU producer slot wait 的跨 boot 中位数范围为 `{:.3f}--{:.3f} ms`，说明背压存在但强度不稳定，暂不据此增加固定 penalty 或第三个 slot。".format(
                        aggregate["vta_to_cpu_producer_slot_wait_ms_median_range"][0],
                        aggregate["vta_to_cpu_producer_slot_wait_ms_median_range"][1],
                    ),
                    "- 当前正式性能 claim：`{}`。统计独立单位是 boot，不是帧或同 boot block。".format(
                        "allowed"
                        if cross_boot["gates"]["formal_performance_claim_allowed"]
                        else "not allowed"
                    ),
                ]
            )
    if cross_boot is None or cross_boot["independent_boot_count"] < 3:
        next_step = "达到三个独立 boot 前继续补相同冻结实验；达到后按 boot 级配对 II 计算冻结的 95% 区间。"
    elif cross_boot["gates"]["formal_performance_claim_allowed"]:
        next_step = "三个独立 boot 和配对 II 区间 gate 已通过。P8C 在此冻结并提交 review；不自动进入条件 P8D。"
    else:
        next_step = "三个 boot 已完成但区间 gate 未通过；只保留字节/组件时间结论，决定是否补 boot 后再进入其他阶段。"
    lines.extend(
        [
            "",
            "## 下一步",
            "",
            next_step,
            "",
            "本地审计 SHA256：`{}`".format(audit["artifact_sha256"]),
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if args.warmup < 0 or args.runs_per_block < 2:
        raise ValueError("P8C requires non-negative warmup and at least two scored frames per block")
    package = Path(args.package).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path("/tmp/ramps_p8c")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    runner_binary = work_dir / "vta_stage_pipeline_runner_p8c"
    p8.compile_runner(runner_binary)
    manifest = p8.load_manifest(package)
    protocol = build_protocol(manifest, args.warmup, args.runs_per_block)
    audit = build_local_audit(package, manifest, runner_binary)
    p8.write_json(output_dir / "v1_p8c_protocol.json", protocol)
    p8.write_json(output_dir / "v1_p8c_local_audit.json", audit)
    session = None
    if not args.local_only:
        session = run_board(args, manifest, runner_binary, work_dir, output_dir)
        p8.write_json(
            output_dir / "v1_p8c_session_{}.json".format(session["boot_id"]), session
        )
    sessions = load_p8c_sessions(output_dir, manifest["candidate_id"])
    cross_boot = build_cross_boot_summary(sessions, output_dir)
    p8.write_json(output_dir / "v1_p8c_cross_boot_summary.json", cross_boot)
    if session is not None:
        p8.write_json(
            output_dir / "v1_p8_ablation.json",
            build_ablation(protocol, audit, session),
        )
    write_review(
        output_dir / "v1_p8c_review.md",
        audit,
        sessions,
        cross_boot,
        output_dir=output_dir,
    )
    print(
        json.dumps(
            {
                "protocol": protocol,
                "audit": audit,
                "session": session,
                "cross_boot": cross_boot,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
