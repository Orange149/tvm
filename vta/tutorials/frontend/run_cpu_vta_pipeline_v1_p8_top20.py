#!/usr/bin/env python3
"""Compare ordinary-copy and dual-slot zero-copy pipelines on the frozen Top-20."""

from __future__ import annotations

import argparse
import datetime
import json
import re
import shlex
import shutil
import statistics
import tarfile
from pathlib import Path

import numpy as np

import run_cpu_vta_pipeline_v1_p8a as p8
import run_cpu_vta_pipeline_v1_p8c as p8c
from freeze_cpu_vta_pipeline_v1 import file_sha256, seal_artifact


ROOT = p8.ROOT
REPORT_ROOT = p8.REPORT_ROOT
DEFAULT_TOP20 = (
    REPORT_ROOT / "v1_p7_top20_board_20260904/top20_board_summary.json"
)
DEFAULT_PACKAGES = REPORT_ROOT / "v1_p7_top20_board_20260904/packages"
DEFAULT_OUTPUT = REPORT_ROOT / "v1_p8_top20_zero_copy"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-host", default="192.168.1.247")
    parser.add_argument("--ssh-user", default="root")
    parser.add_argument("--rpc-port", type=int, default=9090)
    parser.add_argument("--top20-summary", default=str(DEFAULT_TOP20))
    parser.add_argument("--packages-dir", default=str(DEFAULT_PACKAGES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--remote-root", default="/var/volatile/ramps_p8_top20")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs-per-block", type=int, default=10)
    parser.add_argument(
        "--ranks",
        default="",
        help="Comma-separated frozen ranks to run; empty runs all 20.",
    )
    parser.add_argument("--timeout-s", type=int, default=1200)
    parser.add_argument("--keep-remote", action="store_true")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument(
        "--reanalyze-existing",
        action="store_true",
        help="Recompute an existing output directory without board access.",
    )
    return parser.parse_args()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def topology_signature(manifest):
    return tuple(
        (stage["device"], tuple(stage["unit_names"])) for stage in manifest["stages"]
    )


def candidate_signature(candidate_id):
    parts = []
    for token in candidate_id.split("__"):
        match = re.fullmatch(r"(cpu|vta)-(\d+)-(\d+)_t(\d+)", token)
        if match is None:
            raise ValueError("invalid candidate segment {}".format(token))
        device, start_text, end_text, _threads = match.groups()
        start, end = int(start_text), int(end_text)
        parts.append((device, tuple(range(start, end + 1))))
    return tuple(parts)


def unit_index_from_packages(packages):
    names = []
    for package in packages:
        manifest = load_json(package / "manifest.json")
        for stage in manifest["stages"]:
            names.extend(stage["unit_names"])
    ordered = []
    for name in names:
        if name not in ordered:
            ordered.append(name)
    if len(ordered) != 21:
        raise ValueError("expected 21 ordered ResNet-18 units, got {}".format(len(ordered)))
    return {name: index for index, name in enumerate(ordered)}


def indexed_topology_signature(manifest, unit_index):
    return tuple(
        (stage["device"], tuple(unit_index[name] for name in stage["unit_names"]))
        for stage in manifest["stages"]
    )


def discover_packages(packages_dir):
    packages = sorted(Path(packages_dir).glob("*/package"))
    if not packages:
        raise ValueError("no frozen Top-20 packages found")
    unit_index = unit_index_from_packages(packages)
    result = {}
    for package in packages:
        manifest = load_json(package / "manifest.json")
        signature = indexed_topology_signature(manifest, unit_index)
        if signature in result:
            raise ValueError("duplicate package topology")
        result[signature] = (package, manifest)
    return result


def manifest_for_candidate(base_manifest, row):
    manifest = json.loads(json.dumps(base_manifest))
    threads = [int(value) for value in row["stage_runtime_threads"]]
    if len(threads) != len(manifest["stages"]):
        raise ValueError("candidate stage/thread count mismatch")
    thread_map = {"stage{}".format(i): value for i, value in enumerate(threads)}
    affinities = {
        "stage{}".format(i): (list(range(threads[i])) if stage["device"] == "cpu" else [])
        for i, stage in enumerate(manifest["stages"])
    }
    manifest["candidate_id"] = row["candidate_id"]
    manifest["stage_runtime_threads"] = threads
    manifest["pipeline_stage_runtime_threads"] = dict(thread_map)
    manifest["serial_stage_runtime_threads"] = dict(thread_map)
    manifest["stage_cpu_affinity"] = {
        "policy": "independent_overlapping_prefix_masks_v1",
        "pipeline": dict(affinities),
        "serial": dict(affinities),
    }
    return manifest


def make_second_input(package, manifest, output):
    shape = tuple(int(value) for value in manifest["input_shape"])
    source = np.fromfile(Path(package) / manifest["input_file"], dtype=np.float32)
    if source.size != int(np.prod(shape)):
        raise ValueError("input.bin does not match manifest shape")
    shifted = np.roll(source.reshape(shape), shift=1, axis=-1).copy()
    if np.array_equal(source, shifted.reshape(-1)):
        raise ValueError("generated second input is not distinct")
    shifted.tofile(output)


def boundary_contract(manifest, edge_index):
    producer = manifest["stages"][edge_index]
    consumer = manifest["stages"][edge_index + 1]
    producer_slots = producer["output_schema"]["slots"]
    consumer_slots = consumer["input_schema"]["slots"]
    if len(producer_slots) != len(consumer_slots) or not producer_slots:
        raise ValueError("boundary arity mismatch")
    tensors = []
    for left, right in zip(producer_slots, consumer_slots):
        left_contract = (left["dtype"], tuple(left["shape"]))
        right_contract = (right["dtype"], tuple(right["shape"]))
        if left_contract != right_contract:
            raise ValueError("boundary tensor contract mismatch")
        bits = int("".join(ch for ch in left["dtype"] if ch.isdigit()))
        elements = 1
        for extent in left["shape"]:
            elements *= int(extent)
        tensors.append(
            {"shape": list(left["shape"]), "dtype": left["dtype"], "bytes": elements * bits // 8}
        )
    total = sum(tensor["bytes"] for tensor in tensors)
    if total != int(producer["output_bytes"]):
        raise ValueError("boundary byte count mismatch")
    return {
        "producer_stage": edge_index,
        "consumer_stage": edge_index + 1,
        "direction": "{}_to_{}".format(producer["device"], consumer["device"]),
        "tensor_count": len(tensors),
        "tensors": tensors,
        "bytes": total,
    }


def archive_package(package, archive):
    with tarfile.open(archive, "w:gz") as output:
        for path in sorted(Path(package).rglob("*")):
            output.add(path, arcname=str(path.relative_to(package)), recursive=False)


def percentile(values, percentile_value):
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile_value / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def block_window_ii_ms(rows):
    ordered = sorted(rows, key=lambda row: int(row["completion_index"]))
    completion = [float(row["completion_ms"]) for row in ordered]
    if len(completion) < 2 or completion[-1] <= completion[0]:
        raise ValueError("a throughput block needs increasing completion timestamps")
    return (completion[-1] - completion[0]) / (len(completion) - 1)


def block_order(rank):
    return ("B0", "B2", "B2", "B0") if rank % 2 else ("B2", "B0", "B0", "B2")


def parse_ranks(value):
    if not value.strip():
        return None
    ranks = {int(item.strip()) for item in value.split(",") if item.strip()}
    if not ranks or any(rank < 1 or rank > 20 for rank in ranks):
        raise ValueError("--ranks must contain values in 1..20")
    return ranks


def summarize_mode(mode, blocks, manifest):
    stage_count = len(manifest["stages"])
    edge_count = stage_count - 1
    edge_contracts = [boundary_contract(manifest, edge) for edge in range(edge_count)]
    rows = [row for block in blocks for row in block["rows"]]
    intervals = []
    block_interval_medians = []
    block_window_ii = []
    for block in blocks:
        ordered = sorted(block["rows"], key=lambda row: int(row["completion_index"]))
        completion = [float(row["completion_ms"]) for row in ordered]
        current = [right - left for left, right in zip(completion, completion[1:])]
        if not current or any(value <= 0.0 for value in current):
            raise ValueError("invalid completion intervals")
        intervals.extend(current)
        block_interval_medians.append(statistics.median(current))
        block_window_ii.append(block_window_ii_ms(block["rows"]))

    signatures = {
        str(index): sorted(
            {
                tuple(item["fnv1a64"] for item in row["raw_outputs"])
                for row in rows
                if int(row["input_index"]) == index
            }
        )
        for index in (0, 1)
    }
    edge_api = {}
    total_api = [0.0] * len(rows)
    for edge, contract in enumerate(edge_contracts):
        values = [
            float(row["stage{}_get_ms".format(edge)])
            + float(row["stage{}_set_ms".format(edge + 1)])
            for row in rows
        ]
        edge_api[str(edge)] = {
            "direction": contract["direction"],
            "bytes": contract["bytes"],
            "service_ms_median": statistics.median(values),
        }
        total_api = [left + right for left, right in zip(total_api, values)]

    materialization = 2 * sum(contract["bytes"] for contract in edge_contracts) if mode == "B0" else max(
        int(row["p8_framework_materialization_bytes"]) for row in rows
    )
    summary = {
        "mode": mode,
        "scored_frames": len(rows),
        "input_indices": sorted({int(row["input_index"]) for row in rows}),
        "output_signatures_by_input": signatures,
        "latency_ms_median": statistics.median(float(row["total_latency_ms"]) for row in rows),
        "pipeline_interval_ms_median": statistics.median(intervals),
        "pipeline_ii_ms_median": statistics.median(block_window_ii),
        "pipeline_fps": 1000.0 / statistics.median(block_window_ii),
        "throughput_estimator": "median_of_block_completion_window_ii_v1",
        "block_window_ii_ms": block_window_ii,
        "block_interval_median_ms": block_interval_medians,
        "block_ii_ms_medians": block_window_ii,
        "stage_ms_median": {
            "stage{}".format(stage): statistics.median(
                float(row["stage{}_ms".format(stage)]) for row in rows
            )
            for stage in range(stage_count)
        },
        "stage_scheduled_service_ms_median": (
            {
                "stage{}".format(stage): statistics.median(
                    float(row["stage{}_scheduled_service_ms".format(stage)]) for row in rows
                )
                for stage in range(stage_count)
            }
            if "stage0_scheduled_service_ms" in rows[0]
            else None
        ),
        "vta_mutex_wait_ms_median": {
            "stage{}".format(stage): statistics.median(
                float(row["stage{}_vta_mutex_wait_ms".format(stage)]) for row in rows
            )
            for stage, stage_spec in enumerate(manifest["stages"])
            if stage_spec["device"] == "vta"
            and row_has_numeric_field(rows[0], "stage{}_vta_mutex_wait_ms".format(stage))
        },
        "vta_run_call_ms_median": {
            "stage{}".format(stage): statistics.median(
                float(row["stage{}_vta_run_call_ms".format(stage)]) for row in rows
            )
            for stage, stage_spec in enumerate(manifest["stages"])
            if stage_spec["device"] == "vta"
            and row_has_numeric_field(rows[0], "stage{}_vta_run_call_ms".format(stage))
        },
        "edge_api_service": edge_api,
        "boundary_api_service_ms_median_total": statistics.median(total_api),
        "framework_materialization_bytes_per_frame": materialization,
    }
    if mode == "B2":
        boundaries = [
            boundary
            for row in rows
            for boundary in row.get("p8_boundaries", [])
            if boundary is not None
        ]
        summary.update(
            {
                "slot_ids_by_edge": {
                    str(edge): sorted(
                        {
                            int(boundary["slot_id"])
                            for boundary in boundaries
                            if int(boundary["edge_index"]) == edge
                        }
                    )
                    for edge in range(edge_count)
                },
                "every_frame_has_all_edges": all(
                    len(row.get("p8_boundaries", [])) == edge_count
                    and all(boundary is not None for boundary in row["p8_boundaries"])
                    for row in rows
                ),
                "slot_addresses_aligned": all(
                    int(boundary["physical_address"], 16) % 256 == 0
                    for boundary in boundaries
                ),
                "slot_physical_ranges_non_overlapping": all(
                    block_ranges_non_overlapping(block["rows"]) for block in blocks
                ),
                "slot_wait_ms_p95": percentile(
                    [float(row.get("p8_total_slot_wait_ms", 0.0)) for row in rows], 95
                ),
            }
        )
    return summary


def row_has_numeric_field(row, key):
    return key in row and row[key] is not None


def block_ranges_non_overlapping(rows):
    observed = {}
    for row in rows:
        for boundary in row.get("p8_boundaries", []):
            if boundary is None:
                continue
            addresses = boundary.get("physical_addresses") or [boundary["physical_address"]]
            sizes = boundary.get("tensor_bytes") or [boundary["boundary_bytes"]]
            if len(addresses) != len(sizes):
                return False
            for tensor_index, (address, size) in enumerate(zip(addresses, sizes)):
                key = (
                    int(boundary["edge_index"]),
                    int(boundary["slot_id"]),
                    tensor_index,
                )
                value = (int(address, 16), int(size))
                if key in observed and observed[key] != value:
                    return False
                observed[key] = value
    ranges = sorted((start, start + size) for start, size in observed.values())
    return all(left[1] <= right[0] for left, right in zip(ranges, ranges[1:]))


def run_block(args, manifest, mode, occurrence, remote_package, remote_runner):
    name = "{}_{}".format(mode.lower(), occurrence)
    output_name = name + ".jsonl"
    profile_name = "profile_" + name
    runner_args = p8c._runner_args(
        manifest, mode, args.warmup, args.runs_per_block, output_name, profile_name
    )
    command = (
        "cd {package} && export LD_LIBRARY_PATH=$PWD && "
        "export LD_PRELOAD=$PWD/libtvm_runtime.so:$PWD/libvta.so && "
        "export TVM_NUM_THREADS=4 TVM_THREAD_POOL_SPIN_COUNT=0 && "
        "export AXU5EVB_DRIVER_POST_START_SLEEP_NS=1000 "
        "AXU5EVB_DRIVER_POLL_SLEEP_NS=1000 AXU5EVB_DRIVER_SAFE_COPY=0 && "
        "{runner} {args} && cat {output}"
    ).format(
        package=shlex.quote(remote_package),
        runner=shlex.quote(remote_runner),
        args=shlex.join(runner_args),
        output=shlex.quote(output_name),
    )
    stdout = p8._ssh(args, command)
    rows = [json.loads(line) for line in stdout.splitlines() if line.startswith("{")]
    if len(rows) != args.runs_per_block:
        raise RuntimeError("{} returned {} scored rows".format(name, len(rows)))
    profile = load_json_text(
        p8._ssh(
            args,
            "cat {}/profile_{}/benchmark_totals_status.json".format(
                shlex.quote(remote_package), name
            ),
        )
    )
    return {"mode": mode, "occurrence": occurrence, "rows": rows, "profile": profile}


def load_json_text(text):
    return json.loads(text)


def prepare_remote_package(args, package, key, work_dir):
    archive = work_dir / "package_{}.tar.gz".format(key)
    archive_package(package, archive)
    remote_package = args.remote_root.rstrip("/") + "/packages/" + key
    remote_archive = args.remote_root.rstrip("/") + "/" + archive.name
    p8._scp(args, archive, remote_archive)
    p8._ssh(
        args,
        "mkdir -p {package} && cd {package} && tar -xzf {archive}".format(
            package=shlex.quote(remote_package), archive=shlex.quote(remote_archive)
        ),
    )
    p8._ssh(args, "rm -f {}".format(shlex.quote(remote_archive)))
    second_input = work_dir / "input_b_{}.bin".format(key)
    manifest = load_json(package / "manifest.json")
    make_second_input(package, manifest, second_input)
    input_list = work_dir / "p8_inputs_{}.txt".format(key)
    input_list.write_text("input.bin\ninput_b.bin\n", encoding="ascii")
    p8._scp(args, second_input, remote_package + "/input_b.bin")
    p8._scp(args, input_list, remote_package + "/p8_inputs.txt")
    return remote_package


def load_existing_blocks(output_dir, rank):
    occurrences = {"B0": 0, "B2": 0}
    blocks = []
    for order_index, mode in enumerate(block_order(rank)):
        occurrence = occurrences[mode]
        occurrences[mode] += 1
        stem = "rank{:02d}_{}_{}_".format(rank, mode.lower(), occurrence)
        raw_matches = sorted(output_dir.glob(stem + "*.jsonl"))
        if len(raw_matches) != 1:
            raise ValueError("expected one existing raw block for {}, got {}".format(stem, len(raw_matches)))
        raw_path = raw_matches[0]
        profile_matches = sorted(output_dir.glob(raw_path.stem + "_profile.json"))
        if len(profile_matches) != 1:
            raise ValueError(
                "expected one existing profile for {}, got {}".format(stem, len(profile_matches))
            )
        profile_path = profile_matches[0]
        blocks.append(
            {
                "mode": mode,
                "occurrence": occurrence,
                "order_index": order_index,
                "rows": [
                    json.loads(line)
                    for line in raw_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ],
                "profile": load_json(profile_path),
                "raw_artifact": {
                    "path": raw_path.name,
                    "sha256": file_sha256(raw_path),
                    "profile_path": profile_path.name,
                    "profile_sha256": file_sha256(profile_path),
                },
            }
        )
    return blocks


def reanalyze_existing(output_dir, candidates, protocol):
    source_path = output_dir / "v1_p8_top20_summary.json"
    source = load_json(source_path)
    result_rows = []
    for row, _package, manifest in candidates:
        blocks = load_existing_blocks(output_dir, int(row["natural_static_rank"]))
        result = candidate_result(row, manifest, blocks)
        result["raw_artifacts"] = [block["raw_artifact"] for block in blocks]
        result_rows.append(result)
    session = seal_artifact(
        {
            "schema_version": 2,
            "kind": "cpu_vta_pipeline_v1_p8_top20_reanalysis",
            "reanalyzed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "boot_id": source["boot_id"],
            "board": source.get("board"),
            "source_summary": {
                "path": source_path.name,
                "sha256": file_sha256(source_path),
            },
            "protocol_sha256": protocol["artifact_sha256"],
            "throughput_estimator": "median_of_block_completion_window_ii_v1",
            "rows": result_rows,
        }
    )
    session["aggregate"] = aggregate(session)
    session = seal_artifact({key: value for key, value in session.items() if key != "artifact_sha256"})
    p8.write_json(output_dir / "v1_p8_top20_reanalysis.json", session)
    write_review(output_dir / "v1_p8_top20_review_reanalysis.md", session)
    return session


def candidate_result(row, manifest, blocks):
    summaries = {
        mode: summarize_mode(mode, [block for block in blocks if block["mode"] == mode], manifest)
        for mode in ("B0", "B2")
    }
    b0 = summaries["B0"]
    b2 = summaries["B2"]
    edge_count = len(manifest["stages"]) - 1
    expected_slots = {str(edge): [0, 1] for edge in range(edge_count)}
    gate = {
        "B0_B2_outputs_match": b0["output_signatures_by_input"] == b2["output_signatures_by_input"],
        "alternating_inputs_present": b0["input_indices"] == [0, 1] and b2["input_indices"] == [0, 1],
        "B2_uses_both_slots_on_all_edges": b2["slot_ids_by_edge"] == expected_slots,
        "B2_every_frame_has_all_edges": b2["every_frame_has_all_edges"],
        "B2_slot_addresses_aligned": b2["slot_addresses_aligned"],
        "B2_slot_ranges_non_overlapping": b2["slot_physical_ranges_non_overlapping"],
        "B2_framework_materialization_zero": b2["framework_materialization_bytes_per_frame"] == 0,
        "B0_framework_materialization_nonzero": b0["framework_materialization_bytes_per_frame"] > 0,
    }
    gate["passed"] = all(gate.values())
    result = {
        "natural_static_rank": int(row["natural_static_rank"]),
        "candidate_id": row["candidate_id"],
        "stage_runtime_threads": row["stage_runtime_threads"],
        "stage_count": len(manifest["stages"]),
        "mode_order": list(block_order(int(row["natural_static_rank"]))),
        "mode_summaries": summaries,
        "effects": {
            "ii_reduction_ms": b0["pipeline_ii_ms_median"] - b2["pipeline_ii_ms_median"],
            "fps_increase": b2["pipeline_fps"] - b0["pipeline_fps"],
            "fps_increase_percent": 100.0 * (b2["pipeline_fps"] / b0["pipeline_fps"] - 1.0),
            "boundary_api_service_reduction_ms": (
                b0["boundary_api_service_ms_median_total"]
                - b2["boundary_api_service_ms_median_total"]
            ),
            "boundary_api_service_reduction_percent": 100.0
            * (
                1.0
                - b2["boundary_api_service_ms_median_total"]
                / b0["boundary_api_service_ms_median_total"]
            ),
            "framework_materialization_bytes_eliminated_per_frame": b0[
                "framework_materialization_bytes_per_frame"
            ],
        },
        "functional_gate": gate,
    }
    refresh_candidate_estimates(result)
    return result


def refresh_candidate_estimates(result):
    """Use block-level estimates so each repeated process receives equal weight."""
    for mode in ("B0", "B2"):
        summary = result["mode_summaries"][mode]
        if "pipeline_interval_ms_median" not in summary:
            summary["pipeline_interval_ms_median"] = summary["pipeline_ii_ms_median"]
        block_estimates = summary.get("block_window_ii_ms", summary["block_ii_ms_medians"])
        estimate = statistics.median(block_estimates)
        summary["pipeline_ii_ms_median"] = estimate
        summary["pipeline_fps"] = 1000.0 / estimate
    b0 = result["mode_summaries"]["B0"]
    b2 = result["mode_summaries"]["B2"]
    paired_effects = [
        100.0 * (b0_ii / b2_ii - 1.0)
        for b0_ii, b2_ii in zip(
            b0.get("block_window_ii_ms", b0["block_ii_ms_medians"]),
            b2.get("block_window_ii_ms", b2["block_ii_ms_medians"]),
        )
    ]
    result["effects"].update(
        {
            "ii_reduction_ms": b0["pipeline_ii_ms_median"] - b2["pipeline_ii_ms_median"],
            "fps_increase": b2["pipeline_fps"] - b0["pipeline_fps"],
            "fps_increase_percent": 100.0 * (b2["pipeline_fps"] / b0["pipeline_fps"] - 1.0),
            "paired_block_fps_increase_percent": paired_effects,
            "paired_block_effect_direction_consistent": paired_effects[0] * paired_effects[1] > 0.0,
        }
    )
    result["repeatability_diagnostics"] = {
        mode: {
            "block_ii_relative_spread_percent": 100.0
            * (max(summary["block_ii_ms_medians"]) - min(summary["block_ii_ms_medians"]))
            / statistics.mean(summary["block_ii_ms_medians"]),
            "within_5_percent": (
                (max(summary["block_ii_ms_medians"]) - min(summary["block_ii_ms_medians"]))
                / statistics.mean(summary["block_ii_ms_medians"])
                <= 0.05
            ),
        }
        for mode, summary in result["mode_summaries"].items()
    }


def aggregate(session):
    rows = session["rows"]
    uplifts = [float(row["effects"]["fps_increase_percent"]) for row in rows]
    boundary = [
        float(row["effects"]["boundary_api_service_reduction_percent"]) for row in rows
    ]
    return {
        "candidate_count": len(rows),
        "functional_gate_pass_count": sum(row["functional_gate"]["passed"] for row in rows),
        "candidate_count_with_positive_fps_uplift": sum(value > 0.0 for value in uplifts),
        "paired_effect_direction_consistent_count": sum(
            row["effects"]["paired_block_effect_direction_consistent"] for row in rows
        ),
        "both_mode_block_spread_within_5_percent_count": sum(
            all(item["within_5_percent"] for item in row["repeatability_diagnostics"].values())
            for row in rows
        ),
        "fps_increase_percent_mean": statistics.mean(uplifts),
        "fps_increase_percent_median": statistics.median(uplifts),
        "fps_increase_percent_range": [min(uplifts), max(uplifts)],
        "boundary_api_service_reduction_percent_mean": statistics.mean(boundary),
        "boundary_api_service_reduction_percent_median": statistics.median(boundary),
        "single_boot_only": True,
        "formal_cross_boot_claim_allowed": False,
    }


def write_review(path, session):
    summary = session["aggregate"]
    rows = session["rows"]
    lines = [
        "# P8 Top-20 Zero-Copy Board Comparison",
        "",
        "更新日期：{}".format(datetime.date.today().isoformat()),
        "",
        "同一 boot 内对冻结 Top-20 逐项执行 B0 普通拷贝与 B2 双 slot zero-copy。每个模式两个 block，奇偶 rank 采用相反顺序。",
        "",
        "- 功能 gate：`{}/{}`。".format(summary["functional_gate_pass_count"], summary["candidate_count"]),
        "- FPS 提升为正：`{}/{}`。".format(summary["candidate_count_with_positive_fps_uplift"], summary["candidate_count"]),
        "- 两次配对效应同方向：`{}/{}`；B0/B2 两种模式的 block spread 均不超过 5%：`{}/{}`。".format(
            summary["paired_effect_direction_consistent_count"], summary["candidate_count"],
            summary["both_mode_block_spread_within_5_percent_count"], summary["candidate_count"]
        ),
        "- FPS 相对增量：均值 `{:+.2f}%`，中位数 `{:+.2f}%`，范围 `{:+.2f}%--{:+.2f}%`。".format(
            summary["fps_increase_percent_mean"], summary["fps_increase_percent_median"], *summary["fps_increase_percent_range"]
        ),
        "- 边界 API 服务时间减少：均值 `{:.2f}%`，中位数 `{:.2f}%`。".format(
            summary["boundary_api_service_reduction_percent_mean"], summary["boundary_api_service_reduction_percent_median"]
        ),
        "- 吞吐口径：每个 block 使用 `(末次完成-首次完成)/(N-1)`，再对 block 等权汇总；逐帧间隔中位数只作抖动诊断。",
        "- 证据边界：这是单 boot 全候选扫描；候选之间共享同一 boot，不能替代逐候选跨 boot 置信区间。",
        "",
        "| Rank | Stages | Threads | B0 FPS | B2 FPS | FPS 增量 | B0 II (ms) | B2 II (ms) | 边界 API 减少 | 消除字节/帧 | Gate |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        b0 = row["mode_summaries"]["B0"]
        b2 = row["mode_summaries"]["B2"]
        effect = row["effects"]
        lines.append(
            "| {rank} | {stages} | `{threads}` | {b0fps:.3f} | {b2fps:.3f} | {uplift:+.2f}% | {b0ii:.3f} | {b2ii:.3f} | {boundary:.2f}% | {bytes} | {gate} |".format(
                rank=row["natural_static_rank"], stages=row["stage_count"],
                threads=",".join(str(value) for value in row["stage_runtime_threads"]),
                b0fps=b0["pipeline_fps"], b2fps=b2["pipeline_fps"],
                uplift=effect["fps_increase_percent"], b0ii=b0["pipeline_ii_ms_median"],
                b2ii=b2["pipeline_ii_ms_median"], boundary=effect["boundary_api_service_reduction_percent"],
                bytes=effect["framework_materialization_bytes_eliminated_per_frame"],
                gate="pass" if row["functional_gate"]["passed"] else "FAIL",
            )
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if args.warmup < 0 or args.runs_per_block < 2:
        raise ValueError("need non-negative warmup and at least two scored frames")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path("/tmp/ramps_p8_top20")
    runner = work_dir / "vta_stage_pipeline_runner_p8_top20"
    if not args.reanalyze_existing:
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True)
        p8.compile_runner(runner)

    package_index = discover_packages(args.packages_dir)
    top20 = load_json(args.top20_summary)["rows"]
    if len(top20) != 20:
        raise ValueError("frozen summary must contain exactly 20 candidates")
    selected_ranks = parse_ranks(args.ranks)
    candidates = []
    for row in top20:
        if selected_ranks is not None and int(row["natural_static_rank"]) not in selected_ranks:
            continue
        signature = candidate_signature(row["candidate_id"])
        if signature not in package_index:
            raise ValueError("no compiled topology for {}".format(row["candidate_id"]))
        package, base_manifest = package_index[signature]
        candidates.append((row, package, manifest_for_candidate(base_manifest, row)))

    protocol = seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p8_top20_protocol",
            "candidate_count": len(candidates),
            "selected_ranks": (
                sorted(selected_ranks) if selected_ranks is not None else list(range(1, 21))
            ),
            "modes": {"B0": "ordinary_copy_pipeline", "B2": "dual_slot_zero_copy_pipeline"},
            "warmup_per_block": args.warmup,
            "scored_frames_per_block": args.runs_per_block,
            "blocks_per_mode_per_candidate": 2,
            "order": "odd_B0_B2_B2_B0;even_B2_B0_B0_B2",
            "throughput_estimator": "median_of_block_completion_window_ii_v1",
            "throughput_estimator_definition": (
                "for each block, (last_completion-first_completion)/(N-1); "
                "report the median across balanced blocks"
            ),
            "per_frame_interval_median_is_diagnostic_only": True,
            "independence_unit": "board_boot",
            "candidate_throughput_used_for_fit": False,
        }
    )
    p8.write_json(output_dir / "v1_p8_top20_protocol.json", protocol)
    if args.reanalyze_existing:
        session = reanalyze_existing(output_dir, candidates, protocol)
        print(json.dumps(session["aggregate"], indent=2, sort_keys=True))
        return
    if args.local_only:
        print(json.dumps({"protocol": protocol, "candidate_count": len(candidates)}, indent=2))
        return

    from collect_cpu_vta_pipeline_v1_preflight import collect as collect_preflight

    preflight_args = argparse.Namespace(
        board_host=args.board_host,
        rpc_port=args.rpc_port,
        ssh_user=args.ssh_user,
        timeout_s=min(args.timeout_s, 30),
    )
    before = collect_preflight(preflight_args)
    if not before["passed"]:
        raise RuntimeError("Top-20 run refused because board preflight failed")
    properties = p8._board_properties(args)
    remote_root = args.remote_root.rstrip("/")
    p8._ssh(args, "rm -rf {0} && mkdir -p {0}/packages".format(shlex.quote(remote_root)))
    remote_runner = remote_root + "/vta_stage_pipeline_runner_p8_top20"
    p8._scp(args, runner, remote_runner)
    p8._ssh(args, "chmod 700 {}".format(shlex.quote(remote_runner)))

    remote_packages = {}
    for index, package in enumerate(sorted({str(item[1]) for item in candidates})):
        key = "topology{}".format(index)
        remote_packages[package] = prepare_remote_package(
            args, Path(package), key, work_dir
        )

    result_rows = []
    for candidate_index, (row, package, manifest) in enumerate(candidates, start=1):
        rank = int(row["natural_static_rank"])
        print(
            "[TOP20 {}/{}] rank={} {}".format(
                candidate_index, len(candidates), rank, row["candidate_id"]
            ),
            flush=True,
        )
        occurrences = {"B0": 0, "B2": 0}
        blocks = []
        for order_index, mode in enumerate(block_order(rank)):
            occurrence = occurrences[mode]
            occurrences[mode] += 1
            block = run_block(
                args, manifest, mode, occurrence, remote_packages[str(package)], remote_runner
            )
            block["order_index"] = order_index
            raw_name = "rank{:02d}_{}_{}_{}.jsonl".format(
                rank, mode.lower(), occurrence, properties["boot_id"]
            )
            raw_path = output_dir / raw_name
            raw_path.write_text(
                "".join(json.dumps(item, sort_keys=True) + "\n" for item in block["rows"]),
                encoding="utf-8",
            )
            profile_name = "rank{:02d}_{}_{}_{}_profile.json".format(
                rank, mode.lower(), occurrence, properties["boot_id"]
            )
            profile_path = output_dir / profile_name
            p8.write_json(profile_path, block["profile"])
            block["raw_artifact"] = {
                "path": raw_name,
                "sha256": file_sha256(raw_path),
                "profile_path": profile_name,
                "profile_sha256": file_sha256(profile_path),
            }
            blocks.append(block)
        result = candidate_result(row, manifest, blocks)
        result["raw_artifacts"] = [block["raw_artifact"] for block in blocks]
        result_rows.append(result)
        checkpoint = {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p8_top20_checkpoint",
            "boot_id": properties["boot_id"],
            "completed_candidates": len(result_rows),
            "rows": result_rows,
        }
        p8.write_json(output_dir / "v1_p8_top20_checkpoint.json", checkpoint)

    after = collect_preflight(preflight_args)
    session = seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p8_top20_session",
            "observed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "boot_id": properties["boot_id"],
            "board": "{}@{}".format(args.ssh_user, args.board_host),
            "board_properties": properties,
            "preflight_before": before,
            "preflight_after": after,
            "protocol_sha256": protocol["artifact_sha256"],
            "rows": result_rows,
        }
    )
    session["aggregate"] = aggregate(session)
    session = seal_artifact({key: value for key, value in session.items() if key != "artifact_sha256"})
    p8.write_json(
        output_dir / "v1_p8_top20_session_{}.json".format(properties["boot_id"]), session
    )
    p8.write_json(output_dir / "v1_p8_top20_summary.json", session)
    write_review(output_dir / "v1_p8_top20_review.md", session)
    if not args.keep_remote:
        p8._ssh(args, "rm -rf {}".format(shlex.quote(remote_root)))
    print(json.dumps(session["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
