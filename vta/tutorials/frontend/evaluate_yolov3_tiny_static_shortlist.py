#!/usr/bin/env python3
"""Apply the CPU-VTA Pipeline V1 resource ranking to YOLOv3-tiny.

The shortlist is frozen from static graph work and ResNet-derived hardware
prices before any historical YOLO throughput is loaded.  Historical YOLO
measurements are joined afterwards and are used only for retrospective audit.
"""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import itertools
import json
from pathlib import Path
import re
import statistics


THREADS = (1, 2, 3, 4)
PHYSICAL_CPU_CORES = 4
MAX_VTA_ISLANDS = 3
DEFAULT_TOP_K = 20

YOLO_CONVS = (
    # name, cin, cout, h, w, kernel
    ("conv0", 3, 16, 416, 416, 3),
    ("conv2", 16, 32, 208, 208, 3),
    ("conv4", 32, 64, 104, 104, 3),
    ("conv6", 64, 128, 52, 52, 3),
    ("conv8", 128, 256, 26, 26, 3),
    ("conv10", 256, 512, 13, 13, 3),
    ("conv12", 512, 1024, 13, 13, 3),
    ("conv13", 1024, 256, 13, 13, 1),
    ("conv18", 256, 128, 13, 13, 1),
    ("conv21", 384, 256, 26, 26, 3),
    ("conv22", 256, 255, 26, 26, 1),
    ("conv14", 256, 512, 13, 13, 3),
    ("conv15", 512, 255, 13, 13, 1),
)

SPLIT_POINTS = (
    ("data", frozenset(), ("data",)),
    ("pool0", frozenset((0,)), ("pool0",)),
    ("pool1", frozenset((0, 1)), ("pool1",)),
    ("pool2", frozenset((0, 1, 2)), ("pool2",)),
    ("pool3", frozenset((0, 1, 2, 3)), ("pool3",)),
    ("pool4_route", frozenset((0, 1, 2, 3, 4)), ("pool4_route", "route23")),
    ("trunk12", frozenset((0, 1, 2, 3, 4, 5, 6)), ("route23", "trunk34")),
    (
        "shared13",
        frozenset((0, 1, 2, 3, 4, 5, 6, 7)),
        ("route23", "shared38"),
    ),
    (
        "small_pre18",
        frozenset((0, 1, 2, 3, 4, 5, 6, 7, 8)),
        ("route23", "shared38", "small42"),
    ),
    (
        "dual_pre18_14",
        frozenset((0, 1, 2, 3, 4, 5, 6, 7, 8, 11)),
        ("route23", "small42", "big64"),
    ),
    (
        "dual_pre_logits",
        frozenset((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11)),
        ("small_pre26", "big64"),
    ),
    ("logits", frozenset(range(len(YOLO_CONVS))), ("small_logits51", "big_logits66")),
)

OUTPUT_TO_CONV = {
    "pool0": 0,
    "pool1": 1,
    "pool2": 2,
    "pool3": 3,
    "pool4_route": 4,
    "route23": 4,
    "trunk34": 6,
    "shared38": 7,
    "small42": 8,
    "small_pre26": 9,
    "small_logits51": 10,
    "big64": 11,
    "big_logits66": 12,
}


def repo_root():
    return Path(__file__).resolve().parents[3]


DEFAULT_OUTPUT_DIR = repo_root() / "vta/tutorials/frontend/report_out/resource_aware_maxplus"
HISTORICAL_ROOTS = (
    repo_root()
    / "vta/tutorials/frontend/report_out/yolov3_tiny_pipeline_baseline/"
    "20260514_resnet_style_cpu_vta_cpu_100",
    repo_root()
    / "vta/tutorials/frontend/report_out/yolov3_tiny_pipeline_baseline/"
    "20260514_yolo_multisplit100",
)
YOLO_ANCHOR_ROOT = HISTORICAL_ROOTS[0]
YOLO_ANCHOR_TOPOLOGY = "yolo_pool2_to_dual_pre_logits"
LOGITS_CONVS = frozenset((10, 12))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical_json(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def seal_artifact(payload):
    sealed = copy.deepcopy(payload)
    sealed.pop("artifact_sha256", None)
    sealed["artifact_sha256"] = hashlib.sha256(
        canonical_json(sealed).encode("utf-8")
    ).hexdigest()
    return sealed


def write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_csv(path, rows):
    rows = list(rows)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with Path(path).open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def conv_gops(indices):
    total = 0.0
    for index in indices:
        _, cin, cout, height, width, kernel = YOLO_CONVS[index]
        total += 2.0 * height * width * cin * cout * kernel * kernel / 1.0e9
    return total


def conv_output_bytes(index):
    _, _, cout, height, width, _ = YOLO_CONVS[index]
    return int(cout * height * width * 4)


def point_bytes(index):
    outputs = SPLIT_POINTS[index][2]
    if outputs == ("data",):
        return 3 * 416 * 416 * 4
    convs = {OUTPUT_TO_CONV[name] for name in outputs}
    return sum(conv_output_bytes(conv) for conv in convs)


def span_convs(start, end):
    return tuple(sorted(SPLIT_POINTS[end][1] - SPLIT_POINTS[start][1]))


def valid_vta_span(start, end):
    convs = span_convs(start, end)
    return (
        start > 0
        and end > start
        and bool(convs)
        and not (end == len(SPLIT_POINTS) - 1 and len(convs) <= 2)
    )


def enumerate_topologies(max_islands=MAX_VTA_ISLANDS):
    spans = [
        (start, end)
        for start in range(len(SPLIT_POINTS))
        for end in range(len(SPLIT_POINTS))
        if valid_vta_span(start, end)
    ]
    results = []

    def visit(cursor, chosen, remaining):
        if chosen:
            results.append(tuple(chosen))
        if remaining <= 0:
            return
        for start, end in spans:
            if start <= cursor:
                continue
            visit(end, chosen + ((start, end),), remaining - 1)

    visit(0, tuple(), int(max_islands))
    return results


def topology_id(islands):
    labels = [
        "{}_{}".format(SPLIT_POINTS[start][0], SPLIT_POINTS[end][0])
        for start, end in islands
    ]
    return "yolo_multi_{}".format("__".join(labels))


def configuration_id(islands, cpu_threads):
    return "{}__cpu_threads_{}".format(
        topology_id(islands), "_".join(str(value) for value in cpu_threads)
    )


def fit_vta_traffic(local_cost):
    points = {}
    for measurement in local_cost["fit_stage_measurements"]:
        stage = next(item for item in measurement["stages"] if item["device"] == "vta")
        profiler = measurement["serial_vta_profiler"]
        rows = float(measurement["serial_rows"])
        points[tuple(stage["unit_names"])] = (
            float(stage["logical_compute_gops_est"]),
            float(profiler["load_bytes"]) / rows,
            float(profiler["store_bytes"]) / rows,
        )
    denominator = sum(point[0] ** 2 for point in points.values())
    return {
        "load_bytes_per_logical_gop": sum(point[0] * point[1] for point in points.values())
        / denominator,
        "store_bytes_per_logical_gop": sum(point[0] * point[2] for point in points.values())
        / denominator,
    }


def fit_boundary_ownership(local_cost):
    grouped = {"cpu_to_vta": [], "vta_to_cpu": []}
    for measurement in local_cost["fit_stage_measurements"]:
        stages = measurement["stages"]
        grouped["cpu_to_vta"].append(
            (int(stages[0]["output_bytes"]), float(stages[0]["get_ms"]), float(stages[1]["set_ms"]))
        )
        grouped["vta_to_cpu"].append(
            (int(stages[1]["output_bytes"]), float(stages[2]["set_ms"]), float(stages[1]["get_ms"]))
        )
    models = {}
    for direction, raw_points in grouped.items():
        by_bytes = {}
        for byte_count, cpu_ms, vta_ms in raw_points:
            by_bytes.setdefault(byte_count, []).append((cpu_ms, vta_ms))
        points = [
            (byte_count, statistics.median(value[0] for value in values), statistics.median(value[1] for value in values))
            for byte_count, values in by_bytes.items()
        ]
        denominator = sum(point[0] ** 2 for point in points)
        models[direction] = {
            "cpu_owner_ms_per_byte": sum(point[0] * point[1] for point in points) / denominator,
            "vta_mutex_ms_per_byte": sum(point[0] * point[2] for point in points) / denominator,
        }
    return models


def build_transferred_profile(output_dir=DEFAULT_OUTPUT_DIR):
    output_dir = Path(output_dir)
    unit_schema = read_json(output_dir / "v1_unit_and_boundary_schema.json")
    cpu_measurements = read_json(output_dir / "v1_p5b_iteration2_cpu_measurements.json")
    local_cost = read_json(output_dir / "v1_local_cost_table.json")
    dma = read_json(
        output_dir / "stage4a_q_qualification/session1/dma_identification_summary.json"
    )
    total_gops = sum(
        float(unit["static_compute"]["compute_gops_est"])
        for unit in unit_schema["units"]
    )
    cpu_wall = {}
    cpu_core = {}
    for session in cpu_measurements["atomic_sessions"]:
        threads = int(session["threads"])
        cpu_wall[threads] = sum(
            float(unit["run_ms_median"]) for unit in session["unit_measurements"]
        ) / total_gops
        cpu_core[threads] = sum(
            float(unit["run_process_cpu_ms_median"])
            for unit in session["unit_measurements"]
        ) / total_gops
    memory = {
        (item["operation"], int(item["threads"])): float(item["bandwidth_GBps_median"])
        for item in local_cost["shared_ddr"]
    }
    return {
        "cpu_wall_ms_per_gop": cpu_wall,
        "cpu_core_ms_per_gop": cpu_core,
        "cpu_memory_GBps": memory,
        "vta_ms_per_gop": float(local_cost["service_models"]["vta"]["ms_per_logical_gop"]),
        "vta_host_core_ms_per_gop": float(
            local_cost["service_models"]["vta_host_core_demand"]["core_ms_per_logical_gop"]
        ),
        "vta_traffic": fit_vta_traffic(local_cost),
        "vta_load_GBps": float(dma["fitted_load_bandwidth_GBps"]),
        "vta_store_GBps": float(dma["fitted_store_bandwidth_GBps"]),
        "boundary": fit_boundary_ownership(local_cost),
        "source_artifacts": {
            "unit_schema_sha256": unit_schema["artifact_sha256"],
            "cpu_measurements_sha256": cpu_measurements["artifact_sha256"],
            "local_cost_table_sha256": local_cost["artifact_sha256"],
        },
        "calibration_scope": "resnet_transfer_only",
    }


def complete_thread_rates(observed, fallback):
    """Fill missing thread points using the fallback model's relative scaling."""
    completed = dict(observed)
    anchor_thread = max(completed)
    anchor_value = completed[anchor_thread]
    for threads in THREADS:
        if threads not in completed:
            completed[threads] = anchor_value * fallback[threads] / fallback[anchor_thread]
    return completed


def build_minimal_yolo_anchor_profile(transferred, anchor_root=YOLO_ANCHOR_ROOT):
    """Calibrate target shape classes from six serial-only stage profiles."""
    anchor_root = Path(anchor_root)
    pattern = re.compile(r"__rt_s0([1-4])_s2([1-4])_")
    prefix_gops = conv_gops((0, 1, 2))
    vta_gops = conv_gops((3, 4, 5, 6, 7, 8, 9, 11))
    logits_gops = conv_gops(tuple(LOGITS_CONVS))
    regular_samples = {threads: [] for threads in THREADS}
    logits_samples = {threads: [] for threads in THREADS}
    vta_samples = []
    source_files = []
    for directory in sorted(anchor_root.glob("{}__rt_*".format(YOLO_ANCHOR_TOPOLOGY))):
        match = pattern.search(directory.name)
        serial_path = directory / "native_serial_result.jsonl"
        if not match or not serial_path.exists():
            continue
        rows = [
            json.loads(line)
            for line in serial_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not rows:
            continue
        stage0_threads, stage2_threads = (int(value) for value in match.groups())
        regular_samples[stage0_threads].append(
            statistics.median(float(row["stage0_run_ms"]) for row in rows) / prefix_gops
        )
        logits_samples[stage2_threads].append(
            statistics.median(float(row["stage2_run_ms"]) for row in rows) / logits_gops
        )
        vta_samples.append(
            statistics.median(float(row["stage1_run_ms"]) for row in rows) / vta_gops
        )
        source_files.append(str(serial_path.relative_to(repo_root())))
    if len(source_files) != 6:
        raise RuntimeError("expected six YOLO anchor profiles, found {}".format(len(source_files)))
    regular_observed = {
        threads: statistics.median(values)
        for threads, values in regular_samples.items()
        if values
    }
    logits_observed = {
        threads: statistics.median(values)
        for threads, values in logits_samples.items()
        if values
    }
    calibrated = copy.deepcopy(transferred)
    calibrated["cpu_wall_ms_per_gop_by_class"] = {
        "regular": complete_thread_rates(
            regular_observed, transferred["cpu_wall_ms_per_gop"]
        ),
        "logits": complete_thread_rates(logits_observed, transferred["cpu_wall_ms_per_gop"]),
    }
    calibrated["cpu_core_to_wall_ratio"] = {
        threads: transferred["cpu_core_ms_per_gop"][threads]
        / transferred["cpu_wall_ms_per_gop"][threads]
        for threads in THREADS
    }
    calibrated["vta_ms_per_gop"] = statistics.median(vta_samples)
    calibrated["calibration_scope"] = "six_yolo_serial_component_profiles"
    calibrated["target_calibration"] = {
        "anchor_topology": YOLO_ANCHOR_TOPOLOGY,
        "configuration_count": len(source_files),
        "pipeline_throughput_fields_consumed": False,
        "source_files": source_files,
        "source_file_sha256": {
            path: file_sha256(repo_root() / path) for path in source_files
        },
        "observed_regular_cpu_threads": sorted(regular_observed),
        "observed_logits_cpu_threads": sorted(logits_observed),
        "missing_thread_policy": "preserve ResNet relative thread scaling from nearest observed width",
        "cpu_wall_ms_per_gop_by_class": calibrated["cpu_wall_ms_per_gop_by_class"],
        "vta_ms_per_gop": calibrated["vta_ms_per_gop"],
    }
    return calibrated


def cpu_conv_class_gops(convs):
    logits = tuple(index for index in convs if index in LOGITS_CONVS)
    regular = tuple(index for index in convs if index not in LOGITS_CONVS)
    return {"regular": conv_gops(regular), "logits": conv_gops(logits)}


def cpu_stage_service(convs, threads, profile):
    by_class = profile.get("cpu_wall_ms_per_gop_by_class")
    if not by_class:
        wall = conv_gops(convs) * profile["cpu_wall_ms_per_gop"][threads]
        core = conv_gops(convs) * profile["cpu_core_ms_per_gop"][threads]
        return wall, core
    class_gops = cpu_conv_class_gops(convs)
    wall = sum(
        gops * by_class[name][threads] for name, gops in class_gops.items()
    )
    core = wall * profile["cpu_core_to_wall_ratio"][threads]
    return wall, core


def cpu_stage_specs(islands):
    specs = []
    cursor = 0
    for start, end in islands:
        specs.append((cursor, start, span_convs(cursor, start)))
        cursor = end
    final = len(SPLIT_POINTS) - 1
    specs.append((cursor, final, span_convs(cursor, final)))
    return specs


def score_configuration(islands, cpu_threads, profile):
    cpu_specs = cpu_stage_specs(islands)
    cpu_rows = []
    cpu_work_by_threads = [0.0, 0.0, 0.0, 0.0]
    shared_ddr_ms = 0.0
    for stage_index, ((start, end, convs), threads) in enumerate(zip(cpu_specs, cpu_threads)):
        gops = conv_gops(convs)
        service, core = cpu_stage_service(convs, threads, profile)
        input_bytes = point_bytes(start)
        output_bytes = point_bytes(end)
        ddr = input_bytes / (profile["cpu_memory_GBps"][("read", threads)] * 1.0e6)
        ddr += output_bytes / (profile["cpu_memory_GBps"][("write", threads)] * 1.0e6)
        shared_ddr_ms += ddr
        cpu_work_by_threads[threads - 1] += core
        cpu_rows.append(
            {
                "stage_index": stage_index * 2,
                "point_range": [SPLIT_POINTS[start][0], SPLIT_POINTS[end][0]],
                "conv_indices": list(convs),
                "threads": threads,
                "logical_gops": gops,
                "run_ms": service,
                "host_core_ms": core,
                "owned_boundary_ms": 0.0,
                "ddr_service_ms": ddr,
            }
        )

    vta_rows = []
    vta_service_ms = 0.0
    vta_host_core_ms = 0.0
    boundary_host_core_ms = 0.0
    vta_mutex_boundary_ms = 0.0
    for island_index, (start, end) in enumerate(islands):
        convs = span_convs(start, end)
        gops = conv_gops(convs)
        run_ms = gops * profile["vta_ms_per_gop"]
        host_core = gops * profile["vta_host_core_ms_per_gop"]
        load_bytes = gops * profile["vta_traffic"]["load_bytes_per_logical_gop"]
        store_bytes = gops * profile["vta_traffic"]["store_bytes_per_logical_gop"]
        ddr = load_bytes / (profile["vta_load_GBps"] * 1.0e6)
        ddr += store_bytes / (profile["vta_store_GBps"] * 1.0e6)
        shared_ddr_ms += ddr
        vta_service_ms += run_ms
        vta_host_core_ms += host_core
        vta_rows.append(
            {
                "stage_index": island_index * 2 + 1,
                "point_range": [SPLIT_POINTS[start][0], SPLIT_POINTS[end][0]],
                "conv_indices": list(convs),
                "logical_gops": gops,
                "run_ms": run_ms,
                "host_core_ms": host_core,
                "physical_load_bytes_est": int(round(load_bytes)),
                "physical_store_bytes_est": int(round(store_bytes)),
                "ddr_service_ms": ddr,
            }
        )

        for direction, point, cpu_owner_index in (
            ("cpu_to_vta", start, island_index),
            ("vta_to_cpu", end, island_index + 1),
        ):
            byte_count = point_bytes(point)
            model = profile["boundary"][direction]
            cpu_owner_ms = byte_count * model["cpu_owner_ms_per_byte"]
            vta_mutex_ms = byte_count * model["vta_mutex_ms_per_byte"]
            cpu_rows[cpu_owner_index]["owned_boundary_ms"] += cpu_owner_ms
            boundary_host_core_ms += cpu_owner_ms
            vta_mutex_boundary_ms += vta_mutex_ms

    max_cpu_stage_ms = max(
        row["run_ms"] + row["owned_boundary_ms"] for row in cpu_rows
    )
    nested_bounds = []
    confined = 0.0
    for width, work in enumerate(cpu_work_by_threads, 1):
        confined += work
        nested_bounds.append(confined / width)
    total_host_core_ms = (
        sum(row["host_core_ms"] for row in cpu_rows)
        + vta_host_core_ms
        + boundary_host_core_ms
    )
    cpu_core_pool_ms = max(total_host_core_ms / PHYSICAL_CPU_CORES, max(nested_bounds))
    single_vta_ms = vta_service_ms + vta_mutex_boundary_ms
    score = max(max_cpu_stage_ms, single_vta_ms, cpu_core_pool_ms, shared_ddr_ms)
    resources = {
        "max_cpu_stage_ms": max_cpu_stage_ms,
        "single_vta_ms": single_vta_ms,
        "cpu_core_pool_ms": cpu_core_pool_ms,
        "shared_ddr_ms": shared_ddr_ms,
    }
    bottleneck = max(resources, key=resources.get)
    return {
        "candidate_id": configuration_id(islands, cpu_threads),
        "topology_id": topology_id(islands),
        "islands": [
            {
                "start": SPLIT_POINTS[start][0],
                "end": SPLIT_POINTS[end][0],
                "conv_indices": list(span_convs(start, end)),
            }
            for start, end in islands
        ],
        "island_count": len(islands),
        "cpu_threads": list(cpu_threads),
        "cpu_stages": cpu_rows,
        "vta_stages": vta_rows,
        "predicted_ii_ms": score,
        "predicted_fps": 1000.0 / score,
        "resource_bounds_ms": resources,
        "bottleneck_resource": bottleneck,
    }


def rank_candidates(profile, top_k=DEFAULT_TOP_K):
    rows = []
    topology_count = 0
    execution_count = 0
    for islands in enumerate_topologies():
        topology_count += 1
        for cpu_threads in itertools.product(THREADS, repeat=len(islands) + 1):
            execution_count += 1
            rows.append(score_configuration(islands, cpu_threads, profile))
    rows.sort(key=lambda row: (row["predicted_ii_ms"], row["candidate_id"]))
    raw = rows[: int(top_k)]
    diverse = []
    seen = set()
    for row in rows:
        if row["topology_id"] in seen:
            continue
        seen.add(row["topology_id"])
        diverse.append(row)
        if len(diverse) >= int(top_k):
            break
    for rank, row in enumerate(raw, 1):
        row["rank"] = rank
    for rank, row in enumerate(diverse, 1):
        row["diverse_rank"] = rank
    return raw, diverse, topology_count, execution_count


def historical_topology_id(row):
    islands_text = str(row.get("islands") or "").strip()
    if islands_text:
        islands = ast.literal_eval(islands_text)
        return "yolo_multi_{}".format(
            "__".join("{}_{}".format(item["start_name"], item["end_name"]) for item in islands)
        )
    return "yolo_multi_{}_{}".format(row["start_name"], row["end_name"])


def load_historical_rows(roots=HISTORICAL_ROOTS):
    rows = []
    for root in roots:
        with (Path(root) / "summary.csv").open(encoding="utf-8") as inp:
            for row in csv.DictReader(inp):
                if (
                    row.get("mode") == "pipeline"
                    and row.get("status") == "ok"
                    and row.get("throughput_fps")
                ):
                    runtime = ast.literal_eval(row["runtime_config"]) if row.get("runtime_config") else {}
                    rows.append(
                        {
                            "candidate_id": row["candidate_id"],
                            "topology_id": historical_topology_id(row),
                            "measured_fps": float(row["throughput_fps"]),
                            "stage_devices": row["stage_devices"],
                            "stage_mean_ms": ast.literal_eval(row["stage_mean_ms"]),
                            "cpu_threads": (
                                [int(runtime["stage0_threads"]), int(runtime["stage2_threads"])]
                                if runtime
                                else None
                            ),
                            "source": str(Path(root) / "summary.csv"),
                        }
                    )
    rows.sort(key=lambda row: (-row["measured_fps"], row["candidate_id"]))
    for rank, row in enumerate(rows, 1):
        row["measured_rank"] = rank
    return rows


def load_known_buildability(roots=HISTORICAL_ROOTS):
    known = {}
    for root in roots:
        with (Path(root) / "buildability.csv").open(encoding="utf-8") as inp:
            for row in csv.DictReader(inp):
                topology = historical_topology_id(row)
                current = known.setdefault(topology, {"buildable": 0, "failed": 0})
                if row.get("status") == "buildable":
                    current["buildable"] += 1
                else:
                    current["failed"] += 1
    return known


def join_historical(shortlist, historical, known_buildability):
    by_topology = {}
    for row in historical:
        by_topology.setdefault(row["topology_id"], []).append(row)
    joined = []
    for static in shortlist:
        matches = by_topology.get(static["topology_id"], [])
        best = matches[0] if matches else None
        exact = [row for row in matches if row["cpu_threads"] == static["cpu_threads"]]
        best_exact = exact[0] if exact else None
        joined.append(
            {
                "static_rank": static.get("diverse_rank", static.get("rank")),
                "candidate_id": static["candidate_id"],
                "topology_id": static["topology_id"],
                "cpu_threads": static["cpu_threads"],
                "predicted_ii_ms": static["predicted_ii_ms"],
                "predicted_fps": static["predicted_fps"],
                "predicted_bottleneck": static["bottleneck_resource"],
                "historical_exact_configuration_match": bool(exact),
                "historical_exact_configuration_match_count": len(exact),
                "best_exact_historical_candidate_id": (
                    best_exact["candidate_id"] if best_exact else ""
                ),
                "best_exact_historical_fps": best_exact["measured_fps"] if best_exact else "",
                "best_exact_historical_rank_of_169": (
                    best_exact["measured_rank"] if best_exact else ""
                ),
                "historical_topology_match_count": len(matches),
                "best_historical_candidate_id": best["candidate_id"] if best else "",
                "best_historical_fps": best["measured_fps"] if best else "",
                "best_historical_rank_of_169": best["measured_rank"] if best else "",
                "known_buildable_count": known_buildability.get(static["topology_id"], {}).get(
                    "buildable", 0
                ),
                "known_build_failure_count": known_buildability.get(
                    static["topology_id"], {}
                ).get("failed", 0),
            }
        )
    return joined


def run_evaluation(output_dir=DEFAULT_OUTPUT_DIR, top_k=DEFAULT_TOP_K):
    output_dir = Path(output_dir)
    transferred_profile = build_transferred_profile(output_dir)
    profile = build_minimal_yolo_anchor_profile(transferred_profile)

    # Both rankings are frozen before any pipeline-throughput summary is loaded.
    transfer_raw, transfer_diverse, _, _ = rank_candidates(transferred_profile, top_k)
    raw, diverse, topology_count, execution_count = rank_candidates(profile, top_k)

    historical = load_historical_rows()
    buildability = load_known_buildability()
    raw_join = join_historical(raw, historical, buildability)
    diverse_join = join_historical(diverse, historical, buildability)
    transfer_join = join_historical(transfer_diverse, historical, buildability)
    oracle = historical[0]
    covered = [row for row in diverse_join if row["historical_topology_match_count"]]
    selected_best = max(covered, key=lambda row: float(row["best_historical_fps"])) if covered else None
    historical_top20 = {row["topology_id"] for row in historical[:20]}
    diverse_ids = {row["topology_id"] for row in diverse}
    regret_at_k = {}
    for k in (1, 3, 5, 6, 10, 20):
        values = [
            float(row["best_historical_fps"])
            for row in diverse_join[:k]
            if row["best_historical_fps"] != ""
        ]
        best = max(values) if values else None
        regret_at_k[str(k)] = {
            "best_covered_historical_fps": best,
            "regret": (
                (oracle["measured_fps"] - best) / oracle["measured_fps"]
                if best is not None
                else None
            ),
        }
    holdout_historical = [
        row
        for row in historical
        if row["topology_id"] != "yolo_multi_pool2_dual_pre_logits"
    ]
    holdout_oracle = holdout_historical[0]
    holdout_covered = [
        row
        for row in covered
        if row["topology_id"] != "yolo_multi_pool2_dual_pre_logits"
    ]
    holdout_best = (
        max(holdout_covered, key=lambda row: float(row["best_historical_fps"]))
        if holdout_covered
        else None
    )
    report = seal_artifact(
        {
            "schema_version": 1,
            "kind": "yolov3_tiny_cpu_vta_pipeline_v1_static_evaluation",
            "selection_policy": (
                "rank 105696 legal fixed-tile configurations by max(max CPU stage, "
                "single serialized VTA, nested CPU core capacity, shared DDR service); "
                "then retain one thread configuration per topology for board diversity"
            ),
            "search_implementation": (
                "complete fixed-tile enumeration used as an exact host-side oracle; the "
                "resource objective matches V1, but a branch-aware k-best DP is not yet implemented"
            ),
            "candidate_throughput_used_for_ranking": False,
            "historical_labels_joined_after_selection": True,
            "profile_transfer": {
                "source_model": "resnet18",
                "target_model": "yolov3-tiny",
                "transferred_prices": (
                    "CPU wall/core ms per logical GOP by TVM thread parameter, VTA ms per "
                    "logical GOP, VTA physical traffic, DMA bandwidth, boundary ownership, "
                    "and CPU DDR bandwidth"
                ),
                "target_static_work": "13 YOLO convolution shapes and legal branch-aware split points",
                "source_artifacts": profile["source_artifacts"],
                "minimal_target_calibration": profile["target_calibration"],
                "target_cpu_core_policy": (
                    "target CPU wall slopes multiplied by ResNet-measured process-core/wall ratios; "
                    "YOLO serial records do not contain process CPU clocks"
                ),
            },
            "search_space": {
                "legal_topology_count": topology_count,
                "execution_configuration_count": execution_count,
                "max_vta_islands": MAX_VTA_ISLANDS,
                "cpu_thread_choices_per_stage": list(THREADS),
                "tile_search_enabled": False,
            },
            "raw_top20": raw,
            "topology_diverse_top20": diverse,
            "transfer_only_ablation": {
                "raw_top20": transfer_raw,
                "topology_diverse_top20": transfer_diverse,
                "diverse_top20_join": transfer_join,
                "measured_topology_coverage": sum(
                    bool(row["historical_topology_match_count"]) for row in transfer_join
                ),
            },
            "historical_audit": {
                "measured_candidate_count": len(historical),
                "measured_topology_count": len({row["topology_id"] for row in historical}),
                "measured_pool_oracle": oracle,
                "raw_top20_join": raw_join,
                "diverse_top20_join": diverse_join,
                "diverse_top20_measured_topology_coverage": len(covered),
                "diverse_top20_historical_top20_topology_recall": len(
                    diverse_ids & historical_top20
                )
                / len(historical_top20),
                "historical_top20_unique_topology_count": len(historical_top20),
                "retrospective_covered_regret_at_k": regret_at_k,
                "uniform_random_probability_of_including_one_fixed_topology_in_20": (
                    20.0 / topology_count
                ),
                "best_covered_historical_candidate": selected_best,
                "best_covered_regret": (
                    (oracle["measured_fps"] - float(selected_best["best_historical_fps"]))
                    / oracle["measured_fps"]
                    if selected_best
                    else None
                ),
                "anchor_excluded_holdout": {
                    "excluded_topology": "yolo_multi_pool2_dual_pre_logits",
                    "measured_candidate_count": len(holdout_historical),
                    "measured_topology_count": len(
                        {row["topology_id"] for row in holdout_historical}
                    ),
                    "oracle": holdout_oracle,
                    "shortlist_covered_topology_count": len(holdout_covered),
                    "best_covered_candidate": holdout_best,
                    "best_covered_regret": (
                        (
                            holdout_oracle["measured_fps"]
                            - float(holdout_best["best_historical_fps"])
                        )
                        / holdout_oracle["measured_fps"]
                        if holdout_best
                        else None
                    ),
                },
            },
            "evidence_boundary": {
                "supported": (
                    "host-only evidence that a resource model with six target serial component "
                    "profiles selects historically strong YOLO split topologies"
                ),
                "not_supported": (
                    "absolute YOLO FPS accuracy or prospective cross-model generalization; "
                    "the selected thread configurations were not compiled or measured"
                ),
                "known_model_gaps": [
                    "CPU and VTA GOP rates are transferred from ResNet shapes",
                    "only two CPU shape classes are distinguished by the minimal YOLO anchor",
                    "YOLO CPU core demand is inferred rather than directly measured",
                    "YOLO pooling, route, upsample, decode and NMS service are not separately profiled",
                    "shared-DDR contention is represented only by a service lower bound",
                    "candidate-specific lowered VTA traffic and padding are not available",
                    "the current 105696-case implementation is exhaustive, not the scalable DP",
                ],
            },
        }
    )
    json_path = output_dir / "yolov3_tiny_static_v1_evaluation.json"
    csv_path = output_dir / "yolov3_tiny_static_v1_shortlist.csv"
    write_json(json_path, report)
    write_csv(csv_path, diverse_join)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()
    report = run_evaluation(args.output_dir, args.top_k)
    audit = report["historical_audit"]
    print(
        json.dumps(
            {
                "artifact_sha256": report["artifact_sha256"],
                "topology_count": report["search_space"]["legal_topology_count"],
                "execution_configuration_count": report["search_space"][
                    "execution_configuration_count"
                ],
                "top1": report["topology_diverse_top20"][0]["candidate_id"],
                "measured_topology_coverage": audit[
                    "diverse_top20_measured_topology_coverage"
                ],
                "best_covered_historical_candidate": audit[
                    "best_covered_historical_candidate"
                ],
                "best_covered_regret": audit["best_covered_regret"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
