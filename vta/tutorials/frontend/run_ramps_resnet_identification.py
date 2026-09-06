#!/usr/bin/env python3
"""Run the legacy 12 x 3 x 2 RAMPS pilot identification design.

The fixed 72-configuration design has been superseded by sensitivity-driven
D/G-optimal selection.  It is retained for reproducibility and cannot produce
publication-valid records.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from resource_aware_dataset import DEFAULT_RESNET_ROOTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", default="root@192.168.1.228")
    parser.add_argument(
        "--output-dir",
        default=(
            "vta/tutorials/frontend/report_out/resource_aware_maxplus/"
            "20260713_v1/identification/resnet72"
        ),
    )
    parser.add_argument(
        "--cost-model-json",
        default=(
            "vta/tutorials/frontend/report_out/resource_aware_maxplus/"
            "20260713_v1/calibration/cost_model.json"
        ),
    )
    parser.add_argument(
        "--build-cache-dir",
        default=(
            "vta/tutorials/frontend/report_out/native_stage_pipeline_build_cache/"
            "v23_20260506"
        ),
    )
    parser.add_argument(
        "--rpc-baseline-result",
        default=(
            "vta/tutorials/frontend/report_out/native_stage_pipeline_searches/"
            "20260507_v23_warm100/artifacts/rpc_all_vta_baseline.jsonl"
        ),
    )
    parser.add_argument("--remote-dir", default="/var/volatile/ramps_resnet72")
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--queue-depth-options", default="1,2,4")
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--timeout-s", type=int, default=1200)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--allow-pilot-run",
        action="store_true",
        help="Explicitly allow the superseded fixed-72 pilot to access the board",
    )
    parser.add_argument("--ssh-option", action="append", default=[])
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_ints(text: str) -> List[int]:
    return [int(item.strip()) for item in str(text).split(",") if item.strip()]


def historical_rows() -> List[Dict[str, Any]]:
    result = []
    seen = set()
    for root_text in DEFAULT_RESNET_ROOTS:
        for path in sorted(Path(root_text).glob("batch[0-9][0-9][0-9]/summary.json")):
            for row in load_json(path).get("rows", []) or []:
                candidate_id = str(row.get("candidate_id") or row.get("scheme_name") or "")
                if (
                    candidate_id
                    and candidate_id not in seen
                    and row.get("status") == "ok"
                    and row.get("run_kind") == "default"
                    and float(row.get("pipeline_throughput_fps") or 0.0) > 0.0
                    and bool(row.get("passes_correctness_gate"))
                ):
                    item = dict(row)
                    item["source_summary"] = str(path)
                    result.append(item)
                    seen.add(candidate_id)
    if len(result) != 200:
        raise RuntimeError("expected 200 ResNet18 records, found {}".format(len(result)))
    return result


def static_anchor_features(row: Mapping[str, Any]) -> List[float]:
    components = row.get("score_components_json") or {}
    if isinstance(components, str):
        components = json.loads(components)
    stages = components.get("stages", []) or []
    cpu_ms = [float(item.get("static_ms_est") or 0.0) for item in stages if item.get("device") == "cpu"]
    vta_ms = [float(item.get("static_ms_est") or 0.0) for item in stages if item.get("device") == "vta"]
    sram = components.get("sram_utilization_pct") or {}
    return [
        math.log1p(float(row.get("boundary_bytes") or components.get("boundary_bytes") or 0.0)),
        math.log1p(float(row.get("static_dma_bytes_est") or components.get("dma_bytes") or 0.0)),
        float(row.get("static_dma_fragmentation_score_est") or 0.0),
        float(row.get("onchip_peak_util_pct") or sram.get("peak") or 0.0),
        max(cpu_ms or [0.0]),
        sum(vta_ms),
        float(row.get("stage_count") or len(stages)),
    ]


def farthest_point_rows(rows: Sequence[Mapping[str, Any]], count: int) -> List[Mapping[str, Any]]:
    if len(rows) < count:
        raise RuntimeError("not enough rows for anchor stratum: {} < {}".format(len(rows), count))
    matrix = np.asarray([static_anchor_features(row) for row in rows], dtype="float64")
    scale = np.ptp(matrix, axis=0)
    scale[scale == 0.0] = 1.0
    normalized = (matrix - matrix.min(axis=0)) / scale
    # Begin with the largest boundary/DMA/SRAM envelope, then greedily maximize
    # distance to the selected design.  Candidate ids break all ties.
    first = max(
        range(len(rows)),
        key=lambda index: (float(normalized[index, :4].sum()), str(rows[index].get("candidate_id"))),
    )
    selected = [first]
    while len(selected) < count:
        remaining = [index for index in range(len(rows)) if index not in selected]
        next_index = max(
            remaining,
            key=lambda index: (
                min(float(np.linalg.norm(normalized[index] - normalized[chosen])) for chosen in selected),
                str(rows[index].get("candidate_id")),
            ),
        )
        selected.append(next_index)
    return [rows[index] for index in selected]


def select_anchors(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    selected = []
    for island_count in (1, 2, 3):
        pool = [row for row in rows if int(row.get("vta_island_count") or 0) == island_count]
        for row in farthest_point_rows(pool, 4):
            components = row.get("score_components_json") or {}
            if isinstance(components, str):
                components = json.loads(components)
            selected.append(
                {
                    "anchor_index": len(selected) + 1,
                    "candidate_id": row.get("candidate_id") or row.get("scheme_name"),
                    "vta_island_count": island_count,
                    "stage_count": int(row.get("stage_count") or 0),
                    "stage_devices": row.get("stage_devices"),
                    "tuned_threads": row.get("thread_config") or row.get("stage_runtime_threads"),
                    "boundary_bytes": int(row.get("boundary_bytes") or 0),
                    "static_dma_bytes": int(row.get("static_dma_bytes_est") or 0),
                    "dma_fragmentation": float(row.get("static_dma_fragmentation_score_est") or 0.0),
                    "sram_peak_util_pct": float(row.get("onchip_peak_util_pct") or 0.0),
                    "static_bottleneck": components.get("bottleneck_stage", ""),
                    "selection_features": static_anchor_features(row),
                    "source_summary": row.get("source_summary"),
                }
            )
    if len(selected) != 12 or len({row["candidate_id"] for row in selected}) != 12:
        raise RuntimeError("anchor selection did not produce 12 unique candidates")
    return selected


def config_manifest(anchors: Sequence[Mapping[str, Any]], queue_depths: Sequence[int]):
    configs = []
    for anchor in anchors:
        stage_count = int(anchor["stage_count"])
        tuned = [int(item) for item in str(anchor["tuned_threads"]).split(",")]
        if len(tuned) != stage_count:
            raise RuntimeError("invalid tuned thread vector for {}".format(anchor["candidate_id"]))
        for queue_depth in queue_depths:
            for policy, threads in (("tuned", tuned), ("single", [1] * stage_count)):
                config_id = "a{:02d}_q{}_{}".format(
                    int(anchor["anchor_index"]), int(queue_depth), policy
                )
                configs.append(
                    {
                        "config_id": config_id,
                        "candidate_id": anchor["candidate_id"],
                        "anchor_index": anchor["anchor_index"],
                        "vta_island_count": anchor["vta_island_count"],
                        "stage_count": stage_count,
                        "queue_depth": int(queue_depth),
                        "cpu_thread_policy": policy,
                        "stage_threads": ",".join(str(item) for item in threads),
                    }
                )
    if len(configs) != 72:
        raise RuntimeError("expected 72 controlled configs, found {}".format(len(configs)))
    return configs


def command_for_run(args: argparse.Namespace, config: Mapping[str, Any], session: int) -> List[str]:
    label = "{}_session{:02d}".format(config["config_id"], session)
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "search_resnet18_stage_splits.py"),
        "--board",
        args.board,
        "--search-label",
        label,
        "--output-root",
        str(Path(args.output_dir) / "runs"),
        "--include-known",
        "",
        "--candidate-name",
        str(config["candidate_id"]),
        "--max-candidates",
        "1",
        "--max-vta-islands",
        "3",
        "--static-shortlist-n",
        "1",
        "--buildability-top-n",
        "1",
        "--measure-top-n",
        "1",
        "--refine-top-n",
        "0",
        "--max-board-test-configs",
        "1",
        "--runs",
        str(args.runs + args.warmup),
        "--skip-first",
        str(args.warmup),
        "--queue-depth",
        str(config["queue_depth"]),
        "--stage-runtime-num-threads",
        str(config["stage_threads"]),
        "--reuse-buildability-cache",
        "--build-cache-dir",
        args.build_cache_dir,
        "--file-cache-policy",
        "prewarm",
        "--remote-dir",
        args.remote_dir,
        "--correctness-policy",
        "cat_equivalent",
        "--static-cost-model-json",
        args.cost_model_json,
        "--rpc-baseline-result",
        args.rpc_baseline_result,
        "--serial-timeout-s",
        "240",
        "--pipeline-timeout-s",
        "600",
        "--fetch-timeout-s",
        "120",
    ]
    ssh_options = args.ssh_option or [
        "HostKeyAlgorithms=+ssh-rsa",
        "PubkeyAcceptedAlgorithms=+ssh-rsa",
    ]
    for option in ssh_options:
        command.extend(["--ssh-option", option])
    return command


def result_path(args: argparse.Namespace, config: Mapping[str, Any], session: int) -> Path:
    label = "{}_session{:02d}".format(config["config_id"], session)
    return Path(args.output_dir) / "runs" / label / "summary.json"


def run_one(args: argparse.Namespace, config: Mapping[str, Any], session: int) -> Dict[str, Any]:
    summary_path = result_path(args, config, session)
    run_dir = summary_path.parent
    record_path = run_dir / "identification_result.json"
    if args.resume and record_path.exists():
        return load_json(record_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    command = command_for_run(args, config, session)
    (run_dir / "orchestrator_command.txt").write_text(
        " ".join(command) + "\n", encoding="utf-8"
    )
    started = time.monotonic()
    try:
        with (run_dir / "orchestrator.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=args.timeout_s,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError("search process returned {}".format(completed.returncode))
        summary = load_json(summary_path)
        rows = [
            row
            for row in summary.get("rows", []) or []
            if row.get("run_kind") == "default" and row.get("status") not in {"buildable", "static"}
        ]
        if len(rows) != 1:
            raise RuntimeError("expected one measured row, found {}".format(len(rows)))
        row = rows[0]
        result = {
            **dict(config),
            "session": session,
            "status": row.get("status"),
            "failure_type": row.get("failure_type", ""),
            "error_summary": row.get("error_summary", ""),
            "pipeline_throughput_fps": row.get("pipeline_throughput_fps"),
            "measured_cycle_ms": (
                1000.0 / float(row["pipeline_throughput_fps"])
                if float(row.get("pipeline_throughput_fps") or 0.0) > 0.0
                else None
            ),
            "passes_correctness_gate": bool(row.get("passes_correctness_gate")),
            "stage_ms_summary": row.get("stage_ms_summary"),
            "stage_core_demand_ms_json": row.get("stage_core_demand_ms_json"),
            "stage_core_demand_source": row.get("stage_core_demand_source", ""),
            "stage_cpu_time_scope": row.get("stage_cpu_time_scope", ""),
            "stage_runtime_threads": row.get("stage_runtime_threads"),
            "ps_pl_total_dma_bytes": row.get("ps_pl_total_dma_bytes"),
            "dma_fragmentation_score": row.get("dma_fragmentation_score"),
            "elapsed_s": time.monotonic() - started,
            "summary_path": str(summary_path),
        }
    except subprocess.TimeoutExpired as err:
        result = {
            **dict(config),
            "session": session,
            "status": "timeout",
            "failure_type": "timeout",
            "error_summary": str(err),
            "elapsed_s": time.monotonic() - started,
            "summary_path": str(summary_path),
        }
    except Exception as err:  # pylint: disable=broad-except
        result = {
            **dict(config),
            "session": session,
            "status": "failed",
            "failure_type": "orchestration_failed",
            "error_summary": repr(err),
            "elapsed_s": time.monotonic() - started,
            "summary_path": str(summary_path),
        }
    write_json(record_path, result)
    return result


def update_progress(output_dir: Path, completed: int, total: int, current: str, successes: int):
    width = 40
    filled = int(width * completed / max(1, total))
    line = "[{}{}] {}/{} current={} success={} failed={}".format(
        "#" * filled,
        "-" * (width - filled),
        completed,
        total,
        current,
        successes,
        completed - successes,
    )
    (output_dir / "progress.txt").write_text(line + "\n", encoding="utf-8")
    write_json(
        output_dir / "progress.json",
        {
            "completed_processes": completed,
            "total_processes": total,
            "current": current,
            "successes": successes,
            "failures": completed - successes,
        },
    )
    print(line, flush=True)


def aggregate_configs(configs: Sequence[Mapping[str, Any]], results: Sequence[Mapping[str, Any]]):
    rows = []
    for config in configs:
        matches = [row for row in results if row["config_id"] == config["config_id"]]
        valid = [
            row
            for row in matches
            if row.get("status") == "ok"
            and row.get("passes_correctness_gate")
            and float(row.get("pipeline_throughput_fps") or 0.0) > 0.0
        ]
        cycles = [float(row["measured_cycle_ms"]) for row in valid]
        stage_vectors = []
        for row in valid:
            raw_stages = row.get("stage_ms_summary") or []
            if isinstance(raw_stages, str):
                raw_stages = json.loads(raw_stages)
            values = [float(item.get("ms") or 0.0) for item in raw_stages]
            if values:
                stage_vectors.append(values)
        stage_medians = []
        if stage_vectors:
            if any(len(values) != int(config["stage_count"]) for values in stage_vectors):
                raise RuntimeError(
                    "controlled stage vector length mismatch for {}".format(config["config_id"])
                )
            stage_medians = [
                statistics.median(values[index] for values in stage_vectors)
                for index in range(int(config["stage_count"]))
            ]
        core_demand_vectors = []
        for row in valid:
            raw_demands = row.get("stage_core_demand_ms_json") or []
            if isinstance(raw_demands, str):
                raw_demands = json.loads(raw_demands)
            if raw_demands:
                if row.get("stage_cpu_time_scope") != "exclusive_process_cpu_time":
                    raise RuntimeError(
                        "non-exclusive CPU demand sample for {}".format(config["config_id"])
                    )
                core_demand_vectors.append([float(value) for value in raw_demands])
        core_demand_medians = []
        if core_demand_vectors:
            if any(
                len(values) != int(config["stage_count"])
                for values in core_demand_vectors
            ):
                raise RuntimeError(
                    "controlled core-demand vector length mismatch for {}".format(
                        config["config_id"]
                    )
                )
            core_demand_medians = [
                statistics.median(values[index] for values in core_demand_vectors)
                for index in range(int(config["stage_count"]))
            ]
        rows.append(
            {
                **dict(config),
                "completed_sessions": len(matches),
                "successful_sessions": len(valid),
                "failed_sessions": len(matches) - len(valid),
                "cycle_median_ms": statistics.median(cycles) if cycles else "",
                "cycle_mad_ms": (
                    statistics.median(abs(value - statistics.median(cycles)) for value in cycles)
                    if cycles
                    else ""
                ),
                "fps_median": 1000.0 / statistics.median(cycles) if cycles else "",
                "stage_median_ms_json": json.dumps(stage_medians),
                "stage_core_demand_ms_json": json.dumps(core_demand_medians),
                "stage_core_demand_source": (
                    "serial_process_cpu_time" if core_demand_medians else ""
                ),
                "cpu_demand_valid": len(core_demand_vectors) == 3,
                "measurement_complete": len(valid) == 3 and len(core_demand_vectors) == 3,
                "publication_valid": False,
                "evidence_level": "pilot_fixed72",
            }
        )
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as inp:
        for chunk in iter(lambda: inp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    queue_depths = parse_ints(args.queue_depth_options)
    if args.sessions != 3 or args.warmup < 5 or args.runs < 20:
        raise RuntimeError("publication protocol requires sessions=3, warmup>=5, runs>=20")
    if queue_depths != [1, 2, 4]:
        raise RuntimeError("publication protocol requires queue depths 1,2,4")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    anchors = select_anchors(historical_rows())
    configs = config_manifest(anchors, queue_depths)
    write_csv(output_dir / "anchors.csv", anchors)
    write_json(output_dir / "anchors.json", {"rows": anchors})
    write_csv(output_dir / "controlled72_manifest.csv", configs)
    write_json(
        output_dir / "controlled72_protocol.json",
        {
            "frozen": True,
            "evidence_level": "pilot_fixed72",
            "publication_protocol": False,
            "superseded_by": "sensitivity-driven D/G-optimal design",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "board": args.board,
            "anchor_count": 12,
            "configuration_count": 72,
            "independent_processes_per_configuration": 3,
            "total_board_processes": 216,
            "warmup_frames": args.warmup,
            "measured_frames": args.runs,
            "queue_depths": queue_depths,
            "thread_policies": ["tuned", "single"],
            "selection": "static-resource farthest-point within each VTA island count",
            "anchors_sha256": sha256(output_dir / "anchors.csv"),
            "labels_used_for_anchor_selection": False,
        },
    )
    if args.prepare_only:
        print("[RAMPS IDENTIFICATION] prepared", output_dir)
        return
    if not args.allow_pilot_run:
        raise RuntimeError(
            "fixed-72 identification is a superseded pilot; pass --allow-pilot-run "
            "only for reproducibility"
        )
    for path in (Path(args.cost_model_json), Path(args.rpc_baseline_result)):
        if not path.exists():
            raise FileNotFoundError(path)
    cost_model = load_json(Path(args.cost_model_json))
    if cost_model.get("source") != "ramps_measured_calibration":
        raise RuntimeError("controlled identification requires measured RAMPS calibration")
    if "fallback" in json.dumps(cost_model).lower() or "estimated_from" in json.dumps(
        cost_model
    ):
        raise RuntimeError("controlled identification rejects fallback calibration fields")

    schedule = []
    for session in range(1, args.sessions + 1):
        session_configs = list(configs)
        random.Random(args.seed + session).shuffle(session_configs)
        schedule.extend((session, config) for config in session_configs)
    write_csv(
        output_dir / "measurement_order.csv",
        [
            {"order": index + 1, "session": session, **dict(config)}
            for index, (session, config) in enumerate(schedule)
        ],
    )
    results = []
    successes = 0
    for index, (session, config) in enumerate(schedule, start=1):
        update_progress(output_dir, index - 1, len(schedule), config["config_id"], successes)
        result = run_one(args, config, session)
        results.append(result)
        if result.get("status") == "ok" and result.get("passes_correctness_gate"):
            successes += 1
        write_csv(output_dir / "process_results.csv", results)
        write_json(output_dir / "process_results.json", {"rows": results})
        controlled = aggregate_configs(configs, results)
        write_csv(output_dir / "controlled72_summary.csv", controlled)
        write_json(output_dir / "controlled72_summary.json", {"rows": controlled})
    update_progress(output_dir, len(schedule), len(schedule), "complete", successes)
    aggregated = aggregate_configs(configs, results)
    measurement_complete = sum(1 for row in aggregated if row["measurement_complete"])
    write_json(
        output_dir / "completion.json",
        {
            "process_count": len(results),
            "successful_process_count": successes,
            "configuration_count": len(configs),
            "measurement_complete_configuration_count": measurement_complete,
            "publication_valid_configuration_count": 0,
            "evidence_level": "pilot_fixed72",
            "complete": len(results) == 216,
        },
    )


if __name__ == "__main__":
    main()
