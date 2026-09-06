#!/usr/bin/env python3
"""Build and measure the frozen RAMPS YOLOv3-tiny prospective30 cohort."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any, Dict, List, Mapping, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", default="root@192.168.1.228")
    parser.add_argument(
        "--experiment-root",
        default="vta/tutorials/frontend/report_out/resource_aware_maxplus/20260713_v1",
    )
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--queue-depth", type=int, default=2)
    parser.add_argument("--timeout-s", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--ssh-option", action="append", default=[])
    parser.add_argument(
        "--rpc-baseline-json",
        default=(
            "vta/tutorials/frontend/report_out/yolov3_tiny_pipeline_baseline/"
            "20260512_rpc_all_vta_runs20/rpc_all_vta_baseline.json"
        ),
    )
    return parser.parse_args()


def read_ids(path: Path) -> List[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def read_json(path: Path) -> Dict[str, Any]:
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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def common_command(args: argparse.Namespace, shared_root: Path) -> List[str]:
    experiment_root = Path(args.experiment_root)
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "search_yolov3_tiny_stage_splits.py"),
        "--output-root",
        str(shared_root),
        "--heuristic-params-json",
        (
            "vta/tutorials/frontend/report_out/yolov3_tiny_heuristic_search/"
            "heuristic_params_resnet18_v23_200.json"
        ),
        "--resource-model-json",
        str(experiment_root / "models" / "resnet_to_yolo_frozen.json"),
        "--candidate-prior-file",
        str(experiment_root / "prospective30" / "candidate_prior.txt"),
        "--candidate-count",
        "30",
        "--measure-count",
        "1",
        "--max-vta-islands",
        "3",
        "--fine-search-policy",
        "adaptive_balance",
        "--split-backend",
        "relay",
        "--split-quantization-mode",
        "per_stage",
        "--tail-device",
        "cpu",
        "--runner-output-mode",
        "raw_all_stages",
        "--correctness-policy",
        "detection_gate",
        "--board",
        args.board,
        "--rpc-baseline-json",
        args.rpc_baseline_json,
        "--remote-dir",
        "/var/volatile/ramps_yolo_prospective30",
        "--runs",
        str(args.runs),
        "--warmup-runs",
        str(args.warmup),
        "--serial-runs",
        "2",
        "--queue-depth",
        str(args.queue_depth),
        "--serial-timeout-s",
        "240",
        "--pipeline-timeout-s",
        "900",
        "--fetch-timeout-s",
        "180",
    ]
    ssh_options = args.ssh_option or [
        "HostKeyAlgorithms=+ssh-rsa",
        "PubkeyAcceptedAlgorithms=+ssh-rsa",
        "StrictHostKeyChecking=accept-new",
        "UserKnownHostsFile={}".format(experiment_root / "board_known_hosts"),
    ]
    for option in ssh_options:
        command.extend(["--ssh-option", option])
    return command


def run_logged(command: Sequence[str], log_path: Path, timeout_s: int) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            list(command),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError("command returned {}".format(completed.returncode))


def prebuild(args: argparse.Namespace, shared_root: Path) -> List[Dict[str, Any]]:
    buildability = shared_root / "buildability.csv"
    if not (args.resume and buildability.exists()):
        command = common_command(args, shared_root) + ["--mode", "build", "--build-only"]
        (shared_root / "build_command.txt").parent.mkdir(parents=True, exist_ok=True)
        (shared_root / "build_command.txt").write_text(
            " ".join(command) + "\n", encoding="utf-8"
        )
        run_logged(command, shared_root / "build.log", args.timeout_s * 10)
    with buildability.open(encoding="utf-8") as inp:
        return list(csv.DictReader(inp))


def update_progress(root: Path, completed: int, total: int, current: str, ok: int):
    width = 40
    filled = int(width * completed / max(1, total))
    line = "[{}{}] {}/{} current={} ok={} failed={}".format(
        "#" * filled,
        "-" * (width - filled),
        completed,
        total,
        current,
        ok,
        completed - ok,
    )
    (root / "progress.txt").write_text(line + "\n", encoding="utf-8")
    write_json(
        root / "progress.json",
        {"completed": completed, "total": total, "current": current, "ok": ok},
    )
    print(line, flush=True)


def measure_once(
    args: argparse.Namespace,
    shared_root: Path,
    candidate_id: str,
    session: int,
    order: int,
) -> Dict[str, Any]:
    results_root = Path(args.experiment_root) / "prospective30" / "board_results"
    run_dir = results_root / "runs" / "session{:02d}".format(session) / (
        "{:03d}_{}".format(order, candidate_id)
    )
    record_path = run_dir / "orchestrator_result.json"
    if args.resume and record_path.exists():
        return read_json(record_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    command = common_command(args, shared_root) + [
        "--mode",
        "measure",
        "--candidate-id",
        candidate_id,
    ]
    (run_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    started = time.monotonic()
    try:
        run_logged(command, run_dir / "run.log", args.timeout_s)
        summary = read_json(shared_root / "summary.json")
        rows = [
            row
            for row in summary.get("rows", []) or []
            if row.get("candidate_id") == candidate_id and row.get("mode") == "pipeline"
        ]
        if len(rows) != 1:
            raise RuntimeError("expected one pipeline row, found {}".format(len(rows)))
        row = rows[0]
        candidate_output = shared_root / candidate_id
        if candidate_output.exists():
            archived = run_dir / "candidate_output"
            if archived.exists():
                shutil.rmtree(str(archived))
            shutil.copytree(str(candidate_output), str(archived))
        result = {
            "candidate_id": candidate_id,
            "session": session,
            "measurement_order": order,
            "status": row.get("status"),
            "failure_type": row.get("failure_type", ""),
            "error_summary": row.get("error_summary", ""),
            "pipeline_throughput_fps": row.get("pipeline_throughput_fps"),
            "measured_cycle_ms": row.get("measured_cycle_ms")
            or (
                1000.0 / float(row["pipeline_throughput_fps"])
                if float(row.get("pipeline_throughput_fps") or 0.0) > 0.0
                else None
            ),
            "passes_correctness_gate": bool(row.get("passes_correctness_gate")),
            "detection_gate_passed": bool(row.get("detection_gate_passed")),
            "raw_sanity_passed": bool(row.get("raw_sanity_passed")),
            "stage_times_ms": row.get("stage_times_ms"),
            "warmup_runs": row.get("warmup_runs"),
            "measured_runs": row.get("measured_runs"),
            "elapsed_s": time.monotonic() - started,
        }
    except subprocess.TimeoutExpired as err:
        result = {
            "candidate_id": candidate_id,
            "session": session,
            "measurement_order": order,
            "status": "timeout",
            "failure_type": "timeout",
            "error_summary": str(err),
            "elapsed_s": time.monotonic() - started,
        }
    except Exception as err:  # pylint: disable=broad-except
        result = {
            "candidate_id": candidate_id,
            "session": session,
            "measurement_order": order,
            "status": "failed",
            "failure_type": "orchestration_failed",
            "error_summary": repr(err),
            "elapsed_s": time.monotonic() - started,
        }
    write_json(record_path, result)
    return result


def aggregate(candidate_ids: Sequence[str], results: Sequence[Mapping[str, Any]]):
    rows = []
    for candidate_id in candidate_ids:
        matches = [row for row in results if row["candidate_id"] == candidate_id]
        valid = [
            row
            for row in matches
            if row.get("status") == "ok"
            and row.get("passes_correctness_gate")
            and row.get("raw_sanity_passed")
            and float(row.get("pipeline_throughput_fps") or 0.0) > 0.0
        ]
        cycles = [float(row["measured_cycle_ms"]) for row in valid]
        rows.append(
            {
                "candidate_id": candidate_id,
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
                "publication_valid": len(valid) == 3,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.sessions != 3 or args.warmup < 5 or args.runs < 20:
        raise RuntimeError("publication protocol requires sessions=3 warmup>=5 runs>=20")
    experiment_root = Path(args.experiment_root)
    prospective = experiment_root / "prospective30"
    candidate_prior = prospective / "candidate_prior.txt"
    measurement_order = prospective / "measurement_order.txt"
    model_path = experiment_root / "models" / "resnet_to_yolo_frozen.json"
    for path in (candidate_prior, measurement_order, model_path, Path(args.rpc_baseline_json)):
        if not path.exists():
            raise FileNotFoundError(path)
    frozen_model = read_json(model_path)
    if not frozen_model.get("frozen") or not frozen_model.get("calibration_measured"):
        raise RuntimeError("prospective30 requires a frozen model with measured calibration")
    model_sha256 = sha256(model_path)
    candidate_ids = read_ids(candidate_prior)
    base_order = read_ids(measurement_order)
    if len(candidate_ids) != 30 or set(candidate_ids) != set(base_order):
        raise RuntimeError("prospective cohort/order must contain the same 30 unique ids")
    shared_root = prospective / "shared_build_and_measure"
    build_rows = prebuild(args, shared_root)
    buildable = {
        row["candidate_id"] for row in build_rows if row.get("status") == "buildable"
    }
    build_failures = [
        row for row in build_rows if row.get("candidate_id") in candidate_ids and row.get("status") != "buildable"
    ]
    write_csv(prospective / "prospective_buildability.csv", build_rows)
    write_json(
        prospective / "prospective_execution_protocol.json",
        {
            "frozen": True,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "board": args.board,
            "candidate_count": 30,
            "buildable_count": len(buildable & set(candidate_ids)),
            "build_failure_count": len(build_failures),
            "independent_processes_per_buildable_candidate": 3,
            "warmup_frames": args.warmup,
            "measured_frames": args.runs,
            "model_updates_allowed_before_completion": False,
            "frozen_model_sha256": model_sha256,
            "candidate_prior_sha256": sha256(candidate_prior),
            "measurement_order_sha256": sha256(measurement_order),
        },
    )
    if args.build_only:
        return
    schedule = []
    for session in range(1, args.sessions + 1):
        order = list(base_order)
        if session > 1:
            random.Random(args.seed + session).shuffle(order)
        schedule.extend((session, candidate_id) for candidate_id in order if candidate_id in buildable)
    write_csv(
        prospective / "board_measurement_schedule.csv",
        [
            {"order": index + 1, "session": session, "candidate_id": candidate_id}
            for index, (session, candidate_id) in enumerate(schedule)
        ],
    )
    results = []
    ok = 0
    for index, (session, candidate_id) in enumerate(schedule, start=1):
        update_progress(prospective, index - 1, len(schedule), candidate_id, ok)
        row = measure_once(args, shared_root, candidate_id, session, index)
        results.append(row)
        if row.get("status") == "ok" and row.get("passes_correctness_gate"):
            ok += 1
        write_csv(prospective / "board_process_results.csv", results)
        write_json(prospective / "board_process_results.json", {"rows": results})
        summary = aggregate(candidate_ids, results)
        write_csv(prospective / "prospective30_summary.csv", summary)
        write_json(prospective / "prospective30_summary.json", {"rows": summary})
    update_progress(prospective, len(schedule), len(schedule), "complete", ok)
    summary = aggregate(candidate_ids, results)
    write_json(
        prospective / "completion.json",
        {
            "candidate_count": 30,
            "build_failure_count": len(build_failures),
            "board_process_count": len(results),
            "successful_board_process_count": ok,
            "publication_valid_candidate_count": sum(
                1 for row in summary if row["publication_valid"]
            ),
            "complete": len(results) == 3 * len(buildable & set(candidate_ids)),
            "model_may_now_be_updated": True,
            "frozen_model_sha256_before": model_sha256,
            "frozen_model_sha256_after": sha256(model_path),
            "model_unchanged_during_measurement": model_sha256 == sha256(model_path),
        },
    )
    if model_sha256 != sha256(model_path):
        raise RuntimeError("frozen RAMPS model changed during prospective30")


if __name__ == "__main__":
    main()
