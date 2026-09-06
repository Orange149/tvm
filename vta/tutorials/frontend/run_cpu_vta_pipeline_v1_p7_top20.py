#!/usr/bin/env python3
"""Build four ResNet18 topologies and validate the natural static Top-20 on board."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

from compare_cpu_vta_pipeline_v1_reference import compare_files
from freeze_cpu_vta_pipeline_v1 import repo_root


ROOT = repo_root()
REPORT_ROOT = ROOT / "vta/tutorials/frontend/report_out/resource_aware_maxplus"
DEFAULT_RANKING = REPORT_ROOT / "v1_p5b_iteration2_ranked_candidates.json"
DEFAULT_OUTPUT = REPORT_ROOT / "v1_p7_top20_board_20260904"
DEFAULT_REFERENCE = REPORT_ROOT / "v1_p5a_reference.json"
DEPLOY = ROOT / "vta/tutorials/frontend/deploy_classification_stage_pipeline_native.py"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", default="root@192.168.1.247")
    parser.add_argument("--ranking", default=str(DEFAULT_RANKING))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--reference-json", default=str(DEFAULT_REFERENCE))
    parser.add_argument("--stage-cache-dir", default=str(REPORT_ROOT / "v1_p1_stage_cache"))
    parser.add_argument("--remote-root", default="/mnt/sd/v1_p7_top20")
    parser.add_argument("--ranks", default="1-20", help="Comma-separated ranks or ranges")
    parser.add_argument("--runs", type=int, default=22)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--ssh-option", action="append", default=[])
    return parser.parse_args()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_ranks(text):
    result = set()
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            result.update(range(int(left), int(right) + 1))
        else:
            result.add(int(token))
    return result


def topology_key(row):
    encoded = json.dumps(row["scheme_cfg"], sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def package_dir(output_dir, row):
    return Path(output_dir) / "packages" / ("{:02d}_{}".format(row["rank"], row["candidate_id"]))


def package_matches(path, row):
    manifest_path = Path(path) / "package" / "manifest.json"
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    observed = [
        {key: stage.get(key) for key in ("name", "device", "unit_names")}
        for stage in manifest.get("stages", [])
    ]
    expected = [
        {key: stage.get(key) for key in ("name", "device", "unit_names")}
        for stage in row["scheme_cfg"]
    ]
    return observed == expected


def run_logged(command, log_path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[RUN]", " ".join(str(part) for part in command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            [str(part) for part in command],
            cwd=str(ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if proc.returncode:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
        raise RuntimeError("command failed; {}\n{}".format(log_path, "\n".join(tail)))


def build_topology(args, row):
    target = package_dir(args.output_dir, row)
    if not args.force and package_matches(target, row):
        print("[BUILD-RESUME]", target)
        return target / "package"
    scheme_path = target / "scheme.json"
    write_json(
        scheme_path,
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p7_top20_scheme",
            "candidate_id": row["candidate_id"],
            "scheme_name": row["candidate_id"],
            "scheme_cfg": row["scheme_cfg"],
            "stage_runtime_threads": row["stage_runtime_threads"],
            "natural_static_rank": row["rank"],
        },
    )
    command = [
        sys.executable,
        str(DEPLOY),
        "--scheme-config-json",
        str(scheme_path),
        "--candidate-id",
        row["candidate_id"],
        "--stage-runtime-num-threads",
        ",".join(str(value) for value in row["stage_runtime_threads"]),
        "--runtime-num-threads",
        "4",
        "--runs",
        str(args.runs),
        "--queue-depth",
        "2",
        "--run-serial-before-pipeline",
        "--runner-output-mode",
        "raw",
        "--output-dump-dir",
        "correctness_outputs",
        "--stage-build-cache-dir",
        str(Path(args.stage_cache_dir).resolve()),
        "--build-dir",
        str(target),
        "--package-only",
    ]
    run_logged(command, target / "build.log")
    if not package_matches(target, row):
        raise RuntimeError("built package topology does not match {}".format(row["candidate_id"]))
    return target / "package"


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as inp:
        return [json.loads(line) for line in inp if line.strip()]


def output_file(result_dir, row):
    rel = row["raw_output_files"][0]["path"]
    return Path(result_dir) / rel


def analyze_result(args, row, result_dir):
    result_dir = Path(result_dir)
    manifest = json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))
    expected_threads = {
        "stage{}".format(index): int(value)
        for index, value in enumerate(row["stage_runtime_threads"])
    }
    actual_threads = manifest.get("pipeline_stage_runtime_threads")
    if actual_threads != expected_threads:
        raise RuntimeError(
            "rank {} board manifest thread mismatch: expected {}, got {}".format(
                row["rank"], expected_threads, actual_threads
            )
        )
    if manifest.get("candidate_id") != row["candidate_id"]:
        raise RuntimeError("rank {} board manifest candidate mismatch".format(row["rank"]))
    serial = read_jsonl(result_dir / "stage_serial_result.jsonl")
    pipeline = read_jsonl(result_dir / "native_result.jsonl")
    if len(serial) != args.runs or len(pipeline) != args.runs:
        raise RuntimeError("rank {} result count mismatch".format(row["rank"]))
    serial_by_frame = {int(item["frame_id"]): item for item in serial}
    pipeline_by_frame = {int(item["frame_id"]): item for item in pipeline}
    if set(serial_by_frame) != set(pipeline_by_frame):
        raise RuntimeError("rank {} serial/pipeline frame mismatch".format(row["rank"]))
    deterministic_output = None
    for frame_id in serial_by_frame:
        left = output_file(result_dir, serial_by_frame[frame_id])
        right = output_file(result_dir, pipeline_by_frame[frame_id])
        left_bytes = left.read_bytes()
        right_bytes = right.read_bytes()
        if left_bytes != right_bytes:
            raise RuntimeError("rank {} frame {} output mismatch".format(row["rank"], frame_id))
        if deterministic_output is None:
            deterministic_output = right_bytes
        elif right_bytes != deterministic_output:
            raise RuntimeError("rank {} output is not deterministic".format(row["rank"]))

    scored = sorted(pipeline, key=lambda item: int(item["completion_index"]))[args.warmup :]
    if len(scored) < 2:
        raise RuntimeError("not enough scored pipeline rows")
    end_key = "stage{}_end_ms".format(int(scored[0]["stage_count"]) - 1)
    completion = [float(item[end_key]) for item in scored]
    intervals = [right - left for left, right in zip(completion, completion[1:])]
    if any(value <= 0.0 for value in intervals):
        raise RuntimeError("non-positive completion interval")
    cycle_ms = (completion[-1] - completion[0]) / (len(completion) - 1)
    reference = compare_files(args.reference_json, output_file(result_dir, scored[0]))
    write_json(result_dir / "reference_comparison.json", reference)
    if not reference["passed"]:
        raise RuntimeError("rank {} independent reference failed".format(row["rank"]))

    stage_medians = {}
    for index in range(int(scored[0]["stage_count"])):
        stage_medians["stage{}".format(index)] = statistics.median(
            float(item["stage{}_ms".format(index)]) for item in scored
        )
    predicted_raw_ms = float(row["predicted_ii_ms"])
    predicted_fixed_ms = predicted_raw_ms + 34.0
    summary = {
        "schema_version": 1,
        "kind": "cpu_vta_pipeline_v1_p7_top20_board_result",
        "natural_static_rank": int(row["rank"]),
        "candidate_id": row["candidate_id"],
        "stage_runtime_threads": row["stage_runtime_threads"],
        "deployment_receipt": {
            "board": args.board,
            "remote_dir": "{}/rank{:02d}".format(args.remote_root.rstrip("/"), row["rank"]),
            "endpoint_source": "host_deploy_command_and_deploy_log",
            "board_returned_manifest_contains_endpoint": bool(manifest.get("deployment")),
        },
        "prediction": {
            "raw_ii_ms": predicted_raw_ms,
            "raw_fps": 1000.0 / predicted_raw_ms,
            "fixed_residual_ms": 34.0,
            "fixed_residual_ii_ms": predicted_fixed_ms,
            "fixed_residual_fps": 1000.0 / predicted_fixed_ms,
        },
        "measurement": {
            "warmup_frames": int(args.warmup),
            "scored_frames": len(scored),
            "pipeline_cycle_ms": cycle_ms,
            "pipeline_fps": 1000.0 / cycle_ms,
            "completion_interval_median_ms": statistics.median(intervals),
            "completion_interval_min_ms": min(intervals),
            "completion_interval_max_ms": max(intervals),
            "pipeline_latency_median_ms": statistics.median(
                float(item["total_latency_ms"]) for item in scored
            ),
            "pipeline_stage_median_ms": stage_medians,
        },
        "correctness": {
            "serial_pipeline_exact_frames": len(serial),
            "deterministic_output_frames": len(pipeline),
            "board_manifest_threads_match": True,
            "independent_reference_passed": True,
            "cosine_similarity": reference.get("cosine_similarity"),
            "normalized_rmse": reference.get("normalized_rmse"),
            "top_k_overlap": reference.get("top_k_overlap"),
        },
    }
    write_json(result_dir / "summary.json", summary)
    return summary


def deploy_candidate(args, row, source_package):
    result_dir = Path(args.output_dir) / "results" / "rank{:02d}_valid".format(row["rank"])
    summary_path = result_dir / "summary.json"
    if summary_path.exists() and not args.force:
        cached_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        cached_correctness = cached_summary.get("correctness") or {}
        if (
            cached_correctness.get("board_manifest_threads_match") is True
            and cached_summary.get("deployment_receipt")
        ):
            print("[BOARD-RESUME]", summary_path)
            return cached_summary
    complete_outputs = all(
        (result_dir / name).exists()
        for name in ("manifest.json", "stage_serial_result.jsonl", "native_result.jsonl")
    ) and (result_dir / "correctness_outputs").is_dir()
    if complete_outputs and not args.force:
        print("[ANALYZE-RESUME]", result_dir)
        return analyze_result(args, row, result_dir)
    command = [
        sys.executable,
        str(DEPLOY),
        "--board",
        args.board,
        "--remote-dir",
        "{}/rank{:02d}".format(args.remote_root.rstrip("/"), row["rank"]),
        "--remote-min-free-mb",
        "128",
        "--reuse-package-dir",
        str(source_package),
        "--candidate-id",
        row["candidate_id"],
        "--stage-runtime-num-threads",
        ",".join(str(value) for value in row["stage_runtime_threads"]),
        "--runtime-num-threads",
        "4",
        "--runs",
        str(args.runs),
        "--queue-depth",
        "2",
        "--run-serial-before-pipeline",
        "--runner-output-mode",
        "raw",
        "--output-dump-dir",
        "correctness_outputs",
        "--serial-output-jsonl",
        "stage_serial_result.jsonl",
        "--pipeline-output-jsonl",
        "native_result.jsonl",
        "--fetch-results-dir",
        str(result_dir),
        "--compare-serial-pipeline",
        "--cleanup-remote-after-run",
        "--ssh-command-timeout-s",
        "60",
        "--scp-timeout-s",
        "600",
        "--serial-timeout-s",
        "300",
        "--pipeline-timeout-s",
        "300",
        "--fetch-timeout-s",
        "180",
    ]
    for option in args.ssh_option:
        command.extend(["--ssh-option", option])
    run_logged(command, result_dir / "deploy.log")
    return analyze_result(args, row, result_dir)


def write_aggregate(output_dir, summaries, board=""):
    ordered = sorted(summaries, key=lambda item: item["natural_static_rank"])
    measured = sorted(ordered, key=lambda item: item["measurement"]["pipeline_fps"], reverse=True)
    measured_rank = {item["candidate_id"]: index + 1 for index, item in enumerate(measured)}
    for item in ordered:
        item["measured_rank_within_top20"] = measured_rank[item["candidate_id"]]
    oracle_fps = measured[0]["measurement"]["pipeline_fps"]
    raw_errors = []
    fixed_errors = []
    residuals = []
    for item in ordered:
        actual_ms = item["measurement"]["pipeline_cycle_ms"]
        raw_ms = item["prediction"]["raw_ii_ms"]
        fixed_ms = item["prediction"]["fixed_residual_ii_ms"]
        raw_errors.append(abs(raw_ms - actual_ms) / actual_ms)
        fixed_errors.append(abs(fixed_ms - actual_ms) / actual_ms)
        residuals.append(actual_ms - raw_ms)
    static_ranks = [float(item["natural_static_rank"]) for item in ordered]
    actual_ranks = [float(item["measured_rank_within_top20"]) for item in ordered]
    static_mean = statistics.mean(static_ranks)
    actual_mean = statistics.mean(actual_ranks)
    covariance = sum(
        (left - static_mean) * (right - actual_mean)
        for left, right in zip(static_ranks, actual_ranks)
    )
    denominator = (
        sum((value - static_mean) ** 2 for value in static_ranks)
        * sum((value - actual_mean) ** 2 for value in actual_ranks)
    ) ** 0.5
    regret = {}
    for k in (1, 3, 5, 10, 20):
        if k <= len(ordered):
            best = max(item["measurement"]["pipeline_fps"] for item in ordered[:k])
            regret[str(k)] = 1.0 - best / oracle_fps
    threshold = 0.95 * oracle_fps
    evaluations_to_95 = next(
        index
        for index, item in enumerate(ordered, start=1)
        if item["measurement"]["pipeline_fps"] >= threshold
    )
    payload = {
        "schema_version": 1,
        "kind": "cpu_vta_pipeline_v1_p7_top20_board_summary",
        "board": board,
        "selection_policy": "natural_iteration2_static_top20;_34ms_constant_does_not_change_order",
        "measured_candidate_count": len(ordered),
        "summary": {
            "correctness_pass_count": sum(
                bool(item["correctness"]["independent_reference_passed"]) for item in ordered
            ),
            "measured_pool_scope": "these_natural_static_top20_only;_not_global_oracle",
            "measured_pool_oracle_fps": oracle_fps,
            "measured_pool_oracle_candidate_id": measured[0]["candidate_id"],
            "measured_pool_oracle_natural_static_rank": measured[0]["natural_static_rank"],
            "throughput_regret_at_k": regret,
            "evaluations_to_95_percent_measured_pool_oracle": evaluations_to_95,
            "static_vs_measured_rank_spearman": covariance / denominator if denominator else None,
            "raw_cycle_mape": statistics.mean(raw_errors),
            "fixed_34ms_cycle_mape": statistics.mean(fixed_errors),
            "actual_minus_raw_cycle_residual_median_ms": statistics.median(residuals),
            "actual_minus_raw_cycle_residual_mean_ms": statistics.mean(residuals),
            "count_above_prior_10_449_fps_session_result": sum(
                item["measurement"]["pipeline_fps"] > 10.449 for item in ordered
            ),
            "interpretation": (
                "selection reaches a high-performance region, but fine ordering and the "
                "historical 34ms absolute calibration do not transfer accurately"
            ),
        },
        "rows": ordered,
    }
    write_json(Path(output_dir) / "top20_board_summary.json", payload)
    if len(ordered) == 20:
        report = [
            "# ResNet18 Natural Top-20 Board Validation",
            "",
            "- Selection: Iteration 2 natural static Top-20; candidate throughput was not used for ranking.",
            "- Correctness: {}/20 passed exact serial/pipeline output and independent MXNet reference gates.".format(
                payload["summary"]["correctness_pass_count"]
            ),
            "- Best measured candidate: static rank {} at {:.3f} FPS.".format(
                payload["summary"]["measured_pool_oracle_natural_static_rank"], oracle_fps
            ),
            "- Scope: one boot/session; the measured-pool oracle covers only these 20 candidates.",
            "- Raw / +34 ms cycle MAPE: {:.3f}% / {:.3f}%.".format(
                100.0 * payload["summary"]["raw_cycle_mape"],
                100.0 * payload["summary"]["fixed_34ms_cycle_mape"],
            ),
            "- Static-vs-measured rank Spearman: {:.3f}.".format(
                payload["summary"]["static_vs_measured_rank_spearman"]
            ),
            "",
            "| Static rank | Measured rank | Threads | Raw FPS | +34 ms FPS | Measured FPS | Candidate |",
            "|---:|---:|---|---:|---:|---:|---|",
        ]
        for item in ordered:
            report.append(
                "| {static} | {measured} | `{threads}` | {raw:.3f} | {fixed:.3f} | {actual:.3f} | `{candidate}` |".format(
                    static=item["natural_static_rank"],
                    measured=item["measured_rank_within_top20"],
                    threads=",".join(str(value) for value in item["stage_runtime_threads"]),
                    raw=item["prediction"]["raw_fps"],
                    fixed=item["prediction"]["fixed_residual_fps"],
                    actual=item["measurement"]["pipeline_fps"],
                    candidate=item["candidate_id"],
                )
            )
        (Path(output_dir) / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if args.runs <= args.warmup + 1:
        raise RuntimeError("--runs must leave at least two scored frames")
    ranking = json.loads(Path(args.ranking).read_text(encoding="utf-8"))
    wanted = parse_ranks(args.ranks)
    rows = [row for row in ranking["rows"][:20] if int(row["rank"]) in wanted]
    if not rows:
        raise RuntimeError("no requested Top-20 ranks")

    sources = {}
    for row in rows:
        key = topology_key(row)
        if key not in sources:
            sources[key] = build_topology(args, row)
    if args.build_only:
        return 0

    summaries = []
    for row in rows:
        summaries.append(deploy_candidate(args, row, sources[topology_key(row)]))
        write_aggregate(args.output_dir, summaries, board=args.board)
        result = summaries[-1]
        print(
            "[RESULT] rank={} measured_fps={:.6f} measured_cycle_ms={:.6f}".format(
                row["rank"],
                result["measurement"]["pipeline_fps"],
                result["measurement"]["pipeline_cycle_ms"],
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
