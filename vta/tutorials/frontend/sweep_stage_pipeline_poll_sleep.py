#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Sweep AXU5EVB VTA poll sleep values for the native stage pipeline."""

from __future__ import absolute_import, print_function

import argparse
import csv
import json
import os
import shlex
import shutil
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path


DEFAULT_POLL_VALUES = [
    ("poll_0p5us", 500),
    ("poll_1us_repeat", 1000),
    ("poll_2us", 2000),
    ("poll_5us", 5000),
    ("poll_20us", 20000),
    ("poll_50us", 50000),
    ("poll_100us", 100000),
]


def repo_root():
    return Path(__file__).resolve().parents[3]


def parse_poll_value(text):
    if ":" not in text:
        raise argparse.ArgumentTypeError("expected LABEL:NS, for example poll_2us:2000")
    label, ns = text.split(":", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("poll label must be non-empty")
    try:
        ns_int = int(ns)
    except ValueError as err:
        raise argparse.ArgumentTypeError("poll ns must be an integer") from err
    if ns_int < 0:
        raise argparse.ArgumentTypeError("poll ns must be >= 0")
    return label, ns_int


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run and archive a VTA native stage-pipeline poll sleep sweep."
    )
    parser.add_argument("--board", required=True, help="SSH target, for example root@192.168.1.133")
    parser.add_argument("--remote-dir", default="/mnt/sd/vta_stage_pipeline")
    parser.add_argument(
        "--ssh-option",
        action="append",
        default=[],
        help="Extra option passed to deploy_classification_stage_pipeline_native.py",
    )
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--queue-depth", type=int, default=2)
    parser.add_argument("--runtime-num-threads", type=int, default=4)
    parser.add_argument("--stage0-runtime-num-threads", type=int, default=3)
    parser.add_argument("--stage1-runtime-num-threads", type=int, default=1)
    parser.add_argument("--stage2-runtime-num-threads", type=int, default=1)
    parser.add_argument("--scheme", default="three_stage_e")
    parser.add_argument("--image", default="")
    parser.add_argument("--image-dir", default="")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument(
        "--poll-value",
        action="append",
        type=parse_poll_value,
        default=[],
        help="Poll value as LABEL:NS. Repeat to override the default sweep.",
    )
    parser.add_argument(
        "--output-root",
        default="/tmp",
        help="Local root for fetched per-run board results",
    )
    parser.add_argument(
        "--archive-root",
        default="vta/tutorials/frontend/report_out/native_stage_pipeline_runs",
        help="Repo-relative or absolute archive root",
    )
    parser.add_argument(
        "--sweep-label",
        default="",
        help="Aggregate sweep label; default includes date, scheme, runs, and thread config",
    )
    parser.add_argument("--skip-first", type=int, default=3)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse an existing output dir if it already has both JSONL result files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running board experiments or writing archives",
    )
    return parser.parse_args()


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as inp:
        for line in inp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def avg(rows, key, skip_first):
    values = [row[key] for row in rows[skip_first:] if key in row]
    return statistics.mean(values) if values else None


def throughput_from_span(rows, skip_first):
    keep = rows[skip_first:]
    if len(keep) < 2:
        return None
    first_start = keep[0]["stage0_start_ms"]
    last_end = max(row["stage2_end_ms"] for row in keep)
    return len(keep) * 1000.0 / (last_end - first_start)


def stage0_interval(rows, skip_first):
    starts = [row["stage0_start_ms"] for row in rows]
    intervals = [starts[i] - starts[i - 1] for i in range(1, len(starts))]
    intervals = intervals[skip_first - 1 :] if skip_first else intervals
    return statistics.mean(intervals) if intervals else None


def summarize_result(output_dir, label, poll_ns, args, archive_dir):
    output_dir = Path(output_dir)
    serial_path = output_dir / "stage_serial_result.jsonl"
    pipeline_path = output_dir / "native_result.jsonl"
    if not serial_path.exists() or not pipeline_path.exists():
        raise RuntimeError("missing stage_serial_result.jsonl or native_result.jsonl in %s" % output_dir)

    serial = load_jsonl(serial_path)
    pipeline = load_jsonl(pipeline_path)
    keys = [
        "total_latency_ms",
        "stage0_ms",
        "stage1_ms",
        "stage2_ms",
        "stage0_run_ms",
        "stage1_run_ms",
        "stage2_run_ms",
    ]
    summary = {
        "label": label,
        "source_dir": str(output_dir),
        "archive_dir": str(archive_dir),
        "board": args.board,
        "remote_dir": args.remote_dir,
        "scheme": args.scheme,
        "runs": len(pipeline),
        "requested_runs": args.runs,
        "skip_first": args.skip_first,
        "poll_sleep_ns": poll_ns,
        "post_start_sleep_ns": poll_ns,
        "thread_config": {
            "runtime_num_threads": args.runtime_num_threads,
            "stage0": args.stage0_runtime_num_threads,
            "stage1": args.stage1_runtime_num_threads,
            "stage2": args.stage2_runtime_num_threads,
        },
        "top1_match": [row.get("top1") for row in serial] == [row.get("top1") for row in pipeline],
        "serial": {key: avg(serial, key, args.skip_first) for key in keys},
        "pipeline": {key: avg(pipeline, key, args.skip_first) for key in keys},
        "serial_throughput_fps_span": throughput_from_span(serial, args.skip_first),
        "pipeline_throughput_fps_span": throughput_from_span(pipeline, args.skip_first),
        "pipeline_stage0_start_interval_ms": stage0_interval(pipeline, args.skip_first),
        "pipeline_last_stage2_end_ms": pipeline[-1]["stage2_end_ms"] if pipeline else None,
    }
    return summary


def copy_result_archive(output_dir, archive_dir, summary, command_text):
    output_dir = Path(output_dir)
    archive_dir = Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    for name in ["native_result.jsonl", "stage_serial_result.jsonl", "manifest.json"]:
        src = output_dir / name
        if src.exists():
            shutil.copy2(src, archive_dir / name)
    profile_src = output_dir / "profile"
    if profile_src.exists():
        profile_dst = archive_dir / "profile"
        if profile_dst.exists():
            shutil.rmtree(profile_dst)
        shutil.copytree(profile_src, profile_dst)
    (archive_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (archive_dir / "run_command.sh").write_text(command_text, encoding="utf-8")
    (archive_dir / "README.md").write_text(render_run_readme(summary), encoding="utf-8")


def fmt(value):
    return "n/a" if value is None else "%.3f" % value


def render_run_readme(summary):
    return """# Native Stage Pipeline Poll Sweep Run: {label}

- Board: {board}
- Scheme: {scheme}
- Runs: {runs}
- Poll sleep: {poll_sleep_ns} ns
- Post-start sleep: {post_start_sleep_ns} ns
- Stage runtime threads: stage0={stage0}, stage1={stage1}, stage2={stage2}
- Top1 serial/pipeline match: {top1_match}

## Summary, skip first {skip_first} frames

- Serial throughput: {serial_fps} fps
- Pipeline throughput: {pipeline_fps} fps
- Pipeline stage0 submit interval: {stage0_interval} ms
- Pipeline avg stage0: {stage0_ms} ms
- Pipeline avg stage1: {stage1_ms} ms
- Pipeline avg stage2: {stage2_ms} ms

See `run_command.sh` for the exact command.
""".format(
        label=summary["label"],
        board=summary["board"],
        scheme=summary["scheme"],
        runs=summary["runs"],
        poll_sleep_ns=summary["poll_sleep_ns"],
        post_start_sleep_ns=summary["post_start_sleep_ns"],
        stage0=summary["thread_config"]["stage0"],
        stage1=summary["thread_config"]["stage1"],
        stage2=summary["thread_config"]["stage2"],
        top1_match=summary["top1_match"],
        skip_first=summary["skip_first"],
        serial_fps=fmt(summary["serial_throughput_fps_span"]),
        pipeline_fps=fmt(summary["pipeline_throughput_fps_span"]),
        stage0_interval=fmt(summary["pipeline_stage0_start_interval_ms"]),
        stage0_ms=fmt(summary["pipeline"]["stage0_ms"]),
        stage1_ms=fmt(summary["pipeline"]["stage1_ms"]),
        stage2_ms=fmt(summary["pipeline"]["stage2_ms"]),
    )


def build_deploy_command(args, output_dir):
    script = repo_root() / "vta" / "tutorials" / "frontend" / "deploy_classification_stage_pipeline_native.py"
    cmd = [
        sys.executable,
        str(script),
        "--board",
        args.board,
        "--remote-dir",
        args.remote_dir,
        "--runs",
        str(args.runs),
        "--scheme",
        args.scheme,
        "--queue-depth",
        str(args.queue_depth),
        "--runtime-num-threads",
        str(args.runtime_num_threads),
        "--stage0-runtime-num-threads",
        str(args.stage0_runtime_num_threads),
        "--stage1-runtime-num-threads",
        str(args.stage1_runtime_num_threads),
        "--stage2-runtime-num-threads",
        str(args.stage2_runtime_num_threads),
        "--run-serial-before-pipeline",
        "--compare-serial-pipeline",
        "--fetch-results-dir",
        str(output_dir),
    ]
    for opt in args.ssh_option:
        cmd.extend(["--ssh-option", opt])
    if args.image:
        cmd.extend(["--image", args.image])
    if args.image_dir:
        cmd.extend(["--image-dir", args.image_dir])
    if args.max_images:
        cmd.extend(["--max-images", str(args.max_images)])
    return cmd


def command_text(cmd, poll_ns, output_dir):
    env_prefix = [
        "AXU5EVB_DRIVER_POST_START_SLEEP_NS=%s" % poll_ns,
        "AXU5EVB_DRIVER_POLL_SLEEP_NS=%s" % poll_ns,
        "TEST_DATA_ROOT_PATH=${TEST_DATA_ROOT_PATH:-/tmp/tvm_test_data}",
        "MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mpl}",
        "PYTHONUNBUFFERED=1",
    ]
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'OUT_DIR="%s"\n'
        'mkdir -p "$OUT_DIR"\n'
        "%s \\\n%s\n"
        % (
            output_dir,
            " \\\n".join(env_prefix),
            " \\\n".join("  " + shlex.quote(str(part)) for part in cmd),
        )
    )


def run_command(cmd, poll_ns):
    env = os.environ.copy()
    env["AXU5EVB_DRIVER_POST_START_SLEEP_NS"] = str(poll_ns)
    env["AXU5EVB_DRIVER_POLL_SLEEP_NS"] = str(poll_ns)
    env.setdefault("TEST_DATA_ROOT_PATH", "/tmp/tvm_test_data")
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    env["PYTHONUNBUFFERED"] = "1"
    print("[CMD]", " ".join(shlex.quote(str(part)) for part in cmd))
    subprocess.check_call([str(part) for part in cmd], env=env, cwd=str(repo_root()))


def write_sweep_summary(sweep_dir, summaries):
    sweep_dir = Path(sweep_dir)
    sweep_dir.mkdir(parents=True, exist_ok=True)
    sorted_summaries = sorted(
        summaries,
        key=lambda item: (
            -1 if item.get("pipeline_throughput_fps_span") is None else -item["pipeline_throughput_fps_span"],
            item["poll_sleep_ns"],
        ),
    )
    (sweep_dir / "summary.json").write_text(
        json.dumps(sorted_summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fieldnames = [
        "label",
        "poll_sleep_ns",
        "top1_match",
        "pipeline_throughput_fps_span",
        "serial_throughput_fps_span",
        "pipeline_stage0_start_interval_ms",
        "pipeline_stage0_ms",
        "pipeline_stage1_ms",
        "pipeline_stage2_ms",
        "archive_dir",
    ]
    with open(sweep_dir / "summary.csv", "w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        for item in sorted_summaries:
            writer.writerow(
                {
                    "label": item["label"],
                    "poll_sleep_ns": item["poll_sleep_ns"],
                    "top1_match": item["top1_match"],
                    "pipeline_throughput_fps_span": item["pipeline_throughput_fps_span"],
                    "serial_throughput_fps_span": item["serial_throughput_fps_span"],
                    "pipeline_stage0_start_interval_ms": item["pipeline_stage0_start_interval_ms"],
                    "pipeline_stage0_ms": item["pipeline"]["stage0_ms"],
                    "pipeline_stage1_ms": item["pipeline"]["stage1_ms"],
                    "pipeline_stage2_ms": item["pipeline"]["stage2_ms"],
                    "archive_dir": item["archive_dir"],
                }
            )
    lines = [
        "# VTA Poll Sleep Sweep",
        "",
        "| rank | label | ns | fps | top1 | stage0 | stage1 | stage2 |",
        "|---:|---|---:|---:|---|---:|---:|---:|",
    ]
    for idx, item in enumerate(sorted_summaries, 1):
        lines.append(
            "| %d | %s | %d | %s | %s | %s | %s | %s |"
            % (
                idx,
                item["label"],
                item["poll_sleep_ns"],
                fmt(item["pipeline_throughput_fps_span"]),
                item["top1_match"],
                fmt(item["pipeline"]["stage0_ms"]),
                fmt(item["pipeline"]["stage1_ms"]),
                fmt(item["pipeline"]["stage2_ms"]),
            )
        )
    (sweep_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    poll_values = args.poll_value or DEFAULT_POLL_VALUES
    archive_root = Path(args.archive_root)
    if not archive_root.is_absolute():
        archive_root = repo_root() / archive_root
    today = datetime.now().strftime("%Y%m%d")
    if args.sweep_label:
        sweep_label = args.sweep_label
    else:
        sweep_label = (
            "%s_cat_%s_runs%d_threads_%d_%d_%d_poll_sweep"
            % (
                today,
                args.scheme,
                args.runs,
                args.stage0_runtime_num_threads,
                args.stage1_runtime_num_threads,
                args.stage2_runtime_num_threads,
            )
        )
    sweep_dir = archive_root / "poll_sleep_sweeps" / sweep_label
    summaries = []
    for label, poll_ns in poll_values:
        output_dir = Path(args.output_root) / (
            "vta_stage_pipeline_results_%d_%d_%d_%s"
            % (
                args.stage0_runtime_num_threads,
                args.stage1_runtime_num_threads,
                args.stage2_runtime_num_threads,
                label,
            )
        )
        run_label = (
            "%s_cat_%s_runs%d_threads_%d_%d_%d_%s"
            % (
                today,
                args.scheme,
                args.runs,
                args.stage0_runtime_num_threads,
                args.stage1_runtime_num_threads,
                args.stage2_runtime_num_threads,
                label,
            )
        )
        archive_dir = archive_root / run_label
        cmd = build_deploy_command(args, output_dir)
        text = command_text(cmd, poll_ns, output_dir)
        print("=== %s ns=%d output=%s ===" % (label, poll_ns, output_dir))
        if args.dry_run:
            print(text)
            continue
        has_existing = (output_dir / "stage_serial_result.jsonl").exists() and (
            output_dir / "native_result.jsonl"
        ).exists()
        if args.skip_existing and has_existing:
            print("[SKIP] existing result dir:", output_dir)
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            run_command(cmd, poll_ns)
        summary = summarize_result(output_dir, run_label, poll_ns, args, archive_dir)
        copy_result_archive(output_dir, archive_dir, summary, text)
        summaries.append(summary)
        write_sweep_summary(sweep_dir, summaries)
        print(
            "[RESULT] %s fps=%s top1_match=%s"
            % (label, fmt(summary["pipeline_throughput_fps_span"]), summary["top1_match"])
        )
    if summaries:
        write_sweep_summary(sweep_dir, summaries)
        print("[SWEEP]", sweep_dir)


if __name__ == "__main__":
    main()
