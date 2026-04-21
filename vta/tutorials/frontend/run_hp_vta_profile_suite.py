#!/usr/bin/env python3
"""Run the HP-only VTA profiling and validation suite."""

from __future__ import absolute_import, print_function

import argparse
import json
import os
import subprocess
import sys
import time

from tvm import rpc

import vta
from vta_runtime_profile_utils import (
    dump_runtime_snapshot,
    fetch_runtime_profiler_hooks,
    hooks_available,
    read_runtime_status,
    write_manifest,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument(
        "--output-dir",
        default="vta/tutorials/frontend/report_out/hp_profile_suite",
        help="Directory for suite logs, CSVs, and profiler JSON artifacts",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use for subprocess test entrypoints",
    )
    parser.add_argument(
        "--single-op-cases",
        default="s1_conv3x3_64_64,s2_conv3x3_128_128,s4_conv3x3_512_512",
    )
    parser.add_argument(
        "--repeat-case",
        default="s1_conv3x3_64_64",
    )
    parser.add_argument("--repeat-runs", type=int, default=5)
    parser.add_argument("--single-op-number", type=int, default=10)
    parser.add_argument("--single-op-warmup", type=int, default=2)
    parser.add_argument("--repeat-number", type=int, default=1)
    parser.add_argument("--repeat-warmup", type=int, default=0)
    parser.add_argument("--e2e-benchmark-runs", type=int, default=20)
    parser.add_argument("--long-run-benchmark-runs", type=int, default=50)
    parser.add_argument("--long-run-checkpoint-every", type=int, default=10)
    parser.add_argument("--stage-repeat", type=int, default=5)
    parser.add_argument("--stage-warmup-repeat", type=int, default=1)
    parser.add_argument("--events-limit", type=int, default=200)
    parser.add_argument(
        "--steps",
        default="env,build,single_op,repeat,e2e,stage,long_run",
        help="Comma-separated subset of steps to run",
    )
    return parser.parse_args()


def repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def run_and_capture(cmd, log_path, cwd, env):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    start = time.time()
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write("$ {}\n\n".format(" ".join(cmd)))
        log_file.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()
        ret = proc.wait()
    return {
        "cmd": cmd,
        "log_path": log_path,
        "returncode": int(ret),
        "duration_s": time.time() - start,
    }


def connect_remote(host, port):
    return rpc.connect(host, int(port))


def main():
    args = parse_args()
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    root = repo_root()
    out_dir = os.path.abspath(os.path.join(root, args.output_dir))
    os.makedirs(out_dir, exist_ok=True)
    steps = {item.strip() for item in args.steps.split(",") if item.strip()}
    manifest = {
        "started_at": int(time.time()),
        "host": args.host,
        "port": int(args.port),
        "repo_root": root,
        "python": args.python,
        "steps": sorted(steps),
        "results": [],
    }

    if "env" in steps:
        remote = connect_remote(args.host, args.port)
        profiler_hooks = fetch_runtime_profiler_hooks(remote)
        env_info = {
            "target": vta.get_env().TARGET,
            "host": args.host,
            "port": int(args.port),
            "python": args.python,
            "cwd": root,
            "profiler_hooks": {
                "clear": profiler_hooks.get("clear") is not None,
                "status": profiler_hooks.get("status") is not None,
                "events": profiler_hooks.get("events") is not None,
            },
        }
        write_manifest(os.path.join(out_dir, "env_info.json"), env_info)
        if hooks_available(profiler_hooks):
            write_manifest(
                os.path.join(out_dir, "profiler_bootstrap_status.json"),
                read_runtime_status(profiler_hooks),
            )
            dump_runtime_snapshot(
                out_dir,
                "profiler_bootstrap",
                profiler_hooks,
                events_limit=args.events_limit,
                extra={"phase": "bootstrap"},
            )
        manifest["results"].append({"step": "env", "status": "ok"})

    if "build" in steps:
        cmd = [args.python, "vta/tutorials/frontend/test_minimal_ext_dev_hetero_build.py"]
        result = run_and_capture(cmd, os.path.join(out_dir, "minimal_build.log"), root, env)
        manifest["results"].append({"step": "build", **result})
        if result["returncode"] != 0:
            write_manifest(os.path.join(out_dir, "suite_manifest.json"), manifest)
            raise SystemExit(result["returncode"])

    if "single_op" in steps:
        single_profile_dir = os.path.join(out_dir, "single_op")
        cmd = [
            args.python,
            "vta/tutorials/frontend/benchmark_resnet18_single_ops.py",
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--devices",
            "vta",
            "--cases",
            args.single_op_cases,
            "--warmup",
            str(args.single_op_warmup),
            "--number",
            str(args.single_op_number),
            "--check-correctness",
            "--output",
            os.path.join(out_dir, "hp_conv_compare.csv"),
            "--vta-runtime-profile-dir",
            single_profile_dir,
            "--vta-runtime-profile-events-limit",
            str(args.events_limit),
        ]
        result = run_and_capture(cmd, os.path.join(out_dir, "single_op.log"), root, env)
        manifest["results"].append({"step": "single_op", **result})
        if result["returncode"] != 0:
            write_manifest(os.path.join(out_dir, "suite_manifest.json"), manifest)
            raise SystemExit(result["returncode"])

    if "repeat" in steps:
        repeat_dir = os.path.join(out_dir, "repeat_runs")
        os.makedirs(repeat_dir, exist_ok=True)
        for idx in range(1, args.repeat_runs + 1):
            run_dir = os.path.join(repeat_dir, "run_{}".format(idx))
            cmd = [
                args.python,
                "vta/tutorials/frontend/benchmark_resnet18_single_ops.py",
                "--host",
                args.host,
                "--port",
                str(args.port),
                "--devices",
                "vta",
                "--cases",
                args.repeat_case,
                "--warmup",
                str(args.repeat_warmup),
                "--number",
                str(args.repeat_number),
                "--check-correctness",
                "--output",
                os.path.join(repeat_dir, "hp_repeat_{}.csv".format(idx)),
                "--vta-runtime-profile-dir",
                run_dir,
                "--vta-runtime-profile-events-limit",
                str(args.events_limit),
            ]
            result = run_and_capture(cmd, os.path.join(run_dir, "repeat.log"), root, env)
            manifest["results"].append({"step": "repeat_{}".format(idx), **result})
            if result["returncode"] != 0:
                write_manifest(os.path.join(out_dir, "suite_manifest.json"), manifest)
                raise SystemExit(result["returncode"])

    if "e2e" in steps:
        e2e_dir = os.path.join(out_dir, "e2e")
        cmd = [
            args.python,
            "vta/tutorials/frontend/deploy_classification_optimized.py",
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--device",
            "vta",
            "--hetero",
            "--warmup",
            "2",
            "--benchmark-runs",
            str(args.e2e_benchmark_runs),
            "--stage-profile-repeat",
            "3",
            "--vta-runtime-profile-dir",
            e2e_dir,
            "--vta-runtime-profile-events-limit",
            str(args.events_limit),
        ]
        result = run_and_capture(
            cmd, os.path.join(e2e_dir, "deploy_classification_optimized.log"), root, env
        )
        manifest["results"].append({"step": "e2e", **result})
        if result["returncode"] != 0:
            write_manifest(os.path.join(out_dir, "suite_manifest.json"), manifest)
            raise SystemExit(result["returncode"])

    if "stage" in steps:
        stage_dir = os.path.join(out_dir, "stage_profile")
        cmd = [
            args.python,
            "vta/tutorials/frontend/profile_split_resnet18_stages.py",
            "--scheme",
            "three_stage_a",
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--vta-stage-mode",
            "vta_build",
            "--run-stages",
            "--repeat",
            str(args.stage_repeat),
            "--warmup-repeat",
            str(args.stage_warmup_repeat),
            "--print-vta-runtime-profile",
            "--vta-runtime-profile-dir",
            stage_dir,
            "--vta-runtime-profile-events-limit",
            str(args.events_limit),
        ]
        result = run_and_capture(cmd, os.path.join(stage_dir, "stages.log"), root, env)
        manifest["results"].append({"step": "stage", **result})
        if result["returncode"] != 0:
            write_manifest(os.path.join(out_dir, "suite_manifest.json"), manifest)
            raise SystemExit(result["returncode"])

    if "long_run" in steps:
        long_run_dir = os.path.join(out_dir, "long_run")
        cmd = [
            args.python,
            "vta/tutorials/frontend/deploy_classification_optimized.py",
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--device",
            "vta",
            "--hetero",
            "--warmup",
            "2",
            "--benchmark-runs",
            str(args.long_run_benchmark_runs),
            "--stage-profile-repeat",
            "0",
            "--vta-runtime-profile-dir",
            long_run_dir,
            "--vta-runtime-profile-events-limit",
            str(args.events_limit),
            "--vta-runtime-profile-checkpoint-every",
            str(args.long_run_checkpoint_every),
        ]
        result = run_and_capture(cmd, os.path.join(long_run_dir, "bench50.log"), root, env)
        manifest["results"].append({"step": "long_run", **result})
        if result["returncode"] != 0:
            write_manifest(os.path.join(out_dir, "suite_manifest.json"), manifest)
            raise SystemExit(result["returncode"])

    manifest["finished_at"] = int(time.time())
    write_manifest(os.path.join(out_dir, "suite_manifest.json"), manifest)


if __name__ == "__main__":
    main()
