#!/usr/bin/env python3
"""Run staged ResNet18 profile sweeps for one loaded board/runtime config."""

from __future__ import absolute_import, print_function

import argparse
import json
import os
import subprocess
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument(
        "--config-label",
        required=True,
        help="Config name for this loaded bitstream/runtime, e.g. 1hpc or 4hp",
    )
    parser.add_argument(
        "--schemes",
        default="all_vta,three_stage_a,three_stage_b,three_stage_d,three_stage_e",
        help="Comma-separated split schemes to profile",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--warmup-repeat", type=int, default=1)
    parser.add_argument("--timer-number", type=int, default=1)
    parser.add_argument("--timer-repeat", type=int, default=3)
    parser.add_argument("--stage0-num-threads", type=int, default=3)
    parser.add_argument("--stage2-num-threads", type=int, default=1)
    parser.add_argument("--events-limit", type=int, default=200)
    parser.add_argument(
        "--output-dir",
        default="vta/tutorials/frontend/hp_hpc_quant/results/stage_profile_sweeps",
    )
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


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


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as out:
        json.dump(payload, out, indent=2, sort_keys=True)
        out.write("\n")


def main():
    args = parse_args()
    root = repo_root()
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    schemes = [item.strip() for item in args.schemes.split(",") if item.strip()]
    out_dir = os.path.abspath(os.path.join(root, args.output_dir, args.config_label))
    os.makedirs(out_dir, exist_ok=True)
    manifest = {
        "config_label": args.config_label,
        "host": args.host,
        "port": int(args.port),
        "image_size": int(args.image_size),
        "repeat": int(args.repeat),
        "warmup_repeat": int(args.warmup_repeat),
        "schemes": schemes,
        "started_at": int(time.time()),
        "results": [],
    }

    for scheme in schemes:
        scheme_dir = os.path.join(out_dir, scheme)
        profile_dir = os.path.join(scheme_dir, "profile")
        viz_path = os.path.join(scheme_dir, "partition.svg")
        cmd = [
            args.python,
            "vta/tutorials/frontend/profile_split_resnet18_stages.py",
            "--scheme",
            scheme,
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--image-size",
            str(args.image_size),
            "--vta-stage-mode",
            "vta_build",
            "--run-stages",
            "--repeat",
            str(args.repeat),
            "--warmup-repeat",
            str(args.warmup_repeat),
            "--print-vta-runtime-profile",
            "--vta-runtime-profile-dir",
            profile_dir,
            "--vta-runtime-profile-events-limit",
            str(args.events_limit),
            "--save-partition-viz",
            viz_path,
            "--stage0-num-threads",
            str(args.stage0_num_threads),
            "--stage2-num-threads",
            str(args.stage2_num_threads),
            "--timer-number",
            str(args.timer_number),
            "--timer-repeat",
            str(args.timer_repeat),
        ]
        result = run_and_capture(cmd, os.path.join(scheme_dir, "stage_profile.log"), root, env)
        manifest["results"].append({"scheme": scheme, **result})
        write_json(os.path.join(out_dir, "sweep_manifest.json"), manifest)
        if result["returncode"] != 0:
            raise SystemExit(result["returncode"])

    manifest["finished_at"] = int(time.time())
    write_json(os.path.join(out_dir, "sweep_manifest.json"), manifest)


if __name__ == "__main__":
    main()
