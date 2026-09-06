#!/usr/bin/env python3
"""Run the frozen RAMPS publication protocol with strict phase gates."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time
from typing import Any, Dict, List, Mapping, Sequence


PHASES = ("preflight", "calibration", "identification", "freeze", "prospective30")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", default="root@192.168.1.228")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument(
        "--output-root",
        default="vta/tutorials/frontend/report_out/resource_aware_maxplus/20260713_v1",
    )
    parser.add_argument("--phases", default=",".join(PHASES))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ssh-option", action="append", default=[])
    return parser.parse_args()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def board_host(board: str) -> str:
    return str(board).rsplit("@", 1)[-1]


def ssh_options(args: argparse.Namespace) -> List[str]:
    return args.ssh_option or [
        "HostKeyAlgorithms=+ssh-rsa",
        "PubkeyAcceptedAlgorithms=+ssh-rsa",
        "BatchMode=yes",
        "ConnectTimeout=10",
        "StrictHostKeyChecking=accept-new",
        "UserKnownHostsFile={}".format(Path(args.output_root) / "board_known_hosts"),
    ]


def run_preflight(args: argparse.Namespace) -> Dict[str, Any]:
    command = ["ssh"]
    for option in ssh_options(args):
        command.extend(["-o", option])
    command.extend(
        [
            args.board,
            (
                "test -d /mnt/sd/tvm_deploy && "
                "test $(cat /sys/class/fpga_manager/fpga0/state) = operating && "
                "test $(cat /sys/class/u-dma-buf/udmabuf0/size) = 201326592 && "
                "mkdir -p /var/volatile/ramps_resnet72 "
                "/var/volatile/ramps_yolo_prospective30 && "
                "df -Pk /var/volatile"
            ),
        ]
    )
    completed = subprocess.run(command, text=True, capture_output=True, check=False, timeout=30)
    rpc_open = False
    rpc_error = ""
    try:
        with socket.create_connection((board_host(args.board), int(args.port)), timeout=5):
            rpc_open = True
    except OSError as err:
        rpc_error = repr(err)
    payload = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "board": args.board,
        "rpc_port": int(args.port),
        "ssh_command": shlex.join(command),
        "ssh_returncode": completed.returncode,
        "ssh_stdout": completed.stdout,
        "ssh_stderr": completed.stderr,
        "rpc_open": rpc_open,
        "rpc_error": rpc_error,
        "passed": completed.returncode == 0 and rpc_open,
    }
    write_json(Path(args.output_root) / "preflight.json", payload)
    if not payload["passed"]:
        raise RuntimeError(
            "RAMPS preflight failed: ssh_rc={} rpc_open={} ssh_error={} rpc_error={}".format(
                completed.returncode,
                rpc_open,
                completed.stderr.strip(),
                rpc_error,
            )
        )
    return payload


def phase_command(args: argparse.Namespace, phase: str) -> List[str]:
    frontend = Path(__file__).resolve().parent
    output = Path(args.output_root)
    host = board_host(args.board)
    if phase == "calibration":
        command = [
            sys.executable,
            str(frontend / "calibrate_resource_cost_model.py"),
            "--host",
            host,
            "--port",
            str(args.port),
            "--output-dir",
            str(output / "calibration"),
            "--sessions",
            "3",
            "--warmup",
            "5",
            "--number",
            "20",
        ]
    elif phase == "identification":
        command = [
            sys.executable,
            str(frontend / "run_ramps_resnet_identification.py"),
            "--board",
            args.board,
            "--output-dir",
            str(output / "identification" / "resnet72"),
            "--cost-model-json",
            str(output / "calibration" / "cost_model.json"),
            "--sessions",
            "3",
            "--warmup",
            "5",
            "--runs",
            "20",
        ]
        for option in ssh_options(args):
            command.extend(["--ssh-option", option])
    elif phase == "freeze":
        command = [
            sys.executable,
            str(frontend / "fit_resource_aware_maxplus.py"),
            "--output-dir",
            str(output),
            "--calibration-json",
            str(output / "calibration" / "cost_model.json"),
            "--controlled-summary",
            str(output / "identification" / "resnet72" / "controlled72_summary.json"),
            "--require-measured-calibration",
            "--require-controlled",
            "--bootstrap",
            "500",
            "--select-prospective",
        ]
    elif phase == "prospective30":
        command = [
            sys.executable,
            str(frontend / "run_ramps_yolo_prospective30.py"),
            "--board",
            args.board,
            "--experiment-root",
            str(output),
            "--sessions",
            "3",
            "--warmup",
            "5",
            "--runs",
            "20",
        ]
        for option in ssh_options(args):
            command.extend(["--ssh-option", option])
    else:
        raise ValueError(phase)
    if args.resume and phase in {"calibration", "identification", "prospective30"}:
        command.append("--resume")
    return command


def run_phase(args: argparse.Namespace, phase: str) -> Dict[str, Any]:
    output = Path(args.output_root)
    phase_dir = output / "orchestration" / phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    command = phase_command(args, phase)
    (phase_dir / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    started = time.monotonic()
    with (phase_dir / "run.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    payload = {
        "phase": phase,
        "returncode": completed.returncode,
        "elapsed_s": time.monotonic() - started,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "passed": completed.returncode == 0,
    }
    write_json(phase_dir / "result.json", payload)
    if completed.returncode != 0:
        raise RuntimeError(
            "RAMPS phase {} failed; see {}".format(phase, phase_dir / "run.log")
        )
    return payload


def main() -> None:
    args = parse_args()
    requested = [item.strip() for item in args.phases.split(",") if item.strip()]
    unknown = [item for item in requested if item not in PHASES]
    if unknown:
        raise ValueError("unknown phases: {}".format(unknown))
    if any(item in requested for item in ("calibration", "identification", "prospective30")):
        if not os.environ.get("SDKTARGETSYSROOT"):
            raise RuntimeError(
                "source the PetaLinux environment-setup-aarch64-xilinx-linux before board phases"
            )
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for phase in requested:
        print("[RAMPS] phase {} starting".format(phase), flush=True)
        if phase == "preflight":
            result = run_preflight(args)
        else:
            result = run_phase(args, phase)
        results.append(result)
        write_json(output / "orchestration" / "progress.json", {"rows": results})
        print("[RAMPS] phase {} complete".format(phase), flush=True)


if __name__ == "__main__":
    main()
