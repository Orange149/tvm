#!/usr/bin/env python3
"""Run the frozen P7R W04 paired correctness probes on AXU5EVB.

The runner intentionally collects no latency.  It builds each module locally,
uploads through the existing tmpfs-backed RPC server, and checks three seeds
against an independent NumPy reference.  A shell-level timeout is still required
by the caller so a wedged FPGA call can be terminated without rebooting the board.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import tvm
from tvm import autotvm, rpc
from tvm.contrib import cc
import vta

import vta.top.vta_conv2d_residency  # pylint: disable=unused-import
from qualify_vta_residency_fsim import array_sha256, reference_data_from_workload


MODE_NUMBERS = {
    "original": 0,
    "input_stationary": 1,
    "paper_inspired_hybrid": 3,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_contract(path):
    contract = json.loads(Path(path).read_text(encoding="utf-8"))
    if contract.get("schema") not in (
        "c3_p7r_w04_board_contract_v1",
        "c3_p7r_unseen_board_contract_v1",
        "c3_p7r_manifest_attestation_contract_v1",
    ):
        raise ValueError("unexpected contract schema")
    if contract.get("status") != "frozen_before_p7r_board_labels":
        raise ValueError("contract is not frozen")
    for name, expected in contract["source_guards_sha256"].items():
        if sha256_file(name) != expected:
            raise ValueError("source guard changed: {}".format(name))
    for name, expected in contract["input_guards_sha256"].items():
        if sha256_file(name) != expected:
            raise ValueError("input guard changed: {}".format(name))
    return contract


def ssh_preflight(contract, host):
    cutoff = float(contract["board"]["dmesg_error_cutoff_seconds"])
    command = r'''
echo BOOT=$(cat /proc/sys/kernel/random/boot_id)
echo FPGA=$(cat /sys/class/fpga_manager/fpga0/state)
echo UDMABUF=$(cat /sys/class/u-dma-buf/udmabuf0/size)
rpcpid=$(pidof tvm_rpc | awk '{print $1}')
echo RPC=$rpcpid
echo RPC_CWD=$(readlink /proc/$rpcpid/cwd)
mount | grep 'mmcblk1p2 ' | sed 's/^/MOUNT=/'
df -Pk /media/sd-mmcblk1p2 | tail -n 1 | sed 's/^/DF=/'
dmesg | awk '$1 ~ /^\[/ {t=$1; gsub(/\[/,"",t); if ((t+0)>%s && ($0 ~ /EXT4-fs error|mmcblk.*error|I.O error|Buffer I.O/)) print "NEW_STORAGE_ERROR=" $0}'
''' % cutoff
    completed = subprocess.run(
        [
            "ssh",
            "-o", "ConnectTimeout=5",
            "-o", "HostKeyAlgorithms=+ssh-rsa",
            "-o", "PubkeyAcceptedAlgorithms=+ssh-rsa",
            "root@" + host,
            command,
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
    )
    fields = {}
    for line in completed.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key] = value
    expected = contract["board"]
    if fields.get("BOOT") != expected["boot_id"]:
        raise RuntimeError("boot ID changed")
    if fields.get("FPGA") != "operating":
        raise RuntimeError("FPGA is not operating")
    if fields.get("UDMABUF") != str(expected["udmabuf_bytes"]):
        raise RuntimeError("u-dma-buf size changed")
    if not fields.get("RPC") or not fields.get("RPC_CWD", "").startswith(
        expected["rpc_cwd_requirement"]
    ):
        raise RuntimeError("RPC is absent or not tmpfs-backed")
    if "(rw," not in fields.get("MOUNT", ""):
        raise RuntimeError("SD partition is not read-write")
    if "NEW_STORAGE_ERROR" in fields:
        raise RuntimeError(fields["NEW_STORAGE_ERROR"])
    return fields


def cross_options():
    sysroot = os.environ.get("SDKTARGETSYSROOT")
    if not sysroot:
        raise RuntimeError("source the AXU SDK environment first")
    return [
        "--sysroot=" + sysroot,
        "-Wl,-rpath-link," + sysroot + "/lib",
        "-Wl,-rpath-link," + sysroot + "/usr/lib",
        "-L" + sysroot + "/lib",
        "-L" + sysroot + "/usr/lib",
    ]


def make_task(workload, mode, env):
    return autotvm.task.create(
        "conv2d_packed_residency.vta",
        args=tuple(workload[1:]) + (MODE_NUMBERS[mode],),
        target=env.target,
        target_host=env.target_host,
    )


def build_module(workload, mode, config_index, env, directory, expected_tir=None):
    task = make_task(workload, mode, env)
    config = task.config_space.get(int(config_index))
    with task.target:
        schedule, tensors = task.instantiate(config)
    with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
        lowered = tvm.lower(schedule, tensors, name="main")
    tir_hash = hashlib.sha256(tvm.ir.save_json(lowered).encode()).hexdigest()
    if expected_tir is not None and tir_hash != expected_tir:
        raise RuntimeError("frozen TIR mismatch for {} index {}".format(mode, config_index))
    with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
        module = vta.build(
            schedule,
            tensors,
            target=tvm.target.Target(env.target, host=env.target_host),
            name="main",
        )
    binary = directory / ("{}_{}.so".format(mode, config_index))
    module.export_library(
        str(binary),
        fcompile=cc.cross_compiler("aarch64-xilinx-linux-g++"),
        options=cross_options(),
    )
    return binary, tir_hash, config.to_json_dict()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--query-queue-capacity-status", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    output.mkdir(parents=True)
    results_path = output / "correctness.jsonl"
    contract = load_contract(args.contract)
    env = vta.get_env()
    if env.TARGET != "axu5evb":
        raise RuntimeError("VTA TARGET must be axu5evb")
    before = ssh_preflight(contract, args.host)
    remote = rpc.connect(args.host, args.port, session_timeout=120)
    device = remote.ext_dev(0)
    queue_capacity_status = None
    rows = []
    start = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="c3_p7r_board_") as temporary:
            directory = Path(temporary)
            modules = {}
            for entry in contract["execution_order"]:
                key = (entry["mode"], int(entry["config_index"]))
                if key in modules:
                    continue
                binary, tir_hash, config = build_module(
                    contract["workload"],
                    entry["mode"],
                    entry["config_index"],
                    env,
                    directory,
                    entry.get("tir_sha256"),
                )
                remote.upload(str(binary))
                modules[key] = {
                    "function": remote.load_module(binary.name)["main"],
                    "tir_sha256": tir_hash,
                    "config": config,
                    "binary_sha256": sha256_file(binary),
                }

            with results_path.open("x", encoding="utf-8") as stream:
                for ordinal, entry in enumerate(contract["execution_order"]):
                    key = (entry["mode"], int(entry["config_index"]))
                    built = modules[key]
                    seed_rows = []
                    for seed in contract["seeds"]:
                        data, weight, expected = reference_data_from_workload(
                            contract["workload"], int(seed)
                        )
                        buffers = [
                            tvm.nd.array(data, device),
                            tvm.nd.array(weight, device),
                            tvm.nd.empty(expected.shape, "int8", device),
                        ]
                        built["function"](*buffers)
                        actual = buffers[-1].numpy()
                        mismatch = int(np.count_nonzero(actual != expected))
                        seed_rows.append(
                            {
                                "seed": int(seed),
                                "correct": mismatch == 0,
                                "mismatch_count": mismatch,
                                "expected_sha256": array_sha256(expected),
                                "actual_sha256": array_sha256(actual),
                            }
                        )
                    row = {
                        "ordinal": ordinal,
                        "role": entry["role"],
                        "candidate_id": entry.get("candidate_id"),
                        "mode": entry["mode"],
                        "config_index": int(entry["config_index"]),
                        "complete_config_entity": built["config"],
                        "tir_sha256": built["tir_sha256"],
                        "binary_sha256": built["binary_sha256"],
                        "seeds": seed_rows,
                        "correct": all(seed["correct"] for seed in seed_rows),
                        "performance_measurement": "not_collected",
                    }
                    rows.append(row)
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                    stream.flush()
                    print(
                        "{}/{} {} {} correct={}".format(
                            ordinal + 1,
                            len(contract["execution_order"]),
                            entry["mode"],
                            entry["config_index"],
                            row["correct"],
                        ),
                        flush=True,
                    )
                    if not row["correct"]:
                        raise RuntimeError("wrong answer; stop before any timing")
    except Exception as error:
        write_json(output / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise

    if args.query_queue_capacity_status:
        status_func = remote.get_function("vta.runtime.queue_capacity_status")
        queue_capacity_status = json.loads(status_func())
    after = ssh_preflight(contract, args.host)
    summary = {
        "schema": "c3_p7r_w04_board_correctness_v1",
        "status": "passed" if all(row["correct"] for row in rows) else "failed",
        "scope": "real FPGA exact correctness only; no timing or FPS",
        "records": len(rows),
        "seed_checks": sum(len(row["seeds"]) for row in rows),
        "passed_seed_checks": sum(seed["correct"] for row in rows for seed in row["seeds"]),
        "elapsed_seconds": time.monotonic() - start,
        "board_before": before,
        "board_after": after,
        "queue_capacity_status": queue_capacity_status,
    }
    write_json(output / "summary.json", summary)
    (output / "command.txt").write_text(" ".join([os.path.realpath(__file__)] + os.sys.argv) + "\n")
    (output / "STATUS.md").write_text(
        "# P7R W04 board correctness\n\n"
        "- Status: `{}`\n"
        "- Records: {}\n"
        "- Exact seed checks: {}/{}\n"
        "- Performance timing: not collected\n".format(
            summary["status"],
            summary["records"],
            summary["passed_seed_checks"],
            summary["seed_checks"],
        )
    )
    hashes = {
        path.name: sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    write_json(output / "artifact_hashes.json", {"artifacts": hashes})


if __name__ == "__main__":
    main()
