#!/usr/bin/env python3
"""Run one immutable P7 grouped-holdout correctness-only board batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import tvm
from tvm import autotvm, rpc
from tvm.contrib import cc
import vta

import vta.top.vta_conv2d_residency  # pylint: disable=unused-import
from c3_candidate_identity import canonical_json_bytes, normalize_config_entity
from qualify_vta_residency_fsim import array_sha256, reference_data_from_workload


CONTRACT_SCHEMA = "c3_p7_grouped_holdout_contract_v1"
RESULT_SCHEMA = "c3_p7_holdout_correctness_candidate_v1"
MODE_NUMBERS = {
    "original": 0,
    "input_stationary": 1,
    "weight_stationary": 2,
    "paper_inspired_hybrid": 3,
    "weight_stationary_barrier": 4,
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_contract(path):
    path = Path(path).resolve()
    ledger = json.loads((path.parent / "artifact_hashes.json").read_text())
    expected = ledger["output_sha256"].get(path.name)
    if expected != file_sha256(path):
        raise ValueError("P7 contract hash mismatch")
    contract = json.loads(path.read_text())
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise ValueError("unexpected P7 contract schema")
    if contract.get("status") != "frozen_before_any_P7_board_label":
        raise ValueError("P7 contract is not an immutable pre-label contract")
    if contract["pool"].get("candidate_count") != 80:
        raise ValueError("P7 common pool is incomplete")
    return contract


def semantic_config_key(value):
    return canonical_json_bytes(normalize_config_entity(value)).decode("utf-8")


def validate_source_guards(entry):
    observed = {}
    for name, expected in sorted(entry["source_guards_sha256"].items()):
        path = Path(name)
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError("source guard changed or missing: {}".format(path))
        observed[str(path)] = expected
    return observed


def cross_options():
    sysroot = os.environ.get("SDKTARGETSYSROOT")
    if not sysroot:
        raise RuntimeError("SDKTARGETSYSROOT is unset; source the AXU SDK environment")
    return [
        "--sysroot=" + sysroot,
        "-Wl,-rpath-link," + sysroot + "/lib",
        "-Wl,-rpath-link," + sysroot + "/usr/lib",
        "-L" + sysroot + "/lib",
        "-L" + sysroot + "/usr/lib",
    ]


def build_candidate(entry, env, output):
    validate_source_guards(entry)
    mode = entry["residence_mode"]
    task = autotvm.task.create(
        "conv2d_packed_residency.vta",
        args=tuple(entry["workload"][1:]) + (MODE_NUMBERS[mode],),
        target=env.target,
        target_host=env.target_host,
    )
    config = task.config_space.get(int(entry["config_index"]))
    if semantic_config_key(config.to_json_dict()) != semantic_config_key(
        entry["complete_config_entity"]
    ):
        raise RuntimeError("ConfigEntity mismatch")
    with task.target:
        schedule, tensors = task.instantiate(config)
    with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
        lowered = tvm.lower(schedule, tensors, name="main")
    tir_hash = hashlib.sha256(tvm.ir.save_json(lowered).encode()).hexdigest()
    if tir_hash != entry["tir_sha256"]:
        raise RuntimeError("TIR certificate mismatch")
    with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
        module = vta.build(
            schedule,
            tensors,
            target=tvm.target.Target(env.target, host=env.target_host),
            name="main",
        )
    binary = output / (entry["candidate_id"] + ".so")
    module.export_library(
        str(binary),
        fcompile=cc.cross_compiler("aarch64-xilinx-linux-g++"),
        options=cross_options(),
    )
    return binary, tir_hash


def ssh_preflight(contract, host):
    board = contract["board_contract"]
    cutoff = float(board["storage_error_dmesg_after_seconds"])
    command = r'''
boot=$(cat /proc/sys/kernel/random/boot_id)
state=$(cat /sys/class/fpga_manager/fpga0/state)
size=$(cat /sys/class/u-dma-buf/udmabuf0/size)
rpcpid=$(pidof tvm_rpc | awk '{print $1}')
rpccwd=$(readlink /proc/$rpcpid/cwd)
mountline=$(mount | grep 'mmcblk1p2 ')
dfline=$(df -Pk /media/sd-mmcblk1p2 | tail -n 1)
echo BOOT=$boot
echo FPGA=$state
echo SIZE=$size
echo RPC=$rpcpid
echo RPC_CWD=$rpccwd
echo MOUNT=$mountline
echo DF=$dfline
dmesg | awk '$1 ~ /^\[/ {t=$1; gsub(/\[/,"",t); if ((t+0)>%s && ($0 ~ /EXT4-fs error|mmcblk.*error|I.O error|Buffer I.O/)) print "STORAGE_ERROR=" $0}'
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
    if fields.get("BOOT") != board["boot_id"]:
        raise RuntimeError("board boot ID changed")
    if fields.get("FPGA") != board["fpga_state"]:
        raise RuntimeError("FPGA is not operating")
    if fields.get("SIZE") != str(board["udmabuf_bytes"]):
        raise RuntimeError("u-dma-buf size changed")
    if not fields.get("RPC") or not fields.get("RPC_CWD", "").startswith(board["rpc_cwd_prefix"]):
        raise RuntimeError("RPC is not running from tmpfs")
    if "(rw," not in fields.get("MOUNT", ""):
        raise RuntimeError("repaired ext4 partition is not mounted read-write")
    df_fields = fields.get("DF", "").split()
    if len(df_fields) < 5:
        raise RuntimeError("unable to parse ext4 free-space status")
    used_percent = int(df_fields[4].rstrip("%"))
    if 100 - used_percent < int(board["sd_p2_min_free_percent"]):
        raise RuntimeError("ext4 free-space reserve fell below contract")
    if "STORAGE_ERROR" in fields:
        raise RuntimeError("new storage error after repair: " + fields["STORAGE_ERROR"])
    fields["SD_USED_PERCENT"] = used_percent
    return fields


def validate_workload(contract, workload_id):
    if workload_id not in contract["pool"]["holdouts"]:
        raise ValueError("not a frozen grouped holdout")
    workload = contract["workloads"][workload_id]
    entries = {entry["candidate_id"]: entry for entry in workload["candidates"]}
    order = workload["correctness_order"]
    if len(order) != workload["candidate_count"] or set(order) != set(entries):
        raise ValueError("correctness order does not cover the complete workload pool")
    for candidate_id, entry in entries.items():
        if entry["candidate_id"] != candidate_id or entry["workload_id"] != workload_id:
            raise ValueError("candidate identity binding mismatch")
        if entry["residence_mode"] not in MODE_NUMBERS:
            raise ValueError("unsupported frozen residence mode")
    return entries, order


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "correctness.jsonl"
    if results_path.exists() or (output / "failure.json").exists() or (output / "summary.json").exists():
        raise FileExistsError("refusing to overwrite P7 board observations")

    contract = load_contract(args.contract)
    entries, order = validate_workload(contract, args.workload)
    seeds = [int(seed) for seed in contract["measurement"]["correctness_seeds"]]
    env = vta.get_env()
    if env.TARGET != "axu5evb":
        raise RuntimeError("VTA TARGET must be axu5evb")
    before = ssh_preflight(contract, args.host)
    remote = rpc.connect(args.host, args.port, session_timeout=120)
    device = remote.ext_dev(0)
    rows = []
    batch_start = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="c3_p7_correctness_") as temporary:
            binary_dir = Path(temporary)
            with results_path.open("x", encoding="utf-8") as stream:
                for position, candidate_id in enumerate(order, 1):
                    entry = entries[candidate_id]
                    preflight = ssh_preflight(contract, args.host)
                    started = time.monotonic()
                    build_start = time.monotonic()
                    binary, tir_hash = build_candidate(entry, env, binary_dir)
                    build_seconds = time.monotonic() - build_start
                    upload_start = time.monotonic()
                    remote.upload(str(binary))
                    module = remote.load_module(binary.name)
                    upload_seconds = time.monotonic() - upload_start
                    seed_rows = []
                    for seed in seeds:
                        data, weight, expected = reference_data_from_workload(entry["workload"], seed)
                        buffers = [
                            tvm.nd.array(data, device),
                            tvm.nd.array(weight, device),
                            tvm.nd.empty(expected.shape, "int8", device),
                        ]
                        module["main"](*buffers)
                        actual = buffers[-1].numpy()
                        mismatch_count = int(np.count_nonzero(actual != expected))
                        seed_rows.append(
                            {
                                "seed": seed,
                                "correct": mismatch_count == 0,
                                "mismatch_count": mismatch_count,
                                "expected_sha256": array_sha256(expected),
                                "actual_sha256": array_sha256(actual),
                            }
                        )
                        if mismatch_count:
                            raise AssertionError("wrong answer: {} seed {}".format(candidate_id, seed))
                    row = {
                        "schema": RESULT_SCHEMA,
                        "workload_id": args.workload,
                        "correctness_position": position,
                        "candidate_id": candidate_id,
                        "residence_mode": entry["residence_mode"],
                        "config_index": entry["config_index"],
                        "tir_sha256": tir_hash,
                        "build_seconds": build_seconds,
                        "upload_load_seconds": upload_seconds,
                        "candidate_wall_seconds": time.monotonic() - started,
                        "preflight": preflight,
                        "seeds": seed_rows,
                        "overall_status": "passed",
                        "performance_measurement": "not_collected",
                    }
                    rows.append(row)
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                    stream.flush()
                    print(
                        "{}/{} {} mode={} config={} exact=3/3".format(
                            position,
                            len(order),
                            candidate_id[:12],
                            entry["residence_mode"],
                            entry["config_index"],
                        ),
                        flush=True,
                    )
    except Exception as error:
        write_json(
            output / "failure.json",
            {
                "schema": "c3_p7_holdout_correctness_failure_v1",
                "workload_id": args.workload,
                "completed_candidates": len(rows),
                "status": "failed_stop_batch",
                "exception_type": type(error).__name__,
                "message": str(error)[:4000],
                "traceback": traceback.format_exc()[-12000:],
            },
        )
        raise
    after = ssh_preflight(contract, args.host)
    summary = {
        "schema": "c3_p7_holdout_correctness_summary_v1",
        "status": "passed",
        "workload_id": args.workload,
        "candidate_count": len(rows),
        "seed_checks": len(rows) * len(seeds),
        "failed_candidates": 0,
        "performance_measurement": "not_collected",
        "batch_wall_seconds": time.monotonic() - batch_start,
        "board_before": before,
        "board_after": after,
        "contract_sha256": file_sha256(args.contract),
        "correctness_sha256": file_sha256(results_path),
        "next_step": "next frozen workload correctness batch; timing forbidden until all 80 pass",
    }
    write_json(output / "summary.json", summary)
    print("P7 {} correctness passed: {}/{} candidates, {} exact seed checks".format(
        args.workload, len(rows), len(order), summary["seed_checks"]
    ))


if __name__ == "__main__":
    main()
