#!/usr/bin/env python3
"""Validate a compile-derived selected/fallback command allowlist on AXU5EVB.

The positive phase runs the Y00/Y03 selected identities and the Y00 same-tile
original fallback with page-aligned capacities derived from their frozen command
peaks.  The negative phase removes one instruction page and requires the largest
identity to fail before a device submission.  The default tmpfs RPC is restored
and health-checked even when either phase fails.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import tvm
from tvm import rpc
import vta

import vta.top.vta_conv2d_residency  # pylint: disable=unused-import
from prepare_vta_p7r120_yolo_rpc_contract import build_and_export
from run_vta_p7r120_yolo_rpc_board import load_ephemeral, runtime_inventory
from run_vta_p7r132_y00_search_confirmation import (
    allocate_reusable_buffers,
    profiled_reused_call,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE / "report_out" / "stage_tile_cotuning" / "c3_dma_residency_autotune"
P7 = ROOT / "07_grouped_holdout"
DEFAULT_Y00_CONTRACT = P7 / "20260912_p7r132_y00_search_confirmation_contract_run02"
DEFAULT_Y00_BOARD = P7 / "20260912_p7r144_y00_layout_preserving_search_board_run01"
DEFAULT_Y03_CONTRACT = P7 / "20260912_p7r152_y03_search_confirmation_contract_run01"
DEFAULT_Y03_BOARD = P7 / "20260912_p7r154_y03_layout_preserving_search_board_run01"
DEFAULT_OUTPUT = P7 / "20260912_p7r155_selected_allowlist_capacity_run02"
SEEDS = (0, 20250901, 20260910)
PAGE_BYTES = 4096
EXPECTED_DEFAULT_RUNTIME_HASHES = {
    "tvm_rpc": "20bbd3d896d124dd1127343f020c957fbb6034423af83433c2429955b6b316fa",
    "libtvm_runtime.so": "04e894baf305311315fd0c79d03ec2a6375b9591916e9c6f429cf3697c4457b2",
    "libvta.so": "eedfabb0630d58bf2eabaef9b4f2d2e4bfcdfa52a70503090f5608e79404ee5d",
    "start_axu5evb_cpp_rpc.sh": "2f42f2d37d5199fa2b66336b61869d2015f8ec5f0264bc57307789b4c6a5b194",
}
EXPECTED_CAPACITY_RUNTIME_HASHES = {
    **EXPECTED_DEFAULT_RUNTIME_HASHES,
    "libvta.so": "0c39d5f6fcfb1efec6bad715e3cea2279282e3bf887433cd2013b89fff1f4ba4",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify_run(directory):
    directory = Path(directory)
    ledger = read_json(directory / "artifact_hashes.json")
    entries = ledger.get("artifacts", ledger.get("files"))
    if not isinstance(entries, dict):
        raise ValueError("unknown artifact ledger schema: {}".format(directory))
    for name, expected in entries.items():
        path = directory / name
        if sha256_file(path) != expected:
            raise ValueError("artifact hash mismatch: {}".format(path))
    return sha256_file(directory / "artifact_hashes.json")


def align_up(value, alignment=PAGE_BYTES):
    return (int(value) + alignment - 1) // alignment * alignment


def shell_safe(value):
    value = str(value)
    if not re.fullmatch(r"[A-Za-z0-9_./:=+-]+", value):
        raise ValueError("unsafe shell token")
    return value


class Board:
    def __init__(self, host, known_hosts):
        self.host = host
        self.known_hosts = str(Path(known_hosts).resolve())
        self.options = [
            "-o", "ConnectTimeout=8",
            "-o", "HostKeyAlgorithms=+ssh-rsa",
            "-o", "PubkeyAcceptedAlgorithms=+ssh-rsa",
            "-o", "UserKnownHostsFile=" + self.known_hosts,
            "-o", "StrictHostKeyChecking=yes",
            "-o", "PreferredAuthentications=password",
            "-o", "PubkeyAuthentication=no",
            "-o", "NumberOfPasswordPrompts=1",
        ]

    def run(self, command, timeout=30, check=True):
        return subprocess.run(
            ["ssh", *self.options, "root@" + self.host, command],
            check=check,
            text=True,
            capture_output=True,
            timeout=timeout,
            env=os.environ.copy(),
        )

    def state(self):
        command = r'''
echo BOOT=$(cat /proc/sys/kernel/random/boot_id)
echo FPGA=$(cat /sys/class/fpga_manager/fpga0/state)
echo UDMABUF=$(cat /sys/class/u-dma-buf/udmabuf0/size)
echo UDMABUF_PHYS=$(cat /sys/class/u-dma-buf/udmabuf0/phys_addr)
echo UDMABUF_SYNC=$(cat /sys/class/u-dma-buf/udmabuf0/sync_mode)
pid=$(pidof tvm_rpc | cut -d' ' -f1)
echo RPC_PID=$pid
echo RPC_CWD=$(readlink /proc/$pid/cwd)
echo SD_MOUNT=$(mount | grep 'mmcblk1p2 ')
echo SD_DF=$(df -Pk /media/sd-mmcblk1p2 | tail -n 1)
dmesg | grep -Ei 'EXT4-fs.*(error|warning)|mmc.*(error|timeout|I/O)' | sed 's/^/STORAGE_ERROR=/'
'''
        fields = {}
        errors = []
        for line in self.run(command).stdout.splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key == "STORAGE_ERROR":
                errors.append(value)
            else:
                fields[key] = value
        fields["STORAGE_ERRORS"] = errors
        return fields

    def runtime_hashes(self, directory):
        directory = shell_safe(directory.rstrip("/"))
        names = tuple(EXPECTED_DEFAULT_RUNTIME_HASHES)
        stdout = self.run(
            "sha256sum " + " ".join(directory + "/" + name for name in names)
        ).stdout
        return {
            Path(line.split(None, 1)[1]).name: line.split(None, 1)[0]
            for line in stdout.splitlines()
        }

    def rpc_state(self):
        command = r'''
pid=$(pidof tvm_rpc | cut -d' ' -f1)
echo PID=$pid
echo CWD=$(readlink /proc/$pid/cwd)
tr '\0' '\n' < /proc/$pid/environ | grep -E '^(VTA_INSN_BUFFER_BYTES|VTA_UOP_BUFFER_BYTES|VTA_QUEUE_DIAGNOSTICS|VTA_COMMAND_MANIFEST_ID|VTA_REPLAY_POLICY)=' || true
'''
        fields = {}
        for line in self.run(command).stdout.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                fields[key] = value
        return fields

    def stop_rpc(self, expected):
        current = self.rpc_state()
        if current.get("PID") != expected.get("PID") or current.get("CWD") != expected.get("CWD"):
            raise RuntimeError("refusing to stop changed RPC: {} expected {}".format(current, expected))
        pid = int(current["PID"])
        command = (
            "kill {0}; i=0; while kill -0 {0} 2>/dev/null; do "
            "i=$((i+1)); [ $i -lt 50 ] || exit 9; "
            "usleep 100000 2>/dev/null || sleep 1; done"
        ).format(pid)
        self.run(command, timeout=60)

    def start_rpc(self, directory, log_name, environment=None):
        directory = shell_safe(directory.rstrip("/"))
        log_path = directory + "/" + shell_safe(log_name)
        words = []
        for name, value in sorted((environment or {}).items()):
            if not re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
                raise ValueError("unsafe environment name")
            words.append(name + "=" + shell_safe(value))
        env_part = "env " + " ".join(words) if words else "env"
        command = (
            "cd {d}; : > {log}; nohup {env} ./start_axu5evb_cpp_rpc.sh {d} 9090 "
            "> {log} 2>&1 < /dev/null & echo $!"
        ).format(d=directory, log=log_path, env=env_part)
        pid = int(self.run(command).stdout.strip().splitlines()[-1])
        for _ in range(40):
            time.sleep(0.1)
            state = self.rpc_state()
            if state.get("PID") == str(pid) and state.get("CWD") == directory:
                return state
        log = self.run("tail -n 80 " + log_path, check=False).stdout
        raise RuntimeError("RPC failed to start: pid={} log={}".format(pid, log))


def extract_allowlist(contract_dir, board_dir):
    verify_run(contract_dir)
    verify_run(board_dir)
    contract = read_json(Path(contract_dir) / "board_collection_contract.json")
    summary = read_json(Path(board_dir) / "summary.json")
    completed = read_json(Path(board_dir) / "completed_pool.json")
    workload_id = contract["workload_id"]
    workload_pool = completed["workloads"][workload_id]
    by_pool = {row["candidate_id"]: row for row in workload_pool["candidates"]}
    by_contract = {row["candidate_id"]: row for row in contract["candidates"]}
    selected_id = summary["pool_oracle_candidate_id"]
    if selected_id not in by_pool or selected_id not in by_contract:
        raise ValueError("selected identity is absent from frozen pool")
    selected = by_pool[selected_id]
    ids = [(selected_id, "selected")]
    fallback_id = selected.get("same_tile_control_id")
    if fallback_id and fallback_id != selected_id:
        ids.append((fallback_id, "same_tile_original_fallback"))
    rows = []
    for candidate_id, role in ids:
        pool_row = by_pool[candidate_id]
        if candidate_id not in by_contract:
            raise ValueError("allowlist identity absent from source contract")
        if int(pool_row["oracle"]["correctness_seed_checks"]) != 3:
            raise ValueError("allowlist identity lacks three FPGA correctness seeds")
        validity = pool_row["predispatch"]["validity"]
        entry = dict(by_contract[candidate_id])
        entry["allowlist_role"] = role
        entry["frozen_tir_sha256"] = contract["qualification_bindings"][candidate_id]["tir_sha256"]
        entry["command_peak"] = {
            "insn_bytes": int(validity["static_insn_peak_bytes"]),
            "uop_bytes": int(validity["static_uop_peak_bytes"]),
            "submissions_per_call": int(validity["static_submissions"]),
        }
        entry["prior_board_evidence"] = {
            "correctness_seed_checks": 3,
            "latency_ms": float(pool_row["oracle"]["latency_ms"]),
            "board_run": str(Path(board_dir).resolve()),
        }
        rows.append(entry)
    return workload_id, contract["health_canary"], rows


def build_all(entries, health, directory):
    env = vta.get_env()
    if env.TARGET != "axu5evb":
        raise RuntimeError("VTA target must be axu5evb")
    certificates = {"health": build_and_export(health, env, directory)}
    for entry in entries:
        certificates[entry["candidate_id"]] = build_and_export(
            entry, env, directory, entry["frozen_tir_sha256"]
        )
    return certificates


def run_entries(
        host, port, directory, entries, seeds, require_queue_status=False,
        preallocate_bytes=0, module_entries=None):
    remote = rpc.connect(host, port, session_timeout=120)
    inventory, functions = runtime_inventory(remote)
    if require_queue_status and "vta.runtime.queue_capacity_status" not in functions:
        raise RuntimeError("queue capacity status RPC is unavailable")
    device = remote.ext_dev(0)
    # Queue backing is allocated before tensor buffers.  A diagnostic padding
    # buffer can therefore preserve the tensor physical-address layout while a
    # queue-capacity experiment deliberately removes backing bytes.
    layout_padding = None
    if preallocate_bytes:
        layout_padding = tvm.nd.empty((int(preallocate_bytes),), "uint8", device)
    modules = {
        entry["candidate_id"]: load_ephemeral(
            remote, Path(directory) / (entry["candidate_id"] + ".so")
        )
        for entry in (module_entries if module_entries is not None else entries)
    }
    buffers = {}
    rows = []
    for entry in entries:
        workload_id = entry.get("workload_id", entry["identity"].get("workload_id"))
        if workload_id not in buffers:
            buffers[workload_id] = allocate_reusable_buffers(
                device, entry["identity"]["workload"]
            )
        for seed in seeds(entry):
            row = profiled_reused_call(
                modules[entry["candidate_id"]]["main"],
                buffers[workload_id], entry["identity"]["workload"], seed, functions,
            )
            row.update({
                "candidate_id": entry["candidate_id"],
                "workload_id": workload_id,
                "role": entry.get("allowlist_role", entry.get("role")),
                "command_peak": entry.get("command_peak"),
            })
            rows.append(row)
            if not row.get("correct"):
                break
    queue_status = None
    if "vta.runtime.queue_capacity_status" in functions:
        queue_status = json.loads(functions["vta.runtime.queue_capacity_status"]())
    replay_status = None
    if "vta.runtime.replay_status" in functions:
        replay_status = json.loads(functions["vta.runtime.replay_status"]())
    del modules, buffers, layout_padding, functions, device, remote
    gc.collect()
    return {
        "inventory": inventory,
        "rows": rows,
        "queue_status": queue_status,
        "replay_status": replay_status,
    }


def one_seed(_entry):
    return (0,)


def three_seeds(_entry):
    return SEEDS


def check_health(result):
    rows = result["rows"]
    if len(rows) != 3 or not all(row.get("correct") for row in rows):
        raise RuntimeError("W05 health gate failed")


def capacity_environment(capacity, manifest_id):
    return {
        "VTA_INSN_BUFFER_BYTES": str(capacity["insn_bytes"]),
        "VTA_UOP_BUFFER_BYTES": str(capacity["uop_bytes"]),
        "VTA_QUEUE_DIAGNOSTICS": "1",
        "VTA_COMMAND_MANIFEST_ID": manifest_id,
        "VTA_REPLAY_POLICY": "disabled",
    }


def validate_environment(state, expected):
    for name, value in expected.items():
        if state.get(name) != value:
            raise RuntimeError("RPC environment mismatch for {}".format(name))


def finalize(output):
    artifacts = {
        str(path.relative_to(output)): sha256_file(path)
        for path in output.rglob("*")
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    write_json(output / "artifact_hashes.json", {"artifacts": artifacts})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--y00-contract", type=Path, default=DEFAULT_Y00_CONTRACT)
    parser.add_argument("--y00-board", type=Path, default=DEFAULT_Y00_BOARD)
    parser.add_argument("--y03-contract", type=Path, default=DEFAULT_Y03_CONTRACT)
    parser.add_argument("--y03-board", type=Path, default=DEFAULT_Y03_BOARD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--ssh-known-hosts", type=Path, required=True)
    parser.add_argument("--default-runtime", default="/var/volatile/vta_c3_ram/runtime")
    parser.add_argument("--capacity-runtime", default="/var/volatile/vta_c3_ram/capacity_runtime")
    args = parser.parse_args()

    output = args.output_dir
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))
    output.mkdir(parents=True)
    board = Board(args.host, args.ssh_known_hosts)
    original_rpc = None
    active_rpc = None
    restored_rpc = None
    error = None
    positive = None
    negative = None
    health_before = None
    health_after = None
    try:
        y00_id, health, y00 = extract_allowlist(args.y00_contract, args.y00_board)
        y03_id, y03_health, y03 = extract_allowlist(args.y03_contract, args.y03_board)
        if y00_id != "Y00" or y03_id != "Y03" or y03_health["candidate_id"] != health["candidate_id"]:
            raise ValueError("unexpected workload or health identity")
        entries = y00 + y03
        if len(entries) != 3 or len({row["candidate_id"] for row in entries}) != 3:
            raise ValueError("expected three unique selected/fallback identities")
        positive_capacity = {
            "insn_bytes": align_up(max(row["command_peak"]["insn_bytes"] for row in entries)),
            "uop_bytes": align_up(max(row["command_peak"]["uop_bytes"] for row in entries)),
        }
        largest = max(entries, key=lambda row: row["command_peak"]["insn_bytes"])
        negative_capacity = dict(positive_capacity)
        negative_capacity["insn_bytes"] -= PAGE_BYTES
        if not (
            negative_capacity["insn_bytes"] < largest["command_peak"]["insn_bytes"]
            <= positive_capacity["insn_bytes"]
        ):
            raise ValueError("one-page negative control does not straddle the largest identity")

        before = board.state()
        if before.get("FPGA") != "operating" or before.get("UDMABUF") != "201326592":
            raise RuntimeError("FPGA/u-dma-buf preflight failed")
        if before.get("STORAGE_ERRORS"):
            raise RuntimeError("new boot has EXT4/mmc errors: {}".format(before["STORAGE_ERRORS"]))
        default_hashes = board.runtime_hashes(args.default_runtime)
        capacity_hashes = board.runtime_hashes(args.capacity_runtime)
        if default_hashes != EXPECTED_DEFAULT_RUNTIME_HASHES:
            raise RuntimeError("default runtime hash mismatch")
        if capacity_hashes != EXPECTED_CAPACITY_RUNTIME_HASHES:
            raise RuntimeError("capacity runtime hash mismatch")
        original_rpc = board.rpc_state()
        if original_rpc.get("CWD") != args.default_runtime:
            raise RuntimeError("default RPC is not in the frozen tmpfs directory")

        contract = {
            "schema": "c3_p7r155_selected_allowlist_capacity_contract_v1",
            "status": "frozen_before_p7r155_fpga_execution",
            "board": before,
            "host_key_sha256": subprocess.run(
                ["ssh-keygen", "-lf", str(args.ssh_known_hosts)], check=True,
                text=True, capture_output=True,
            ).stdout.strip(),
            "source_runs": {
                "Y00_contract": str(args.y00_contract.resolve()),
                "Y00_board": str(args.y00_board.resolve()),
                "Y03_contract": str(args.y03_contract.resolve()),
                "Y03_board": str(args.y03_board.resolve()),
            },
            "health_canary": health,
            "allowlist": entries,
            "capacity_derivation": {
                "page_bytes": PAGE_BYTES,
                "rule": "page_align(max exact frozen peak across selected plus qualified fallback)",
                "positive": positive_capacity,
                "negative_one_insn_page": negative_capacity,
                "negative_target_candidate_id": largest["candidate_id"],
            },
            "runtime_hashes": {
                "default": EXPECTED_DEFAULT_RUNTIME_HASHES,
                "capacity": EXPECTED_CAPACITY_RUNTIME_HASHES,
            },
            "runtime_directories": {
                "default": args.default_runtime,
                "capacity": args.capacity_runtime,
            },
            "seeds": list(SEEDS),
            "failure_policy": "always restore default RPC; negative target must fail before device submission",
            "performance_measurement": "not_collected",
            "persistent_board_writes": False,
        }
        write_json(output / "contract.json", contract)
        write_json(output / "pre_execution_hashes.json", {
            "contract.json": sha256_file(output / "contract.json")
        })

        with tempfile.TemporaryDirectory(prefix="c3_p7r155_cross_") as temporary:
            directory = Path(temporary)
            certificates = build_all(entries, health, directory)
            write_json(output / "cross_compile_certificates.json", certificates)

            health_entry = dict(health)
            health_entry["allowlist_role"] = "health_canary"
            health_before = run_entries(
                args.host, args.port, directory, [health_entry], three_seeds
            )
            check_health(health_before)
            write_json(output / "health_before.json", health_before)

            board.stop_rpc(original_rpc)
            positive_manifest = canonical_sha256({"allowlist": [row["candidate_id"] for row in entries],
                                                   "capacity": positive_capacity})
            positive_env = capacity_environment(positive_capacity, positive_manifest)
            active_rpc = board.start_rpc(args.capacity_runtime, "p7r155_positive.log", positive_env)
            validate_environment(active_rpc, positive_env)
            positive = run_entries(
                args.host, args.port, directory, entries, three_seeds,
                require_queue_status=True,
            )
            if len(positive["rows"]) != 9 or not all(row.get("correct") for row in positive["rows"]):
                raise RuntimeError("positive capacity did not pass 9/9 exact checks")
            status = positive["queue_status"]
            if (
                int(status["insn_capacity_bytes"]) != positive_capacity["insn_bytes"]
                or int(status["uop_capacity_bytes"]) != positive_capacity["uop_bytes"]
                or int(status["insn_peak_bytes"]) != max(row["command_peak"]["insn_bytes"] for row in entries)
                or int(status["uop_peak_bytes"]) != max(row["command_peak"]["uop_bytes"] for row in entries)
                or int(status["submissions"]) != 9
                or status.get("command_manifest_id") != positive_manifest
                or status.get("replay_policy") != "disabled"
            ):
                raise RuntimeError("positive queue status differs from derived allowlist")
            write_json(output / "positive_capacity.json", positive)
            board.stop_rpc(active_rpc)
            active_rpc = None

            negative_manifest = canonical_sha256({"negative_target": largest["candidate_id"],
                                                   "capacity": negative_capacity})
            negative_env = capacity_environment(negative_capacity, negative_manifest)
            active_rpc = board.start_rpc(args.capacity_runtime, "p7r155_negative.log", negative_env)
            validate_environment(active_rpc, negative_env)
            smaller = next(row for row in entries if row["command_peak"]["insn_bytes"] <= negative_capacity["insn_bytes"])
            negative_order = [smaller, largest]
            negative = run_entries(
                args.host, args.port, directory, negative_order, one_seed,
                require_queue_status=True,
            )
            if len(negative["rows"]) != 2 or not negative["rows"][0].get("correct"):
                raise RuntimeError("negative control did not first prove an in-capacity identity")
            rejected = negative["rows"][1]
            message = rejected.get("message", "")
            profile = rejected.get("runtime_profile_complete") or {}
            if (
                rejected.get("status") != "execution_failed"
                or "queue backing capacity exceeded before submission" not in message
                or int(profile.get("driver_run_calls", -1)) != 0
                or int(negative["queue_status"].get("submissions", -1)) != 1
            ):
                raise RuntimeError("undersized capacity was not rejected before device submission")
            write_json(output / "negative_capacity.json", negative)
            board.stop_rpc(active_rpc)
            active_rpc = None

            restored_rpc = board.start_rpc(args.default_runtime, "p7r155_restored.log", None)
            forbidden = set(capacity_environment(positive_capacity, positive_manifest))
            if forbidden.intersection(restored_rpc):
                raise RuntimeError("restored default RPC retained capacity environment")
            health_after = run_entries(
                args.host, args.port, directory, [health_entry], three_seeds
            )
            check_health(health_after)
            write_json(output / "health_after.json", health_after)

        after = board.state()
        if after.get("BOOT") != before.get("BOOT") or after.get("STORAGE_ERRORS"):
            raise RuntimeError("postflight board/storage state changed")
        summary = {
            "schema": "c3_p7r155_selected_allowlist_capacity_summary_v1",
            "status": "passed",
            "boot_id": before["BOOT"],
            "allowlist_identities": len(entries),
            "allowlist": [
                {
                    "candidate_id": row["candidate_id"],
                    "workload_id": row["workload_id"],
                    "role": row["allowlist_role"],
                    "insn_peak_bytes": row["command_peak"]["insn_bytes"],
                    "uop_peak_bytes": row["command_peak"]["uop_bytes"],
                }
                for row in entries
            ],
            "derived_capacity": positive_capacity,
            "positive_exact_correctness": "9/9",
            "positive_observed_queue_status": positive["queue_status"],
            "undersized_capacity": negative_capacity,
            "negative_in_capacity_control": "passed",
            "negative_target": largest["candidate_id"],
            "negative_rejection": "queue backing capacity exceeded before submission",
            "negative_target_driver_run_calls": 0,
            "negative_observed_submissions": 1,
            "health_before": "3/3",
            "health_after_restore": "3/3",
            "default_rpc_restored": True,
            "performance_measurement": "not_collected",
            "sd_experiment_writes": "none; runtime/log/upload paths are tmpfs",
            "claim_boundary": "three exact Y00/Y03 selected/fallback identities on one boot",
        }
        write_json(output / "board_postflight.json", after)
        write_json(output / "summary.json", summary)
        (output / "STATUS.md").write_text(
            "# P7R155 selected-allowlist capacity validation\n\n"
            "- Status: `passed`\n"
            "- Derived capacity: instruction 16 KiB + UOP 4 KiB\n"
            "- Positive phase: 3 identities x 3 seeds = 9/9 correct\n"
            "- Negative phase: 12 KiB instruction accepts a smaller identity, then rejects the "
            "13,488 B fallback before device submission\n"
            "- Default tmpfs RPC restored and W05 health is 3/3 after restoration\n"
            "- No latency measurement and no persistent SD experiment write\n",
            encoding="utf-8",
        )
    except Exception as caught:  # Always leave a reviewable fail-closed artifact.
        error = caught
        write_json(output / "failure.json", {
            "status": "failed_closed",
            "exception_type": type(caught).__name__,
            "message": (str(caught) or type(caught).__name__)[:4000],
            "traceback": traceback.format_exc()[-16000:],
        })
    finally:
        try:
            if active_rpc is not None:
                current = board.rpc_state()
                if current.get("PID") == active_rpc.get("PID"):
                    board.stop_rpc(active_rpc)
            current = board.rpc_state()
            if current.get("CWD") != args.default_runtime:
                if current.get("PID") and current.get("CWD"):
                    board.stop_rpc(current)
                restored_rpc = board.start_rpc(args.default_runtime, "p7r155_emergency_restore.log", None)
        except Exception as restore_error:
            write_json(output / "restoration_failure.json", {
                "exception_type": type(restore_error).__name__,
                "message": (str(restore_error) or type(restore_error).__name__)[:4000],
            })
            if error is None:
                error = restore_error
        finalize(output)
    if error is not None:
        raise error
    print(json.dumps(read_json(output / "summary.json"), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
