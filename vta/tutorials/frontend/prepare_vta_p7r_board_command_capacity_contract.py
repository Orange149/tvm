#!/usr/bin/env python3
"""Freeze the real-FPGA W05 4 KiB+4 KiB command-capacity contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def verify_frozen_file(path):
    path = Path(path).resolve()
    ledger = load_json(path.parent / "artifact_hashes.json")
    hashes = ledger.get("artifacts", ledger.get("output_sha256", ledger))
    observed = sha256_file(path)
    if hashes.get(path.name) != observed:
        raise ValueError("frozen artifact hash mismatch: {}".format(path))
    return {"path": str(path), "sha256": observed}


def board_probe(host, cutoff):
    def ssh_run(command):
        return subprocess.run(
            [
                "ssh", "-o", "ConnectTimeout=5", "-o", "HostKeyAlgorithms=+ssh-rsa",
                "-o", "PubkeyAcceptedAlgorithms=+ssh-rsa", "root@" + host, command,
            ],
            check=True,
            text=True,
            capture_output=True,
            timeout=20,
        ).stdout

    core_command = r'''
echo BOOT=$(cat /proc/sys/kernel/random/boot_id)
echo FPGA=$(cat /sys/class/fpga_manager/fpga0/state)
echo UDMABUF=$(cat /sys/class/u-dma-buf/udmabuf0/size)
rpcpid=$(pidof tvm_rpc | awk '{print $1}')
echo RPC=$rpcpid
echo RPC_CWD=$(readlink /proc/$rpcpid/cwd)
grep ' /media/sd-mmcblk1p2 ' /proc/mounts | sed 's/^/MOUNT=/'
df -Pk /media/sd-mmcblk1p2 | tail -n 1 | sed 's/^/DF=/'
'''
    hash_command = "sha256sum " + " ".join(
        "/var/volatile/vta_c3_ram/runtime/" + name
        for name in ("libvta.so", "libtvm_runtime.so", "tvm_rpc", "start_axu5evb_cpp_rpc.sh")
    )
    error_command = (
        "dmesg | awk '$1 ~ /^\\[/ {t=$1; gsub(/\\[/,\"\",t); "
        "if ((t+0)>%s && ($0 ~ /EXT4-fs error|mmcblk.*error|I.O error|Buffer I.O/)) "
        "print \"NEW_STORAGE_ERROR=\" $0}'"
    ) % cutoff
    stdout = ssh_run(core_command)
    hash_stdout = ssh_run(hash_command)
    error_stdout = ssh_run(error_command)
    fields = {}
    remote_hashes = {}
    for line in stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key] = value
    for line in hash_stdout.splitlines():
        digest, name = line.split(None, 1)
        remote_hashes[name] = digest
    for line in error_stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key] = value
    fields["REMOTE_SHA256"] = remote_hashes
    return fields


def hash_remote_runtime(host, runtime):
    names = ("libvta.so", "libtvm_runtime.so", "tvm_rpc", "start_axu5evb_cpp_rpc.sh")
    completed = subprocess.run(
        [
            "ssh", "-o", "ConnectTimeout=5", "-o", "HostKeyAlgorithms=+ssh-rsa",
            "-o", "PubkeyAcceptedAlgorithms=+ssh-rsa", "root@" + host,
            "sha256sum " + " ".join(runtime.rstrip("/") + "/" + name for name in names),
        ],
        check=True,
        text=True,
        capture_output=True,
        timeout=20,
    )
    return {line.split(None, 1)[1]: line.split(None, 1)[0] for line in completed.stdout.splitlines()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-contract", required=True)
    parser.add_argument("--board-correctness-summary", required=True)
    parser.add_argument("--resource-signatures", required=True)
    parser.add_argument("--resource-plan", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument(
        "--capacity-runtime",
        default="/var/volatile/vta_c3_ram/capacity_runtime",
    )
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    output.mkdir(parents=True)

    paths = {
        "board_contract": Path(args.board_contract),
        "board_correctness_summary": Path(args.board_correctness_summary),
        "resource_signatures": Path(args.resource_signatures),
        "resource_plan": Path(args.resource_plan),
    }
    guards = {name: verify_frozen_file(path) for name, path in paths.items()}
    old_contract = load_json(paths["board_contract"])
    correctness = load_json(paths["board_correctness_summary"])
    plan = load_json(paths["resource_plan"])
    if correctness.get("status") != "passed" or correctness.get("passed_seed_checks") != 18:
        raise ValueError("prior W05 FPGA correctness gate is not 18/18")
    deploy = plan["certified_dispatch_set"]
    if (deploy["insn_capacity_bytes"], deploy["uop_capacity_bytes"]) != (4096, 4096):
        raise ValueError("expected frozen W05 capacity 4096+4096 bytes")
    entries = old_contract["execution_order"]
    incumbent = next(row for row in entries if row["role"] == "tophub_incumbent_before")
    hybrid = next(
        row for row in entries
        if row["role"] == "bounded_hybrid_t2" and int(row["config_index"]) == 574
    )
    cutoff = float(old_contract["board"]["dmesg_error_cutoff_seconds"])
    board = board_probe(args.host, cutoff)
    expected_board = old_contract["board"]
    if board.get("BOOT") != expected_board["boot_id"]:
        raise RuntimeError("board boot changed")
    if board.get("FPGA") != "operating" or board.get("UDMABUF") != str(expected_board["udmabuf_bytes"]):
        raise RuntimeError("FPGA/u-dma-buf state changed")
    if board.get("RPC_CWD") != expected_board["rpc_cwd_requirement"]:
        raise RuntimeError("RPC is not running from the frozen tmpfs directory")
    if " rw," not in " " + board.get("MOUNT", ""):
        raise RuntimeError("SD is not read-write")
    if "NEW_STORAGE_ERROR" in board:
        raise RuntimeError(board["NEW_STORAGE_ERROR"])

    root = Path(__file__).resolve().parents[3]
    sources = [
        root / "vta/tutorials/frontend/run_vta_p7r_joint_correctness.py",
        root / "vta/python/vta/top/vta_conv2d.py",
        root / "vta/python/vta/top/vta_conv2d_residency.py",
        root / "vta/runtime/runtime.cc",
        root / "vta/runtime/queue_capacity.h",
    ]
    capacity_runtime = args.capacity_runtime.rstrip("/")
    remote_capacity_hashes = hash_remote_runtime(args.host, capacity_runtime)
    contract = {
        "schema": "c3_p7r_unseen_board_contract_v1",
        "status": "frozen_before_p7r_board_labels",
        "purpose": "real u-dma-buf validation of W05 compile-derived command capacities",
        "workload_id": "W05",
        "workload": old_contract["workload"],
        "seeds": [0, 20250901, 20260910],
        "execution_order": [
            {**incumbent, "role": "capacity_tophub575"},
            {**hybrid, "role": "capacity_hybrid574"},
        ],
        "selection_rule": {
            "performance_labels_used": False,
            "identities_already_fpga_correct": True,
            "capacity_derived_from_local_command_peaks": True,
        },
        "command_capacity": {
            "insn_bytes": 4096,
            "uop_bytes": 4096,
            "legacy_total_bytes": 67108864,
            "expected_submissions": 6,
            "diagnostics": "query vta.runtime.queue_capacity_status through the same RPC session",
        },
        "board": {
            "boot_id": board["BOOT"],
            "fpga_state": board["FPGA"],
            "udmabuf_bytes": int(board["UDMABUF"]),
            "rpc_cwd_requirement": capacity_runtime,
            "default_rpc_cwd": board["RPC_CWD"],
            "capacity_rpc_cwd": capacity_runtime,
            "sd_mount": "/media/sd-mmcblk1p2",
            "sd_used_percent_at_freeze": int(board["DF"].split()[4].rstrip("%")),
            "dmesg_error_cutoff_seconds": cutoff,
            "rpc_pid_at_freeze": int(board["RPC"]),
            "remote_runtime_sha256": board["REMOTE_SHA256"],
            "remote_capacity_runtime_sha256": remote_capacity_hashes,
            "safety": "tmpfs only; stop exact RPC PID; always restore default RPC; never reboot or poweroff",
        },
        "source_guards_sha256": {
            str(path.relative_to(root)): sha256_file(path) for path in sources
        },
        "input_guards_sha256": {
            str(path.resolve()): value["sha256"] for path, value in (
                (paths[name], guards[name]) for name in paths
            )
        },
        "claim_boundary": "single boot, two exact W05 identities; correctness/capacity only, no timing or FPS",
    }
    contract_path = output / "contract.json"
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    (output / "STATUS.md").write_text(
        "# W05 real-FPGA command-capacity contract\n\n"
        "- Status: `frozen_before_p7r_board_labels`\n"
        "- Identities: TopHub575 and bounded-hybrid574\n"
        "- Capacity: instruction 4096 B + UOP 4096 B\n"
        "- Correctness: 3 seeds each; no timing\n"
        "- Safety: tmpfs only, exact RPC PID replacement, restore default RPC, no reboot/poweroff\n"
    )
    hashes = {
        path.name: sha256_file(path)
        for path in output.iterdir()
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    (output / "artifact_hashes.json").write_text(
        json.dumps({"artifacts": hashes}, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"contract": str(contract_path), "board": board, "capacity": contract["command_capacity"]}, indent=2))


if __name__ == "__main__":
    main()
