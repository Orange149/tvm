#!/usr/bin/env python3
"""Freeze a manifest-bound, correctness-only VTA board experiment.

The contract is composed only after the current ARM runtime has been copied to
an isolated tmpfs directory.  It never modifies the default runtime or the SD
card and contains no performance labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from build_vta_command_resource_certificate import validate_certificate
from build_vta_deployment_manifest import validate_manifest


SSH_OPTIONS = [
    "-o", "ConnectTimeout=5",
    "-o", "HostKeyAlgorithms=+ssh-rsa",
    "-o", "PubkeyAcceptedAlgorithms=+ssh-rsa",
]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_frozen_file(path):
    path = Path(path).resolve()
    ledger = load_json(path.parent / "artifact_hashes.json")
    expected = ledger.get("artifacts", ledger.get("output_sha256", ledger)).get(path.name)
    observed = sha256_file(path)
    if expected != observed:
        raise ValueError("frozen artifact hash mismatch: {}".format(path))
    return {"path": str(path), "sha256": observed}


def ssh(host, command):
    return subprocess.run(
        ["ssh", *SSH_OPTIONS, "root@" + host, command],
        check=True,
        text=True,
        capture_output=True,
        timeout=20,
    ).stdout


def remote_hashes(host, runtime):
    names = ("libvta.so", "libtvm_runtime.so", "tvm_rpc", "start_axu5evb_cpp_rpc.sh")
    output = ssh(
        host,
        "sha256sum " + " ".join(runtime.rstrip("/") + "/" + name for name in names),
    )
    return {
        line.split(None, 1)[1]: line.split(None, 1)[0]
        for line in output.splitlines()
    }


def board_probe(host, cutoff):
    command = r'''
echo BOOT=$(cat /proc/sys/kernel/random/boot_id)
echo FPGA=$(cat /sys/class/fpga_manager/fpga0/state)
echo UDMABUF=$(cat /sys/class/u-dma-buf/udmabuf0/size)
pid=$(pidof tvm_rpc | awk '{print $1}')
echo RPC=$pid
echo RPC_CWD=$(readlink /proc/$pid/cwd)
echo TMPFS=$(df -Pk /var/volatile | tail -n 1)
dmesg | awk '$1 ~ /^\[/ {t=$1; gsub(/\[/,"",t); if ((t+0)>%s && ($0 ~ /EXT4-fs error|mmcblk.*error|I.O error|Buffer I.O/)) print "NEW_STORAGE_ERROR=" $0}'
''' % cutoff
    fields = {}
    for line in ssh(host, command).splitlines():
        if "=" in line:
            name, value = line.split("=", 1)
            fields[name] = value
    return fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-board-contract", required=True)
    parser.add_argument("--resource-certificate", required=True)
    parser.add_argument("--pending-manifest", required=True)
    parser.add_argument("--runtime-build-manifest", required=True)
    parser.add_argument("--capacity-runtime", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="192.168.1.247")
    args = parser.parse_args()

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[3]

    input_paths = {
        "base_board_contract": Path(args.base_board_contract),
        "resource_certificate": Path(args.resource_certificate),
        "pending_manifest": Path(args.pending_manifest),
    }
    inputs = {name: verify_frozen_file(path) for name, path in input_paths.items()}
    build_manifest_path = Path(args.runtime_build_manifest).resolve()
    build = load_json(build_manifest_path)
    build_dir = build_manifest_path.parent
    for name, expected in build["artifacts"].items():
        if sha256_file(build_dir / name) != expected:
            raise ValueError("local runtime build artifact changed: {}".format(name))
    inputs["runtime_build_manifest"] = {
        "path": str(build_manifest_path),
        "sha256": sha256_file(build_manifest_path),
    }

    base = load_json(input_paths["base_board_contract"])
    certificate = load_json(input_paths["resource_certificate"])
    manifest = load_json(input_paths["pending_manifest"])
    validate_certificate(certificate, repo_root=root)
    validate_manifest(manifest, certificate)
    if certificate["status"] != "local_provisional" or manifest["status"] != "qualification_pending":
        raise ValueError("board experiment requires a provisional certificate and pending manifest")
    identity_by_key = {
        (row["residence_mode"], int(row["config_index"]), row["tir_sha256"]): row
        for row in certificate["identities"]
    }
    old_by_key = {
        (row["mode"], int(row["config_index"]), row["tir_sha256"]): row
        for row in base["execution_order"]
    }
    if not set(identity_by_key).issubset(old_by_key):
        raise ValueError("base board contract does not cover the exact manifest identities")
    execution_order = []
    for key, identity in sorted(identity_by_key.items(), key=lambda item: item[1]["candidate_id"]):
        row = dict(old_by_key[key])
        row["candidate_id"] = identity["candidate_id"]
        row["role"] = (
            "manifest_primary" if identity["candidate_id"] == manifest["primary_candidate_id"]
            else "manifest_qualified_alternative"
        )
        execution_order.append(row)

    cutoff = float(base["board"]["dmesg_error_cutoff_seconds"])
    board = board_probe(args.host, cutoff)
    if board.get("BOOT") != base["board"]["boot_id"]:
        raise RuntimeError("board boot changed")
    if board.get("FPGA") != "operating" or board.get("UDMABUF") != str(
        base["board"]["udmabuf_bytes"]
    ):
        raise RuntimeError("FPGA/u-dma-buf state changed")
    if not board.get("RPC") or not board.get("RPC_CWD", "").startswith("/var/volatile/"):
        raise RuntimeError("default RPC is absent or not tmpfs-backed")
    if "NEW_STORAGE_ERROR" in board:
        raise RuntimeError(board["NEW_STORAGE_ERROR"])

    capacity_runtime = args.capacity_runtime.rstrip("/")
    remote_capacity = remote_hashes(args.host, capacity_runtime)
    remote_lib = capacity_runtime + "/libvta.so"
    if remote_capacity.get(remote_lib) != build["artifacts"]["libvta.so"]:
        raise RuntimeError("remote manifest runtime differs from the cross build")
    default_runtime = board["RPC_CWD"]
    remote_default = remote_hashes(args.host, default_runtime)
    per_instance = manifest["command_memory"]["per_instance"]
    manifest_environment = {
        "VTA_COMMAND_MANIFEST_ID": manifest["manifest_id"],
        "VTA_REPLAY_POLICY": manifest["replay_policy"]["mode"],
    }
    launcher_environment = {
        **manifest["launcher_environment"],
        "VTA_QUEUE_DIAGNOSTICS": "1",
    }
    contract = {
        "schema": "c3_p7r_manifest_attestation_contract_v1",
        "status": "frozen_before_p7r_board_labels",
        "purpose": "prospective real-FPGA attestation of the exact deployment manifest and command backing",
        "workload_id": "W05",
        "workload": base["workload"],
        "seeds": [0, 20250901, 20260910],
        "execution_order": execution_order,
        "selection_rule": {
            "performance_labels_used": False,
            "exact_manifest_identity_set": True,
            "capacity_formula": certificate["derivation"]["formula"],
        },
        "command_capacity": {
            "insn_bytes": int(per_instance["insn_capacity_bytes"]),
            "uop_bytes": int(per_instance["uop_capacity_bytes"]),
            "legacy_total_bytes": 67108864,
            "expected_submissions": len(execution_order) * 3,
            "diagnostics": "same-session capacity, peak, manifest ID, and replay-policy query",
        },
        "manifest": {
            "manifest_id": manifest["manifest_id"],
            "resource_certificate_key": certificate["certificate_key"],
            "candidate_ids": sorted(row["candidate_id"] for row in certificate["identities"]),
        },
        "manifest_environment": manifest_environment,
        "launcher_environment": launcher_environment,
        "board": {
            "boot_id": board["BOOT"],
            "fpga_state": board["FPGA"],
            "udmabuf_bytes": int(board["UDMABUF"]),
            "rpc_cwd_requirement": capacity_runtime,
            "default_rpc_cwd": default_runtime,
            "capacity_rpc_cwd": capacity_runtime,
            "capacity_rpc_log": "/var/volatile/vta_c3_ram/p7r103_manifest_rpc.log",
            "restored_default_rpc_log": "/var/volatile/vta_c3_ram/p7r103_restored_default_rpc.log",
            "dmesg_error_cutoff_seconds": cutoff,
            "rpc_pid_at_freeze": int(board["RPC"]),
            "remote_runtime_sha256": remote_default,
            "remote_capacity_runtime_sha256": remote_capacity,
            "tmpfs_at_freeze": board.get("TMPFS"),
            "safety": "isolated tmpfs only; exact PID replacement; always restore; never reboot/poweroff",
        },
        "source_guards_sha256": {
            **certificate["source_guards_sha256"],
            "vta/tutorials/frontend/run_vta_p7r_joint_correctness.py": sha256_file(
                root / "vta/tutorials/frontend/run_vta_p7r_joint_correctness.py"
            ),
            "vta/tutorials/frontend/run_vta_p7r_board_command_capacity.py": sha256_file(
                root / "vta/tutorials/frontend/run_vta_p7r_board_command_capacity.py"
            ),
        },
        "input_guards_sha256": {
            value["path"]: value["sha256"] for value in inputs.values()
        },
        "claim_boundary": (
            "same boot, one queue instance, two exact W05 identities; correctness, command capacity, "
            "manifest ID, and replay policy only; no timing, FPS, replay, SD, or cross-boot claim"
        ),
    }
    contract_path = output / "contract.json"
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    (output / "STATUS.md").write_text(
        "# VTA manifest-attestation board contract\n\n"
        "- Status: `frozen_before_p7r_board_labels`\n"
        "- Manifest: `{}`\n"
        "- Exact identities: {}\n"
        "- Capacity: instruction {} B + UOP {} B\n"
        "- Safety: isolated tmpfs, exact RPC PID, restore default, no reboot/poweroff\n".format(
            manifest["manifest_id"], len(execution_order),
            per_instance["insn_capacity_bytes"], per_instance["uop_capacity_bytes"],
        )
    )
    hashes = {
        path.name: sha256_file(path)
        for path in output.iterdir()
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    (output / "artifact_hashes.json").write_text(
        json.dumps({"artifacts": hashes}, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "contract": str(contract_path),
        "manifest_id": manifest["manifest_id"],
        "rpc_pid": int(board["RPC"]),
        "capacity": contract["command_capacity"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
