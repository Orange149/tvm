#!/usr/bin/env python3
"""Validate frozen W05 command capacities on real FPGA and restore the default RPC."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from collect_vta_full_pool_fsim_commands import sanitize_stderr


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
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def ssh(host, command, timeout=20, check=True):
    return subprocess.run(
        ["ssh", *SSH_OPTIONS, "root@" + host, command],
        check=check,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def rpc_state(host):
    command = r'''
pid=$(pidof tvm_rpc | awk '{print $1}')
echo PID=$pid
echo CWD=$(readlink /proc/$pid/cwd)
tr '\0' '\n' < /proc/$pid/environ | grep -E '^(VTA_INSN_BUFFER_BYTES|VTA_UOP_BUFFER_BYTES|VTA_QUEUE_DIAGNOSTICS|VTA_COMMAND_MANIFEST_ID|VTA_REPLAY_POLICY)=' || true
'''
    fields = {}
    for line in ssh(host, command).stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key] = value
    return fields


def stop_exact_rpc(host, pid, expected_cwd):
    state = rpc_state(host)
    if state.get("PID") != str(pid) or state.get("CWD") != expected_cwd:
        raise RuntimeError("refusing to stop RPC whose exact PID/cwd changed: {}".format(state))
    command = (
        "kill {pid}; i=0; while kill -0 {pid} 2>/dev/null; do "
        "i=$((i+1)); [ $i -lt 50 ] || exit 9; usleep 100000 2>/dev/null || sleep 1; done"
    ).format(pid=int(pid))
    ssh(host, command, timeout=60)


def start_rpc(host, runtime, log_path, environment=None):
    env_words = []
    for name, value in sorted((environment or {}).items()):
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", name) or not re.fullmatch(
            r"[a-zA-Z0-9_.-]+", str(value)
        ):
            raise ValueError("unsafe RPC environment entry")
        env_words.append("{}={}".format(name, value))
    launched = "exec env {env} ./start_axu5evb_cpp_rpc.sh {runtime} 9090".format(
        env=" ".join(env_words), runtime=runtime
    )
    command = (
        "cd {runtime}; : > {log}; nohup sh -c '{launched}' "
        "> {log} 2>&1 < /dev/null & echo $!"
    ).format(runtime=runtime, log=log_path, launched=launched)
    completed = ssh(host, command)
    pid = int(completed.stdout.strip().splitlines()[-1])
    for _ in range(30):
        time.sleep(0.1)
        state = rpc_state(host)
        if state.get("PID") == str(pid) and state.get("CWD") == runtime:
            return state
    raise RuntimeError("RPC did not reach the expected state: pid={}".format(pid))


def board_guard(host, contract):
    cutoff = float(contract["board"]["dmesg_error_cutoff_seconds"])
    command = (
        "echo BOOT=$(cat /proc/sys/kernel/random/boot_id); "
        "echo FPGA=$(cat /sys/class/fpga_manager/fpga0/state); "
        "echo UDMABUF=$(cat /sys/class/u-dma-buf/udmabuf0/size); "
        "dmesg | awk '$1 ~ /^\\[/ {{t=$1; gsub(/\\[/,\"\",t); "
        "if ((t+0)>{} && ($0 ~ /EXT4-fs error|mmcblk.*error|I.O error|Buffer I.O/)) "
        "print \"NEW_STORAGE_ERROR=\" $0}}'"
    ).format(cutoff)
    fields = {}
    for line in ssh(host, command).stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key] = value
    expected = contract["board"]
    if fields.get("BOOT") != expected["boot_id"]:
        raise RuntimeError("board boot changed")
    if fields.get("FPGA") != "operating" or fields.get("UDMABUF") != str(expected["udmabuf_bytes"]):
        raise RuntimeError("FPGA/u-dma-buf state changed")
    if "NEW_STORAGE_ERROR" in fields:
        raise RuntimeError(fields["NEW_STORAGE_ERROR"])
    return fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--correctness-timeout-seconds", type=int, default=180)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    output.mkdir(parents=True)

    contract_path = Path(args.contract).resolve()
    contract = load_json(contract_path)
    contract_ledger = load_json(contract_path.parent / "artifact_hashes.json")["artifacts"]
    if contract_ledger.get(contract_path.name) != sha256_file(contract_path):
        raise ValueError("contract artifact hash mismatch")
    for path, expected in contract["source_guards_sha256"].items():
        if sha256_file(path) != expected:
            raise ValueError("source guard changed: {}".format(path))
    for path, expected in contract["input_guards_sha256"].items():
        if sha256_file(path) != expected:
            raise ValueError("input guard changed: {}".format(path))
    before = board_guard(args.host, contract)
    old_state = rpc_state(args.host)
    if old_state.get("PID") != str(contract["board"]["rpc_pid_at_freeze"]):
        raise RuntimeError("default RPC PID changed after contract freeze")
    capacities = {
        "insn_bytes": int(contract["command_capacity"]["insn_bytes"]),
        "uop_bytes": int(contract["command_capacity"]["uop_bytes"]),
    }
    launcher_environment = dict(contract.get("launcher_environment") or {})
    expected_launcher_environment = {
        "VTA_INSN_BUFFER_BYTES": str(capacities["insn_bytes"]),
        "VTA_UOP_BUFFER_BYTES": str(capacities["uop_bytes"]),
        "VTA_QUEUE_DIAGNOSTICS": "1",
    }
    expected_launcher_environment.update(contract.get("manifest_environment") or {})
    if launcher_environment and launcher_environment != expected_launcher_environment:
        raise ValueError("frozen launcher environment is inconsistent")
    if not launcher_environment:
        launcher_environment = expected_launcher_environment
    preregistered = {
        "schema": "c3_p7r_board_command_capacity_execution_v1",
        "status": "frozen_before_rpc_replacement_or_fpga_execution",
        "contract": {"path": str(contract_path), "sha256": sha256_file(contract_path)},
        "old_rpc": old_state,
        "capacity": capacities,
        "launcher_environment": launcher_environment,
        "execution": "two exact identities, three seeds each, correctness and structural queue fields only",
        "restoration": "always stop exact capacity RPC and restore default tmpfs RPC; never reboot/poweroff",
    }
    prereg_path = output / "preregistered.json"
    write_json(prereg_path, preregistered)
    write_json(output / "pre_execution_hashes.json", {"preregistered.json": sha256_file(prereg_path)})

    capacity_log = contract["board"].get(
        "capacity_rpc_log", "/var/volatile/vta_c3_ram/p7r90_capacity_rpc.log"
    )
    default_log = contract["board"].get(
        "restored_default_rpc_log", "/var/volatile/vta_c3_ram/p7r90_restored_default_rpc.log"
    )
    default_runtime = contract["board"]["default_rpc_cwd"]
    capacity_runtime = contract["board"]["capacity_rpc_cwd"]
    remote_hash_command = "sha256sum " + " ".join(
        contract["board"]["remote_capacity_runtime_sha256"]
    )
    observed_remote_hashes = {
        line.split(None, 1)[1]: line.split(None, 1)[0]
        for line in ssh(args.host, remote_hash_command).stdout.splitlines()
    }
    if observed_remote_hashes != contract["board"]["remote_capacity_runtime_sha256"]:
        raise RuntimeError("isolated capacity runtime hashes changed after contract freeze")
    capacity_state = None
    restored_state = None
    error = None
    try:
        stop_exact_rpc(args.host, int(old_state["PID"]), default_runtime)
        capacity_state = start_rpc(
            args.host, capacity_runtime, capacity_log, launcher_environment
        )
        for name, value in launcher_environment.items():
            if capacity_state.get(name) != str(value):
                raise RuntimeError("capacity RPC {} environment mismatch".format(name))
        correctness_dir = output / "correctness"
        command = [
            sys.executable,
            str(Path(__file__).with_name("run_vta_p7r_joint_correctness.py")),
            "--contract", str(contract_path),
            "--output-dir", str(correctness_dir),
            "--host", args.host,
            "--port", str(args.port),
            "--query-queue-capacity-status",
        ]
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
            timeout=args.correctness_timeout_seconds,
            env=os.environ.copy(),
        )
        (output / "runner.stdout").write_text(completed.stdout)
        (output / "runner.stderr").write_text(sanitize_stderr(completed.stderr))
        if completed.returncode != 0:
            raise RuntimeError("correctness runner failed with exit {}".format(completed.returncode))
        correctness = load_json(correctness_dir / "summary.json")
        if correctness.get("status") != "passed" or correctness.get("passed_seed_checks") != 6:
            raise RuntimeError("real-FPGA reduced-capacity correctness is not 6/6")
        capacity_status = correctness.get("queue_capacity_status") or {}
        if capacity_status.get("schema") != "vta_queue_capacity_status_v1":
            raise RuntimeError("queue capacity status RPC is absent")
        if (
            int(capacity_status["insn_capacity_bytes"]) != capacities["insn_bytes"]
            or int(capacity_status["uop_capacity_bytes"]) != capacities["uop_bytes"]
            or int(capacity_status["insn_peak_bytes"]) > capacities["insn_bytes"]
            or int(capacity_status["uop_peak_bytes"]) > capacities["uop_bytes"]
            or int(capacity_status["submissions"])
            != contract["command_capacity"]["expected_submissions"]
        ):
            raise RuntimeError("queried real-FPGA queue status violates the frozen capacity")
        manifest_environment = contract.get("manifest_environment") or {}
        if manifest_environment and (
            capacity_status.get("command_manifest_id")
            != manifest_environment.get("VTA_COMMAND_MANIFEST_ID")
            or capacity_status.get("replay_policy")
            != manifest_environment.get("VTA_REPLAY_POLICY")
        ):
            raise RuntimeError("runtime manifest/replay attestation mismatch")
        write_json(output / "queue_capacity_status.json", capacity_status)
    except Exception as caught:
        error = caught
        write_json(output / "failure.json", {"type": type(caught).__name__, "message": str(caught)})
    finally:
        try:
            if capacity_state is not None:
                current = rpc_state(args.host)
                if current.get("PID") == capacity_state.get("PID"):
                    stop_exact_rpc(args.host, int(capacity_state["PID"]), capacity_runtime)
            restored_state = start_rpc(args.host, default_runtime, default_log, None)
            if any(
                name in restored_state
                for name in (
                    "VTA_INSN_BUFFER_BYTES", "VTA_UOP_BUFFER_BYTES", "VTA_QUEUE_DIAGNOSTICS",
                    "VTA_COMMAND_MANIFEST_ID", "VTA_REPLAY_POLICY",
                )
            ):
                raise RuntimeError("restored default RPC retained capacity environment")
        except Exception as restore_error:
            write_json(
                output / "restoration_failure.json",
                {"type": type(restore_error).__name__, "message": str(restore_error)},
            )
            if error is None:
                error = restore_error

    after = board_guard(args.host, contract)
    summary = {
        "schema": "c3_p7r_real_fpga_command_capacity_summary_v1",
        "status": "passed" if error is None else "failed",
        "board_before": before,
        "capacity_rpc": capacity_state,
        "restored_default_rpc": restored_state,
        "board_after": after,
        "capacity": capacities,
        "correctness_seed_checks_passed": 6 if error is None else None,
        "queue_status_submissions": contract["command_capacity"]["expected_submissions"] if error is None else None,
        "manifest_attested": bool(contract.get("manifest_environment")) and error is None,
        "manifest_id": (contract.get("manifest_environment") or {}).get(
            "VTA_COMMAND_MANIFEST_ID"
        ),
        "legacy_total_bytes": contract["command_capacity"]["legacy_total_bytes"],
        "requested_byte_reduction_fraction": 1.0
        - (capacities["insn_bytes"] + capacities["uop_bytes"])
        / contract["command_capacity"]["legacy_total_bytes"],
        "performance_measurement": "not_collected",
        "sd_writes_for_experiment": "none; RPC logs and uploads used /var/volatile tmpfs",
        "claim_boundary": contract["claim_boundary"],
    }
    write_json(output / "summary.json", summary)
    (output / "STATUS.md").write_text(
        "# W05 real-FPGA reduced command-capacity validation\n\n"
        "- Status: `{}`\n"
        "- Capacity: instruction {} B + UOP {} B\n"
        "- Exact correctness: {}/6\n"
        "- Queried queue submissions: {}/6\n"
        "- Default RPC restored: `{}`\n"
        "- No performance timing, reboot, poweroff, or SD experiment writes\n".format(
            summary["status"], capacities["insn_bytes"], capacities["uop_bytes"],
            summary["correctness_seed_checks_passed"] or 0,
            summary["queue_status_submissions"] or 0,
            restored_state is not None and error is None,
        )
    )
    hashes = {
        str(path.relative_to(output)): sha256_file(path)
        for path in output.rglob("*")
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    write_json(output / "artifact_hashes.json", {"artifacts": hashes})
    print(json.dumps(summary, indent=2, sort_keys=True))
    if error is not None:
        raise error


if __name__ == "__main__":
    main()
