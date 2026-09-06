#!/usr/bin/env python3
"""Collect auditable board evidence for the CPU-VTA Pipeline V1 P0 gate."""

import argparse
import datetime
import json
import socket
import subprocess
from pathlib import Path

from tvm import rpc

from freeze_cpu_vta_pipeline_v1 import file_sha256, repo_root, seal_artifact


REMOTE_RUNTIME_DIR = "/mnt/sd/tvm_deploy/hpc"
REMOTE_BITSTREAM = "/mnt/sd/tvm_deploy/firmware/vta_hpc.bit"
REQUIRED_RPC_FUNCTIONS = [
    "runtime.NumThreads",
    "runtime.config_threadpool",
    "vta.runtime.profiler_clear",
    "vta.runtime.profiler_status",
    "vta.runtime.profiler_events",
]


def _parse_key_values(stdout):
    values = {}
    for line in stdout.splitlines():
        if "\t" not in line:
            continue
        key, value = line.split("\t", 1)
        values[key] = value
    return values


def _remote_probe(ssh_target, ssh_options, timeout_s):
    script = r"""
set -eu
runtime_dir=/mnt/sd/tvm_deploy/hpc
bitstream=/mnt/sd/tvm_deploy/firmware/vta_hpc.bit
emit() { printf '%s\t%s\n' "$1" "$2"; }
emit uname "$(uname -srm)"
emit board_model "$(tr -d '\000' </sys/firmware/devicetree/base/model 2>/dev/null || echo unavailable)"
emit cpu_count "$(nproc)"
emit ethernet_mac "$(cat /sys/class/net/eth0/address 2>/dev/null || echo unavailable)"
emit fpga_state "$(cat /sys/class/fpga_manager/fpga0/state)"
emit fpga_firmware "$(cat /sys/class/fpga_manager/fpga0/firmware 2>/dev/null || echo unavailable)"
emit udmabuf0_size "$(cat /sys/class/u-dma-buf/udmabuf0/size 2>/dev/null || echo unavailable)"
emit udmabuf0_device "$(test -c /dev/udmabuf0 && echo present || echo unavailable)"
sd_requested_path=/mnt/sd
sd_resolved_path=$(readlink -f "$sd_requested_path" 2>/dev/null || echo unavailable)
emit sd_requested_path "$sd_requested_path"
emit sd_resolved_path "$sd_resolved_path"
sd_row=$(df -Pk "$sd_resolved_path" 2>/dev/null | awk 'NR==2 {print $1 " " $4 " " $6}')
emit sd_filesystem_device "$(printf '%s' "$sd_row" | awk '{print $1}')"
emit sd_available_kb "$(printf '%s' "$sd_row" | awk '{print $2}')"
emit sd_mount "$(printf '%s' "$sd_row" | awk '{print $3}')"
storage_errors=$(dmesg 2>/dev/null | grep -E 'EXT4-fs error|Buffer I/O error|I/O error.*mmcblk' || true)
emit current_boot_storage_error_count "$(printf '%s\n' "$storage_errors" | sed '/^$/d' | wc -l)"
emit current_boot_storage_errors "$(printf '%s' "$storage_errors" | tr '\n' ';')"
rpc_pid=$(pidof tvm_rpc || true)
emit rpc_runtime_cwd "$(test -n "$rpc_pid" && readlink -f "/proc/$rpc_pid/cwd" || echo unavailable)"
emit bitstream_sha256 "$(sha256sum "$bitstream" | awk '{print $1}')"
emit tvm_rpc_sha256 "$(sha256sum "$runtime_dir/tvm_rpc" | awk '{print $1}')"
emit libtvm_runtime_sha256 "$(sha256sum "$runtime_dir/libtvm_runtime.so" | awk '{print $1}')"
emit libvta_sha256 "$(sha256sum "$runtime_dir/libvta.so" | awk '{print $1}')"
boot_hash=unavailable
for candidate in /mnt/sd/BOOT.BIN /boot/BOOT.BIN /media/sd-mmcblk1p2/BOOT.BIN; do
  if [ -f "$candidate" ]; then boot_hash=$(sha256sum "$candidate" | awk '{print $1}'); break; fi
done
emit boot_image_sha256 "$boot_hash"
freqs=
governors=
for cpu_dir in /sys/devices/system/cpu/cpu[0-9]*; do
  freq=$(cat "$cpu_dir/cpufreq/scaling_cur_freq" 2>/dev/null || echo unavailable)
  governor=$(cat "$cpu_dir/cpufreq/scaling_governor" 2>/dev/null || echo unavailable)
  freqs=${freqs}${freqs:+,}${freq}
  governors=${governors}${governors:+,}${governor}
done
emit cpu_frequencies_khz "$freqs"
emit cpu_governors "$governors"
"""
    command = ["ssh"] + list(ssh_options) + [ssh_target, script]
    proc = subprocess.run(command, text=True, capture_output=True, timeout=timeout_s, check=False)
    if proc.returncode != 0:
        raise RuntimeError("SSH preflight failed: {}".format(proc.stderr.strip()))
    return _parse_key_values(proc.stdout), proc.stderr.strip()


def _rpc_probe(host, port, timeout_s):
    with socket.create_connection((host, int(port)), timeout=timeout_s):
        pass
    remote = rpc.connect(host, int(port), session_timeout=max(1, int(timeout_s)))
    available = {}
    for name in REQUIRED_RPC_FUNCTIONS:
        try:
            available[name] = remote.get_function(name) is not None
        except (RuntimeError, ValueError):
            available[name] = False
    num_threads = None
    if available["runtime.NumThreads"]:
        num_threads = int(remote.get_function("runtime.NumThreads")())
    return {"required_functions": available, "runtime_num_threads": num_threads}


def storage_gate(remote, min_free_mb=128):
    try:
        sd_available_kb = int(remote.get("sd_available_kb", "0"))
    except (TypeError, ValueError):
        sd_available_kb = 0
    requested_path = remote.get("sd_requested_path", "/mnt/sd")
    resolved_path = remote.get("sd_resolved_path", requested_path)
    return (
        requested_path == "/mnt/sd"
        and resolved_path not in {None, "", "unavailable"}
        and remote.get("sd_mount") == resolved_path
        and str(remote.get("sd_filesystem_device", "")).startswith("/dev/")
        and sd_available_kb >= int(min_free_mb) * 1024
        and remote.get("current_boot_storage_error_count") == "0"
    )


def collect(args):
    root = repo_root()
    ssh_options = [
        "-o",
        "HostKeyAlgorithms=+ssh-rsa",
        "-o",
        "PubkeyAcceptedAlgorithms=+ssh-rsa",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout={}".format(int(args.timeout_s)),
    ]
    remote, ssh_stderr = _remote_probe(
        "{}@{}".format(args.ssh_user, args.board_host), ssh_options, args.timeout_s
    )
    rpc_evidence = _rpc_probe(args.board_host, args.rpc_port, args.timeout_s)

    local_paths = {
        "bitstream": root / "apps" / "vta_rpc" / "firmware" / "vta_hpc.bit",
        "tvm_rpc": root / "build_axu_aarch64" / "tvm_rpc",
        "libtvm_runtime.so": root / "build_axu_aarch64" / "libtvm_runtime.so",
        "libvta.so": root / "build_axu_aarch64" / "libvta.so",
    }
    missing_local = [name for name, path in local_paths.items() if not path.is_file()]
    if missing_local:
        raise RuntimeError("missing local P0 components: {}".format(", ".join(missing_local)))
    local_hashes = {name: file_sha256(path) for name, path in local_paths.items()}
    remote_hashes = {
        "bitstream": remote.get("bitstream_sha256"),
        "tvm_rpc": remote.get("tvm_rpc_sha256"),
        "libtvm_runtime.so": remote.get("libtvm_runtime_sha256"),
        "libvta.so": remote.get("libvta_sha256"),
    }
    hash_matches = {
        name: remote_hashes.get(name) == local_hashes.get(name) for name in local_hashes
    }
    rpc_functions_ok = all(rpc_evidence["required_functions"].values())
    frequencies = remote.get("cpu_frequencies_khz", "").split(",")
    governors = remote.get("cpu_governors", "").split(",")
    homogeneous_cpu = (
        remote.get("cpu_count") == "4"
        and len(frequencies) == 4
        and len(set(frequencies)) == 1
        and frequencies[0] != "unavailable"
        and len(governors) == 4
        and len(set(governors)) == 1
    )
    storage_healthy = storage_gate(remote)
    passed = (
        all(hash_matches.values())
        and rpc_functions_ok
        and rpc_evidence["runtime_num_threads"] == 4
        and homogeneous_cpu
        and remote.get("fpga_state") == "operating"
        and remote.get("udmabuf0_size") == "201326592"
        and remote.get("udmabuf0_device") == "present"
        and storage_healthy
        and remote.get("rpc_runtime_cwd") in {
            REMOTE_RUNTIME_DIR,
            "/media/sd-mmcblk1p2/tvm_deploy/hpc",
        }
    )
    board_proxy = {
        "device_tree_model": remote.get("board_model"),
        "ethernet_mac": remote.get("ethernet_mac"),
        "boot_image_sha256": remote.get("boot_image_sha256"),
        "identity_strength": "board_instance_proxy_not_silicon_serial",
    }
    return seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_preflight_evidence",
            "protocol_id": "cpu_vta_pipeline_v1",
            "observed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "passed": passed,
            "endpoint": {
                "ssh_target": "{}@{}".format(args.ssh_user, args.board_host),
                "ssh_options": ssh_options,
                "rpc_host": args.board_host,
                "rpc_port": int(args.rpc_port),
            },
            "ssh": {"status": "connected", "stderr": ssh_stderr or None},
            "rpc": {"status": "connected", **rpc_evidence},
            "remote": remote,
            "expected_runtime": {
                "bitstream_name": "vta_hpc.bit",
                "bitstream_path": REMOTE_BITSTREAM,
                "runtime_dir": REMOTE_RUNTIME_DIR,
            },
            "board_instance_proxy": board_proxy,
            "component_hashes": {
                "local": local_hashes,
                "remote": remote_hashes,
                "matches": hash_matches,
            },
            "gate_checks": {
                "all_component_hashes_match": all(hash_matches.values()),
                "required_rpc_functions_available": rpc_functions_ok,
                "four_homogeneous_cpu_cores": homogeneous_cpu,
                "runtime_num_threads_four": rpc_evidence["runtime_num_threads"] == 4,
                "hpc_runtime_and_fpga_operating": remote.get("fpga_state") == "operating"
                and remote.get("rpc_runtime_cwd")
                in {REMOTE_RUNTIME_DIR, "/media/sd-mmcblk1p2/tvm_deploy/hpc"},
                "native_vta_udmabuf_ready": remote.get("udmabuf0_size") == "201326592"
                and remote.get("udmabuf0_device") == "present",
                "sd_space_and_current_boot_log_healthy": storage_healthy,
            },
        }
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-host", default="192.168.1.247")
    parser.add_argument("--rpc-port", type=int, default=9090)
    parser.add_argument("--ssh-user", default="root")
    parser.add_argument("--timeout-s", type=int, default=10)
    parser.add_argument(
        "--output",
        default=str(
            repo_root()
            / "vta/tutorials/frontend/report_out/resource_aware_maxplus/v1_preflight_evidence.json"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    payload = collect(args)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("{} {} passed={}".format(path, payload["artifact_sha256"], payload["passed"]))
    if not payload["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
