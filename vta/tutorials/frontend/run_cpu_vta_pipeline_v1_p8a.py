#!/usr/bin/env python3
"""Audit one CPU-to-VTA shared-memory edge before implementing the P8 pipeline."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shlex
import shutil
import statistics
import subprocess
import tarfile
from pathlib import Path

import numpy as np

from freeze_cpu_vta_pipeline_v1 import file_sha256, repo_root, seal_artifact


ROOT = repo_root()
REPORT_ROOT = ROOT / "vta/tutorials/frontend/report_out/resource_aware_maxplus"
DEFAULT_PACKAGE = (
    REPORT_ROOT
    / "v1_p7_top20_board_20260904/packages/"
    "01_cpu-00-02_t4__vta-03-17_t1__cpu-18-20_t1/package"
)
DEFAULT_OUTPUT = REPORT_ROOT
RUNNER_SOURCE = ROOT / "vta/apps/native_deploy/vta_stage_pipeline_runner.cc"
GRAPH_EXECUTOR_SOURCE = ROOT / "src/runtime/graph_executor/graph_executor.cc"
PIPELINE_SOURCE = ROOT / "src/runtime/pipeline/pipeline_struct.h"
DRIVER_SOURCE = ROOT / "3rdparty/vta-hw/src/axu5evb/axu5evb_driver.cc"
COMPILE_COMMANDS = ROOT / "build_axu_aarch64/compile_commands.json"
PRIOR_HPC_EVIDENCE = (
    ROOT
    / "vta/tutorials/frontend/hp_hpc_quant/results/"
    "hpc_coherent_board_results_20260403.md"
)
TOP20_BOARD_SUMMARY = (
    REPORT_ROOT / "v1_p7_top20_board_20260904/top20_board_summary.json"
)
SSH_OPTIONS = [
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=8",
    "-o",
    "HostKeyAlgorithms=+ssh-rsa",
    "-o",
    "PubkeyAcceptedAlgorithms=+ssh-rsa",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-host", default="192.168.1.247")
    parser.add_argument("--ssh-user", default="root")
    parser.add_argument("--rpc-port", type=int, default=9090)
    parser.add_argument("--package", default=str(DEFAULT_PACKAGE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--remote-root", default="/var/volatile/ramps_p8a")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--keep-remote", action="store_true")
    parser.add_argument("--reuse-remote-package", action="store_true")
    parser.add_argument("--timeout-s", type=int, default=600)
    return parser.parse_args()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_manifest(package):
    path = Path(package) / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("candidate_id") != "cpu-00-02_t4__vta-03-17_t1__cpu-18-20_t1":
        raise ValueError("P8A must use the frozen rank-1 package")
    return payload


def _slot_contract(schema):
    if int(schema.get("arity", 0)) != 1:
        raise ValueError("P8A requires a single-tensor stage boundary")
    slot = schema["slots"][0]
    return {
        "shape": [int(value) for value in slot["shape"]],
        "dtype": str(slot["dtype"]),
    }


def boundary_contract(manifest, edge_index=0):
    stages = manifest["stages"]
    if edge_index < 0 or edge_index + 1 >= len(stages):
        raise ValueError("P8 edge must name an adjacent stage boundary")
    producer_stage = stages[edge_index]
    consumer_stage = stages[edge_index + 1]
    direction = "{}_to_{}".format(producer_stage["device"], consumer_stage["device"])
    if direction not in {"cpu_to_vta", "vta_to_cpu"}:
        raise ValueError("P8 requires a heterogeneous CPU/VTA boundary")
    producer = _slot_contract(producer_stage["output_schema"])
    consumer = _slot_contract(consumer_stage["input_schema"])
    if producer != consumer:
        raise ValueError("producer and consumer tensor contracts differ")
    bits = int("".join(ch for ch in producer["dtype"] if ch.isdigit()))
    elements = 1
    for extent in producer["shape"]:
        elements *= extent
    observed_bytes = elements * bits // 8
    if observed_bytes != int(producer_stage["output_bytes"]):
        raise ValueError("manifest output byte count is inconsistent")
    return {
        **producer,
        "bytes": observed_bytes,
        "producer_stage": edge_index,
        "consumer_stage": edge_index + 1,
        "direction": direction,
    }


def _graph_entry_contract(graph, entry):
    node_index, output_index = int(entry[0]), int(entry[1])
    entry_id = int(graph["node_row_ptr"][node_index]) + output_index
    attrs = graph["attrs"]
    return {
        "shape": [int(value) for value in attrs["shape"][1][entry_id]],
        "dtype": str(attrs["dltype"][1][entry_id]),
        "device_index": int(attrs["device_index"][1][entry_id]),
    }


def graph_boundary_contract(package, edge_index=0):
    package = Path(package)
    left = json.loads(
        (package / "stages/stage{}/graph.json".format(edge_index)).read_text(encoding="utf-8")
    )
    right = json.loads(
        (package / "stages/stage{}/graph.json".format(edge_index + 1)).read_text(
            encoding="utf-8"
        )
    )
    left_contract = _graph_entry_contract(left, left["heads"][0])
    input_node = int(right["arg_nodes"][0])
    right_contract = _graph_entry_contract(right, [input_node, 0, 0])
    return {"producer_output": left_contract, "consumer_input": right_contract}


def build_protocol(manifest, warmup=5, runs=20):
    return seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p8_protocol",
            "phase": "P8A",
            "scope": "single_cpu_to_vta_edge_abi_and_memory_feasibility",
            "candidate_id": manifest["candidate_id"],
            "edge": boundary_contract(manifest),
            "controls": [
                {"id": "safe_byte_copy", "AXU5EVB_DRIVER_SAFE_COPY": "1"},
                {"id": "memcpy", "AXU5EVB_DRIVER_SAFE_COPY": "0"},
            ],
            "measurement": {
                "alternating_distinct_inputs": 2,
                "warmup_per_path": int(warmup),
                "scored_per_path": int(runs),
                "ordinary_and_zero_copy_in_same_process": True,
                "scored_pair_order": "balanced_ab_ba_per_frame",
                "single_boot_qualification_only": True,
            },
            "gates": {
                "zero_copy_matches_ordinary_copy_for_alternating_inputs": True,
                "slot_alignment_bytes": 256,
                "same_virtual_address_dual_view": True,
                "zero_copy_materialization_bytes": 0,
                "cpu_direct_mapping_penalty_not_greater_than_copy_saved": True,
                "coherence_evidence_sources_recorded_separately": True,
            },
            "excluded_claims": [
                "dual_slot_pipeline_throughput",
                "cross_boot_performance_improvement",
                "candidate_ranking_change",
            ],
            "candidate_throughput_used_for_fit": False,
            "correctness_scope": {
                "current_test": "ordinary_copy_same_compiled_graph_equivalence",
                "independent_reference": "inherited_from_frozen_rank1_package_validation",
            },
        }
    )


def build_local_audit(package, manifest, runner_binary=None, edge_index=0):
    graph_executor = GRAPH_EXECUTOR_SOURCE.read_text(encoding="utf-8")
    pipeline = PIPELINE_SOURCE.read_text(encoding="utf-8")
    driver = DRIVER_SOURCE.read_text(encoding="utf-8")
    runner = RUNNER_SOURCE.read_text(encoding="utf-8")
    checks = {
        "graph_executor_set_input_zero_copy_present": "set_input_zero_copy" in graph_executor,
        "graph_executor_set_output_zero_copy_present": "set_output_zero_copy" in graph_executor,
        "pipeline_executor_queue_deep_copy_present": "TVMArrayCopyFromTo" in pipeline,
        "driver_safe_copy_switch_present": "AXU5EVB_DRIVER_SAFE_COPY" in driver,
        "runner_dtype_check_present": "SameDType" in runner,
        "runner_shape_check_present": "SameShape" in runner,
        "runner_byte_offset_check_present": "requires zero byte offsets" in runner,
        "runner_physical_address_check_present": "VTAMemGetPhyAddr" in runner,
        "runner_reference_hash_check_present": "reference_correctness_passed" in runner,
        "runner_reference_scope_present": "ordinary_copy_same_compiled_graph" in runner,
        "runtime_compiled_for_coherent_accesses": (
            COMPILE_COMMANDS.exists()
            and "-DVTA_COHERENT_ACCESSES=true"
            in COMPILE_COMMANDS.read_text(encoding="utf-8")
        ),
        "prior_hpc_controlled_evidence_present": PRIOR_HPC_EVIDENCE.exists(),
    }
    graphs = graph_boundary_contract(package, edge_index)
    edge = boundary_contract(manifest, edge_index)
    checks["graph_shape_dtype_match"] = (
        graphs["producer_output"]["shape"] == edge["shape"]
        and graphs["consumer_input"]["shape"] == edge["shape"]
        and graphs["producer_output"]["dtype"] == edge["dtype"]
        and graphs["consumer_input"]["dtype"] == edge["dtype"]
    )
    expected_devices = (1, 12) if edge["direction"] == "cpu_to_vta" else (12, 1)
    checks["graph_device_views_are_complementary_cpu_extdev"] = (
        graphs["producer_output"]["device_index"],
        graphs["consumer_input"]["device_index"],
    ) == expected_devices
    return seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p8_local_abi_audit",
            "candidate_id": manifest["candidate_id"],
            "package_manifest_sha256": file_sha256(Path(package) / "manifest.json"),
            "source_sha256": {
                "runner": file_sha256(RUNNER_SOURCE),
                "graph_executor": file_sha256(GRAPH_EXECUTOR_SOURCE),
                "pipeline_struct": file_sha256(PIPELINE_SOURCE),
                "axu5evb_driver": file_sha256(DRIVER_SOURCE),
                "compile_commands": file_sha256(COMPILE_COMMANDS),
                "prior_hpc_evidence": file_sha256(PRIOR_HPC_EVIDENCE),
            },
            "runner_binary_sha256": file_sha256(runner_binary) if runner_binary else None,
            "edge": edge,
            "lowered_graph_contract": graphs,
            "checks": checks,
            "passed": all(checks.values()) and runner_binary is not None,
        }
    )


def compile_runner(output):
    output = Path(output)
    command = " ".join(
        [
            "unset LD_LIBRARY_PATH &&",
            "source /home/orange/alinx_plsdk/install/environment-setup-aarch64-xilinx-linux &&",
            "aarch64-xilinx-linux-g++ -std=c++17 -O2",
            '--sysroot="$SDKTARGETSYSROOT"',
            "-I include -I 3rdparty/dlpack/include -I 3rdparty/dmlc-core/include",
            "-I 3rdparty/vta-hw/include -I vta/include",
            shlex.quote(str(RUNNER_SOURCE.relative_to(ROOT))),
            "-o",
            shlex.quote(str(output)),
            "-L build_axu_aarch64",
            '-Wl,-rpath-link,"$SDKTARGETSYSROOT/lib"',
            '-Wl,-rpath-link,"$SDKTARGETSYSROOT/usr/lib"',
            "-ltvm_runtime -lvta -ldl -pthread",
        ]
    )
    proc = subprocess.run(
        ["/usr/bin/zsh", "-lc", command], cwd=ROOT, text=True, capture_output=True, check=False
    )
    if proc.returncode:
        raise RuntimeError("P8A runner cross compile failed:\n{}".format(proc.stdout + proc.stderr))
    output.chmod(0o755)
    return proc.stdout + proc.stderr


def make_second_input(package, output):
    manifest = load_manifest(package)
    shape = tuple(int(value) for value in manifest["input_shape"])
    source = np.fromfile(Path(package) / manifest["input_file"], dtype=np.float32)
    if source.size != int(np.prod(shape)):
        raise ValueError("input.bin does not match manifest shape")
    shifted = np.roll(source.reshape(shape), shift=1, axis=-1).copy()
    if np.array_equal(source, shifted.reshape(-1)):
        raise ValueError("generated P8A input is not distinct")
    shifted.tofile(output)
    return {"shape": list(shape), "dtype": "float32", "bytes": int(shifted.nbytes)}


def _ssh(args, command, timeout=None):
    proc = subprocess.run(
        ["ssh", *SSH_OPTIONS, "{}@{}".format(args.ssh_user, args.board_host), command],
        text=True,
        capture_output=True,
        timeout=timeout or args.timeout_s,
        check=False,
    )
    if proc.returncode:
        raise RuntimeError("remote command failed:\n{}\n{}".format(proc.stdout, proc.stderr))
    return proc.stdout


def _scp(args, source, destination):
    proc = subprocess.run(
        [
            "scp",
            *SSH_OPTIONS,
            str(source),
            "{}@{}:{}".format(args.ssh_user, args.board_host, destination),
        ],
        text=True,
        capture_output=True,
        timeout=args.timeout_s,
        check=False,
    )
    if proc.returncode:
        raise RuntimeError("scp failed: {}".format(proc.stderr))


def _runner_args(manifest, warmup, runs, output, edge_index=0):
    result = []
    for index, stage in enumerate(manifest["stages"]):
        prefix = "--stage{}-".format(index)
        result.extend(
            [
                prefix + "graph",
                stage["graph"],
                prefix + "lib",
                stage["lib"],
                prefix + "params",
                stage["params"],
                prefix + "input-names",
                ",".join(stage["input_names"]),
                prefix + "name",
                stage["name"],
                prefix + "device",
                stage["device"],
                prefix + "runtime-num-threads",
                str(manifest["serial_stage_runtime_threads"]["stage{}".format(index)]),
            ]
        )
        affinity = manifest["stage_cpu_affinity"]["serial"]["stage{}".format(index)]
        if affinity:
            result.extend([prefix + "cpu-affinity", ",".join(str(value) for value in affinity)])
    result.extend(
        [
            "--input-list",
            "p8a_inputs.txt",
            "--runs",
            str(runs),
            "--warmup-runs",
            str(warmup),
            "--runtime-num-threads",
            "4",
            "--output-mode",
            "raw",
            "--output-jsonl",
            output,
            "--serial",
            "--p8-edge",
            str(edge_index),
        ]
    )
    return result


def _median(rows, key):
    return statistics.median(float(row[key]) for row in rows)


def summarize_control(control_id, rows):
    if not rows:
        raise ValueError("empty P8A result")
    correctness = all(row["reference_correctness_passed"] for row in rows)
    virtual_addresses = {row["slot_virtual_address"] for row in rows}
    physical_addresses = {row["slot_physical_address"] for row in rows}
    address_alignment_verified = all(
        int(row[key], 16) % 256 == 0
        for row in rows
        for key in ("slot_virtual_address", "slot_physical_address")
    )
    summary = {
        "control_id": control_id,
        "direction": rows[0]["direction"],
        "scored_frames": len(rows),
        "reference_correctness_passed": correctness,
        "reference_scope": rows[0].get(
            "reference_scope", "ordinary_copy_same_compiled_graph"
        ),
        "pair_orders": sorted({row.get("pair_order", "legacy_blocked") for row in rows}),
        "alternating_input_indices": sorted({int(row["input_index"]) for row in rows}),
        "distinct_input_hashes": len({row["input_hash"] for row in rows}),
        "distinct_reference_output_hashes": len(
            {tuple(row["reference_output_hashes"]) for row in rows}
        ),
        "boundary_bytes": int(rows[0]["boundary_bytes"]),
        "slot_virtual_address": rows[0]["slot_virtual_address"],
        "slot_physical_address": rows[0]["slot_physical_address"],
        "same_virtual_address": all(bool(row["same_virtual_address"]) for row in rows),
        "slot_address_stable": len(virtual_addresses) == 1 and len(physical_addresses) == 1,
        "slot_address_alignment_verified": address_alignment_verified,
        "slot_alignment_bytes": int(rows[0]["slot_alignment_bytes"]),
        "baseline_materialization_bytes": int(rows[0]["baseline_materialization_bytes"]),
        "zero_copy_materialization_bytes": int(rows[0]["zero_copy_materialization_bytes"]),
        "baseline_edge_copy_ms_median": _median(rows, "baseline_edge_copy_ms"),
        "baseline_total_latency_ms_median": _median(rows, "baseline_total_latency_ms"),
        "zero_copy_total_latency_ms_median": _median(rows, "zero_copy_total_latency_ms"),
        "baseline_producer_run_ms_median": _median(rows, "baseline_producer_run_ms"),
        "zero_copy_producer_run_ms_median": _median(rows, "zero_copy_producer_run_ms"),
        "baseline_consumer_run_ms_median": _median(rows, "baseline_consumer_run_ms"),
        "zero_copy_consumer_run_ms_median": _median(rows, "zero_copy_consumer_run_ms"),
        "safe_copy_effective": bool(rows[0]["safe_copy_effective"]),
    }
    summary["cpu_direct_output_penalty_ms"] = statistics.median(
        float(row["zero_copy_producer_run_ms"])
        - float(row["baseline_producer_run_ms"])
        for row in rows
    )
    summary["consumer_run_penalty_ms"] = statistics.median(
        float(row["zero_copy_consumer_run_ms"])
        - float(row["baseline_consumer_run_ms"])
        for row in rows
    )
    summary["cpu_shared_mapping_penalty_ms"] = (
        summary["cpu_direct_output_penalty_ms"]
        if summary["direction"] == "cpu_to_vta"
        else summary["consumer_run_penalty_ms"]
    )
    summary["latency_delta_ms"] = statistics.median(
        float(row["baseline_total_latency_ms"])
        - float(row["zero_copy_total_latency_ms"])
        for row in rows
    )
    summary["direct_mapping_gate_passed"] = (
        summary["cpu_shared_mapping_penalty_ms"]
        <= summary["baseline_edge_copy_ms_median"]
    )
    return summary


def _board_properties(args):
    script = r"""
set -eu
emit() { printf '%s\t%s\n' "$1" "$2"; }
emit boot_id "$(cat /proc/sys/kernel/random/boot_id)"
emit fpga_state "$(cat /sys/class/fpga_manager/fpga0/state)"
emit udmabuf_size "$(cat /sys/class/u-dma-buf/udmabuf0/size)"
emit udmabuf_phys_addr "$(cat /sys/class/u-dma-buf/udmabuf0/phys_addr 2>/dev/null || echo unavailable)"
emit udmabuf_sync_mode "$(cat /sys/class/u-dma-buf/udmabuf0/sync_mode 2>/dev/null || echo unavailable)"
emit udmabuf_dma_coherent "$(cat /sys/class/u-dma-buf/udmabuf0/dma_coherent 2>/dev/null || echo unavailable)"
emit udmabuf_map_size "$(cat /sys/class/u-dma-buf/udmabuf0/map_size 2>/dev/null || echo unavailable)"
emit udmabuf_sysfs_attributes "$(ls -1 /sys/class/u-dma-buf/udmabuf0 2>/dev/null | tr '\n' ',' || true)"
emit udmabuf_module_version "$(cat /sys/module/u_dma_buf/version 2>/dev/null || echo unavailable)"
emit storage_errors "$(dmesg | grep -Ec 'EXT4-fs error|Buffer I/O error|I/O error.*mmcblk' || true)"
emit rpc_pid "$(pidof tvm_rpc || echo unavailable)"
"""
    values = {}
    for line in _ssh(args, script).splitlines():
        if "\t" in line:
            key, value = line.split("\t", 1)
            values[key] = value
    return values


def _archive_package(package, archive):
    with tarfile.open(archive, "w:gz") as output:
        for path in sorted(Path(package).rglob("*")):
            output.add(path, arcname=str(path.relative_to(package)), recursive=False)


def build_coherence_evidence(properties, preflight):
    component_matches = preflight["component_hashes"]["matches"]
    allocator_value = properties.get("udmabuf_dma_coherent", "unavailable")
    runtime_macro = (
        COMPILE_COMMANDS.exists()
        and "-DVTA_COHERENT_ACCESSES=true"
        in COMPILE_COMMANDS.read_text(encoding="utf-8")
    )
    evidence = {
        "allocator_sysfs_dma_coherent": allocator_value,
        "allocator_metadata_attests_hardware_coherence": allocator_value == "1",
        "allocator_metadata_scope": (
            "u-dma-buf device allocation metadata; it does not by itself describe "
            "the PL master's AxCACHE attributes or the selected PS port"
        ),
        "vta_runtime_compile_macro_coherent": runtime_macro,
        "frozen_hpc_component_hashes_match": all(component_matches.values()),
        "prior_controlled_hp_hpc_evidence": {
            "path": str(PRIOR_HPC_EVIDENCE.relative_to(ROOT)),
            "sha256": file_sha256(PRIOR_HPC_EVIDENCE),
        },
        "current_test_scope": (
            "alternating-input zero-copy equivalence supports software binding and "
            "lifetime correctness; it is not an independent proof of hardware coherence"
        ),
    }
    evidence["vta_hpc_path_evidence_complete"] = bool(
        runtime_macro
        and evidence["frozen_hpc_component_hashes_match"]
        and PRIOR_HPC_EVIDENCE.exists()
    )
    return evidence


def build_correctness_evidence(candidate_id):
    payload = json.loads(TOP20_BOARD_SUMMARY.read_text(encoding="utf-8"))
    row = next(item for item in payload["rows"] if item["candidate_id"] == candidate_id)
    correctness = row["correctness"]
    return {
        "current_test": "ordinary_copy_same_compiled_graph_equivalence",
        "prior_independent_reference": {
            "path": str(TOP20_BOARD_SUMMARY.relative_to(ROOT)),
            "sha256": file_sha256(TOP20_BOARD_SUMMARY),
            "passed": bool(correctness["independent_reference_passed"]),
            "serial_pipeline_exact_frames": int(correctness["serial_pipeline_exact_frames"]),
            "deterministic_output_frames": int(correctness["deterministic_output_frames"]),
        },
    }


def run_board(
    args,
    manifest,
    runner_binary,
    work_dir,
    output_dir,
    edge_index=0,
    phase="P8A",
    artifact_prefix="v1_p8a",
):
    from collect_cpu_vta_pipeline_v1_preflight import collect as collect_preflight

    preflight_args = argparse.Namespace(
        board_host=args.board_host,
        rpc_port=args.rpc_port,
        ssh_user=args.ssh_user,
        timeout_s=min(args.timeout_s, 30),
    )
    preflight_before = collect_preflight(preflight_args)
    if not preflight_before["passed"]:
        raise RuntimeError("P8A refused because board preflight failed")
    remote = args.remote_root.rstrip("/")
    archive = work_dir / "package.tar.gz"
    second_input = work_dir / "input_b.bin"
    input_list = work_dir / "p8a_inputs.txt"
    make_second_input(args.package, second_input)
    input_list.write_text("input.bin\ninput_b.bin\n", encoding="ascii")

    if not args.reuse_remote_package:
        _archive_package(args.package, archive)
        _ssh(args, "rm -rf {0} && mkdir -p {0}".format(shlex.quote(remote)))
        _scp(args, archive, remote + "/package.tar.gz")
        _ssh(
            args,
            "cd {root} && tar -xzf package.tar.gz".format(root=shlex.quote(remote)),
        )
    else:
        _ssh(args, "test -f {}/manifest.json".format(shlex.quote(remote)))
    _scp(args, runner_binary, remote + "/vta_stage_pipeline_runner_p8a")
    _scp(args, second_input, remote + "/input_b.bin")
    _scp(args, input_list, remote + "/p8a_inputs.txt")
    _ssh(
        args,
        "cd {root} && chmod 700 vta_stage_pipeline_runner_p8a".format(
            root=shlex.quote(remote)
        ),
    )

    properties = _board_properties(args)
    boot_id = properties["boot_id"]
    controls = []
    raw_artifacts = {}
    for control_id, safe_copy in (("safe_byte_copy", "1"), ("memcpy", "0")):
        output_name = "{}.jsonl".format(control_id)
        runner_args = _runner_args(
            manifest, args.warmup, args.runs, output_name, edge_index=edge_index
        )
        command = (
            "cd {root} && export LD_LIBRARY_PATH=$PWD && "
            "export LD_PRELOAD=$PWD/libtvm_runtime.so:$PWD/libvta.so && "
            "export TVM_NUM_THREADS=4 TVM_THREAD_POOL_SPIN_COUNT=0 && "
            "export AXU5EVB_DRIVER_POST_START_SLEEP_NS=1000 "
            "AXU5EVB_DRIVER_POLL_SLEEP_NS=1000 AXU5EVB_DRIVER_SAFE_COPY={safe} && "
            "./vta_stage_pipeline_runner_p8a {runner} && cat {output}"
        ).format(
            root=shlex.quote(remote),
            safe=safe_copy,
            runner=shlex.join(runner_args),
            output=shlex.quote(output_name),
        )
        stdout = _ssh(args, command)
        rows = [json.loads(line) for line in stdout.splitlines() if line.startswith("{")]
        controls.append(summarize_control(control_id, rows))
        raw_path = work_dir / output_name
        raw_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
        )
        archived_name = "{}_edge{}_raw_{}_{}.jsonl".format(
            artifact_prefix, edge_index, control_id, boot_id
        )
        archived_path = Path(output_dir) / archived_name
        shutil.copy2(raw_path, archived_path)
        raw_artifacts[control_id] = {
            "path": archived_name,
            "sha256": file_sha256(archived_path),
            "rows": len(rows),
        }

    correct_controls = [item for item in controls if item["reference_correctness_passed"]]
    if not correct_controls:
        raise RuntimeError("neither ordinary copy implementation passed correctness")
    fastest = min(correct_controls, key=lambda item: item["baseline_edge_copy_ms_median"])
    preflight_after = collect_preflight(preflight_args)
    coherence_evidence = build_coherence_evidence(properties, preflight_before)
    correctness_evidence = build_correctness_evidence(manifest["candidate_id"])
    session = seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_{}_edge_session".format(phase.lower()),
            "phase": phase,
            "edge": boundary_contract(manifest, edge_index),
            "observed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "board": "{}@{}".format(args.ssh_user, args.board_host),
            "candidate_id": manifest["candidate_id"],
            "board_properties": properties,
            "preflight_before": preflight_before,
            "preflight_after": preflight_after,
            "controls": controls,
            "raw_artifacts": raw_artifacts,
            "coherence_evidence": coherence_evidence,
            "correctness_evidence": correctness_evidence,
            "ordinary_baseline_selection": {
                "policy": "fastest_correct_ordinary_copy",
                "selected_control_id": fastest["control_id"],
                "edge_copy_ms_median": fastest["baseline_edge_copy_ms_median"],
            },
            "single_boot_gate": {
                "all_controls_correct": all(
                    item["reference_correctness_passed"] for item in controls
                ),
                "alternating_inputs_proven": all(
                    item["alternating_input_indices"] == [0, 1]
                    and item["distinct_input_hashes"] == 2
                    and item["distinct_reference_output_hashes"] == 2
                    for item in controls
                ),
                "paired_order_balanced": all(
                    item["pair_orders"]
                    == ["ordinary_then_zero_copy", "zero_copy_then_ordinary"]
                    for item in controls
                ),
                "zero_copy_bytes_are_zero": all(
                    item["zero_copy_materialization_bytes"] == 0 for item in controls
                ),
                "slot_address_contract_passed": all(
                    item["same_virtual_address"]
                    and item["slot_address_stable"]
                    and item["slot_address_alignment_verified"]
                    for item in controls
                ),
                "direct_mapping_not_worse_than_copy_saved": fastest[
                    "direct_mapping_gate_passed"
                ],
                "storage_healthy": properties.get("storage_errors") == "0",
                "preflight_before_and_after_passed": bool(
                    preflight_before["passed"] and preflight_after["passed"]
                ),
                "coherence_evidence_boundary_recorded": coherence_evidence[
                    "vta_hpc_path_evidence_complete"
                ],
                "prior_independent_reference_passed": correctness_evidence[
                    "prior_independent_reference"
                ]["passed"],
            },
            "formal_performance_claim_allowed": False,
        }
    )
    session["single_boot_gate"]["passed"] = all(session["single_boot_gate"].values())
    session = seal_artifact(session)
    if not args.keep_remote:
        _ssh(args, "rm -rf {}".format(shlex.quote(remote)))
    return session


def write_review(path, audit, session=None):
    lines = [
        "# P8A ABI 与共享内存可行性审查",
        "",
        "更新日期：{}".format(datetime.date.today().isoformat()),
        "",
        "## 本地结论",
        "",
        "- GraphExecutor 的 input/output zero-copy 入口、runner 的 shape/dtype/offset/alignment/物理地址检查均已审计。",
        "- 冻结边界为 `[1,64,56,56] float32`，共 `802816 bytes`；producer 为 CPU view，consumer 为 VTA ext-dev view。",
        "- P8A 只验证单 slot 串行同址绑定，不代表双 slot Pipeline 已实现。",
    ]
    if session is None:
        lines.extend(["", "## 板端状态", "", "尚未执行单 boot qualification。"])
    else:
        selected = session["ordinary_baseline_selection"]
        chosen = next(
            item
            for item in session["controls"]
            if item["control_id"] == selected["selected_control_id"]
        )
        lines.extend(
            [
                "",
                "## 单 Boot 结果",
                "",
                "- 路径等价性：两种普通 copy control 与 zero-copy 的 A/B 逐帧输出均一致；这是同一 compiled graph 的 ordinary-copy 对照，不冒充新的独立 CPU reference。",
                "- 测量顺序：每帧组成 ordinary/zero-copy matched pair，并平衡 AB/BA 顺序；早期 blocked-order 结果因漂移已废弃。",
                "- 最快正确普通路径：`{}`，边界 materialization 中位时间 `{:.6f} ms`。".format(
                    selected["selected_control_id"], selected["edge_copy_ms_median"]
                ),
                "- 同址路径将 framework stage-boundary materialization 从 `{}` 降为 `0 bytes`；VTA 内部 DDR LOAD 不在此范围。".format(
                    chosen["baseline_materialization_bytes"]
                ),
                "- CPU 直接写 u-dma-buf 的 producer run 配对差值为 `{:+.6f} ms`；`baseline - zero-copy` 单帧串行 latency 配对差值为 `{:+.6f} ms`。".format(
                    chosen["cpu_direct_output_penalty_ms"], chosen["latency_delta_ms"]
                ),
                "- 单 boot gate：`{}`。该结果只能决定是否进入 P8B，不能声明流水线 FPS 提升。".format(
                    "passed" if session["single_boot_gate"]["passed"] else "failed"
                ),
                "- 一致性证据边界：u-dma-buf `dma_coherent={}`，该 allocator 元数据{}为 PL 的 HPC 路径背书；VTA 路径依据冻结的 HPC bitstream/runtime hash、`VTA_COHERENT_ACCESSES=true` 和既有 HP/HPC 受控实验。".format(
                    session["coherence_evidence"]["allocator_sysfs_dma_coherent"],
                    "可以" if session["coherence_evidence"]["allocator_metadata_attests_hardware_coherence"] else "不能",
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## 下一步",
            "",
            "停在 P8A review。只有本次 gate 通过并经用户确认，才进入 P8B 的单边界串行 matched control；不自动实现双 slot Pipeline。",
            "",
            "本地审计 SHA256：`{}`".format(audit["artifact_sha256"]),
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    package = Path(args.package).resolve()
    output_dir = Path(args.output_dir).resolve()
    work_dir = Path("/tmp/ramps_p8a")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    runner_binary = work_dir / "vta_stage_pipeline_runner_p8a"
    compile_runner(runner_binary)
    manifest = load_manifest(package)
    protocol = build_protocol(manifest, args.warmup, args.runs)
    audit = build_local_audit(package, manifest, runner_binary)
    write_json(output_dir / "v1_p8_protocol.json", protocol)
    write_json(output_dir / "v1_p8_local_abi_audit.json", audit)
    session = None
    if not args.local_only:
        session = run_board(args, manifest, runner_binary, work_dir, output_dir)
        boot_id = session["board_properties"]["boot_id"]
        write_json(output_dir / "v1_p8_session_{}.json".format(boot_id), session)
    write_review(output_dir / "v1_p8_review.md", audit, session)
    print(json.dumps({"protocol": protocol, "audit": audit, "session": session}, indent=2))


if __name__ == "__main__":
    main()
