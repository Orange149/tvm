#!/usr/bin/env python3
"""Execute the frozen C3-P6b mechanism correctness canary on a RAM-only VTA board.

The offline P6b contract remains immutable.  This runner revalidates its source
records, stable candidate identities, complete ConfigEntity payloads, and TIR
hashes before uploading cross-compiled modules to the existing RPC server.  It
collects no latency: each candidate must first match an independent NumPy oracle
for all three preregistered seeds.
"""

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


SCHEMA = "c3_p6c_ram_board_correctness_v1"
SUMMARY_SCHEMA = "c3_p6c_ram_board_correctness_summary_v1"
MODE_NUMBERS = {
    "original": 0,
    "input_stationary": 1,
    "weight_stationary_barrier": 4,
}
REQUIRED_COUNTERS = (
    "load_buffer_2d_calls",
    "load_buffer_2d_bytes",
    "load_buffer_2d_inp_bytes",
    "load_buffer_2d_wgt_bytes",
    "load_buffer_2d_small_calls",
    "load_buffer_2d_strided_calls",
    "load_buffer_2d_padded_calls",
    "store_buffer_2d_calls",
    "store_buffer_2d_bytes",
    "driver_run_insns",
    "synchronize_insns",
)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def semantic_config_key(value):
    return canonical_json_bytes(normalize_config_entity(value)).decode("utf-8")


def validate_contract(manifest_path, plan_path, repo_root):
    manifest_path = Path(manifest_path).resolve()
    plan_path = Path(plan_path).resolve()
    manifest = load_json(manifest_path)
    plan = load_json(plan_path)
    if manifest.get("schema") != "c3_p6b_mechanism_canary_manifest_v1":
        raise ValueError("unexpected mechanism manifest schema")
    if plan.get("schema") != "c3_p6b_mechanism_canary_dispatch_plan_v1":
        raise ValueError("unexpected dispatch plan schema")
    if manifest.get("board_executed") is not False or plan.get("board_executed") is not False:
        raise ValueError("P6b inputs are no longer an immutable offline contract")
    seeds = tuple(int(seed) for seed in plan["correctness"]["seeds"])
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("P6b must freeze exactly three unique correctness seeds")

    source_records = {}
    for source_name in ("P4b", "P4g"):
        source = manifest["source_runs"][source_name]
        run_dir = Path(source["run_dir"])
        if not run_dir.is_absolute():
            run_dir = repo_root / run_dir
        results_path = run_dir / "results.jsonl"
        if file_sha256(results_path) != source["results_sha256"]:
            raise ValueError("{} results hash mismatch".format(source_name))
        for record in load_jsonl(results_path):
            candidate_id = record.get("candidate_id")
            if candidate_id in source_records:
                continue
            source_records[candidate_id] = record

    selected = []
    seen = set()
    for workload_id in sorted(manifest["workloads"]):
        entries = manifest["workloads"][workload_id]
        if len(entries) != 3:
            raise ValueError("{} does not contain exactly three candidates".format(workload_id))
        for entry in entries:
            candidate_id = entry["candidate_id"]
            if candidate_id in seen:
                raise ValueError("duplicate selected candidate ID")
            seen.add(candidate_id)
            record = source_records.get(candidate_id)
            if not record:
                raise ValueError("selected candidate missing from P4b/P4g")
            identity = record["identity"]
            if hashlib.sha256(canonical_json_bytes(identity)).hexdigest() != candidate_id:
                raise ValueError("stable identity mismatch: {}".format(candidate_id))
            if identity["workload"] is None:
                raise ValueError("candidate workload is missing")
            expected = dict(identity["complete_config_entity"])
            expected["index"] = int(record["debug"]["config_index"])
            if semantic_config_key(expected) != semantic_config_key(entry["complete_config_entity"]):
                raise ValueError("manifest/source ConfigEntity mismatch: {}".format(candidate_id))
            if record["tir_sha256"] != entry["tir_sha256"]:
                raise ValueError("manifest/source TIR mismatch: {}".format(candidate_id))
            if entry["residence_mode"] not in MODE_NUMBERS:
                raise ValueError("unsupported residence mode")
            selected.append((entry, record))
    if len(selected) != 9:
        raise ValueError("P6b must contain exactly nine candidates")
    return manifest, plan, seeds, selected


def validate_source_guards(entry):
    observed = {}
    for name, expected in sorted(entry["qualification_source_guards_sha256"].items()):
        path = Path(name)
        if not path.is_file():
            raise ValueError("source guard is missing: {}".format(path))
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError("source guard changed: {}".format(path))
        observed[str(path)] = actual
    return observed


def make_task(entry, record, env):
    workload = record["identity"]["workload"]
    mode = entry["residence_mode"]
    # P4e/P4f qualified every mode through the isolated residency template.
    # Mode 0 delegates to the original schedule while retaining the protected
    # incumbent's historical identity/template label and ConfigSpace index.
    name = "conv2d_packed_residency.vta"
    args = tuple(workload[1:]) + (MODE_NUMBERS[mode],)
    expected_label = "conv2d_packed.vta" if mode == "original" else name
    if expected_label != entry["template_name"]:
        raise ValueError("template identity label mismatch: {}".format(entry["candidate_id"]))
    return autotvm.task.create(
        name,
        args=args,
        target=env.target,
        target_host=env.target_host,
    )


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


def build_candidate(entry, record, env, binary_dir):
    validate_source_guards(entry)
    task = make_task(entry, record, env)
    config = task.config_space.get(int(entry["config_index"]))
    if semantic_config_key(config.to_json_dict()) != semantic_config_key(
        entry["complete_config_entity"]
    ):
        raise RuntimeError("current ConfigSpace no longer matches frozen ConfigEntity")
    with task.target:
        schedule, tensors = task.instantiate(config)
    with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
        lowered = tvm.lower(schedule, tensors, name="main")
    tir_hash = hashlib.sha256(tvm.ir.save_json(lowered).encode("utf-8")).hexdigest()
    if tir_hash != entry["tir_sha256"]:
        raise RuntimeError("re-lowered TIR hash differs from frozen certificate")
    with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
        module = vta.build(
            schedule,
            tensors,
            target=tvm.target.Target(env.target, host=env.target_host),
            name="main",
        )
    binary = binary_dir / (entry["candidate_id"] + ".so")
    module.export_library(
        str(binary),
        fcompile=cc.cross_compiler("aarch64-xilinx-linux-g++"),
        options=cross_options(),
    )
    return binary, tir_hash


def ssh_preflight(host, expected_boot_id, dmesg_after):
    command = (
        "boot=$(cat /proc/sys/kernel/random/boot_id); "
        "state=$(cat /sys/class/fpga_manager/fpga0/state); "
        "size=$(cat /sys/class/u-dma-buf/udmabuf0/size); "
        "mountline=$(mount | grep 'mmcblk1p2.*(ro,'); "
        "rpcpid=$(pidof tvm_rpc); "
        "echo BOOT=$boot; echo FPGA=$state; echo SIZE=$size; "
        "echo RPC=$rpcpid; echo SD_RO=$mountline; "
        "dmesg | awk '$1 ~ /^\\[/ {t=$1; gsub(/\\[/,\"\",t); "
        "if ((t+0)>%d && ($0 ~ /EXT4-fs|mmcblk.*error|I.O error|Buffer I.O/)) print}'"
    ) % int(dmesg_after)
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
        timeout=15,
    )
    lines = completed.stdout.splitlines()
    fields = {}
    extra = []
    for line in lines:
        if "=" in line and line.split("=", 1)[0] in {"BOOT", "FPGA", "SIZE", "RPC", "SD_RO"}:
            key, value = line.split("=", 1)
            fields[key] = value
        elif line.strip():
            extra.append(line)
    if fields.get("BOOT") != expected_boot_id:
        raise RuntimeError("board boot ID changed")
    if fields.get("FPGA") != "operating":
        raise RuntimeError("FPGA is not operating")
    if fields.get("SIZE") != "201326592":
        raise RuntimeError("unexpected u-dma-buf size")
    if not fields.get("RPC"):
        raise RuntimeError("RAM RPC server is not running")
    if "ro," not in fields.get("SD_RO", ""):
        raise RuntimeError("damaged SD ext4 partition is not read-only")
    if extra:
        raise RuntimeError("new storage error after RAM switchover: {}".format(extra[-1]))
    return fields


def run_candidate(entry, record, seeds, remote, clear, status, env, binary_dir):
    result = {
        "schema": SCHEMA,
        "candidate_id": entry["candidate_id"],
        "workload_id": entry["workload_id"],
        "mechanism_role": entry["mechanism_role"],
        "residence_mode": entry["residence_mode"],
        "config_index": entry["config_index"],
        "build": "not_attempted",
        "seeds": [],
        "overall_status": "not_attempted",
        "performance_timing": "not_collected",
        "failure": None,
    }
    started = time.monotonic()
    try:
        binary, tir_hash = build_candidate(entry, record, env, binary_dir)
        result.update(
            build="passed",
            relowered_tir_sha256=tir_hash,
            binary_sha256=file_sha256(binary),
            binary_size_bytes=binary.stat().st_size,
        )
        remote.upload(str(binary))
        module = remote.load_module(binary.name)
        device = remote.ext_dev(0)
        function = module["main"]
        workload = record["identity"]["workload"]
        for seed in seeds:
            data, weight, expected = reference_data_from_workload(workload, seed)
            buffers = [
                tvm.nd.array(data, device),
                tvm.nd.array(weight, device),
                tvm.nd.empty(expected.shape, "int8", device),
            ]
            clear()
            function(*buffers)
            actual = buffers[-1].numpy()
            profile = json.loads(status())
            missing = [name for name in REQUIRED_COUNTERS if name not in profile]
            if missing:
                raise RuntimeError("runtime profile lacks counters: {}".format(missing))
            mismatches = int(np.count_nonzero(actual != expected))
            seed_result = {
                "seed": int(seed),
                "correct": mismatches == 0,
                "mismatch_count": mismatches,
                "max_absolute_error": int(
                    np.max(np.abs(actual.astype("int32") - expected.astype("int32")))
                ),
                "expected_sha256": array_sha256(expected),
                "actual_sha256": array_sha256(actual),
                "runtime_profile": {name: profile[name] for name in REQUIRED_COUNTERS},
            }
            result["seeds"].append(seed_result)
            if mismatches:
                raise AssertionError("{} output elements differ".format(mismatches))
        result["overall_status"] = "passed"
    except Exception as error:  # Keep the failure record even if RPC/device execution fails.
        result["overall_status"] = "failed"
        result["failure"] = {
            "exception_type": type(error).__name__,
            "message": (str(error) or type(error).__name__)[:4000],
            "traceback": traceback.format_exc()[-12000:],
        }
    result["elapsed_seconds"] = time.monotonic() - started
    return result


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dispatch-plan", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--expected-boot-id", required=True)
    parser.add_argument("--dmesg-after", type=int, default=467)
    parser.add_argument("--candidate-id", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "results.jsonl"
    if results_path.exists():
        raise FileExistsError("refusing to overwrite existing results: {}".format(results_path))
    manifest, plan, seeds, selected = validate_contract(
        args.manifest, args.dispatch_plan, repo_root
    )
    requested = set(args.candidate_id)
    if requested:
        unknown = requested - {entry["candidate_id"] for entry, _ in selected}
        if unknown:
            raise ValueError("unknown requested candidate IDs: {}".format(sorted(unknown)))
        selected = [(entry, record) for entry, record in selected if entry["candidate_id"] in requested]

    env = vta.get_env()
    if env.TARGET != "axu5evb":
        raise RuntimeError("VTA TARGET must be axu5evb")
    preflight = ssh_preflight(args.host, args.expected_boot_id, args.dmesg_after)
    if args.dry_run:
        for entry, record in selected:
            validate_source_guards(entry)
            task = make_task(entry, record, env)
            config = task.config_space.get(int(entry["config_index"]))
            if semantic_config_key(config.to_json_dict()) != semantic_config_key(
                entry["complete_config_entity"]
            ):
                raise RuntimeError("ConfigEntity dry-run mismatch")
        write_json(
            output / "dry_run.json",
            {
                "status": "passed",
                "candidate_count": len(selected),
                "board_preflight": preflight,
                "performance_timing": "not_collected",
            },
        )
        print("dry_run=passed candidates={}".format(len(selected)))
        return

    remote = rpc.connect(args.host, args.port, session_timeout=120)
    clear = remote.get_function("vta.runtime.profiler_clear")
    status = remote.get_function("vta.runtime.profiler_status")
    records = []
    with tempfile.TemporaryDirectory(prefix="c3_p6c_board_") as temporary:
        binary_dir = Path(temporary)
        with results_path.open("x", encoding="utf-8") as stream:
            for entry, record in selected:
                ssh_preflight(args.host, args.expected_boot_id, args.dmesg_after)
                result = run_candidate(
                    entry, record, seeds, remote, clear, status, env, binary_dir
                )
                stream.write(json.dumps(result, sort_keys=True) + "\n")
                stream.flush()
                records.append(result)
                print(
                    "candidate={} workload={} mode={} status={}".format(
                        entry["candidate_id"][:12],
                        entry["workload_id"],
                        entry["residence_mode"],
                        result["overall_status"],
                    ),
                    flush=True,
                )
                if result["overall_status"] != "passed":
                    break
    final_preflight = ssh_preflight(args.host, args.expected_boot_id, args.dmesg_after)
    passed = sum(row["overall_status"] == "passed" for row in records)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "passed" if passed == len(selected) else "failed_or_incomplete",
        "selected_candidates": len(selected),
        "executed_candidates": len(records),
        "passed_all_three_seeds": passed,
        "failed": sum(row["overall_status"] == "failed" for row in records),
        "correctness_seeds": list(seeds),
        "board_host": args.host,
        "board_boot_id": args.expected_boot_id,
        "board_preflight_before": preflight,
        "board_preflight_after": final_preflight,
        "ram_only_execution": True,
        "performance_timing": "not_collected",
        "source_manifest_sha256": file_sha256(args.manifest),
        "source_dispatch_plan_sha256": file_sha256(args.dispatch_plan),
    }
    write_json(output / "summary.json", summary)
    print(
        "summary={} passed={}/{}".format(summary["status"], passed, len(selected)),
        flush=True,
    )
    if summary["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
