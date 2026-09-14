#!/usr/bin/env python3
"""Execute a frozen single-workload full-pool correctness/timing confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
from tvm import rpc
import tvm
import vta

import vta.top.vta_conv2d_residency  # pylint: disable=unused-import
import replay_vta_equal_budget_search as equal_budget_search
from prepare_vta_p7r120_yolo_rpc_contract import build_and_export, write_json
from replay_vta_equal_budget_search import (
    POLICIES,
    RESOURCE_KEYS,
    protocol as search_protocol,
    render_results,
    run_offline,
    validate_pool,
)
from run_vta_p7r120_yolo_rpc_board import load_ephemeral, runtime_inventory
from qualify_vta_residency_fsim import array_sha256, reference_data_from_workload


HERE = Path(__file__).resolve().parent
ROOT = HERE / "report_out" / "stage_tile_cotuning" / "c3_dma_residency_autotune"
P7 = ROOT / "07_grouped_holdout"
DEFAULT_CONTRACT = P7 / "20260912_p7r132_y00_search_confirmation_contract_run02"
DEFAULT_OUTPUT = P7 / "20260912_p7r133_y00_search_confirmation_board_run01"
SEEDS = (0, 20250901, 20260910)
ROUNDS = 7
BUDGETS = (1, 2, 4, 8)
EXPECTED_DEFAULT_RUNTIME_HASHES = {
    "tvm_rpc": "20bbd3d896d124dd1127343f020c957fbb6034423af83433c2429955b6b316fa",
    "libtvm_runtime.so": "04e894baf305311315fd0c79d03ec2a6375b9591916e9c6f429cf3697c4457b2",
    "libvta.so": "eedfabb0630d58bf2eabaef9b4f2d2e4bfcdfa52a70503090f5608e79404ee5d",
    "start_axu5evb_cpp_rpc.sh": "2f42f2d37d5199fa2b66336b61869d2015f8ec5f0264bc57307789b4c6a5b194",
}


class CleanStartBoard:
    """Narrow SSH control used only for the pre-RPC clean-start gate."""

    def __init__(self, host, known_hosts):
        self.host = host
        self.options = [
            "-o", "ConnectTimeout=8",
            "-o", "HostKeyAlgorithms=+ssh-rsa",
            "-o", "PubkeyAcceptedAlgorithms=+ssh-rsa",
            "-o", "UserKnownHostsFile=" + str(Path(known_hosts).resolve()),
            "-o", "StrictHostKeyChecking=yes",
            "-o", "PreferredAuthentications=publickey,password",
            "-o", "PubkeyAuthentication=yes",
            "-o", "NumberOfPasswordPrompts=1",
        ]

    def run(self, command, timeout=60):
        return subprocess.run(
            ["ssh", *self.options, "root@" + self.host, command], check=True,
            text=True, capture_output=True, timeout=timeout, env=os.environ.copy(),
        )

    def state(self):
        command = r'''
echo BOOT=$(cat /proc/sys/kernel/random/boot_id)
echo FPGA=$(cat /sys/class/fpga_manager/fpga0/state)
echo UDMABUF=$(cat /sys/class/u-dma-buf/udmabuf0/size)
pid=$(pidof tvm_rpc | cut -d' ' -f1)
echo RPC_PID=$pid
echo RPC_CWD=$(readlink /proc/$pid/cwd)
dmesg | grep -Ei 'EXT4-fs.*(error|warning)|mmc.*(error|timeout|I/O)' | sed 's/^/STORAGE_ERROR=/'
'''
        fields, errors = {}, []
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
        if not re.fullmatch(r"[A-Za-z0-9_./-]+", directory):
            raise ValueError("unsafe runtime directory")
        stdout = self.run(
            "sha256sum " + " ".join(
                directory.rstrip("/") + "/" + name for name in EXPECTED_DEFAULT_RUNTIME_HASHES
            )
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
'''
        return dict(
            line.split("=", 1) for line in self.run(command).stdout.splitlines() if "=" in line
        )

    def stop_rpc(self, expected):
        current = self.rpc_state()
        if current != expected:
            raise RuntimeError("refusing to stop changed RPC")
        pid = int(current["PID"])
        self.run(
            "kill {0}; i=0; while kill -0 {0} 2>/dev/null; do "
            "i=$((i+1)); [ $i -lt 50 ] || exit 9; "
            "usleep 100000 2>/dev/null || sleep 1; done".format(pid)
        )

    def start_rpc(self, directory, log_name, environment=None):
        if not re.fullmatch(r"[A-Za-z0-9_./-]+", directory + log_name):
            raise ValueError("unsafe RPC path")
        environment = environment or {}
        for key in environment:
            if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
                raise ValueError("unsafe RPC environment key: " + key)
        env_prefix = ""
        if environment:
            env_prefix = "env " + " ".join(
                shlex.quote(key + "=" + str(value))
                for key, value in sorted(environment.items())
            ) + " "
        directory = directory.rstrip("/")
        command = (
            "cd {0}; : > {0}/{1}; nohup {2}./start_axu5evb_cpp_rpc.sh {0} 9090 "
            "> {0}/{1} 2>&1 < /dev/null & echo $!"
        ).format(directory, log_name, env_prefix)
        pid = int(self.run(command).stdout.strip().splitlines()[-1])
        for _ in range(40):
            time.sleep(0.1)
            state = self.rpc_state()
            if state.get("PID") == str(pid) and state.get("CWD") == directory:
                return state
        raise RuntimeError("fresh default RPC failed to start")

    def reload_frozen_bitstream(self):
        command = (
            "test $(sha256sum /lib/firmware/vta_hpc.bit | cut -d' ' -f1) = "
            "7bf1ac95b1182c670cd25241111f33725b4ea37ed6d66f2df37b0b2e7df528d6; "
            "echo vta_hpc.bit > /sys/class/fpga_manager/fpga0/firmware; "
            "test $(cat /sys/class/fpga_manager/fpga0/state) = operating; "
            "cat /sys/class/fpga_manager/fpga0/state"
        )
        if self.run(command).stdout.strip() != "operating":
            raise RuntimeError("frozen bitstream reload failed")
        return {
            "firmware": "vta_hpc.bit",
            "sha256": "7bf1ac95b1182c670cd25241111f33725b4ea37ed6d66f2df37b0b2e7df528d6",
            "state": "operating",
        }


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def zero_resource_cost():
    return {name: 0.0 for name in RESOURCE_KEYS}


def profile_resource_cost(profile):
    """Logical VTA/u-dma-buf traffic, not physical AXI burst accounting."""
    if not isinstance(profile, dict):
        return None
    required = (
        "load_buffer_2d_bytes", "store_buffer_2d_bytes",
        "load_buffer_2d_calls", "store_buffer_2d_calls",
    )
    if any(name not in profile for name in required):
        return None
    return {
        "logical_vta_load_bytes": float(profile["load_buffer_2d_bytes"]),
        "logical_vta_store_bytes": float(profile["store_buffer_2d_bytes"]),
        "logical_vta_dma_calls": float(
            profile["load_buffer_2d_calls"] + profile["store_buffer_2d_calls"]
        ),
        "fpga_kernel_invocations": float(profile.get("driver_run_calls", 1)),
    }


def aggregate_row_resources(rows):
    costs = [profile_resource_cost(row.get("runtime_profile_complete")) for row in rows]
    if not costs or any(cost is None for cost in costs):
        return None
    return {name: sum(cost[name] for cost in costs) for name in RESOURCE_KEYS}


def representative_timing_cost(rows):
    costs = [profile_resource_cost(row.get("runtime_profile_complete")) for row in rows]
    walls = [row.get("host_wall_ms") for row in rows]
    if not costs or any(cost is None for cost in costs) or any(wall is None for wall in walls):
        return {"wall_ms": None, "resource_cost": None}
    return {
        "wall_ms": statistics.median(float(wall) for wall in walls),
        "resource_cost": {
            name: statistics.median(cost[name] for cost in costs) for name in RESOURCE_KEYS
        },
    }


def allocate_reusable_buffers(device, workload):
    data, weight, expected = reference_data_from_workload(workload, 0)
    # Preserve the allocation layout used by the frozen FSim and fresh-buffer
    # FPGA qualification path: output, then input, then weight.  The AXU5EVB
    # backend is a non-freeing physical bump allocator, so allocation order is
    # part of the concrete shared-memory deployment state and must not change
    # silently when converting the harness to buffer reuse.
    output = tvm.nd.empty(expected.shape, "int8", device)
    remote_data = tvm.nd.array(data, device)
    remote_weight = tvm.nd.array(weight, device)
    return {
        "data": remote_data,
        "weight": remote_weight,
        "output": output,
        "allocation_order": ["output", "data", "weight"],
        "data_shape": tuple(data.shape),
        "weight_shape": tuple(weight.shape),
        "output_shape": tuple(expected.shape),
    }


def prepare_reused_buffers(buffers, workload, seed):
    data, weight, expected = reference_data_from_workload(workload, int(seed))
    if tuple(data.shape) != buffers["data_shape"] or tuple(weight.shape) != buffers["weight_shape"] \
            or tuple(expected.shape) != buffers["output_shape"]:
        raise RuntimeError("workload shape changed inside a reusable-buffer pool")
    buffers["data"].copyfrom(data)
    buffers["weight"].copyfrom(weight)
    buffers["output"].copyfrom(np.full(expected.shape, -113, dtype="int8"))
    return expected


def profiled_reused_call(function, buffers, workload, seed, functions):
    expected = prepare_reused_buffers(buffers, workload, seed)
    clear = functions.get("vta.runtime.profiler_clear")
    if clear is None:
        raise RuntimeError("profiler_clear is required for per-call complete profiles")
    clear()
    try:
        function(buffers["data"], buffers["weight"], buffers["output"])
        actual = buffers["output"].numpy()
        mismatch = int(np.count_nonzero(actual != expected))
        row = {
            "seed": int(seed),
            "correct": mismatch == 0,
            "status": "passed" if mismatch == 0 else "wrong_answer",
            "mismatch_count": mismatch,
            "expected_sha256": array_sha256(expected),
            "actual_sha256": array_sha256(actual),
        }
    except Exception as error:
        row = {
            "seed": int(seed),
            "correct": False,
            "status": "execution_failed",
            "exception_type": type(error).__name__,
            "message": (str(error) or type(error).__name__)[:4000],
        }
    try:
        row["runtime_profile_complete"] = json.loads(
            functions["vta.runtime.profiler_status"]()
        )
    except Exception as error:
        row["runtime_profile_error"] = (str(error) or type(error).__name__)[:2000]
    return row


def timed_reused_call(module, device, buffers, workload, seed, functions):
    expected = prepare_reused_buffers(buffers, workload, seed)
    clear = functions.get("vta.runtime.profiler_clear")
    if clear is None:
        raise RuntimeError("profiler_clear is required for per-call complete profiles")
    clear()
    timer = module.time_evaluator("main", device, number=1, repeat=1)
    measured = timer(buffers["data"], buffers["weight"], buffers["output"])
    latency_ms = float(measured.mean * 1000.0)
    actual = buffers["output"].numpy()
    mismatch = int(np.count_nonzero(actual != expected))
    profile = json.loads(functions["vta.runtime.profiler_status"]())
    row = {
        "seed": int(seed),
        "number": 1,
        "repeat": 1,
        "time_evaluator_total_kernel_invocations": 2,
        "latency_ms": latency_ms,
        "correct": mismatch == 0,
        "mismatch_count": mismatch,
        "expected_sha256": array_sha256(expected),
        "actual_sha256": array_sha256(actual),
        "runtime_profile_complete": profile,
        "remote_buffer_policy": "one_shape_compatible_set_reused",
    }
    if not np.isfinite(latency_ms) or latency_ms <= 0:
        raise RuntimeError("invalid timed result")
    if mismatch:
        raise RuntimeError("timed invocation wrong answer: {} mismatches".format(mismatch))
    return row


def verify_run(directory):
    directory = Path(directory)
    ledger_path = directory / "artifact_hashes.json"
    ledger = read_json(ledger_path)
    entries = ledger.get("artifacts", ledger.get("files"))
    if not isinstance(entries, dict):
        raise ValueError("unknown artifact ledger schema")
    for name, expected in entries.items():
        if sha256(directory / name) != expected:
            raise ValueError("artifact hash mismatch: " + str(directory / name))
    return sha256(ledger_path)


def qualification_costs(contract_dir, candidate_ids, workload_id="Y00"):
    bindings = read_json(Path(contract_dir) / "input_bindings.json")
    local = bindings["local_v2"]
    local_dir = Path(local["path"])
    if verify_run(local_dir) != local["ledger_sha256"]:
        raise ValueError("P7R127 local qualification ledger binding mismatch")
    static_rows = read_jsonl(local_dir / "static_results.jsonl")
    fsim_rows = read_jsonl(local_dir / "fsim_results.jsonl")
    static_by_id = {row["candidate_id"]: row for row in static_rows}
    fsim_by_id = {row["candidate_id"]: row for row in fsim_rows}
    per_candidate = {}
    for candidate_id in candidate_ids:
        static = static_by_id[candidate_id]
        fsim = fsim_by_id[candidate_id]
        if static["status"] != "ok" or fsim["status"] != "passed":
            raise ValueError("board pool contains a non-qualified identity")
        per_candidate[candidate_id] = {
            "lower_wall_ms": float(static["diagnostic_wall_seconds"]) * 1000.0,
            "fsim_wall_ms": float(fsim["diagnostic_wall_seconds"]) * 1000.0,
        }
    preprocessing = {
        "scope": "common {}-identity {} qualification before the {}-candidate board search pool".format(
            len(static_rows), workload_id, len(candidate_ids)
        ),
        "static_identity_count": len(static_rows),
        "static_pass_count": sum(row["status"] == "ok" for row in static_rows),
        "fsim_pass_count": sum(row["status"] == "passed" for row in fsim_rows),
        "static_known_wall_ms": 1000.0 * sum(
            float(row["diagnostic_wall_seconds"]) for row in static_rows
            if row.get("diagnostic_wall_seconds") is not None
        ),
        "fsim_known_wall_ms": 1000.0 * sum(
            float(row["diagnostic_wall_seconds"]) for row in fsim_rows
            if row.get("diagnostic_wall_seconds") is not None
        ),
        "included_in_equal_budget_replay": False,
        "interpretation": "reported as common prequalification overhead; replay compares ordering inside the frozen FSim-pass pool",
    }
    return per_candidate, preprocessing


def finalize(output):
    artifacts = {
        path.name: sha256(path) for path in sorted(Path(output).iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    write_json(Path(output) / "artifact_hashes.json", {
        "artifacts": artifacts,
        "source_sha256": {
            str(Path(__file__).resolve()): sha256(__file__),
            str(Path(equal_budget_search.__file__).resolve()): sha256(equal_budget_search.__file__),
        },
    })


def timing_orders(candidate_ids, rounds=ROUNDS):
    ids = list(candidate_ids)
    if not ids:
        raise ValueError("cannot time an empty candidate set")
    orders = []
    for round_index in range(rounds):
        offset = round_index % len(ids)
        order = ids[offset:] + ids[:offset]
        if round_index % 2:
            order = list(reversed(order))
        orders.append(order)
    return orders


def quartiles(values):
    return {
        "values_ms": list(values),
        "median_ms": statistics.median(values),
        "q1_ms": float(np.percentile(values, 25)),
        "q3_ms": float(np.percentile(values, 75)),
        "iqr_ms": float(np.percentile(values, 75) - np.percentile(values, 25)),
    }


def summarize_timing(rows, correct_ids):
    by_id = {candidate_id: [] for candidate_id in correct_ids}
    for row in rows:
        by_id[row["candidate_id"]].append(float(row["latency_ms"]))
    if any(len(values) != ROUNDS for values in by_id.values()):
        raise ValueError("each correct identity must have seven timing observations")
    stats = {candidate_id: quartiles(values) for candidate_id, values in by_id.items()}
    oracle_id = min(stats, key=lambda candidate_id: (stats[candidate_id]["median_ms"], candidate_id))
    return {
        "candidate_stats": stats,
        "pool_oracle_candidate_id": oracle_id,
        "pool_oracle_latency_ms": stats[oracle_id]["median_ms"],
    }


def completed_pool(prospective_pool, correctness_rows, timing_summary,
                   qualification=None, compile_walls=None, timing_rows=None):
    pool = json.loads(json.dumps(prospective_pool))
    correct_by_id = {row["candidate_id"]: row for row in correctness_rows}
    stats = timing_summary["candidate_stats"]
    qualification = qualification or {}
    compile_walls = compile_walls or {}
    timing_rows = timing_rows or []
    if len(pool["workloads"]) != 1:
        raise ValueError("board confirmation requires exactly one workload")
    workload_id = next(iter(pool["workloads"]))
    for candidate in pool["workloads"][workload_id]["candidates"]:
        candidate_id = candidate["candidate_id"]
        correctness = correct_by_id[candidate_id]
        passed = correctness["status"] == "passed"
        candidate_qualification = qualification.get(candidate_id, {})
        correctness_walls = [row.get("host_wall_ms") for row in correctness["seeds"]]
        fpga_wall = None if any(value is None for value in correctness_walls) else sum(
            float(value) for value in correctness_walls
        )
        fpga_resource = aggregate_row_resources(correctness["seeds"])
        measure_cost = representative_timing_cost(
            [row for row in timing_rows if row["candidate_id"] == candidate_id]
        ) if passed else {"wall_ms": None, "resource_cost": None}
        phases = {
            "lower": {
                "status": "ok", "wall_ms": candidate_qualification.get("lower_wall_ms"),
                "resource_cost": zero_resource_cost(),
            },
            "fsim": {
                "status": "ok", "wall_ms": candidate_qualification.get("fsim_wall_ms"),
                "resource_cost": zero_resource_cost(),
            },
            "compile": {
                "status": "ok", "wall_ms": compile_walls.get(candidate_id),
                "resource_cost": zero_resource_cost(),
            },
            "fpga": {
                "status": "ok" if passed else "invalid", "wall_ms": fpga_wall,
                "resource_cost": fpga_resource,
            },
            "measure": {
                "status": "ok" if passed else "not_run",
                "wall_ms": measure_cost["wall_ms"] if passed else None,
                "resource_cost": measure_cost["resource_cost"] if passed else None,
            },
        }
        candidate["oracle"] = {
            "phases": phases,
            "latency_ms": stats[candidate_id]["median_ms"] if passed else None,
            "delta_t_ms": None,
            "correctness_seed_checks": len(correctness["seeds"]),
        }
    if str(pool.get("claim_status", "")).startswith("recovery_"):
        pool["claim_status"] = "recovery_{}_latency_confirmation_labels_complete".format(
            workload_id.lower()
        )
    else:
        pool["claim_status"] = "prospective_{}_full_pool_labels_complete".format(
            workload_id.lower()
        )
    validate_pool(pool, require_oracle=True)
    return pool


def run(args):
    contract_dir = Path(args.contract_dir)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable P7R133 output " + str(output))
    ledger_sha = verify_run(contract_dir)
    contract = read_json(contract_dir / "board_collection_contract.json")
    workload_id = contract["workload_id"]
    pool = read_json(contract_dir / "prospective_pool.json")
    validate_pool(pool, require_oracle=False)
    expected_status = "frozen_before_{}_fpga_correctness_or_latency".format(workload_id.lower())
    recovery_status = "frozen_before_{}_latency_recovery_confirmation".format(workload_id.lower())
    if contract.get("status") not in (expected_status, recovery_status):
        raise ValueError("search confirmation contract is not prospective")
    candidates = contract["candidates"]
    order = contract["candidate_order_for_unbiased_full_label_collection"]
    if not candidates or set(order) != {row["candidate_id"] for row in candidates}:
        raise ValueError("expected a non-empty frozen complete board pool")
    by_id = {row["candidate_id"]: row for row in candidates}
    health = contract["health_canary"]
    execution_contract = {
        "schema": "c3_{}_search_confirmation_board_v2".format(workload_id.lower()),
        "status": "frozen_before_current_board_observation",
        "p7r132": {
            "path": str(contract_dir.resolve()),
            "artifact_ledger_sha256": ledger_sha,
        },
        "executor_source": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(__file__),
        },
        "health_canary": health,
        "candidate_order": order,
        "seeds": list(SEEDS),
        "timing_rounds": ROUNDS,
        "search_budgets": list(BUDGETS),
        "search_policies": list(POLICIES),
        "rpc": {"host": args.host, "port": args.port, "session_timeout": args.session_timeout},
        "clean_board_start": {
            "enabled": bool(args.reset_bitstream_before_rpc),
            "operation": (
                "stop exact default RPC, reload frozen bitstream, start fresh default RPC before "
                "the first health/tensor allocation"
                if args.reset_bitstream_before_rpc else "not requested"
            ),
        },
        "failure_policy": "health fail stops before {}; candidate first error rejects identity; any timed mismatch stops".format(workload_id),
        "shared_memory_buffer_policy": {
            "reuse": "one shape-compatible output/input/weight set per workload",
            "allocation_order": ["output", "data", "weight"],
            "reason": "match frozen qualification layout while avoiding non-freeing u-dma-buf exhaustion",
        },
        "persistent_board_write": False,
    }
    output.mkdir(parents=True)
    write_json(output / "contract.json", execution_contract)
    write_json(output / "pre_observation_hashes.json", {
        "artifacts": {"contract.json": sha256(output / "contract.json")}
    })

    temporary = None
    correctness_rows = []
    timing_rows = []
    health_rows = []
    try:
        env = vta.get_env()
        if env.TARGET != "axu5evb":
            raise RuntimeError("P7R133 requires TARGET=axu5evb")
        temporary = tempfile.TemporaryDirectory(prefix="c3_p7r133_cross_")
        directory = Path(temporary.name)
        qualification, preprocessing = qualification_costs(contract_dir, order, workload_id)
        write_json(output / "prequalification_costs.json", preprocessing)
        started = time.perf_counter()
        health_certificate = build_and_export(health, env, directory)
        health_certificate["cross_compile_wall_ms"] = (time.perf_counter() - started) * 1000.0
        certificates = {"health_canary": health_certificate}
        for candidate in candidates:
            expected_tir = contract["qualification_bindings"][candidate["candidate_id"]]["tir_sha256"]
            started = time.perf_counter()
            certificate = build_and_export(candidate, env, directory, expected_tir)
            certificate["cross_compile_wall_ms"] = (time.perf_counter() - started) * 1000.0
            certificates[candidate["candidate_id"]] = certificate
        write_json(output / "fresh_cross_certificates.json", certificates)
        write_json(output / "pre_rpc_hashes.json", {"artifacts": {
            "contract.json": sha256(output / "contract.json"),
            "fresh_cross_certificates.json": sha256(output / "fresh_cross_certificates.json"),
        }})
        if args.prepare_only:
            write_json(output / "summary.json", {
                "schema": execution_contract["schema"],
                "status": "cross_compile_only_complete_no_board_contact",
                "candidate_count": len(order),
                "health_canary_cross_compiled": True,
                "candidate_tir_bound_to_p7r132": len(certificates) - 1,
                "board_contacted": False,
                "performance_labels_collected": False,
            })
            (output / "command.txt").write_text(
                " ".join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]) + "\n",
                encoding="utf-8",
            )
            finalize(output)
            return

        if args.reset_bitstream_before_rpc:
            if args.ssh_known_hosts is None:
                raise ValueError("--ssh-known-hosts is required for clean board start")
            board = CleanStartBoard(args.host, args.ssh_known_hosts)
            before_reset = board.state()
            if before_reset.get("FPGA") != "operating" or before_reset.get("UDMABUF") != "201326592":
                raise RuntimeError("clean-start FPGA/u-dma-buf preflight failed")
            if before_reset.get("STORAGE_ERRORS"):
                raise RuntimeError("clean-start preflight found EXT4/mmc errors")
            if board.runtime_hashes(args.default_runtime) != EXPECTED_DEFAULT_RUNTIME_HASHES:
                raise RuntimeError("clean-start default runtime hash mismatch")
            prior_rpc = board.rpc_state()
            if prior_rpc.get("CWD") != args.default_runtime:
                raise RuntimeError("clean-start default RPC is not active")
            board.stop_rpc(prior_rpc)
            reload_result = board.reload_frozen_bitstream()
            fresh_rpc = board.start_rpc(args.default_runtime, "p7r_clean_start.log")
            write_json(output / "clean_board_start.json", {
                "status": "passed",
                "before": before_reset,
                "stopped_rpc": prior_rpc,
                "bitstream_reload": reload_result,
                "fresh_rpc": fresh_rpc,
                "allocator_effect": "fresh process resets the non-freeing u-dma-buf bump cursor",
                "health_or_target_allocation_before_fresh_rpc": False,
            })

        remote = rpc.connect(args.host, args.port, session_timeout=args.session_timeout)
        inventory, functions = runtime_inventory(remote)
        write_json(output / "rpc_inventory.json", inventory)
        device = remote.ext_dev(0)
        health_module = load_ephemeral(remote, directory / (health["candidate_id"] + ".so"))
        loaded = {
            candidate_id: load_ephemeral(remote, directory / (candidate_id + ".so"))
            for candidate_id in order
        }
        health_buffers = allocate_reusable_buffers(device, health["identity"]["workload"])
        target_buffers = allocate_reusable_buffers(device, candidates[0]["identity"]["workload"])

        for seed in SEEDS:
            started = time.perf_counter()
            row = profiled_reused_call(
                health_module["main"], health_buffers,
                health["identity"]["workload"], seed, functions
            )
            row["host_wall_ms"] = (time.perf_counter() - started) * 1000.0
            health_rows.append(row)
            if not row.get("correct"):
                break
        health_pass = len(health_rows) == 3 and all(row.get("correct") for row in health_rows)
        write_json(output / "health_gate.json", {
            "status": "passed" if health_pass else "failed_stop", "seeds": health_rows
        })
        if not health_pass:
            raise RuntimeError("W05 health gate failed; no {} candidate dispatched".format(workload_id))

        with (output / "correctness.jsonl").open("x", encoding="utf-8") as stream:
            for position, candidate_id in enumerate(order):
                candidate = by_id[candidate_id]
                seed_rows = []
                for seed in SEEDS:
                    started = time.perf_counter()
                    result = profiled_reused_call(
                        loaded[candidate_id]["main"], target_buffers,
                        candidate["identity"]["workload"], seed, functions,
                    )
                    result["host_wall_ms"] = (time.perf_counter() - started) * 1000.0
                    seed_rows.append(result)
                    if not result.get("correct"):
                        break
                row = {
                    "candidate_id": candidate_id,
                    "family_id": candidate["family_id"],
                    "public_mode": candidate["public_mode"],
                    "position": position,
                    "status": "passed" if len(seed_rows) == 3 and all(
                        item.get("correct") for item in seed_rows
                    ) else "failed",
                    "seeds": seed_rows,
                }
                correctness_rows.append(row)
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
                print(f"correctness {position + 1}/{len(order)} {row['status']}", flush=True)

        correct_ids = [row["candidate_id"] for row in correctness_rows if row["status"] == "passed"]
        if not correct_ids:
            raise RuntimeError("{} full pool has no FPGA-correct candidate; timing forbidden".format(workload_id))
        orders = timing_orders(correct_ids)
        write_json(output / "timing_orders.json", orders)
        with (output / "timing.jsonl").open("x", encoding="utf-8") as stream, (
            output / "timing_health_brackets.jsonl"
        ).open("x", encoding="utf-8") as health_stream:
            for round_index, round_order in enumerate(orders):
                for bracket in ("before",):
                    started = time.perf_counter()
                    row = timed_reused_call(
                        health_module, device, health_buffers,
                        health["identity"]["workload"], 0, functions
                    )
                    row["host_wall_ms"] = (time.perf_counter() - started) * 1000.0
                    row.update(round=round_index, bracket=bracket)
                    health_stream.write(json.dumps(row, sort_keys=True) + "\n"); health_stream.flush()
                for position, candidate_id in enumerate(round_order):
                    candidate = by_id[candidate_id]
                    started = time.perf_counter()
                    row = timed_reused_call(
                        loaded[candidate_id], device, target_buffers,
                        candidate["identity"]["workload"], 0, functions
                    )
                    row["host_wall_ms"] = (time.perf_counter() - started) * 1000.0
                    row.update(candidate_id=candidate_id, round=round_index, position=position)
                    timing_rows.append(row)
                    stream.write(json.dumps(row, sort_keys=True) + "\n"); stream.flush()
                started = time.perf_counter()
                row = timed_reused_call(
                    health_module, device, health_buffers,
                    health["identity"]["workload"], 0, functions
                )
                row["host_wall_ms"] = (time.perf_counter() - started) * 1000.0
                row.update(round=round_index, bracket="after")
                health_stream.write(json.dumps(row, sort_keys=True) + "\n"); health_stream.flush()
                print(f"timing round {round_index + 1}/{ROUNDS}", flush=True)

        timing_summary = summarize_timing(timing_rows, correct_ids)
        compile_walls = {
            candidate_id: certificates[candidate_id]["cross_compile_wall_ms"]
            for candidate_id in order
        }
        complete = completed_pool(
            pool, correctness_rows, timing_summary, qualification, compile_walls, timing_rows
        )
        runs, search_summary = run_offline(complete, BUDGETS, 20)
        proto = search_protocol(BUDGETS, 20)
        write_json(output / "timing_summary.json", timing_summary)
        write_json(output / "completed_pool.json", complete)
        (output / "search_runs.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in runs), encoding="utf-8"
        )
        write_json(output / "search_summary.json", search_summary)
        (output / "SEARCH_RESULTS.md").write_text(
            render_results(complete, search_summary, proto), encoding="utf-8"
        )
        write_json(output / "summary.json", {
            "schema": execution_contract["schema"],
            "status": "completed_full_pool_correctness_timing_and_search_replay",
            "candidate_count": len(order),
            "correct_candidate_count": len(correct_ids),
            "correctness_invocations": sum(len(row["seeds"]) for row in correctness_rows),
            "full_three_seed_invocations": len(order) * 3,
            "timing_samples": len(timing_rows),
            "all_timed_calls_correct": True,
            "pool_oracle_candidate_id": timing_summary["pool_oracle_candidate_id"],
            "pool_oracle_latency_ms": timing_summary["pool_oracle_latency_ms"],
            "search_cost_scope": "five phases: lower/FSim/cross-compile/FPGA correctness/one representative timing call",
            "shared_memory_cost_scope": "logical VTA LOAD/STORE runtime counters; not physical AXI bursts",
            "common_prequalification_overhead_in_replay": False,
            "remote_buffer_policy": "one shape-compatible output/input/weight set reused per workload",
            "remote_buffer_allocation_order": ["output", "data", "weight"],
            "claim_boundary": contract.get("claim_boundary"),
        })
        (output / "command.txt").write_text(
            " ".join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]) + "\n",
            encoding="utf-8",
        )
        finalize(output)
    except Exception as error:
        write_json(output / "failure.json", {
            "status": "failed_closed",
            "exception_type": type(error).__name__,
            "message": (str(error) or type(error).__name__)[:4000],
            "traceback": traceback.format_exc()[-12000:],
            "health_seed_checks": len(health_rows),
            "candidate_identities_completed": len(correctness_rows),
            "timing_samples_completed": len(timing_rows),
        })
        finalize(output)
        raise
    finally:
        if temporary is not None:
            temporary.cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract-dir", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--session-timeout", type=int, default=120)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--reset-bitstream-before-rpc", action="store_true")
    parser.add_argument("--ssh-known-hosts", type=Path)
    parser.add_argument("--default-runtime", default="/var/volatile/vta_c3_ram/runtime")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
