#!/usr/bin/env python3
"""Fail-closed source-TopHub versus residency-adapter correctness diagnostic.

This experiment deliberately collects no latency.  It builds only the genuine
``conv2d_packed.vta`` TopHub task and the mode-0 residency adapter, checks three
local FSim seeds, and then checks the same two modules and seeds through direct
RPC.  SSH, board restart/reconfiguration, persistent board writes, and P7R120
candidate dispatch are outside the contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import tvm
from tvm import autotvm, rpc
from tvm.autotvm.task.space import ConfigEntity
from tvm.contrib import cc
import vta

import vta.top.vta_conv2d_residency  # pylint: disable=unused-import
from c3_candidate_identity import canonical_json_bytes, normalize_config_entity
from prepare_vta_p7r120_yolo_rpc_contract import (
    GUARD_PATHS,
    REPO,
    cross_options,
    load_json,
    sha256_file,
    write_json,
)
from qualify_vta_residency_fsim import array_sha256, reference_data_from_workload
from run_vta_p7r120_yolo_rpc_board import (
    PROFILE_FIELDS,
    load_ephemeral,
    runtime_inventory,
)
from tune_resnet18_vta import register_vta_conv2d_template


SCHEMA = "c3_p7r121_tophub_adapter_diagnostic_v1"
SEEDS = (0, 20250901, 20260910)
HERE = Path(__file__).resolve().parent
C3 = HERE / "report_out" / "stage_tile_cotuning" / "c3_dma_residency_autotune"
DEFAULT_PARENT = (
    C3
    / "07_grouped_holdout"
    / "20260911_p7r120_y02_rpc_contract_v2_run01"
    / "contract.json"
)
DEFAULT_OUTPUT = (
    C3
    / "07_grouped_holdout"
    / "20260912_p7r121_tophub_adapter_diagnostic_run01"
)


def canonical_sha256(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def verify_parent(path):
    path = Path(path).resolve()
    ledger_path = path.parent / "artifact_hashes.json"
    ledger = load_json(ledger_path)
    if ledger["artifacts"].get(path.name) != sha256_file(path):
        raise ValueError("P7R120 parent contract hash mismatch")
    contract = load_json(path)
    if contract.get("schema") != "c3_p7r120_y02_rpc_board_contract_v2":
        raise ValueError("unexpected parent contract schema")
    guards = contract["fresh_hardware_certificate_v2"]["source_guards_sha256"]
    observed = {str(source.relative_to(REPO)): sha256_file(source) for source in GUARD_PATHS}
    if observed != guards:
        raise ValueError("source/hardware guards changed after P7R120 freeze")
    return contract, sha256_file(ledger_path)


def variant_specs(parent):
    sealed = parent["sealed_reference"]
    identity = sealed["identity"]
    common = {
        "workload": identity["workload"],
        "semantic_knobs": identity["semantic_knobs"],
        "selection_or_training_label": False,
        "performance_measurement": "not_collected",
    }
    source_identity = {
        "schema": "c3_p7r121_diagnostic_variant_v1",
        "variant": "source_tophub",
        "template": "conv2d_packed.vta",
        "workload": identity["source_workload"],
        "complete_config_entity": identity["source_tophub_complete_config_entity"],
        "semantic_knobs": identity["semantic_knobs"],
    }
    adapter_identity = {
        "schema": "c3_p7r121_diagnostic_variant_v1",
        "variant": "residency_mode0_adapter",
        "template": "conv2d_packed_residency.vta",
        "implementation_mode": 0,
        "workload": identity["workload"],
        "complete_config_entity": identity["complete_config_entity"],
        "semantic_knobs": identity["semantic_knobs"],
    }
    return [
        {
            **common,
            "variant": "source_tophub",
            "variant_id": canonical_sha256(source_identity),
            "identity": source_identity,
            "config_index": int(sealed["debug"]["source_tophub_config_index"]),
        },
        {
            **common,
            "variant": "residency_mode0_adapter",
            "variant_id": canonical_sha256(adapter_identity),
            "identity": adapter_identity,
            "config_index": int(sealed["debug"]["config_index"]),
        },
    ]


def make_task(spec, env):
    if spec["variant"] == "source_tophub":
        # This is the actual fused tuning template used by the source task: packed
        # convolution followed by shift, clip, and cast.  Creating the bare TOPI
        # compute directly is not equivalent and fails when output == conv stage.
        register_vta_conv2d_template()
        name = "conv2d_packed.vta"
        args = tuple(spec["identity"]["workload"][1:])
    elif spec["variant"] == "residency_mode0_adapter":
        name = "conv2d_packed_residency.vta"
        args = tuple(spec["identity"]["workload"][1:]) + (0,)
    else:
        raise ValueError("unknown diagnostic variant")
    return autotvm.task.create(name, args=args, target=env.target, target_host=env.target_host)


def config_entity(spec):
    entity = spec["identity"]["complete_config_entity"]
    return ConfigEntity.from_json_dict(
        {
            "index": int(spec["config_index"]),
            "code_hash": entity.get("code_hash"),
            "entity": entity["entity"],
        }
    )


def instantiate(spec, env):
    task = make_task(spec, env)
    config = config_entity(spec)
    if normalize_config_entity(config.to_json_dict()) != normalize_config_entity(
        spec["identity"]["complete_config_entity"]
    ):
        raise RuntimeError("complete ConfigEntity changed")
    with task.target:
        schedule, tensors = task.instantiate(config)
    return schedule, tensors


def lower_and_build(spec, env):
    schedule, tensors = instantiate(spec, env)
    with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
        lowered = tvm.lower(schedule, tensors, name="main")
    lowered_json = tvm.ir.save_json(lowered)
    with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
        module = vta.build(
            schedule,
            tensors,
            target=tvm.target.Target(env.target, host=env.target_host),
            name="main",
        )
    return module, lowered_json


def result_detail(function, device, workload, seed):
    data, weight, expected = reference_data_from_workload(workload, int(seed))
    output = tvm.nd.empty(expected.shape, "int8", device)
    function(tvm.nd.array(data, device), tvm.nd.array(weight, device), output)
    actual = output.numpy()
    locations = np.argwhere(actual != expected)
    examples = []
    for location in locations[:8]:
        index = tuple(int(value) for value in location)
        examples.append(
            {
                "index": list(index),
                "expected": int(expected[index]),
                "actual": int(actual[index]),
            }
        )
    return {
        "seed": int(seed),
        "correct": len(locations) == 0,
        "mismatch_count": int(len(locations)),
        "expected_sha256": array_sha256(expected),
        "actual_sha256": array_sha256(actual),
        "mismatch_examples_first8": examples,
        "expected_range": [int(expected.min()), int(expected.max())],
        "actual_range": [int(actual.min()), int(actual.max())],
    }


def fsim_worker(contract_path, result_path):
    # Importing the simulator registers the local ext_dev DeviceAPI.  Merely
    # compiling for TARGET=sim is insufficient to allocate ext_dev NDArrays.
    from vta.testing import simulator  # pylint: disable=import-outside-toplevel

    contract = load_json(contract_path)
    if contract.get("schema") != SCHEMA:
        raise ValueError("unexpected diagnostic contract")
    env = vta.get_env()
    if env.TARGET != "sim" or not simulator.enabled():
        raise RuntimeError("P7R121 FSim worker requires TARGET=sim")
    rows = []
    tir = {}
    for spec in contract["variants"]:
        start = time.monotonic()
        try:
            module, lowered_json = lower_and_build(spec, env)
            tir[spec["variant"]] = {
                "sha256": hashlib.sha256(lowered_json.encode("utf-8")).hexdigest(),
                "bytes": len(lowered_json.encode("utf-8")),
            }
            function = module["main"]
            seeds = [
                result_detail(function, tvm.device("ext_dev", 0), spec["workload"], seed)
                for seed in contract["seeds"]
            ]
            rows.append(
                {
                    "variant": spec["variant"],
                    "status": "passed" if all(row["correct"] for row in seeds) else "wrong_answer",
                    "seeds": seeds,
                    "diagnostic_wall_seconds_not_performance": time.monotonic() - start,
                }
            )
        except Exception as error:  # Preserve a fail-closed local diagnostic.
            rows.append(
                {
                    "variant": spec["variant"],
                    "status": "failed",
                    "seeds": [],
                    "exception_type": type(error).__name__,
                    "message": (str(error) or type(error).__name__)[:4000],
                    "traceback": traceback.format_exc()[-8000:],
                }
            )
    write_json(
        result_path,
        {
            "schema": SCHEMA,
            "phase": "local_fsim_three_seed_correctness",
            "performance_measurement": "not_collected",
            "tir": tir,
            "variants": rows,
        },
    )


def run_fsim(contract_path, fsim_hw_path, timeout):
    with tempfile.TemporaryDirectory(prefix="c3_p7r121_fsim_") as directory:
        result_path = Path(directory) / "result.json"
        environment = os.environ.copy()
        environment["VTA_HW_PATH"] = str(Path(fsim_hw_path).resolve())
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-contract",
            str(Path(contract_path).resolve()),
            "--worker-output",
            str(result_path),
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=environment,
        )
        if completed.returncode or not result_path.is_file():
            raise RuntimeError(
                "FSim worker failed exit={} stderr={}".format(
                    completed.returncode, completed.stderr[-4000:]
                )
            )
        return load_json(result_path)


def profile_snapshot(functions):
    status = json.loads(functions["vta.runtime.profiler_status"]())
    return {name: status.get(name) for name in PROFILE_FIELDS}


def board_seed(function, device, workload, seed, functions):
    clear = functions.get("vta.runtime.profiler_clear")
    if clear is not None:
        clear()
    try:
        row = result_detail(function, device, workload, seed)
        row["status"] = "passed" if row["correct"] else "wrong_answer"
    except Exception as error:
        row = {
            "seed": int(seed),
            "correct": False,
            "status": "execution_failed",
            "exception_type": type(error).__name__,
            "message": (str(error) or type(error).__name__)[:4000],
        }
    try:
        row["runtime_profile"] = profile_snapshot(functions)
    except Exception as error:
        row["runtime_profile_error"] = (str(error) or type(error).__name__)[:2000]
    return row


def outcome(rows):
    passed = {
        row["variant"]: len(row["seeds"]) == 3
        and all(seed.get("correct") for seed in row["seeds"])
        for row in rows
    }
    if passed == {"source_tophub": True, "residency_mode0_adapter": False}:
        return "source_pass_adapter_fail_v3_source_canary_required"
    if passed == {"source_tophub": False, "residency_mode0_adapter": False}:
        return "both_fail_p7r120_remains_blocked"
    if passed == {"source_tophub": True, "residency_mode0_adapter": True}:
        return "both_pass_prior_failure_not_reproduced"
    return "source_fail_adapter_pass_inconsistent_mapping"


def finalize(output):
    hashes = {
        path.name: sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    write_json(
        output / "artifact_hashes.json",
        {
            "artifacts": hashes,
            "source_sha256": {str(Path(__file__).resolve()): sha256_file(__file__)},
        },
    )


def run(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable diagnostic {}".format(output))
    parent, parent_ledger = verify_parent(args.parent_contract)
    if args.host != parent["rpc_contract"]["host"] or args.port != parent["rpc_contract"]["port"]:
        raise ValueError("RPC endpoint differs from frozen parent")
    specs = variant_specs(parent)
    contract = {
        "schema": SCHEMA,
        "status": "frozen_before_local_fsim_or_board_observations",
        "parent_contract": {
            "path": str(Path(args.parent_contract).resolve()),
            "sha256": sha256_file(args.parent_contract),
            "artifact_ledger_sha256": parent_ledger,
        },
        "variants": specs,
        "variant_order": ["source_tophub", "residency_mode0_adapter"],
        "seeds": list(SEEDS),
        "local_protocol": "two modules x three exact FSim seeds",
        "board_protocol": "two modules x three exact seeds in frozen order",
        "rpc": {
            "host": args.host,
            "port": args.port,
            "transport": "direct TVM RPC only",
            "module_transport": "ephemeral upload/load_module/immediate remote.remove",
            "ssh_forbidden": True,
            "restart_or_reconfiguration_forbidden": True,
            "persistent_board_write_forbidden": True,
            "boot_id": "unknown_not_exposed_by_rpc",
        },
        "candidate_dispatch": "forbidden",
        "latency_or_performance_measurement": "forbidden",
        "decision": {
            "source_pass_adapter_fail": "generate fresh v3 with genuine source-task canary",
            "both_fail": "retain fail-closed P7R120 block and mismatch/profile evidence",
        },
    }
    output.mkdir(parents=True)
    write_json(output / "diagnostic_contract.json", contract)
    write_json(
        output / "pre_observation_hashes.json",
        {"artifacts": {"diagnostic_contract.json": sha256_file(output / "diagnostic_contract.json")}},
    )

    temporary = None
    try:
        env = vta.get_env()
        if env.TARGET != "axu5evb":
            raise RuntimeError("P7R121 board controller requires TARGET=axu5evb")

        fsim = run_fsim(output / "diagnostic_contract.json", args.fsim_hw_path, args.fsim_timeout)
        write_json(output / "local_fsim.json", fsim)

        temporary = tempfile.TemporaryDirectory(prefix="c3_p7r121_cross_")
        directory = Path(temporary.name)
        binaries = {}
        axu_tir = {}
        for spec in specs:
            module, lowered_json = lower_and_build(spec, env)
            (output / (spec["variant"] + "_axu_lowered_tir.json")).write_text(
                lowered_json, encoding="utf-8"
            )
            binary = directory / (spec["variant_id"] + ".so")
            module.export_library(
                str(binary),
                fcompile=cc.cross_compiler("aarch64-xilinx-linux-g++"),
                options=cross_options(),
            )
            binaries[spec["variant"]] = binary
            axu_tir[spec["variant"]] = {
                "sha256": hashlib.sha256(lowered_json.encode("utf-8")).hexdigest(),
                "bytes": len(lowered_json.encode("utf-8")),
                "binary_sha256": sha256_file(binary),
                "binary_bytes": binary.stat().st_size,
            }
        axu_tir["identical_lowered_tir"] = (
            axu_tir["source_tophub"]["sha256"]
            == axu_tir["residency_mode0_adapter"]["sha256"]
        )
        write_json(output / "lowering_comparison.json", axu_tir)

        remote = rpc.connect(args.host, args.port, session_timeout=args.session_timeout)
        inventory, functions = runtime_inventory(remote)
        write_json(output / "rpc_inventory.json", inventory)
        device = remote.ext_dev(0)
        board_rows = []
        for spec in specs:
            module = load_ephemeral(remote, binaries[spec["variant"]])
            function = module["main"]
            seeds = [
                board_seed(function, device, spec["workload"], seed, functions)
                for seed in SEEDS
            ]
            board_rows.append(
                {
                    "variant": spec["variant"],
                    "variant_id": spec["variant_id"],
                    "status": "passed" if all(row.get("correct") for row in seeds) else "failed",
                    "seeds": seeds,
                    "performance_measurement": "not_collected",
                }
            )
            print("board {} {}".format(spec["variant"], board_rows[-1]["status"]), flush=True)
        board = {
            "schema": SCHEMA,
            "phase": "rpc_only_two_by_three_correctness",
            "boot_id": "unknown_not_exposed_by_rpc",
            "ssh_used": False,
            "restart_or_reconfiguration": False,
            "persistent_board_write": False,
            "candidate_dispatches": 0,
            "latency_samples": 0,
            "variants": board_rows,
            "outcome": outcome(board_rows),
        }
        write_json(output / "board_correctness.json", board)
        write_json(
            output / "summary.json",
            {
                "schema": SCHEMA,
                "status": board["outcome"],
                "local_fsim": {row["variant"]: row["status"] for row in fsim["variants"]},
                "board": {row["variant"]: row["status"] for row in board_rows},
                "lowered_tir_identical": axu_tir["identical_lowered_tir"],
                "candidate_dispatches": 0,
                "latency_samples": 0,
                "claim_boundary": "correctness diagnostic only; no performance conclusion",
            },
        )
        (output / "command.txt").write_text(
            " ".join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]) + "\n",
            encoding="utf-8",
        )
        finalize(output)
        print(json.dumps(load_json(output / "summary.json"), indent=2, sort_keys=True))
    except Exception as error:
        write_json(
            output / "failure.json",
            {
                "status": "failed_closed",
                "exception_type": type(error).__name__,
                "message": (str(error) or type(error).__name__)[:4000],
                "traceback": traceback.format_exc()[-12000:],
                "candidate_dispatches": 0,
                "latency_samples": 0,
                "ssh_used": False,
            },
        )
        finalize(output)
        raise
    finally:
        if temporary is not None:
            temporary.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-contract", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--session-timeout", type=int, default=120)
    parser.add_argument("--fsim-hw-path", type=Path, default=Path("/tmp/vta-hw-fsim"))
    parser.add_argument("--fsim-timeout", type=int, default=900)
    parser.add_argument("--worker-contract", type=Path)
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args()
    if args.worker_contract:
        if not args.worker_output:
            raise ValueError("worker output is required")
        fsim_worker(args.worker_contract, args.worker_output)
    else:
        run(args)


if __name__ == "__main__":
    main()
