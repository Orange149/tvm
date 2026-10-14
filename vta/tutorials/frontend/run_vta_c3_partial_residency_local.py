#!/usr/bin/env python3
"""Local-only R18-H1 T8 P1/P2/P4 residency qualification and command audit."""

from __future__ import annotations

import argparse
import copy
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

import run_vta_c3_reuse_supplement as s


HERE, ROOT = s.HERE, s.ROOT
BASE = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune"
PARENT = BASE / "07_grouped_holdout/20260922_c3_thread_sync_prepare_run07"
SCHEMA = "c3_partial_weight_residency_local_v1"
SEEDS = (0, 20250901, 20260910)
QUEUE_ENV = {
    "VTA_INSN_BUFFER_BYTES": "33554432",
    "VTA_UOP_BUFFER_BYTES": "33554432",
    "VTA_INSN_SUBMIT_THRESHOLD_BYTES": "33554432",
    "VTA_REPLAY_POLICY": "disabled",
    "VTA_RUNTIME_TRACE": "0",
    "VTA_QUEUE_BOUNDARY_AUDIT": "0",
    "VTA_QUEUE_DRAIN_AUDIT": "0",
    "VTA_QUEUE_DIAGNOSTICS": "1",
}


def _baseline():
    rows = s.read(PARENT / "candidates.json")
    row = copy.deepcopy(next(item for item in rows if item["candidate_id"] == "T8_A"))
    expected = {
        "tile_b": 1,
        "tile_h": 8,
        "tile_w": 56,
        "tile_ci": 1,
        "tile_co": 2,
        "oc_nthread": 2,
        "h_nthread": 1,
    }
    if row["status"] != "passed" or row["visible_features"] != expected:
        raise ValueError("sealed T8 genuine TopHub identity changed")
    return row


def candidates():
    baseline = _baseline()
    shared = {
        "family_id": "R18-H1-T8-partial-residency-local",
        "workload_id": "R18-H1",
        "identity": baseline["identity"],
        "complete_config_entity": baseline["complete_config_entity"],
        "visible_features": baseline["visible_features"],
    }
    return [
        {
            **shared,
            "candidate_id": "T8_O2",
            "variant": "O2",
            "implementation_mode": 0,
            "resident_groups": 0,
            "source_tophub": True,
        },
        {
            **shared,
            "candidate_id": "T8_P1S",
            "variant": "P1S",
            "implementation_mode": 8,
            "resident_groups": 1,
            "source_tophub": False,
        },
        {
            **shared,
            "candidate_id": "T8_P2S",
            "variant": "P2S",
            "implementation_mode": 8,
            "resident_groups": 2,
            "source_tophub": False,
        },
        {
            **shared,
            "candidate_id": "T8_P4S",
            "variant": "P4S",
            "implementation_mode": 8,
            "resident_groups": 4,
            "source_tophub": False,
        },
    ]


def instantiate(row):
    import vta
    from tvm import autotvm
    from tvm.autotvm.task.space import ConfigEntity
    import vta.top.vta_conv2d_residency  # noqa: F401
    from tune_resnet18_vta import register_vta_conv2d_template

    env = vta.get_env()
    workload = row["identity"]["workload"]
    if row["source_tophub"]:
        register_vta_conv2d_template()
        name, args = workload[0], tuple(workload[1:])
    else:
        name = "conv2d_packed_residency.vta"
        args = tuple(workload[1:]) + (row["implementation_mode"],)
    task = autotvm.task.create(name, args=args, target=env.target, target_host=env.target_host)
    entity = ConfigEntity.from_json_dict(
        {"index": -1, "code_hash": None, **row["complete_config_entity"]}
    )
    with task.target:
        return task.instantiate(entity)


def build_config(row, **kwargs):
    import vta
    from vta.top.vta_partial_weight import partial_weight_build_config
    from vta.top.vta_staged_weight import staged_weight_build_config

    groups = row["resident_groups"]
    if groups == 0:
        return vta.build_config(**kwargs)
    if groups in (1, 2):
        return partial_weight_build_config(groups, **kwargs)
    if groups == 4:
        return staged_weight_build_config(**kwargs)
    raise ValueError("unknown residency group count")


def _bytes_before_first_mac(address):
    transferred = 0
    for event in address["ordered_events"]:
        if event[0] == "VTALoadBuffer2D" and event[1][-1] == 1:
            transferred += event[1][1] * event[1][2] * 256
        if event[0] == "kernel" and any(
            uop["args"][0] == 0 and uop["args"][1] == 0 for uop in event[1]
        ):
            return transferred
    raise ValueError("no effective MAC in final TIR")


def lower(row, out):
    import tvm
    import vta
    from c3_reuse_semantics import capture_lower, extract_requests
    from c3_thread_sync_audit import final_tir_audit
    from run_vta_c3_resnet18_local_qualification import dependency_audit, sync_multiplicity

    schedule, tensors = instantiate(row)
    module, storage, arenas = capture_lower(
        schedule,
        tensors,
        build_config_factory=lambda **kwargs: build_config(row, **kwargs),
    )
    semantics = extract_requests(module, vta.get_env())
    dependencies = dependency_audit(module)
    address = final_tir_audit(module, semantics)
    sync = sync_multiplicity(module)
    if (
        not dependencies["balanced"]
        or dependencies["forbidden_direct_1_3_count"]
        or address["status"] != "passed"
        or sync != [1]
    ):
        raise ValueError("local dependency/address/single-submit qualification failed")
    (out / "tir").mkdir(exist_ok=True)
    tir_path = out / "tir" / (row["candidate_id"] + ".json")
    storage_path = out / "tir" / (row["candidate_id"] + ".storage.json")
    tir_path.write_text(tvm.ir.save_json(module))
    storage_path.write_text(tvm.ir.save_json(storage))
    return {
        "tir_sha256": s.file_hash(tir_path),
        "storage_tir_sha256": s.file_hash(storage_path),
        "semantics": semantics,
        "arenas": arenas["scope_arena_span_sum_upper_bound_bytes"],
        "dependency_audit": dependencies,
        "address_audit": address,
        "sync_multiplicity": sync,
        "pre_first_mac_weight_bytes": _bytes_before_first_mac(address),
    }


def worker(request_path, response_path):
    import tvm
    import vta
    from qualify_vta_residency_fsim import array_sha256, reference_data_from_workload

    request = s.read(request_path)
    row = request["candidate"]
    ctypes.CDLL(request["backend"], mode=ctypes.RTLD_GLOBAL)
    schedule, tensors = instantiate(row)
    with build_config(row, disabled_pass={"tir.CommonSubexprElimTIR"}):
        module = vta.build(
            schedule,
            tensors,
            target=tvm.target.Target(vta.get_env().target, host="llvm"),
            name="main",
        )
    function = module["main"]
    device = tvm.device("ext_dev", 0)
    result = {"status": "failed", "candidate_id": row["candidate_id"], "seeds": []}
    try:
        first_data, first_weight, first_expected = reference_data_from_workload(
            row["identity"]["workload"], SEEDS[0]
        )
        data_handle = tvm.nd.array(first_data, device)
        weight_handle = tvm.nd.array(first_weight, device)
        output_handle = tvm.nd.array(
            np.full(first_expected.shape, -113, dtype="int8"), device
        )
        function(data_handle, weight_handle, output_handle)
        np.testing.assert_array_equal(output_handle.numpy(), first_expected)
        for seed in SEEDS:
            data, weight, expected = reference_data_from_workload(
                row["identity"]["workload"], seed
            )
            data_handle.copyfrom(data)
            weight_handle.copyfrom(weight)
            output_handle.copyfrom(np.full(expected.shape, -113, dtype="int8"))
            tvm.get_global_func("vta.runtime.profiler_clear")()
            tvm.get_global_func("vta.simulator.profiler_clear")()
            print("[C3_R18_SEED] " + str(seed), file=sys.stderr, flush=True)
            function(data_handle, weight_handle, output_handle)
            actual = output_handle.numpy()
            np.testing.assert_array_equal(actual, expected)
            runtime = json.loads(tvm.get_global_func("vta.runtime.profiler_status")())
            simulator = json.loads(tvm.get_global_func("vta.simulator.profiler_status")())
            result["seeds"].append(
                {
                    "seed": seed,
                    "correct": True,
                    "mismatch_count": 0,
                    "actual_sha256": array_sha256(actual),
                    "expected_sha256": array_sha256(expected),
                    "runtime_logical_dma": {
                        key: value
                        for key, value in runtime.items()
                        if (
                            key.startswith(("load_buffer_2d_", "store_buffer_2d_"))
                            and key.endswith(("_bytes", "_calls"))
                        )
                        or key in {"synchronize_calls", "driver_run_calls"}
                    },
                    "fsim_instruction_bytes": {
                        key: value
                        for key, value in simulator.items()
                        if key.endswith("_nbytes")
                    },
                    "complete_function_invocations": 1,
                }
            )
        result["status"] = "passed"
    except Exception as error:  # preserve exact failure
        result["failure"] = {"type": type(error).__name__, "message": str(error)}
    s.write(response_path, result)


def qualify_commands(row, static, out, timeout):
    from run_vta_c3_resnet18_local_qualification import attach_command_features

    with tempfile.TemporaryDirectory(prefix="c3_partial_weight_") as directory:
        request = Path(directory) / "request.json"
        response = Path(directory) / "response.json"
        s.write(
            request,
            {
                "candidate": row,
                "backend": str(ROOT / "build/libvta_fsim.so"),
            },
        )
        environment = os.environ.copy()
        environment.update(QUEUE_ENV)
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker-request",
                str(request),
                "--worker-response",
                str(response),
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=environment,
            check=False,
        )
        (out / "logs").mkdir(exist_ok=True)
        (out / "logs" / (row["candidate_id"] + ".stderr.log")).write_text(
            completed.stderr
        )
        if completed.returncode or not response.is_file():
            raise ValueError(
                "FSim worker failed for {}: {}".format(
                    row["candidate_id"], completed.stderr[-2000:]
                )
            )
        result = s.read(response)
        if result["status"] != "passed":
            raise ValueError("FSim qualification failed: {}".format(result.get("failure")))
        result["implementation_mode"] = row["implementation_mode"]
        result = attach_command_features(
            result,
            completed.stderr,
            {"sync": {"explicit_residency_drains": 0}},
        )
        if result["status"] != "passed":
            raise ValueError("command qualification failed: {}".format(result.get("failure")))
        expected = static["semantics"]["totals"]
        for seed in result["seeds"]:
            for key, value in expected.items():
                if seed["runtime_logical_dma"].get(key) != value:
                    raise ValueError("static/runtime request mismatch: " + key)
        return result


def run(out, timeout):
    started = time.monotonic()
    rows = candidates()
    results = []
    for row in rows:
        static = lower(row, out)
        command = qualify_commands(row, static, out, timeout)
        values = command["command_features"]["values"]
        totals = static["semantics"]["totals"]
        result = {
            **row,
            "static": static,
            "fsim": command,
            "metrics": {
                "wgt_arena_bytes": static["arenas"]["local.wgt_buffer"],
                "wgt_load_bytes": totals["load_buffer_2d_wgt_bytes"],
                "wgt_load_calls": totals["load_buffer_2d_wgt_calls"],
                "inp_load_bytes": totals["load_buffer_2d_inp_bytes"],
                "inp_load_calls": totals["load_buffer_2d_inp_calls"],
                "pre_first_mac_weight_bytes": static["pre_first_mac_weight_bytes"],
                "uop_load_bytes": values["totals"]["uop_bytes"],
                "instruction_bytes": values["totals"]["insn_bytes"],
                "submission_count": values["submissions"],
                "max_kernel_uops": static["address_audit"]["max_kernel_uops"],
                "explicit_drains": 0,
            },
        }
        results.append(result)
        print(row["candidate_id"], json.dumps(result["metrics"], sort_keys=True), flush=True)

    by_variant = {row["variant"]: row for row in results}
    shared_checks = {
        "all_three_seed_fsim_exact": all(
            item["correct"] and item["actual_sha256"] == item["expected_sha256"]
            for row in results
            for item in row["fsim"]["seeds"]
        ),
        "all_single_submission": all(row["metrics"]["submission_count"] == 1 for row in results),
        "all_no_explicit_drains": all(row["metrics"]["explicit_drains"] == 0 for row in results),
        "same_nonweight_requests": len(
            {
                (
                    row["metrics"]["inp_load_bytes"],
                    row["metrics"]["inp_load_calls"],
                    row["static"]["semantics"]["totals"]["store_buffer_2d_out_bytes"],
                    row["static"]["semantics"]["totals"]["store_buffer_2d_out_calls"],
                )
                for row in results
            }
        )
        == 1,
        "p1_reduces_uop_vs_p4": by_variant["P1S"]["metrics"]["uop_load_bytes"]
        < by_variant["P4S"]["metrics"]["uop_load_bytes"],
        "p2_reduces_uop_vs_p4": by_variant["P2S"]["metrics"]["uop_load_bytes"]
        < by_variant["P4S"]["metrics"]["uop_load_bytes"],
    }
    status = (
        "LOCAL_GATE_PASS"
        if all(shared_checks.values())
        else "LOCAL_MECHANISM_VALID_BUT_UOP_GATE_NOT_MET"
        if all(
            shared_checks[key]
            for key in (
                "all_three_seed_fsim_exact",
                "all_single_submission",
                "all_no_explicit_drains",
                "same_nonweight_requests",
            )
        )
        else "LOCAL_GATE_REJECTED"
    )
    s.write(out / "candidates.json", results)
    s.write(out / "checks.json", shared_checks)
    s.write(
        out / "contract.json",
        {
            "schema": SCHEMA,
            "phase": "local",
            "scope": "R18-H1 T8 O2/P1-S/P2-S/P4-S local qualification",
            "parent_directory": str(PARENT),
            "parent_manifest_sha256": s.file_hash(PARENT / "artifact_hashes.json"),
            "board_contacted": False,
            "seeds": list(SEEDS),
            "queue_environment": QUEUE_ENV,
        },
    )
    s.write(
        out / "summary.json",
        {
            "status": status,
            "checks": shared_checks,
            "board_contacted": False,
            "performance_labels_collected": False,
            "wall_seconds": time.monotonic() - started,
        },
    )
    lines = [
        "# R18-H1 T8 P1/P2/P4 本地驻留资格",
        "",
        "| 版本 | WGT arena | WGT请求 B/calls | 首MAC前WGT B | UOP LOAD B | 指令 B | 提交 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in ("O2", "P1S", "P2S", "P4S"):
        metric = by_variant[variant]["metrics"]
        lines.append(
            "| {} | {} | {}/{} | {} | {} | {} | {} |".format(
                variant,
                metric["wgt_arena_bytes"],
                metric["wgt_load_bytes"],
                metric["wgt_load_calls"],
                metric["pre_first_mac_weight_bytes"],
                metric["uop_load_bytes"],
                metric["instruction_bytes"],
                metric["submission_count"],
            )
        )
    lines += ["", "状态：`{}`。".format(status), "", json.dumps(shared_checks, indent=2), ""]
    (out / "README.md").write_text("\n".join(lines))
    s.seal(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--worker-request", type=Path)
    parser.add_argument("--worker-response", type=Path)
    args = parser.parse_args()
    if args.worker_request:
        if not args.worker_response:
            parser.error("worker-response is required")
        worker(args.worker_request, args.worker_response)
        return
    if not args.output_dir:
        parser.error("output-dir is required")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    try:
        run(out, args.timeout)
    except BaseException as error:
        s.write(
            out / "invalid_session.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "board_contacted": False,
            },
        )
        s.seal(out)
        raise


if __name__ == "__main__":
    main()
