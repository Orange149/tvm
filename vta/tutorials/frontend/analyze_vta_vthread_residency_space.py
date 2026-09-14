#!/usr/bin/env python3
"""Certify a label-free VTA residency/virtual-thread joint space.

This tool never contacts RPC and never measures latency.  It compares each
input-stationary candidate with the exact same ConfigEntity under the original
schedule after the complete VTA lowering pipeline, so capacity and DMA-shape
failures are rejected before FSim or board dispatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import Counter
from pathlib import Path

import tvm
from tvm import autotvm
import vta

import vta.top.vta_conv2d_residency  # pylint: disable=unused-import
from extract_static_vta_dma import extract_module_dma_compact


SCHEMA = "c3_p7r_vthread_residency_space_v1"
TEMPLATE = "conv2d_packed_residency.vta"
RESIDENCY_MODE_NUMBERS = {
    "input_stationary": 1,
    "paper_inspired_hybrid": 3,
}
EXTRA_GEOMETRIES = {
    "E00": {
        "incumbent_index": None,
        "workload": [
            "conv2d_packed.vta",
            ["TENSOR", [1, 6, 42, 42, 1, 16], "int8"],
            ["TENSOR", [10, 6, 3, 3, 16, 16], "int8"],
            [1, 1],
            [1, 1, 1, 1],
            [1, 1],
            "NCHW1n16c",
            "int32",
        ],
    },
    "E01": {
        "incumbent_index": None,
        "workload": [
            "conv2d_packed.vta",
            ["TENSOR", [1, 12, 21, 21, 1, 16], "int8"],
            ["TENSOR", [20, 12, 1, 1, 16, 16], "int8"],
            [1, 1],
            [0, 0, 0, 0],
            [1, 1],
            "NCHW1n16c",
            "int32",
        ],
    },
    "E02": {
        "incumbent_index": None,
        "workload": [
            "conv2d_packed.vta",
            ["TENSOR", [1, 20, 11, 11, 1, 16], "int8"],
            ["TENSOR", [24, 20, 3, 3, 16, 16], "int8"],
            [2, 2],
            [1, 1, 1, 1],
            [1, 1],
            "NCHW1n16c",
            "int32",
        ],
    },
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def workload_ids(text):
    result = []
    for item in text.split(","):
        item = item.strip().upper()
        if not item.startswith(("W", "E")) or not item[1:].isdigit():
            raise argparse.ArgumentTypeError("workloads must look like W04,E00")
        result.append(item)
    return tuple(result)


def knob_values(config):
    values = {}
    for name in ("tile_b", "tile_h", "tile_w", "tile_ci", "tile_co"):
        values[name] = int(config[name].size[-1])
    values["oc_nthread"] = int(config["oc_nthread"].val)
    values["h_nthread"] = int(config["h_nthread"].val)
    return values


def failure_category(error):
    message = str(error)
    if "Allocation exceed bound" in message:
        return "sram_capacity"
    if "cannot detect 2d pattern" in message.lower():
        return "dma_2d_pattern"
    if "Cannot match copy pattern" in message:
        return "dma_copy_pattern"
    if "Do not support pad" in message:
        return "padding_shape"
    if "Only input-stationary" in message or "h_nthread" in message:
        return "unsupported_vthread_combination"
    return "other_lower"


def lower_candidate(task, config, env):
    try:
        with task.target:
            schedule, tensors = task.instantiate(config)
        with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
            module = tvm.lower(schedule, tensors, name="main")
        dma = extract_module_dma_compact(module, env)["totals"]
        return {
            "status": "ok",
            "tir_sha256": hashlib.sha256(tvm.ir.save_json(module).encode()).hexdigest(),
            "dma": dma,
            "failure": None,
        }
    except Exception as error:  # TVM uses several exception classes for rejection.
        return {
            "status": "rejected",
            "tir_sha256": None,
            "dma": None,
            "failure": {
                "category": failure_category(error),
                "type": type(error).__name__,
                "message": str(error).splitlines()[-1][:1000],
            },
        }


def task_for(selected, mode, env):
    return autotvm.task.create(
        TEMPLATE,
        args=tuple(selected["workload"][1:]) + (mode,),
        target=env.target,
        target_host=env.target_host,
    )


def analytic_certificate(selected, knobs, env):
    """Return necessary, label-free feasibility conditions before full lowering."""

    workload = selected["workload"]
    data_shape = workload[1][1]
    weight_shape = workload[2][1]
    stride_h, stride_w = workload[3]
    co_blocks = int(weight_shape[0])
    kernel_h, kernel_w = int(weight_shape[2]), int(weight_shape[3])
    threads = knobs["oc_nthread"]
    effective_groups = co_blocks / float(threads * knobs["tile_co"])
    input_h = (knobs["tile_h"] - 1) * int(stride_h) + kernel_h
    input_w = (knobs["tile_w"] - 1) * int(stride_w) + kernel_w
    required = {
        "effective_output_channel_groups": effective_groups,
        "minimum_acc_vectors": knobs["tile_h"] * knobs["tile_w"] * co_blocks,
        "maximum_input_vectors": threads * knobs["tile_ci"] * input_h * input_w,
        "maximum_weight_vectors": threads
        * knobs["tile_co"]
        * knobs["tile_ci"]
        * kernel_h
        * kernel_w,
    }
    limits = {
        "acc_vectors": env.ACC_BUFF_SIZE // env.ACC_ELEM_BYTES,
        "input_vectors": env.INP_BUFF_SIZE // env.INP_ELEM_BYTES,
        "weight_vectors": env.WGT_BUFF_SIZE // env.WGT_ELEM_BYTES,
    }
    reasons = []
    if effective_groups <= 1:
        reasons.append("analytic_no_cross_context_reuse")
    if required["minimum_acc_vectors"] > limits["acc_vectors"]:
        reasons.append("analytic_acc_capacity")
    if required["maximum_input_vectors"] > limits["input_vectors"]:
        reasons.append("analytic_input_capacity")
    if required["maximum_weight_vectors"] > limits["weight_vectors"]:
        reasons.append("analytic_weight_capacity")
    return {"passed": not reasons, "reasons": reasons, "required": required, "limits": limits}


def scan_workload(
    workload_id, selected, env, progress_every, schedule_sha256, oc_nthread=2,
    residency_mode="input_stationary",
):
    mode_number = RESIDENCY_MODE_NUMBERS[residency_mode]
    tasks = {mode: task_for(selected, mode, env) for mode in (0, mode_number)}
    rows = []
    attempted = 0
    for index in range(len(tasks[mode_number].config_space)):
        config = tasks[mode_number].config_space.get(index)
        knobs = knob_values(config)
        if knobs["oc_nthread"] != oc_nthread or knobs["h_nthread"] != 1:
            continue
        attempted += 1
        analytic = analytic_certificate(selected, knobs, env)
        not_attempted = {
            "status": "not_attempted",
            "tir_sha256": None,
            "dma": None,
            "failure": {"category": "analytic_prefilter"},
        }
        if analytic["passed"]:
            original = lower_candidate(tasks[0], tasks[0].config_space.get(index), env)
            residency = (
                lower_candidate(tasks[mode_number], config, env)
                if original["status"] == "ok"
                else {
                    "status": "not_attempted",
                    "tir_sha256": None,
                    "dma": None,
                    "failure": {"category": "original_rejected"},
                }
            )
        else:
            original = dict(not_attempted)
            residency = dict(not_attempted)
        reduction = None
        eligible = False
        if original["status"] == residency["status"] == "ok":
            before = int(original["dma"].get("load_buffer_2d_inp_bytes", 0))
            after = int(residency["dma"].get("load_buffer_2d_inp_bytes", 0))
            reduction = None if before == 0 else (before - after) / before
            eligible = reduction is not None and reduction > 0
        if not analytic["passed"]:
            reject_reason = analytic["reasons"][0]
        elif original["status"] != "ok":
            reject_reason = "same_tile_original_rejected"
        elif residency["status"] != "ok":
            reject_reason = residency["failure"]["category"]
        elif not eligible:
            reject_reason = "no_input_dma_reduction"
        else:
            reject_reason = None
        identity = {
            "workload_id": workload_id,
            "template": TEMPLATE,
            "schedule_sha256": schedule_sha256,
            "hardware": {
                "target": env.TARGET,
                "batch": env.BATCH,
                "block_in": env.BLOCK_IN,
                "block_out": env.BLOCK_OUT,
                "inp_buff_size": env.INP_BUFF_SIZE,
                "wgt_buff_size": env.WGT_BUFF_SIZE,
                "acc_buff_size": env.ACC_BUFF_SIZE,
                "uop_buff_size": env.UOP_BUFF_SIZE,
            },
            "mode": residency_mode,
            "workload": selected["workload"],
            "config": config.to_json_dict(),
        }
        rows.append(
            {
                "schema": SCHEMA,
                "candidate_id": canonical_sha256(identity),
                "workload_id": workload_id,
                "candidate_role": "p7r_joint_candidate" if eligible else "p7r_rejected",
                "status": "ok" if eligible else "rejected",
                "residence_mode": residency_mode,
                "config_index": index,
                "debug": {"config_index": index},
                "identity": {
                    "workload": selected["workload"],
                    "complete_config_entity": config.to_json_dict(),
                    "schedule_sha256": schedule_sha256,
                },
                "tir_sha256": residency["tir_sha256"],
                "tir_hash_encoding": "tvm.ir.save_json",
                "knobs": knobs,
                "analytic_certificate": analytic,
                "original": original,
                residency_mode: residency,
                "input_dma_reduction_fraction": reduction,
                "locally_eligible": eligible,
                "reject_reason": reject_reason,
            }
        )
        if progress_every and attempted % progress_every == 0:
            print(
                "{}: scanned {}, eligible {}".format(
                    workload_id, attempted, sum(row["locally_eligible"] for row in rows)
                ),
                flush=True,
            )
    return rows


def summarize(rows, selected_by_id):
    reasons = Counter(row["reject_reason"] for row in rows if row["reject_reason"])
    eligible = [row for row in rows if row["locally_eligible"]]
    eligible.sort(
        key=lambda row: (
            -row["input_dma_reduction_fraction"],
            -row["knobs"]["tile_h"]
            * row["knobs"]["tile_w"]
            * row["knobs"]["tile_co"],
            row["config_index"],
        )
    )
    workloads = {}
    for workload_id in sorted({row["workload_id"] for row in rows}):
        group = [row for row in rows if row["workload_id"] == workload_id]
        workloads[workload_id] = {
            "incumbent_index": (
                None
                if selected_by_id[workload_id].get("incumbent_index") is None
                else int(selected_by_id[workload_id]["incumbent_index"])
            ),
            "attempted": len(group),
            "locally_eligible": sum(row["locally_eligible"] for row in group),
            "top_label_free_candidates": [
                {
                    "candidate_id": row["candidate_id"],
                    "config_index": row["config_index"],
                    "knobs": row["knobs"],
                    "input_dma_reduction_fraction": row["input_dma_reduction_fraction"],
                    "original_input_bytes": row["original"]["dma"][
                        "load_buffer_2d_inp_bytes"
                    ],
                    "residency_input_bytes": row[row["residence_mode"]]["dma"][
                        "load_buffer_2d_inp_bytes"
                    ],
                }
                for row in eligible
                if row["workload_id"] == workload_id
            ][:10],
        }
    return {
        "schema": SCHEMA,
        "scope": "local lowering and logical DMA only; no FSim, RPC, board, latency, or FPS",
        "attempted": len(rows),
        "locally_eligible": len(eligible),
        "reject_reasons": dict(sorted(reasons.items())),
        "workloads": workloads,
    }


def main():
    base = Path(__file__).resolve().parent / "report_out" / "stage_tile_cotuning"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selected-tasks", default=str(base / "iteration3_tophub_incumbent" / "selected_tasks.json")
    )
    parser.add_argument("--workloads", type=workload_ids, default=("W04",))
    parser.add_argument(
        "--extra-geometries",
        help="Optional JSON object mapping new workload IDs to workload/incumbent records",
    )
    parser.add_argument("--oc-nthread", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--residency-mode",
        choices=tuple(RESIDENCY_MODE_NUMBERS),
        default="input_stationary",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--progress-every", type=int, default=20)
    args = parser.parse_args()

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    output.mkdir(parents=True)
    selected_path = Path(args.selected_tasks)
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    selected_by_id = {"W{:02d}".format(index): row for index, row in enumerate(selected)}
    selected_by_id.update(EXTRA_GEOMETRIES)
    if args.extra_geometries:
        extra_path = Path(args.extra_geometries)
        extras = json.loads(extra_path.read_text(encoding="utf-8"))
        if not isinstance(extras, dict):
            raise ValueError("extra geometries must be a JSON object")
        overlap = set(extras) & set(selected_by_id)
        if overlap:
            raise ValueError("extra geometry IDs already exist: {}".format(sorted(overlap)))
        selected_by_id.update(extras)
    missing = set(args.workloads) - set(selected_by_id)
    if missing:
        raise ValueError("unknown workload IDs: {}".format(sorted(missing)))

    env = vta.get_env()
    source = Path(__file__).resolve()
    schedule = source.parents[2] / "python" / "vta" / "top" / "vta_conv2d.py"
    schedule_sha256 = sha256_file(schedule)
    rows = []
    for workload_id in args.workloads:
        rows.extend(
            scan_workload(
                workload_id,
                selected_by_id[workload_id],
                env,
                args.progress_every,
                schedule_sha256,
                args.oc_nthread,
                args.residency_mode,
            )
        )
    summary = summarize(rows, selected_by_id)

    protocol = (
        base
        / "c3_dma_residency_autotune"
        / "00_governance"
        / "P7R_VTHREAD_JOINT_METHOD.md"
    )
    manifest = {
        "schema": SCHEMA,
        "status": "completed",
        "environment": {
            "python": sys.executable,
            "python_version": platform.python_version(),
            "tvm_version": tvm.__version__,
            "vta_target": env.TARGET,
            "board_contacted": False,
        },
        "filter": {
            "oc_nthread": args.oc_nthread,
            "h_nthread": 1,
            "residency_mode": args.residency_mode,
        },
        "inputs": {
            "selected_tasks": str(selected_path),
            "selected_tasks_sha256": sha256_file(selected_path),
            "protocol": str(protocol),
            "protocol_sha256": sha256_file(protocol),
            "scanner_sha256": sha256_file(source),
            "schedule_sha256": sha256_file(schedule),
        },
        "workloads": list(args.workloads),
    }
    if args.extra_geometries:
        manifest["inputs"]["extra_geometries"] = str(Path(args.extra_geometries).resolve())
        manifest["inputs"]["extra_geometries_sha256"] = sha256_file(args.extra_geometries)
    (output / "results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "command.txt").write_text(" ".join([sys.executable] + sys.argv) + "\n", encoding="utf-8")
    status = [
        "# P7R local joint-space certificate",
        "",
        "- Status: `completed`",
        "- Scope: local VTA lowering and logical DMA only; no board/latency/FPS",
        "- Workloads: {}".format(", ".join(args.workloads)),
        "- Joint candidates attempted: {}".format(summary["attempted"]),
        "- Locally eligible (both paths lower and input bytes decrease): {}".format(
            summary["locally_eligible"]
        ),
        "- Reject reasons: `{}`".format(json.dumps(summary["reject_reasons"], sort_keys=True)),
        "",
        "Eligibility is not a speed or correctness claim. Next gate is three-seed FSim, followed by",
        "cross-compilation and real-FPGA correctness before any timing.",
    ]
    (output / "STATUS.md").write_text("\n".join(status) + "\n", encoding="utf-8")
    hashes = {
        path.name: sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    (output / "artifact_hashes.json").write_text(
        json.dumps({"artifacts": hashes}, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
