#!/usr/bin/env python3
"""Select the first statically feasible latency-unseen YOLOv3-tiny resolution.

No RPC connection or board timing is permitted.  Resolutions are attempted in
the frozen order.  A passing result has completed Darknet import, one full-graph
quantization, graph_pack, AutoTVM task extraction, a conservative graph-memory
contract, and label-free candidate-identity generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import tvm
from tvm import autotvm, relay
from tvm.relay.testing.darknet import __darknetffi__
import vta
from vta.top import graph_pack

from build_vta_c3_real_task_inventory import darknet_call_instances
from c3_candidate_identity import canonical_json_bytes
from prepare_vta_p7r115_yolo_confirmation import create_tasks, enumerate_shared_domain
from tune_resnet18_vta import register_vta_conv2d_template


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_OUTPUT = GROUPED / "20260924_c3_yolo_unseen_resolution_prepare_run05"
DEFAULT_CFG = Path("/home/orange/.tvm_test_data/darknet/yolov3-tiny.cfg")
DEFAULT_WEIGHTS = Path("/tmp/tvm_test_data/darknet/yolov3-tiny.weights")
DEFAULT_DARKNET_LIB = Path("/home/orange/.tvm_test_data/darknet/libdarknet2.0.so")
RESOLUTIONS = (608, 576, 544, 512, 480, 448, 384, 352)
LATENCY_EXPOSURE = {
    608: {
        "status": "latency_exposed",
        "evidence": "20260924_c3_yolo608_composable_confirmation_boot1_run01",
        "reason": "full-graph stock/composable latency was observed before this run03 selection",
    },
    576: {
        "status": "latency_exposed",
        "evidence": "20260924_c3_yolo576_transition_safe_timing_boot1_run01",
        "reason": (
            "complete operator-pool latency was observed before selector-v4 holdout selection"
        ),
    },
    544: {
        "status": "latency_exposed",
        "evidence": "20260924_c3_yolo544_selector_v4_timing_boot1_run01",
        "reason": (
            "complete operator-pool latency was observed before the production-gated "
            "selector-v4 holdout selection"
        ),
    },
}
EXPECTED_WEIGHTS_SHA256 = "dccea06f59b781ec1234ddf8d1e94b9519a97f4245748a7d4db75d5b7080a42c"
UDMABUF_BYTES = 192 * 1024 * 1024
COMMAND_BACKING_BYTES = 2 * 32 * 1024 * 1024
SCHEMA = "c3_yolo_unseen_resolution_prepare_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def derive_cfg(base: Path, resolution: int) -> str:
    lines = base.read_text().splitlines(keepends=True)
    in_net = False
    changed = Counter()
    result = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_net = stripped == "[net]"
        if in_net and "=" in line:
            key, _ = line.split("=", 1)
            if key.strip() in ("width", "height"):
                newline = "\n" if line.endswith("\n") else ""
                result.append(f"{key.rstrip()}={resolution}{newline}")
                changed[key.strip()] += 1
                continue
        result.append(line)
    if changed != Counter({"width": 1, "height": 1}):
        raise ValueError(f"expected exactly one width and height in [net], got {changed}")
    return "".join(result)


def resolution_admission(resolution: int) -> Dict[str, Any]:
    exposure = LATENCY_EXPOSURE.get(int(resolution))
    return {
        "resolution": int(resolution),
        "eligible": exposure is None,
        "exposure": exposure,
        "rule": "exclude any resolution with previously observed target-board full-graph latency",
    }


def dtype_bytes(dtype: str) -> int:
    parsed = tvm.runtime.DataType(dtype)
    return math.ceil(parsed.bits * parsed.lanes / 8)


def align(value: int, alignment: int = 256) -> int:
    return (value + alignment - 1) // alignment * alignment


def graph_memory_contract(graph_text: str) -> Dict[str, Any]:
    graph = json.loads(graph_text)
    attrs = {name: value[1] for name, value in graph["attrs"].items()}
    shapes = attrs["shape"]
    dtypes = attrs["dltype"]
    storage_ids = attrs["storage_id"]
    devices = attrs.get("device_index", [None] * len(storage_ids))
    if not (len(shapes) == len(dtypes) == len(storage_ids) == len(devices)):
        raise ValueError("inconsistent GraphExecutor entry attributes")
    pools: Dict[tuple, int] = {}
    for shape, dtype, storage_id, device in zip(shapes, dtypes, storage_ids, devices):
        if int(storage_id) < 0:
            continue
        nbytes = math.prod(int(value) for value in shape) * dtype_bytes(dtype)
        key = (device, int(storage_id))
        pools[key] = max(pools.get(key, 0), nbytes)
    aligned = {(device, sid): align(value) for (device, sid), value in pools.items()}
    by_device: Dict[str, int] = {}
    for (device, _), value in aligned.items():
        by_device[str(device)] = by_device.get(str(device), 0) + value
    graph_bound = sum(aligned.values())
    total = graph_bound + COMMAND_BACKING_BYTES
    return {
        "scope": "conservative all-device GraphExecutor storage pools plus one VTA insn/uop queue pair",
        "graph_pool_aligned_bytes": graph_bound,
        "graph_pool_aligned_bytes_by_device": by_device,
        "storage_pool_count": len(aligned),
        "command_backing_bytes": COMMAND_BACKING_BYTES,
        "static_contract_bytes": total,
        "udmabuf_capacity_bytes": UDMABUF_BYTES,
        "fits": total <= UDMABUF_BYTES,
        "caveat": "static conservative admission bound; not measured allocator high-water",
    }


def build_variant(cfg_path: Path, weights: Path, darknet_lib: Path, env) -> Dict[str, Any]:
    register_vta_conv2d_template()
    net = __darknetffi__.dlopen(str(darknet_lib)).load_network(
        str(cfg_path).encode(), str(weights).encode(), 0
    )
    shape = (env.BATCH, net.c, net.h, net.w)
    mod, params = relay.frontend.from_darknet(net, dtype="float32", shape=shape)
    imported = tvm.transform.Sequential([relay.transform.InferType()])(mod)
    with tvm.transform.PassContext(opt_level=3):
        with relay.quantize.qconfig(global_scale=23.0, skip_conv_layers=[0],
                                    store_lowbit_output=True, round_for_shift=True):
            quantized = relay.quantize.quantize(imported, params=params)
    quantized = relay.transform.InferType()(quantized)
    packed = graph_pack(
        quantized["main"], env.BATCH, env.BLOCK_OUT, env.WGT_WIDTH,
        start_name="nn.max_pool2d", stop_name="cast", start_name_idx=4, stop_name_idx=186,
    )
    packed_mod = relay.transform.InferType()(tvm.IRModule.from_expr(packed))
    tasks = autotvm.task.extract_from_program(
        packed_mod, params={}, ops=(relay.op.get("nn.conv2d"),),
        target=env.target, target_host=env.target_host,
    )
    tasks = [task for task in tasks if len(task.args[0][1]) > 4 and "conv" in task.name]
    unique = {}
    for task in tasks:
        unique.setdefault(digest(task.workload), task)

    task_rows = []
    for task_id, task in sorted(unique.items()):
        modes = create_tasks(task.workload, env)
        _, selected, per_mode = enumerate_shared_domain(task_id, task.workload, modes, env, 2)
        identities = []
        for family in selected:
            for mode in sorted(modes):
                identities.append(digest({"workload": task.workload, "mode": mode,
                                          "config": family["complete_config_entity"]}))
        task_rows.append({
            "task_id": task_id,
            "workload": task.workload,
            "occurrences": sum(digest(item.workload) == task_id for item in tasks),
            "config_space_by_mode": {mode: values["config_count"] for mode, values in per_mode.items()},
            "selected_static_family_count": len(selected),
            "static_candidate_identity_count": len(set(identities)),
            "candidate_identity_status": "label-free static predicate; lowering/FSim equivalence pending",
            "selected_families": selected,
        })

    target = tvm.target.Target(env.target, host=env.target_host)
    with vta.build_config(opt_level=3, disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"}):
        built = relay.build(packed_mod, target=target, params={})
    if hasattr(built, "get_graph_json"):
        graph_text = built.get_graph_json()
    else:
        graph_text = built[0]
    return {
        "network_shape": list(shape),
        "imported_ir_sha256": hashlib.sha256(tvm.ir.save_json(imported).encode()).hexdigest(),
        "quantized_ir_sha256": hashlib.sha256(tvm.ir.save_json(quantized).encode()).hexdigest(),
        "packed_ir_sha256": hashlib.sha256(tvm.ir.save_json(packed_mod).encode()).hexdigest(),
        "extracted_task_count": len(tasks),
        "unique_vta_execution_task_count": len(task_rows),
        "tasks_with_at_least_eight_static_identities": sum(
            row["static_candidate_identity_count"] >= 8 for row in task_rows
        ),
        "tasks": task_rows,
        "memory_contract": graph_memory_contract(graph_text),
        "graph_json_sha256": hashlib.sha256(graph_text.encode()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-cfg", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--darknet-lib", type=Path, default=DEFAULT_DARKNET_LIB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite immutable output: {args.output}")
    if sha256(args.weights) != EXPECTED_WEIGHTS_SHA256:
        raise SystemExit("YOLO weights do not match the historical frozen SHA256")
    args.output.mkdir(parents=True)
    base_copy = args.output / "yolov3-tiny.base.cfg"
    base_copy.write_bytes(args.base_cfg.read_bytes())
    env = vta.get_env()
    attempts = []
    selected = None
    for resolution in RESOLUTIONS:
        attempt_dir = args.output / f"resolution_{resolution}"
        attempt_dir.mkdir()
        cfg_path = attempt_dir / f"yolov3-tiny-{resolution}.cfg"
        cfg_path.write_text(derive_cfg(args.base_cfg, resolution))
        admission = resolution_admission(resolution)
        row: Dict[str, Any] = {
            "resolution": resolution,
            "status": "failed",
            "base_cfg_sha256": sha256(args.base_cfg),
            "derived_cfg_sha256": sha256(cfg_path),
            "weights_sha256": sha256(args.weights),
            "darknet_lib_sha256": sha256(args.darknet_lib),
            "board_contacted": False,
            "latency_read": False,
            "latency_exposure_admission": admission,
        }
        if not admission["eligible"]:
            row["status"] = "excluded_latency_exposed"
            write_json(attempt_dir / "result.json", row)
            attempts.append(row)
            continue
        try:
            calls = darknet_call_instances(cfg_path)
            row["darknet_convolution_calls"] = len(calls)
            row.update(build_variant(cfg_path, args.weights, args.darknet_lib, env))
            row["gates"] = {
                "darknet_import_quantize_graph_pack": True,
                "static_memory_at_most_192_mib": row["memory_contract"]["fits"],
                "at_least_five_vta_tasks": row["unique_vta_execution_task_count"] >= 5,
                "at_least_two_tasks_with_eight_static_identities": (
                    row["tasks_with_at_least_eight_static_identities"] >= 2
                ),
                "local_lowering_fsim_non_equivalence": "pending",
            }
            row["status"] = "static_feasible_pending_local_program_qualification" if all(
                value is True for key, value in row["gates"].items()
                if key != "local_lowering_fsim_non_equivalence"
            ) else "static_gate_failed"
        except Exception as error:  # TVM has heterogeneous compiler errors.
            row["failure"] = {"type": type(error).__name__, "message": str(error)[:4000],
                              "traceback": traceback.format_exc()[-8000:]}
        write_json(attempt_dir / "result.json", row)
        attempts.append(row)
        if row["status"] == "static_feasible_pending_local_program_qualification":
            selected = row
            break

    write_jsonl(args.output / "attempts.jsonl", attempts)
    contract = {
        "schema": SCHEMA,
        "status": ("selected_static_feasible_resolution_pending_local_qualification"
                   if selected else "no_static_feasible_resolution"),
        "resolution_order": list(RESOLUTIONS),
        "latency_exposure_registry": LATENCY_EXPOSURE,
        "selected_resolution": selected["resolution"] if selected else None,
        "selection_rule": (
            "first non-exposed static-feasible resolution in frozen order; exposure status is a "
            "hard exclusion and no latency value is read"
        ),
        "base_cfg": {"path": str(args.base_cfg), "sha256": sha256(args.base_cfg)},
        "derived_cfg_change_allowlist": ["net.width", "net.height"],
        "weights": {"path": str(args.weights), "sha256": sha256(args.weights)},
        "darknet_lib": {"path": str(args.darknet_lib), "sha256": sha256(args.darknet_lib)},
        "board_contacted": False,
        "latency_values_read": False,
        "claim_boundary": "static feasibility only; candidate lowering/FSim equivalence remains required",
    }
    write_json(args.output / "contract.json", contract)
    write_json(args.output / "summary.json", contract)
    write_json(args.output / "artifact_hashes.json", {
        "base_cfg": sha256(base_copy), "attempts": sha256(args.output / "attempts.jsonl"),
        "contract": sha256(args.output / "contract.json"), "summary": sha256(args.output / "summary.json"),
        "source": sha256(Path(__file__).resolve()),
    })
    print(json.dumps(contract, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
