#!/usr/bin/env python3
"""Lower two fusion contexts for each frozen C1 E4 cross-model tile."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import tvm
from tvm import autotvm, relay, tir
import vta
from vta.top.residency_dispatch import ExplicitResidencyDispatch

from audit_vta_compile_context import signature
from extract_static_vta_dma import extract_module_dma_compact
from fusion_aware_transfer import digest_value, read_json, sha256_file
from prepare_vta_c1_cross_model_e4 import canonical, seal


ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "vta/tutorials/frontend/report_out"
DEFAULT_FREEZE = (
    REPORT
    / "stage_tile_cotuning/compile_context_audit/c1_e4_fixed_tiles_run1/fixed_tiles.json"
)


def validate_sealed(payload, label):
    expected = payload.get("artifact_sha256")
    unsealed = dict(payload)
    unsealed.pop("artifact_sha256", None)
    observed = hashlib.sha256(canonical(unsealed).encode("utf-8")).hexdigest()
    if expected != observed:
        raise ValueError(label + " artifact hash mismatch")


def relay_module(workload, with_residual):
    data_desc, weight_desc = workload[1], workload[2]
    strides, padding, dilation, layout, out_dtype = workload[3:]
    data = relay.var("data", shape=tuple(data_desc[1]), dtype=data_desc[2])
    weight = relay.var("weight", shape=tuple(weight_desc[1]), dtype=weight_desc[2])
    channels = int(weight_desc[1][0] * weight_desc[1][4])
    kernel_size = tuple(int(value) for value in weight_desc[1][2:4])
    conv = relay.nn.conv2d(
        data,
        weight,
        strides=tuple(strides),
        padding=tuple(padding[:2]),
        dilation=tuple(dilation),
        groups=1,
        channels=channels,
        kernel_size=kernel_size,
        data_layout=layout,
        kernel_layout="OIHW16o16i",
        out_layout=layout,
        out_dtype=out_dtype,
    )
    params = [data, weight]
    value = conv
    if with_residual:
        inferred = relay.transform.InferType()(tvm.IRModule.from_expr(relay.Function(params, conv)))
        output_shape = tuple(int(value) for value in inferred["main"].body.checked_type.shape)
        residual = relay.var("residual", shape=output_shape, dtype=out_dtype)
        params.append(residual)
        value = relay.add(value, residual)
    shifted = relay.right_shift(value, relay.const(8, "int32"))
    output = relay.cast(relay.clip(shifted, a_min=0, a_max=127), "int8")
    return tvm.IRModule.from_expr(relay.Function(params, output))


def primitive_inventory(mod):
    result = []

    def visit(node):
        if not (
            isinstance(node, relay.Call)
            and isinstance(node.op, relay.Function)
            and node.op.attrs is not None
            and "Primitive" in node.op.attrs.keys()
            and int(node.op.attrs["Primitive"]) == 1
        ):
            return
        digest, normalized = signature(node.op)
        ops = []

        def inside(item):
            if isinstance(item, relay.Call) and isinstance(item.op, tvm.ir.Op):
                ops.append(item.op.name)

        relay.analysis.post_order_visit(node.op.body, inside)
        result.append({"signature": digest, "ops": ops, "normalized": normalized})

    relay.analysis.post_order_visit(mod["main"].body, visit)
    return result


@tvm.instrument.pass_instrument
class Archive:
    def __init__(self, env):
        self.env = env
        self.fused = []
        self.lowered = []

    def run_after_pass(self, mod, info):
        if info.name == "FuseOps":
            self.fused = primitive_inventory(mod)
        if info.name != "tir.vta.CPUAccessRewrite":
            return
        functions = []
        for global_var, function in mod.functions.items():
            if not isinstance(function, tir.PrimFunc):
                continue
            single = tvm.IRModule({"main": function})
            functions.append(
                {
                    "global_var": str(global_var.name_hint),
                    "dma": extract_module_dma_compact(single, self.env),
                }
            )
        self.lowered.append(
            {
                "json": tvm.ir.save_json(mod),
                "tir": mod.script(),
                "functions": functions,
            }
        )


def route_from_frozen(row):
    return {
        "candidate_id": row["candidate_id"],
        "identity": {
            "workload": row["workload"],
            "complete_config_entity": row["config"],
            "public_mode": "original",
            "implementation_mode": 0,
        },
    }


def verify_dispatch_identity(audit, row):
    """Fail closed unless the scheduled route is the exact frozen identity."""
    hits = audit.get("schedule_hits", [])
    if len(hits) != 1:
        raise RuntimeError(
            "expected exactly one frozen schedule hit, observed {}".format(len(hits))
        )
    hit = hits[0]
    expected = {
        "candidate_id": row["candidate_id"],
        "complete_config_entity": row["config"],
        "public_mode": "original",
        "implementation_mode": 0,
        "workload": row["workload"],
    }
    mismatches = [
        key for key, value in expected.items() if canonical(hit.get(key)) != canonical(value)
    ]
    if mismatches:
        raise RuntimeError(
            "scheduled route differs from frozen identity: {}".format(
                ", ".join(mismatches)
            )
        )
    return {
        "exact": True,
        "scheduled_identity_sha256": digest_value(expected),
        "frozen_tile_identity_sha256": row["tile_identity_sha256"],
    }


def build_context(row, context_name, output, env):
    local = output / row["workload_id"] / context_name
    local.mkdir(parents=True)
    with_residual = context_name == "conv_residual_quant"
    archive = Archive(env)
    route = route_from_frozen(row)
    target = tvm.target.Target(env.target, host=env.target_host)
    relay.backend.te_compiler.get().clear()
    started = time.perf_counter()
    with autotvm.tophub.context(target):
        with ExplicitResidencyDispatch([route]) as dispatch:
            with vta.build_config(
                opt_level=3,
                disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"},
                instruments=[archive],
            ):
                graph, _lib, params = relay.build(
                    relay_module(row["workload"], with_residual), target=target
                )
    audit = dispatch.summary()
    if not audit["all_routes_scheduled"]:
        raise RuntimeError("frozen workload/config route was not scheduled")
    dispatch_identity = verify_dispatch_identity(audit, row)
    if not archive.fused or not archive.lowered:
        raise RuntimeError("fusion or CPUAccessRewrite archive is empty")
    (local / "graph.json").write_text(graph, encoding="utf-8")
    (local / "params.bin").write_bytes(relay.save_param_dict(params))
    snapshots = []
    for index, snapshot in enumerate(archive.lowered):
        stem = "lowered_{:03d}".format(index)
        json_path = local / (stem + ".json")
        tir_path = local / (stem + ".tir")
        json_path.write_text(snapshot["json"], encoding="utf-8")
        tir_path.write_text(snapshot["tir"], encoding="utf-8")
        snapshots.append(
            {
                "json": json_path.name,
                "json_sha256": sha256_file(json_path),
                "tir": tir_path.name,
                "tir_sha256": sha256_file(tir_path),
                "functions": snapshot["functions"],
            }
        )
    result = {
        "context": context_name,
        "with_residual": with_residual,
        "build_seconds": time.perf_counter() - started,
        "candidate_id": row["candidate_id"],
        "tile_identity_sha256": row["tile_identity_sha256"],
        "dispatch_audit": audit,
        "dispatch_identity": dispatch_identity,
        "primitive_inventory": archive.fused,
        "primitive_signature_multiset": sorted(
            item["signature"] for item in archive.fused
        ),
        "snapshots": snapshots,
        "graph_sha256": sha256_file(local / "graph.json"),
        "params_sha256": sha256_file(local / "params.bin"),
        "status": "complete",
    }
    (local / "build.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-tiles", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workloads", nargs="*", default=[])
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite {}".format(args.output_dir))
    fixed = read_json(args.fixed_tiles)
    validate_sealed(fixed, "fixed tile freeze")
    if fixed.get("status") != "fixed_before_cross_context_lowering":
        raise ValueError("fixed tiles were not frozen before cross-context lowering")
    selected = fixed["rows"]
    if args.workloads:
        wanted = set(args.workloads)
        selected = [row for row in selected if row["workload_id"] in wanted]
        if {row["workload_id"] for row in selected} != wanted:
            raise ValueError("unknown requested workload")
    args.output_dir.mkdir(parents=True)
    env = vta.get_env()
    results = []
    for row in selected:
        builds = []
        failure = None
        try:
            for context in ("conv_quant", "conv_residual_quant"):
                print("[E4] {} {}".format(row["workload_id"], context), flush=True)
                builds.append(build_context(row, context, args.output_dir, env))
        except Exception as error:
            failure = {
                "exception_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
        complete = failure is None and len(builds) == 2
        results.append(
            {
                "workload_id": row["workload_id"],
                "candidate_id": row["candidate_id"],
                "tile_identity_sha256": row["tile_identity_sha256"],
                "builds": builds,
                "same_tile_identity_across_contexts": complete
                and len({item["tile_identity_sha256"] for item in builds}) == 1,
                "fusion_contexts_distinct": complete
                and builds[0]["primitive_signature_multiset"]
                != builds[1]["primitive_signature_multiset"],
                "failure": failure,
                "status": "complete" if complete else "failed_closed",
            }
        )
        (args.output_dir / "progress.json").write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    complete = all(row["status"] == "complete" for row in results)
    summary = seal(
        {
            "schema": "vta_c1_cross_model_e4_context_lowering_v1",
            "status": "complete" if complete else "completed_with_fail_closed_cases",
            "scope": (
                "compiler-only paired fusion contexts using pre-frozen model-local tiles; "
                "not actual partition placement, shared-edge policy, FPGA, or speed evidence"
            ),
            "fixed_tiles": {
                "path": str(args.fixed_tiles.resolve()),
                "sha256": sha256_file(args.fixed_tiles),
                "artifact_sha256": fixed["artifact_sha256"],
            },
            "builder": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__)),
            },
            "environment": {
                "vta_config": env.cfg_dict,
                "target": str(env.target),
                "host": str(env.target_host),
                "libtvm_sha256": sha256_file(Path(tvm._ffi.base._LIB._name)),
                "process": {
                    key: os.environ.get(key)
                    for key in (
                        "TVM_NUM_THREADS",
                        "TVM_THREAD_POOL_SPIN_COUNT",
                        "OMP_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                    )
                },
            },
            "rows": results,
            "summary": {
                "requested_workloads": len(selected),
                "complete_workloads": sum(row["status"] == "complete" for row in results),
                "same_tile_count": sum(
                    row["same_tile_identity_across_contexts"] for row in results
                ),
                "distinct_fusion_context_count": sum(
                    row["fusion_contexts_distinct"] for row in results
                ),
            },
            "e4_complete": False,
            "remaining_gate": (
                "repeat with actual model partition placements and query exact shared-edge "
                "certificates; these synthetic paired fusion contexts are a mechanism check"
            ),
        }
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["summary"], indent=2))


if __name__ == "__main__":
    main()
