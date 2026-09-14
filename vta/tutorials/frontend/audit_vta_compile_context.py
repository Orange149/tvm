#!/usr/bin/env python3
"""E0/E1: audit the frozen legal domain before codegen; qualify three real builds."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess

import tvm
from tvm import autotvm, relay, te
from mxnet.gluon.model_zoo import vision
import vta

from profile_split_resnet18_stages import prepare_vta_stage, build_vta_stage
from split_resnet18_stages import (UNIT_ORDER, build_resnet18_unit_blocks,
    build_resnet18_unit_metadata, relay_inputs_for_stage, make_stage_block, lower_stage_to_relay)
from vta_tuning_history import audited_history, tophub_identity

ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "vta/tutorials/frontend/report_out"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def value(obj):
    if obj is None or isinstance(obj, (bool, str, int, float)):
        return obj
    if isinstance(obj, (tvm.tir.IntImm, tvm.tir.FloatImm, tvm.tir.StringImm)):
        return obj.value
    if isinstance(obj, tvm.ir.container.Array):
        return [value(x) for x in obj]
    return re.sub(r"id=[0-9a-f]+", "id=normalized", str(obj))


def attrs(obj):
    if obj is None:
        return {}
    # Compiler-generated labels and placement metadata are not fusion structure.
    # The actual target/device configuration is separately frozen in provenance.
    return {str(k): value(obj[k]) for k in sorted(obj.keys())
            if str(k) not in {"hash", "global_symbol", "virtual_device"}}


def constant(node):
    data = node.data.numpy()
    return {"shape": list(data.shape), "dtype": str(data.dtype), "data_sha256": sha(data.tobytes())}


def signature(expr):
    """Alpha-invariant typed expression DAG, preserving attrs and scalar constants."""
    nodes, seen, variables = [], {}, {}

    def visit(node):
        if node in seen:
            return seen[node]
        if isinstance(node, relay.Var):
            if node not in variables:
                variables[node] = len(variables)
            item = ["var", variables[node], str(node.checked_type)]
        elif isinstance(node, relay.Constant):
            item = ["constant", constant(node)]
        elif isinstance(node, relay.Call):
            op = node.op.name if isinstance(node.op, tvm.ir.Op) else ["expr", visit(node.op)]
            item = ["call", op, [visit(a) for a in node.args], attrs(node.attrs), str(node.checked_type)]
        elif isinstance(node, relay.Function):
            item = ["function", [visit(p) for p in node.params], visit(node.body),
                    str(node.ret_type), attrs(node.attrs)]
        elif isinstance(node, relay.Tuple):
            item = ["tuple", [visit(x) for x in node.fields]]
        elif isinstance(node, relay.TupleGetItem):
            item = ["get", visit(node.tuple_value), node.index]
        elif isinstance(node, relay.Let):
            item = ["let", visit(node.var), visit(node.value), visit(node.body)]
        else:
            raise TypeError("Unsupported Relay node: " + str(type(node)))
        index = len(nodes)
        nodes.append(item)
        seen[node] = index
        return index

    root = visit(expr)
    payload = {"nodes": nodes, "root": root}
    return sha(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()), payload


@tvm.instrument.pass_instrument
class Capture:
    def __init__(self):
        self.before_fuse = []
        self.after_fuse = []
        self.tir_passes = []

    def run_before_pass(self, mod, info):
        if info.name == "FuseOps":
            self.before_fuse.append(mod)

    def run_after_pass(self, mod, info):
        if info.name == "FuseOps":
            self.after_fuse.append(mod)
        if str(info.name).startswith("tir."):
            self.tir_passes.append(str(info.name))


def conv_contexts(mod, target):
    records, primitives = [], []

    def visit(call):
        if not (isinstance(call, relay.Call) and isinstance(call.op, relay.Function)
                and call.op.attrs is not None and "Primitive" in call.op.attrs.keys()
                and int(call.op.attrs["Primitive"]) == 1):
            return
        function = call.op
        digest, normalized = signature(function)
        convs, ops = [], []
        def inside(node):
            if isinstance(node, relay.Call) and isinstance(node.op, tvm.ir.Op):
                ops.append(node.op.name)
                if node.op.name == "nn.conv2d":
                    convs.append(node)
        relay.analysis.post_order_visit(function.body, inside)
        primitives.append({"signature": digest, "ops": ops, "normalized": normalized})
        for conv in convs:
            types = [a.checked_type for a in conv.args[:2]]
            tensors = [te.placeholder(tuple(int(d) for d in t.shape), dtype=t.dtype) for t in types]
            workload = autotvm.task.args_to_workload((tensors[0], tensors[1],
                tuple(int(x) for x in conv.attrs.strides), tuple(int(x) for x in conv.attrs.padding),
                tuple(int(x) for x in conv.attrs.dilation), str(conv.attrs.data_layout),
                str(conv.attrs.out_dtype)), "conv2d_packed.vta")
            cfg = autotvm.task.DispatchContext.current.query(target, workload)
            binding = dict(zip(function.params, call.args))
            weight = binding.get(conv.args[1], conv.args[1])
            if not isinstance(weight, relay.Constant):
                raise ValueError("Convolution weight not a constant at pre-codegen invocation")
            records.append({"primitive_signature": digest, "primitive_ops": ops,
                            "weight": constant(weight), "workload": workload,
                            "config": cfg.to_json_dict(), "is_fallback": bool(cfg.is_fallback),
                            "kernel_size": [int(x) for x in conv.attrs.kernel_size],
                            "input_types": [str(p.checked_type) for p in function.params],
                            "return_type": str(function.ret_type)})
    relay.analysis.post_order_visit(mod["main"].body, visit)
    return records, primitives


def map_occurrences(rows, metadata):
    registry = {}
    by_id = {r["segment_id"]: r for r in rows if r["status"] == "ok"}
    for index, unit in enumerate(UNIT_ORDER):
        count = metadata[unit]["conv_count"]
        if not count or unit == "stem":
            continue
        start = index - 1 if unit.endswith("skip_proj") else index
        ref = by_id.get(f"vta:{start:02d}:{index:02d}")
        if ref is None:
            continue
        convs = ref["convolutions"]
        if unit.endswith("skip_proj"):
            convs = [c for c in convs if c["kernel_size"] == [1, 1]]
        assert len(convs) == count, (unit, len(convs), count)
        for ordinal, conv in enumerate(convs):
            key = json.dumps(conv["weight"], sort_keys=True)
            occurrence = f"unit{index:02d}.conv{ordinal}"
            assert key not in registry, ("ambiguous quantized weight identity", occurrence)
            registry[key] = {"occurrence": occurrence, "unit": unit, "reference_segment": ref["segment_id"]}
    groups, unresolved = defaultdict(list), []
    for row in rows:
        if row["status"] != "ok":
            continue
        seen = set()
        for conv in row["convolutions"]:
            identity = registry.get(json.dumps(conv["weight"], sort_keys=True))
            if identity is None:
                unresolved.append({"segment": row["segment_id"], "weight": conv["weight"]})
                continue
            assert identity["unit"] in row["unit_names"]
            assert identity["occurrence"] not in seen
            seen.add(identity["occurrence"])
            conv["semantic_identity"] = identity
            groups[identity["occurrence"]].append({"segment": row["segment_id"],
                "primitive_signature": conv["primitive_signature"],
                "workload": conv["workload"], "config": conv["config"]})
    return {"weight_identity_registry_size": len(registry), "unresolved": unresolved,
            "occurrences": [{"occurrence": k, "placements": v,
                "unique_primitive_contexts": len({x["primitive_signature"] for x in v}),
                "unique_workloads": len({json.dumps(x["workload"]) for x in v}),
                "unique_configs": len({json.dumps(x["config"], sort_keys=True) for x in v})}
                for k, v in sorted(groups.items())]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--segments", nargs="*", help="Default: all 87 manifest segments")
    parser.add_argument("--qualify-builds", action="store_true", help="Build only 15:15/16/17 for path equivalence")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    profile_path = REPORT / "resource_aware_maxplus/v1_profile_manifest.json"
    domain = [s for s in json.loads(profile_path.read_text())["segments"] if s["device"] == "vta"]
    assert len(domain) == 87
    if args.segments:
        domain = [s for s in domain if s["segment_id"] in args.segments]
        assert len(domain) == len(args.segments)
    env = vta.get_env()
    model = vision.get_model("resnet18_v1", pretrained=True)
    features, head = list(model.features), model.output
    units = build_resnet18_unit_blocks(features, head)
    metadata = build_resnet18_unit_metadata(units, env.BATCH, 224)
    sources = ["vta/tutorials/frontend/split_resnet18_stages.py",
        "vta/tutorials/frontend/profile_split_resnet18_stages.py",
        "vta/tutorials/frontend/deploy_classification_stage_pipeline_native.py",
        "vta/tutorials/frontend/vta_tuning_history.py", "src/relay/backend/build_module.cc",
        "src/relay/transforms/fuse_ops.cc", "python/tvm/relay/build_module.py",
        "python/tvm/relay/quantize/quantize.py", "vta/python/vta/top/graphpack.py",
        "vta/python/vta/top/vta_conv2d.py", "vta/python/vta/build_module.py"]
    save(args.output / "provenance.json", {"git_head": subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
         "dirty_worktree": True, "sources": {p: sha((ROOT/p).read_bytes()) for p in sources},
         "script_sha256": sha(Path(__file__).read_bytes()), "profile_sha256": sha(profile_path.read_bytes()),
         "target": str(env.target), "host": str(env.target_host), "tophub": tophub_identity(env.target),
         "quantize": {"global_scale":8.0,"skip_conv_layers":[]},
         "disabled_passes":["AlterOpLayout","tir.CommonSubexprElimTIR"],
         "domain": [s["segment_id"] for s in domain], "full_segment_compile_sweep_forbidden": True,
         "qualification_builds": ["vta:15:15","vta:15:16","vta:15:17"] if args.qualify_builds else []})
    rows = []
    for number, segment in enumerate(domain):
        sid = segment["segment_id"]
        print(f"[AUDIT {number+1}/{len(domain)}] {sid}", flush=True)
        local = args.output / sid.replace(":", "_")
        local.mkdir()
        row = {"segment_id": sid, "unit_names": segment["unit_names"],
               "input_contract": segment["input_contract"], "output_contract": segment["output_contract"]}
        try:
            stage = {"name": "stage_audit", "device":"vta", "unit_names":segment["unit_names"]}
            block = make_stage_block(features,head,stage,unit_blocks=units)
            inputs = relay_inputs_for_stage(stage,metadata)
            mod,params = lower_stage_to_relay(block,inputs)
            (local/"original.relay").write_text(mod.astext(show_meta_data=False))
            save(local/"params_identity.json", {k:constant(relay.const(v)) for k,v in params.items()})
            def observer(phase, observed):
                (local/(phase+".relay")).write_text(observed.astext(show_meta_data=False))
            packed,target = prepare_vta_stage(mod["main"],params,env,observer=observer)
            prepared = tvm.IRModule.from_expr(packed).with_attr("executor", relay.backend.Executor("graph"))
            prepared = prepared.with_attr("runtime", relay.backend.Runtime("cpp"))
            capture, audit = Capture(), {}
            with audited_history("",audit,True,tophub_targets=target), vta.build_config(
                    opt_level=3,disabled_pass={"AlterOpLayout","tir.CommonSubexprElimTIR"}, instruments=[capture]):
                optimized,_ = relay.optimize(prepared,target=target,params=params)
                convolutions,primitives = conv_contexts(optimized,env.target)
            assert len(capture.after_fuse) == 1
            assert not capture.tir_passes, "pre-codegen audit unexpectedly entered TIR"
            expected_count = sum(metadata[u]["conv_count"] for u in segment["unit_names"])
            assert len(convolutions) == expected_count
            (local/"optimized.relay").write_text(optimized.astext(show_meta_data=False))
            save(local/"primitives.json",primitives)
            row.update(status="ok",convolutions=convolutions,primitive_count=len(primitives),
                       tuning=audit, expected_conv_count=expected_count,
                       dispatch_scope="explicit query from optimized packed conv; actual lowering checked only on qualification builds")
            if args.qualify_builds and sid in {"vta:15:15","vta:15:16","vta:15:17"}:
                full, full_audit = Capture(), {}
                graph,lib,_ = build_vta_stage("stage_audit",mod["main"],params,env,
                    tuning_audit=full_audit,require_tuned=True,compile_instruments=[full])
                (local/"graph.json").write_text(graph)
                lib.save(str(local/"graphlib.o"))
                exact = {"before_fuse": tvm.ir.structural_equal(capture.before_fuse[0],full.before_fuse[0],map_free_vars=True),
                         "after_fuse": tvm.ir.structural_equal(capture.after_fuse[0],full.after_fuse[0],map_free_vars=True)}
                normalize = lambda a: sorted(json.dumps([r["workload"],r["config"]],sort_keys=True) for r in a["rows"])
                exact["workload_config_dispatch"] = normalize(audit)==normalize(full_audit)
                row["qualification"] = dict(exact, actual_tuning=full_audit, tir_passes=full.tir_passes)
                assert all(exact.values()), ("optimize/build path mismatch",exact)
            print("[OK]",sid,"convs",len(convolutions),"primitives",len(primitives),flush=True)
        except Exception as error:
            row.update(status="failed",error=str(error))
            print("[FAIL]",sid,str(error)[:500],flush=True)
        rows.append(row)
        save(local/"audit.json",row)
        save(args.output/"summary.json",{"rows":rows,"complete":False})
    mapping = map_occurrences(rows,metadata)
    save(args.output/"occurrence_contexts.json",mapping)
    save(args.output/"summary.json",{"rows":rows,"complete":True,
         "domain_segments":len(domain),"successful_segments":sum(r["status"]=="ok" for r in rows),
         "mapping_unresolved":len(mapping["unresolved"]),
         "level2_complete":False,"full_segment_tir_dma_complete":False})
    print("[COMPLETE] segments",len(rows),"unresolved",len(mapping["unresolved"]),flush=True)
    if any(r["status"] != "ok" for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
