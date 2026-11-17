"""Isolated SDK compiler for the frozen ResNet50 common graph.

No board or performance files are read. One interval module serves all thread
counts; only two additional CPU programs provide independent ARM references.
"""
import argparse
import json
import os
from pathlib import Path
import time

from cpu_vta_partitioning.common import digest, inventory, read, sha256, verify_inventory, write


def program_ir(graph, spec):
    import tvm
    from tvm import relay
    from .units import outputs
    n = len(graph.units)
    if spec["kind"] == "interval":
        mod, incoming, outgoing = graph.stage(spec["begin"], spec["end"])
        graph.recompose([(i, i + 1) for i in range(spec["begin"])]
                        + [(spec["begin"], spec["end"])]
                        + [(i, i + 1) for i in range(spec["end"], n)])
    else:
        incoming = list(graph.mod["main"].params)
        if spec["kind"] == "complete_graph":
            outgoing = outputs(graph.mod["main"].body)
        elif spec["kind"] == "complete_frontier_probe":
            outgoing = list(dict.fromkeys(node for boundary in range(n + 1)
                                          for node in graph.frontier(boundary)))
        else:
            raise ValueError("unknown program kind: " + spec["kind"])
        mod = relay.transform.InferType()(tvm.IRModule.from_expr(
            relay.Function(incoming, relay.Tuple(outgoing))))
    return mod, incoming, outgoing


def packed_stage(mod, env):
    import tvm
    from tvm import relay
    from vta.top.graphpack import graph_pack
    from .units import Replace
    function = mod["main"]
    replacements = {p: relay.Call(tvm.ir.Op.get("annotation.bitpack_start"), [p])
                    for p in function.params}
    body = Replace(replacements).visit(function.body)
    body = relay.Tuple([relay.Call(tvm.ir.Op.get("annotation.bitpack_end"), [x])
                        for x in body.fields])
    wrapped = relay.transform.InferType()(tvm.IRModule.from_expr(
        relay.Function(list(function.params), body)))
    packed = relay.transform.InferType()(tvm.IRModule.from_expr(graph_pack(
        wrapped["main"], env.BATCH, env.BLOCK_OUT, env.WGT_WIDTH,
        start_name=None, stop_name=None, boundary_bridge=True)))
    if not tvm.ir.structural_equal(function.checked_type, packed["main"].checked_type):
        raise ValueError("VTA packing changed the external frontier types")
    return packed


def _history_context(history):
    """Use the frozen project's native dispatch semantics, adding only audit.

    Native ApplyHistoryBest requires the full workload, first matching the model
    and then a target key. Requiring an identical target string here would reject
    compatible records the existing ResNet18 and YOLO compilers already use.
    Kernel record costs select schedules; they never become stage costs.
    """
    from tvm import autotvm
    from tvm.autotvm.task.dispatcher import ApplyHistoryBest

    class AuditedHistory(ApplyHistoryBest):
        def __init__(self):
            super().__init__(str(history))
            self.queries = {}
            self.sources = {}
            for inp, result in list(self.best_by_model.values()) + list(self.best_by_targetkey.values()):
                source = {"target": str(inp.target), "model": str(inp.target.model),
                          "target_keys": list(map(str, inp.target.keys)),
                          "workload": repr(inp.task.workload), "config_index": inp.config.index,
                          "config_sha256": digest(inp.config.to_json_dict()),
                          "kernel_history_cost_seconds": sum(map(float, result.costs)) / len(result.costs)}
                self.sources.setdefault(id(inp.config), source)

        def _query_inside(self, target, workload):
            if target is None:
                raise ValueError("schedule history queried without a target")
            # Delegate selection unchanged, including native user updates.
            configuration = super()._query_inside(target, workload)
            key = (str(target), workload)
            source = self.sources.get(id(configuration)) if configuration is not None else None
            model_key = (target.model, workload)
            if model_key in self._best_user_defined:
                policy = "native_user_update"
            elif model_key in self.best_by_model:
                policy = "native_model_and_full_workload"
            else:
                policy = "framework_default_fallback_no_new_tuning"
                for target_key in target.keys:
                    query_key = (target_key, workload)
                    if query_key in self._best_user_defined:
                        policy = "native_user_update"
                        break
                    if query_key in self.best_by_targetkey:
                        policy = "native_target_key_and_full_workload"
                        break
            self.queries[key] = {"target": str(target), "workload": repr(workload),
                                 "history_match": source is not None,
                                 "native_dispatch_hit": configuration is not None,
                                 "policy": policy, "source_record": source}
            return configuration
    return AuditedHistory()


def compile_program(workspace, name):
    import tvm
    from tvm import relay
    from tvm.contrib import cc
    import vta
    from .units import UnitGraph, outputs

    workspace = Path(workspace).resolve()
    identity = read(workspace / "plans/compiler_identity.json")
    verify_inventory(workspace, identity["files"])
    build_identity = digest(identity)
    specs = read(workspace / "plans/programs.json")
    selected = [s for s in specs if s["name"] == name]
    if len(selected) != 1:
        raise ValueError("program name is not unique in the frozen build plan")
    spec = selected[0]
    destination = workspace / "programs" / name
    marker = destination / "record.json"
    if marker.exists():
        record = read(marker)
        if record["build_identity"] != build_identity or record["spec"] != spec:
            raise ValueError("compiled program belongs to another frozen build")
        verify_inventory(destination, record["files"])
        return record
    destination.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    graph = UnitGraph(tvm.ir.load_json((workspace / "sources/canonical_qir.json").read_text()))
    original, incoming, outgoing = program_ir(graph, spec)
    env = vta.get_env()
    lowered = packed_stage(original, env) if spec["device"] == "vta" else original
    (destination / "lowering_qir.json").write_text(tvm.ir.save_json(lowered))
    target = tvm.target.Target(env.target if spec["device"] == "vta" else env.target_vta_cpu,
                               host=env.target_host)
    context = vta.build_config if spec["device"] == "vta" else tvm.transform.PassContext
    from tvm.autotvm.task import dispatcher
    if sha256(Path(dispatcher.__file__)) != sha256(workspace / "sources/repository/python/tvm/autotvm/task/dispatcher.py"):
        raise ValueError("loaded native AutoTVM dispatcher differs from the frozen source")
    history = _history_context(workspace / "sources/tuning_history.log")
    with history, context(opt_level=3, disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"}):
        factory = relay.build(lowered, target=target)
    (destination / "graph.json").write_text(factory.get_graph_json())
    (destination / "params.bin").write_bytes(relay.save_param_dict(factory.get_params()))
    sysroot = os.environ["SDKTARGETSYSROOT"]
    factory.get_lib().export_library(str(destination / "module.so"),
        fcompile=cc.cross_compiler("aarch64-xilinx-linux-g++"), options=[
            f"--sysroot={sysroot}", f"-Wl,-rpath-link,{sysroot}/lib",
            f"-Wl,-rpath-link,{sysroot}/usr/lib", f"-L{sysroot}/lib", f"-L{sysroot}/usr/lib"])
    def typed(node):
        return {"dtype": str(node.checked_type.dtype), "shape": list(map(int, node.checked_type.shape))}
    terminals = outputs(graph.mod["main"].body)
    queries = sorted(history.queries.values(), key=lambda x: (x["target"], x["workload"]))
    write(destination / "schedule_history.json", {
        "records_sha256": sha256(workspace / "sources/tuning_history.log"),
        "dispatcher": "native_ApplyHistoryBest_model_then_target_key_full_workload",
        "dispatcher_source_sha256": sha256(Path(dispatcher.__file__)),
        "queries": queries, "matched": sum(q["history_match"] for q in queries),
        "fallback": sum(not q["history_match"] for q in queries), "new_tuning": False})
    record = {**spec, "spec": spec, "model": "mxnet_gluon_resnet50_v2",
              "unit_count": len(graph.units), "build_identity": build_identity,
              "canonical_sha256": sha256(workspace / "sources/canonical_qir.json"),
              "target": str(target), "input_names": [p.name_hint for p in original["main"].params],
              "input_node_ids": [graph.ids[n] for n in incoming],
              "output_node_ids": [graph.ids[n] for n in outgoing],
              "input_types": [typed(n) for n in incoming], "output_types": [typed(n) for n in outgoing],
              "terminal_output_indices": [outgoing.index(n) if n in outgoing else None for n in terminals],
              "extra_quantize_calls": 0, "recomposition_exact": True,
              "compile_seconds": time.monotonic() - started,
              "files": inventory(destination, ["record.json"])}
    write(marker, record)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--program", required=True)
    args = p.parse_args()
    result = compile_program(args.workspace, args.program)
    print(json.dumps({"name": args.program, "compile_seconds": result["compile_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
