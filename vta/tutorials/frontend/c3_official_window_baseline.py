#!/usr/bin/env python3
"""Official-cache identity and lossless post-Fold capture, without board access.

The capture hook is inserted in the existing lowering pipeline.  B_rt loads the
same IRModule at that point and resumes the remaining passes exactly once.
CPU operators are captured only by the ordinary build and never routed to VTA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
RUN = GROUPED / "20261014_c3_official_valid_window_run01"
LOG = Path("/home/orange/.tvm/tophub/vta_v0.10.log")
LOG_SHA = "8eea6e4b4edf4e800b7782b93e17db4450d01d8817220fcbcf6ef8ef5f68a6dc"
SOURCE_MODEL = "pynq_1x16_i8w8a32_15_15_18_17"
STOCK = {
    "resnet18_224": GROUPED / "20260914_p7r516f_resnet18_fullgraph_fallback_build_run01/stock_reference",
    "yolov3_tiny_416": GROUPED / "20260926_c3_yolo416_stock_graph_build_run01/stock",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return [plain(v) for v in value]
    except TypeError:
        return str(value)


def key(value):
    return json.dumps(plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def select_records(path=LOG):
    """Select exact PYNQ 1x16 workload minima; retain full original records."""
    from tvm import autotvm
    if digest(path) != LOG_SHA:
        raise ValueError("OFFICIAL_LOG_IDENTITY_CHANGED")
    all_records, selected = [], {}
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        inp, result = autotvm.record.decode(line)
        if int(result.error_no) != 0:
            continue
        if inp.target.model != SOURCE_MODEL or inp.task.name != "conv2d_packed.vta":
            continue
        costs = [float(v) for v in result.costs]
        if not costs or not all(__import__("math").isfinite(v) for v in costs):
            raise ValueError("INVALID_OFFICIAL_COST")
        workload = plain(inp.task.workload)
        row = {"line": line_number, "raw_record": json.loads(line),
               "record_sha256": hashlib.sha256(line.encode()).hexdigest(),
               "workload": workload, "source_target": str(inp.target),
               "source_model": inp.target.model, "search_budget": "unknown",
               "historical_cost_mean": sum(costs) / len(costs),
               "complete_config_entity": plain(inp.config.to_json_dict())}
        all_records.append((inp, result))
        identity = key(workload)
        if identity not in selected or row["historical_cost_mean"] < selected[identity]["historical_cost_mean"]:
            selected[identity] = row
    if not selected:
        raise ValueError("NO_OFFICIAL_RECORDS")
    return all_records, selected


def assert_effective_config(selected, actual):
    if actual is None or actual.is_fallback:
        raise ValueError("OFFICIAL_CONFIG_FALLBACK")
    config = plain(actual.to_json_dict())
    if config != selected["complete_config_entity"]:
        raise ValueError("OFFICIAL_CONFIG_EFFECTIVE_MISMATCH")
    return config


def is_vta_mac(function):
    from tvm import tir
    found = []
    def visit(node):
        if isinstance(node, tir.Call) and node.op.name == "tir.vta.uop_push":
            if isinstance(node.args[0], tir.IntImm) and int(node.args[0]) == 0:
                if isinstance(node.args[1], tir.IntImm) and int(node.args[1]) == 0:
                    found.append(True)
    tir.stmt_functor.post_order_visit(function.body, visit)
    return bool(found)


def module_functions(module):
    import tvm
    return [(gv, f) for gv, f in module.functions.items()
            if isinstance(f, tvm.tir.PrimFunc) and is_vta_mac(f)]


def buffer_abi(function):
    result = []
    for ordinal, param in enumerate(function.params):
        if param not in function.buffer_map:
            result.append({"ordinal": ordinal, "kind": "scalar", "dtype": str(param.dtype)})
            continue
        buf = function.buffer_map[param]
        result.append({"ordinal": ordinal, "kind": "buffer", "dtype": str(buf.dtype),
                       "shape": [int(v) for v in buf.shape],
                       "strides": [str(v) for v in buf.strides],
                       "elem_offset": str(buf.elem_offset), "scope": buf.scope()})
    return result


def normalize_llvm(text):
    # LLVM TBAA labels embed the compiler's host allocation pointer. Rebind
    # those labels bijectively, retaining the full alias graph and all fields.
    # SSA, initializer fields, static handles and control flow remain intact.
    aliases = {}
    lines = []
    for line in text.splitlines():
        if line.startswith("; ModuleID =") or line.startswith("source_filename ="):
            continue
        if re.match(r'^!\d+ = !\{!"', line):
            def name(match):
                pointer = match.group(0)
                return "compiler_buffer_%d" % aliases.setdefault(pointer, len(aliases))
            line = re.sub(r'0x[0-9a-fA-F]+(?=(?:\.w\d+\.b\d+)?")', name, line)
        lines.append(line)
    return "\n".join(lines)


def parameter_semantics(parameters):
    result = []
    for name in sorted(parameters):
        array = parameters[name].numpy()
        result.append({"name": name, "shape": list(array.shape), "dtype": str(array.dtype),
                       "contents_sha256": hashlib.sha256(array.tobytes()).hexdigest()})
    return result


def restore_external_global_bindings(original, loaded):
    """Retain GVar identities referenced by the enclosing Relay compiler.

    JSON legitimately allocates fresh GlobalVars.  Returning those keys from an
    in-pipeline pass severs the enclosing compiler's existing references, even
    though the serialized module is structurally equal.  Only module binding
    objects are reattached; loaded PrimFunc contents and all attrs are retained.
    """
    import tvm
    from tvm import tir
    if {str(v.name_hint) for v in original.functions} != {str(v.name_hint) for v in loaded.functions}:
        raise ValueError("ROUNDTRIP_GLOBAL_BINDING_SET_CHANGED")
    originals = {str(v.name_hint): v for v in original.functions}
    def rebind_call(node):
        if isinstance(node, tir.Call) and isinstance(node.op, tvm.ir.GlobalVar):
            name = str(node.op.name_hint)
            if name not in originals:
                raise ValueError("ROUNDTRIP_UNKNOWN_GLOBAL_REFERENCE")
            return tir.Call(node.dtype, originals[name], node.args, node.span)
    functions = {}
    for gv, f in loaded.functions.items():
        if isinstance(f, tir.PrimFunc):
            f = f.with_body(tir.stmt_functor.ir_transform(f.body, None, rebind_call, ["tir.Call"]))
        functions[originals[str(gv.name_hint)]] = f
    result = tvm.IRModule(functions, type_definitions=loaded.type_definitions, attrs=loaded.attrs)
    if not tvm.ir.structural_equal(original, result, map_free_vars=False):
        raise ValueError("ROUNDTRIP_BINDING_REATTACH_CHANGED_CONTENT")
    return result


def compiler_sources():
    files = [REPO / "build/libtvm.so", REPO / "build/libtvm_runtime.so",
             REPO / "python/tvm/autotvm/task/dispatcher.py",
             REPO / "python/tvm/autotvm/task/space.py",
             REPO / "vta/python/vta/build_module.py",
             REPO / "vta/python/vta/transform.py",
             REPO / "vta/python/vta/top/vta_conv2d.py",
             REPO / "src/node/serialization.cc", Path(__file__),
             HERE / "build_vta_resnet50_relay_residency_dispatch_pair.py",
             HERE / "build_vta_yolov3_tiny_relay_residency_dispatch_pair.py"]
    return {str(p): digest(p) for p in files}


def make_program(network, env):
    if network == "resnet18_224":
        from build_vta_resnet50_relay_residency_dispatch_pair import make_relay_testing_resnet_program
        return make_relay_testing_resnet_program(env, 18, device_annot=True)
    from types import SimpleNamespace
    from build_vta_yolov3_tiny_relay_residency_dispatch_pair import make_relay_program
    cfg = GROUPED / "20260924_c3_yolo_unseen_resolution_prepare_run05/yolov3-tiny.base.cfg"
    weights = REPO / ".tvm_test_data/darknet/yolov3-tiny.weights"
    darknet = Path("/home/orange/.tvm_test_data/darknet/libdarknet2.0.so")
    if digest(weights) != "dccea06f59b781ec1234ddf8d1e94b9519a97f4245748a7d4db75d5b7080a42c":
        raise ValueError("YOLO_WEIGHT_IDENTITY_CHANGED")
    program, params, shape = make_relay_program(env, SimpleNamespace(
        cfg=str(cfg), weights=str(weights), darknet_lib=str(darknet)))
    if tuple(shape) != (1, 3, 416, 416):
        raise ValueError("YOLO_RESOLUTION_CHANGED")
    return program, params


def capture(network, output):
    import tvm
    from tvm import autotvm, relay, tir
    import vta
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records, selected = select_records()
    env = vta.get_env()
    if env.cfg_dict["TARGET"] != "axu5evb":
        raise ValueError("CURRENT_TARGET_NOT_AXU5EVB")
    program, params = make_program(network, env)
    parent_program = tvm.IRModule.from_expr(program) if not isinstance(program, tvm.IRModule) else program
    (output / "packed_relay.json").write_text(tvm.ir.save_json(parent_program))
    (output / "packed_relay.txt").write_text(parent_program.astext(show_meta_data=False))
    (output / "params_input.bin").write_bytes(relay.save_param_dict(params))
    target = tvm.target.Target(env.target, host=env.target_host)
    build_target = ({"cpu": tvm.target.Target(env.target_vta_cpu, host=env.target_host),
                     "ext_dev": target} if network == "resnet18_224" else target)
    sources = compiler_sources()
    write(output / "compiler_identity.json", {"tvm_version": tvm.__version__,
        "target": str(target), "target_host": str(env.target_host), "vta_config": env.cfg_dict,
        "source_sha256": sources, "official_log": str(LOG), "official_log_sha256": digest(LOG),
        "source_platform": "PYNQ", "search_budget": "unknown", "board_contacted": False})

    class AuditHistory(autotvm.task.ApplyHistoryBest):
        def __init__(self):
            super().__init__(records)
            self.queries = []
        def query(self, queried_target, workload):
            if str(workload[0]) != "conv2d_packed.vta":
                return super().query(queried_target, workload)
            identity = key(workload)
            if identity not in selected:
                raise ValueError("OFFICIAL_CONFIG_MISS: " + identity)
            actual = self._query_inside(queried_target, workload)
            config = assert_effective_config(selected[identity], actual)
            self.queries.append({"queried_target": str(queried_target), "workload": plain(workload),
                                 "effective_config_entity": config, "source_record": selected[identity],
                                 "all_config_fields_match": True})
            return actual

    builds = {}
    for variant in ("B_raw", "B_rt"):
        local = output / variant
        local.mkdir()
        snapshots = {}
        history = AuditHistory()
        ordinal = [0]
        @tvm.transform.module_pass(opt_level=0, name="tir.c3.OfficialPostFoldCapture")
        def capture_hook(mod, _ctx):
            if not module_functions(mod):
                return mod
            names = tuple(str(gv.name_hint) for gv, _ in module_functions(mod))
            if names in snapshots:
                raise ValueError("DUPLICATE_PARENT_CAPTURE")
            stem = "parent_%03d" % ordinal[0]
            ordinal[0] += 1
            serialized = tvm.ir.save_json(mod)
            (local / (stem + ".json")).write_text(serialized)
            (local / (stem + ".tir")).write_text(mod.script())
            loaded = tvm.ir.load_json(serialized)
            if not tvm.ir.structural_equal(mod, loaded, map_free_vars=False):
                raise ValueError("PARENT_SERIALIZATION_CHANGED")
            snapshots[names] = {"stem": stem, "parent": mod, "roundtrip": loaded,
                                 "parent_structural_equal": True}
            return restore_external_global_bindings(mod, loaded) if variant == "B_rt" else mod

        @tvm.instrument.pass_instrument
        class FinalArchive:
            def __init__(self):
                self.pass_names = []
            def run_after_pass(self, mod, info):
                self.pass_names.append(str(info.name))
                if info.name != "tir.vta.CPUAccessRewrite" or not module_functions(mod):
                    return
                names = tuple(str(gv.name_hint) for gv, _ in module_functions(mod))
                if names not in snapshots:
                    raise ValueError("FINAL_WITHOUT_PARENT")
                row = snapshots[names]
                if "final" in row:
                    raise ValueError("DUPLICATE_CPU_ACCESS_REWRITE")
                row["final"] = mod
                (local / (row["stem"] + "_final.json")).write_text(tvm.ir.save_json(mod))
                (local / (row["stem"] + "_final.tir")).write_text(mod.script())
        archive = FinalArchive()
        base = vta.build_config(debug_flag=0, opt_level=3,
            disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"})
        lower_passes = list(base.config["tir.add_lower_pass"])
        matches = [i for i, (_, p) in enumerate(lower_passes) if p.info.name == "tir.vta.FoldUopLoop"]
        if len(matches) != 1:
            raise ValueError("UNKNOWN_FOLD_PASS_POSITION")
        insertion = matches[0] + 1
        if lower_passes[insertion][1].info.name != "tir.vta.CPUAccessRewrite":
            raise ValueError("UNKNOWN_CPU_PASS_POSITION")
        lower_passes.insert(insertion, (3, capture_hook))
        configuration = dict(base.config)
        configuration["tir.add_lower_pass"] = lower_passes
        relay.backend.te_compiler.get().clear()
        started = time.monotonic()
        with history:
            with tvm.transform.PassContext(opt_level=3, config=configuration,
                    disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"}, instruments=[archive]):
                graph, lib, lowered_params = relay.build(program, target=build_target, params=params)
        (local / "graph.json").write_text(graph)
        (local / "params.bin").write_bytes(relay.save_param_dict(lowered_params))
        llvm = lib.get_source("ll")
        (local / "module.ll").write_text(llvm)
        write(local / "queries.json", history.queries)
        write(local / "pass_sequence.json", {"configured_lower_passes": [
            {"phase": int(phase), "name": str(p.info.name)} for phase, p in lower_passes],
            "actual_passes": archive.pass_names, "build_seconds": time.monotonic() - started})
        if any("final" not in row for row in snapshots.values()):
            raise ValueError("MISSING_REMAINING_LOWERING")
        builds[variant] = {"snapshots": snapshots, "graph": graph, "llvm": llvm,
                           "queries": history.queries, "params_sha256": digest(local / "params.bin"),
                           "parameters": lowered_params}
    raw, rebuilt = builds["B_raw"], builds["B_rt"]
    if raw["snapshots"].keys() != rebuilt["snapshots"].keys():
        raise ValueError("ROUNDTRIP_FUSED_MODULE_SET_CHANGED")
    if json.loads(raw["graph"]) != json.loads(rebuilt["graph"]):
        raise ValueError("ROUNDTRIP_GRAPH_CHANGED")
    # save_param_dict serializes unordered map entries; compare actual keys,
    # dtypes, shapes and every value instead of requiring binary map ordering.
    import numpy as np
    if set(raw["parameters"]) != set(rebuilt["parameters"]) or any(
            raw["parameters"][name].dtype != rebuilt["parameters"][name].dtype or
            not np.array_equal(raw["parameters"][name].numpy(), rebuilt["parameters"][name].numpy())
            for name in raw["parameters"]):
        raise ValueError("ROUNDTRIP_PARAMETERS_CHANGED")
    write(output / "parameter_roundtrip.json", {"status": "ALL_PARAMETER_VALUES_EQUAL",
        "raw_serialized_sha256": raw["params_sha256"], "rt_serialized_sha256": rebuilt["params_sha256"],
        "normalized_parameter_inventory": parameter_semantics(raw["parameters"])})
    if normalize_llvm(raw["llvm"]) != normalize_llvm(rebuilt["llvm"]):
        raise ValueError("ROUNDTRIP_LLVM_CHANGED")
    graph = json.loads(raw["graph"])
    rows = []
    for names, captured in raw["snapshots"].items():
        other = rebuilt["snapshots"][names]
        if not tvm.ir.structural_equal(captured["parent"], other["parent"], map_free_vars=False):
            raise ValueError("ROUNDTRIP_PARENT_CHANGED")
        if not tvm.ir.structural_equal(captured["final"], other["final"], map_free_vars=False):
            raise ValueError("ROUNDTRIP_FINAL_CHANGED")
        for gv, func in module_functions(captured["parent"]):
            symbol = str(func.attrs["global_symbol"])
            abi = buffer_abi(func)
            if len(abi) < 3 or any(p["kind"] != "buffer" for p in abi):
                raise ValueError("UNSUPPORTED_OFFICIAL_BUFFER_ABI")
            matches = [q for q in raw["queries"] if q["workload"][1][1] == abi[0]["shape"]
                       and q["workload"][2][1] == abi[1]["shape"]]
            unique = {key(q["workload"]): q for q in matches}
            if len(unique) != 1:
                raise ValueError("FUSED_FUNCTION_QUERY_BINDING_AMBIGUOUS: " + symbol)
            query = next(iter(unique.values()))
            nodes = [i for i, node in enumerate(graph["nodes"])
                     if node.get("attrs", {}).get("func_name") == symbol]
            if not nodes:
                raise ValueError("FUSED_FUNCTION_MISSING_GRAPH_CALLSITE")
            for node in nodes:
                semantic = hashlib.sha256(tvm.ir.save_json(func).encode()).hexdigest()
                rows.append({"object_id": network + "_node%03d" % node,
                    "network": network, "earliest_graph_node": node, "global_symbol": symbol,
                    "global_var": str(gv.name_hint), "semantic_hash": semantic,
                    "abi": abi, "workload": query["workload"], "source_record": query["source_record"],
                    "effective_config_entity": query["effective_config_entity"],
                    "official_baseline_bound": True, "roundtrip_passed": True,
                    "parent_module": str(output / "B_raw" / (captured["stem"] + ".json")),
                    "final_module": str(output / "B_raw" / (captured["stem"] + "_final.json")),
                    "parent_module_sha256": digest(output / "B_raw" / (captured["stem"] + ".json")),
                    "final_module_sha256": digest(output / "B_raw" / (captured["stem"] + "_final.json"))})
    expected = 20 if network == "resnet18_224" else 12
    if len(rows) != expected:
        raise ValueError("EXISTING_CALLSITE_COVERAGE_CHANGED: %d != %d" % (len(rows), expected))
    unique_workloads = len({key(row["workload"]) for row in rows})
    if unique_workloads != 11:
        raise ValueError("EXISTING_WORKLOAD_COVERAGE_CHANGED")
    old_graph = json.loads((STOCK[network] / "graph.json").read_text())
    if sources != compiler_sources():
        raise ValueError("COMPILER_SOURCE_CHANGED_DURING_CAPTURE")
    write(output / "baseline_identity.json", {"status": "OFFICIAL_QUERY_AND_ROUNDTRIP_PASSED",
        "network": network, "callsites": rows, "vta_callsites": len(rows),
        "unique_workloads": unique_workloads, "unique_fused_modules": len(raw["snapshots"]),
        "same_graph_as_existing_stock": old_graph == graph,
        "existing_stock_graph_sha256": digest(STOCK[network] / "graph.json"),
        "roundtrip": {"parent_structural_equal": True, "final_structural_equal": True,
            "llvm_complete_initializer_and_handle_content_equal": True,
            "graph_equal": True, "parameters_equal": True,
            "aggregation_enabled": False, "common_boundary_repairs": []},
        "board_contacted": False, "historical_latency_used_as_current_latency": False})
    return {"network": network, "status": "OFFICIAL_QUERY_AND_ROUNDTRIP_PASSED",
            "callsites": len(rows), "workloads": unique_workloads}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", choices=tuple(STOCK), required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(capture(args.network, args.output_dir), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
