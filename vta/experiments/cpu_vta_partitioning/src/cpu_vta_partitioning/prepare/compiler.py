"""Isolated SDK worker. No transport imports or board commands are permitted."""
import argparse
import os
from pathlib import Path
import time
import traceback

from cpu_vta_partitioning.common import read, write, sha256, inventory


def packed_stage(mod, env):
    import tvm
    from tvm import relay
    from vta.top.graphpack import graph_pack
    from thesis_canonical_graph_split import Replace
    function = mod["main"]
    replacement = {p: relay.Call(tvm.ir.Op.get("annotation.bitpack_start"), [p])
                   for p in function.params}
    body = Replace(replacement).visit(function.body)
    body = relay.Tuple([relay.Call(tvm.ir.Op.get("annotation.bitpack_end"), [x]) for x in body.fields])
    wrapped = relay.transform.InferType()(tvm.IRModule.from_expr(relay.Function(list(function.params), body)))
    packed = relay.transform.InferType()(tvm.IRModule.from_expr(graph_pack(
        wrapped["main"], env.BATCH, env.BLOCK_OUT, env.WGT_WIDTH,
        start_name=None, stop_name=None, boundary_bridge=True)))
    if not tvm.ir.structural_equal(mod["main"].checked_type, packed["main"].checked_type):
        raise ValueError("packing changed frontier types")
    return packed


def compile_stage(bundle, spec):
    import tvm
    from tvm import autotvm, relay
    from tvm.contrib import cc
    import vta
    from cpu_vta_partitioning.adapters import legacy_graph_class
    start = time.monotonic()
    bundle = Path(bundle)
    output = bundle / "programs" / spec["name"]
    record_path = output / "record.json"
    from cpu_vta_partitioning.common import digest
    build_identity = digest(read(bundle / "plans/compiler_identity.json"))
    if record_path.exists():
        from cpu_vta_partitioning.common import verify_inventory
        record = read(record_path)
        if record.get("build_identity") == build_identity:
            verify_inventory(output, record["files"])
            return record
    output.mkdir(parents=True, exist_ok=True)
    graph = legacy_graph_class()(tvm.ir.load_json((bundle / "sources/canonical_qir.json").read_text()))
    env = vta.get_env()
    original, incoming, outgoing = graph.stage(spec["begin"], spec["end"])
    full_ranges = [(i, i + 1) for i in range(spec["begin"])] + [(spec["begin"], spec["end"])] + [
        (i, i + 1) for i in range(spec["end"], 21)]
    graph.recompose(full_ranges)
    mod = packed_stage(original, env) if spec["device"] == "vta" else original
    (output / "lowering_qir.json").write_text(tvm.ir.save_json(mod))
    target = tvm.target.Target(env.target if spec["device"] == "vta" else env.target_vta_cpu,
                               host=env.target_host)
    context = vta.build_config if spec["device"] == "vta" else tvm.transform.PassContext
    with autotvm.apply_history_best([str(bundle / "sources/tuning_history.log")]), context(
            opt_level=3, disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"}):
        factory = relay.build(mod, target=target)
    (output / "graph.json").write_text(factory.get_graph_json())
    (output / "params.bin").write_bytes(relay.save_param_dict(factory.get_params()))
    sysroot = os.environ["SDKTARGETSYSROOT"]
    factory.get_lib().export_library(str(output / "module.so"),
        fcompile=cc.cross_compiler("aarch64-xilinx-linux-g++"),
        options=[f"--sysroot={sysroot}", f"-Wl,-rpath-link,{sysroot}/lib",
                 f"-Wl,-rpath-link,{sysroot}/usr/lib", f"-L{sysroot}/lib", f"-L{sysroot}/usr/lib"])
    record = {**spec, "build_identity": build_identity,
              "build_backend": "vta" if spec["device"] == "vta" else "tvm_cpu_standard",
              "input_names": [p.name_hint for p in original["main"].params],
              "input_node_ids": [graph.ids[n] for n in incoming],
              "output_node_ids": [graph.ids[n] for n in outgoing],
              "input_types": [{"dtype": str(p.checked_type.dtype), "shape": list(map(int, p.checked_type.shape))}
                              for p in original["main"].params],
              "output_types": [{"dtype": str(n.checked_type.dtype), "shape": list(map(int, n.checked_type.shape))}
                               for n in outgoing],
              "canonical_sha256": sha256(bundle / "sources/canonical_qir.json"),
              "tuning_sha256": sha256(bundle / "sources/tuning_history.log"),
              "extra_quantize_calls": 0, "recomposition_exact": True,
              "compile_seconds": time.monotonic() - start, "files": inventory(output, ["record.json"])}
    write(record_path, record, replace=True)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    spec = read(args.bundle / "plans/segments.json")[args.index]
    try:
        row = compile_stage(args.bundle, spec)
        print(row["name"], "compiled", round(row["compile_seconds"], 2), flush=True)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
