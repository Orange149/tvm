#!/usr/bin/env python3
"""E3 host qualification: split frozen, once-quantized Relay without requantizing.

Preserves legacy int8 wrap semantics deliberately; not a numerical-policy fix,
VTA/u-dma-buf qualification, whole-network accuracy test, or timing experiment.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import tvm
from tvm import relay
from tvm.contrib import graph_executor

from analyze_vta_tail_quantization import terminal_scale
from audit_vta_compile_context import signature


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Replace(relay.ExprMutator):
    def __init__(self, replacements):
        super().__init__()
        self.replacements = replacements

    def visit(self, expr):
        if expr in self.replacements:
            return self.replacements[expr]
        return super().visit(expr)


def typed(body, params):
    return relay.transform.InferType()(tvm.IRModule.from_expr(relay.Function(params, body)))


def tail_edges(mod):
    terminal_scale(mod)  # Reject an unexpected tail before selecting cut nodes.
    value = mod["main"].body.args[0]
    for _ in range(4):  # float cast, annotation, int8 cast, relu -> int32 add
        value = value.args[0]
    return [branch.args[0] for branch in value.args]


def split(mod, edges, float_transport=False):
    assert len(edges) == 2 and edges[0] != edges[1]
    for edge in edges:
        assert str(edge.checked_type.dtype) == "int8"
    params = [relay.var(f"edge{i}", shape=e.checked_type.shape,
                        dtype="float32" if float_transport else "int8")
              for i,e in enumerate(edges)]
    transported = edges
    replacements = params
    if float_transport:
        # Exact for int8 values and binary scale 1/16. No calibrate/quantize call.
        transported = [relay.cast(e,"float32")*relay.const(0.0625,"float32") for e in edges]
        replacements = [relay.cast(p*relay.const(16.0,"float32"),"int8") for p in params]
    producer = typed(relay.Tuple(transported), list(mod["main"].params))
    consumer = typed(Replace(dict(zip(edges,replacements))).visit(mod["main"].body), params)
    consumer_params=list(consumer["main"].params)
    assert set(relay.analysis.free_vars(consumer["main"].body)) == set(consumer_params), "Cut is not dependency-closed"
    if not float_transport:
        reconstructed = relay.bind(consumer["main"].body, dict(zip(consumer_params,edges)))
        assert tvm.ir.structural_equal(reconstructed,mod["main"].body), "Split changed Relay expression"
    return producer, consumer


@tvm.instrument.pass_instrument
class FusionCapture:
    def __init__(self):
        self.snapshots = []

    def run_after_pass(self, mod, info):
        if info.name == "FuseOps":
            signatures = []
            def visit(node):
                if isinstance(node,relay.Call) and isinstance(node.op,relay.Function):
                    if node.op.attrs is not None and node.op.attrs.get_int("Primitive") == 1:
                        signatures.append(signature(node.op)[0])
            relay.analysis.post_order_visit(mod["main"].body,visit)
            self.snapshots.append(dict(Counter(signatures)))


def build(mod, directory):
    directory.mkdir()
    (directory/"relay.json").write_text(tvm.ir.save_json(mod))
    (directory/"relay.txt").write_text(mod.astext(show_meta_data=False))
    capture = FusionCapture()
    with tvm.transform.PassContext(opt_level=3,disabled_pass=["AlterOpLayout"],instruments=[capture]):
        factory = relay.build(mod,target="llvm")
    (directory/"graph.json").write_text(factory.get_graph_json())
    save(directory/"fusion.json",capture.snapshots)
    return factory,capture.snapshots


def executor(factory):
    return graph_executor.GraphModule(factory["default"](tvm.cpu()))


def pointer(array):
    tensor=array.handle.contents
    return int(tensor.data or 0)+int(tensor.byte_offset)


def output(exe):
    return exe.get_output(0).numpy()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit",type=Path,default=Path(
        "vta/tutorials/frontend/report_out/stage_tile_cotuning/compile_context_audit"))
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    pilot=args.audit/"e3_tail_pilot_run1"
    sources=[Path(__file__),Path(__file__).with_name("analyze_vta_tail_quantization.py"),
             Path(__file__).with_name("audit_vta_compile_context.py"),
             Path("src/runtime/graph_executor/graph_executor.cc")]
    for start in (5,10,15):
        sources.append(pilot/f"vta_{start:02d}_{start+2:02d}"/"quantized.json")
        sources.extend(args.audit/"board_e2_run1"/f"vta_{start:02d}_{start+1:02d}"/f"inputs_{i}.npz" for i in range(3))
        sources.extend(pilot/f"layer{start//5+1}_case{i}.npz" for i in range(3))
    save(args.output/"preregistered.json",{
        "scope":"Host LLVM numerical and compiler-context qualification; no timing or board claims",
        "arms":["A original joined", "B joined plus stop_fusion at both cut edges",
                "C two host Executors sharing int8 output/input storage", "D two host Executors with int8 copy",
                "C_float exact float32 transport shared", "D_float exact float32 transport copied"],
        "reference":"frozen once-quantized local joined stage, including existing wrap behavior",
        "cases":["frozen uniform", "frozen normal including wrap counterexamples", "frozen zeros"],
        "threshold":"bit-exact output and exact transport at all edges; no accuracy-policy repair",
        "sources_sha256":{str(p):sha(p) for p in sources}})
    results=[]
    for start in (5,10,15):
        layer=start//5+1
        local=args.output/f"layer{layer}"
        local.mkdir()
        mod=tvm.ir.load_json((pilot/f"vta_{start:02d}_{start+2:02d}"/"quantized.json").read_text())
        edges=tail_edges(mod)
        original,fa=build(mod,local/"A")
        blocked=typed(Replace({e:relay.annotation.stop_fusion(e) for e in edges}).visit(mod["main"].body),list(mod["main"].params))
        barrier,fb=build(blocked,local/"B")
        a,b=executor(original),executor(barrier)
        transports={}
        for use_float in (False,True):
            label="float32" if use_float else "int8"
            pm,cm=split(mod,edges,use_float)
            pf,_=build(pm,local/(label+"_producer"))
            cf,_=build(cm,local/(label+"_consumer"))
            p,c_shared,c_copy=executor(pf),executor(cf),executor(cf)
            arrays=[p.get_output(i) for i in range(2)]
            for i,array in enumerate(arrays):
                c_shared.set_input_zero_copy(f"edge{i}",array)
            transports[label]=(p,c_shared,c_copy,arrays)
        row={"layer":layer,"native_recomposition_structurally_equal":True,
             "existing_stop_fusion_on_cut":[isinstance(e,relay.Call) and e.op.name=="annotation.stop_fusion" for e in edges],
             "A_B_primitive_signatures_equal":fa==fb,"cases":[]}
        for case in range(3):
            path=args.audit/"board_e2_run1"/f"vta_{start:02d}_{start+1:02d}"/f"inputs_{case}.npz"
            with np.load(path) as data:
                values={k:data[k] for k in data.files}
            a.set_input(**values); a.run()
            b.set_input(**values); b.run()
            expected=output(a)
            with np.load(pilot/f"layer{layer}_case{case}.npz") as prior:
                assert np.array_equal(expected,prior["tail_inside"]),"Frozen reference changed"
            outputs={"A":expected,"B":output(b)}
            transport_rows={}
            for label,(p,c_shared,c_copy,arrays) in transports.items():
                p.set_input(**values); p.run()
                checks=[]
                for i,array in enumerate(arrays):
                    # get_input exposes the original storage pool, not the rebound
                    # op DLTensor. Do not claim it measures the zero-copy pointer.
                    pool=c_shared.get_input(f"edge{i}")
                    pool.copyfrom(np.zeros(tuple(pool.shape),dtype=str(pool.dtype)))
                    c_copy.set_input(f"edge{i}",array)
                    checks.append({"tensor":i,"shape":list(array.shape),"dtype":str(array.dtype),
                        "bytes":int(array.numpy().nbytes),
                        "producer_pointer_stable":pointer(p.get_output(i))==pointer(array),
                        "shared_op_pointer_introspection":None,
                        "original_consumer_pool_distinct":pointer(pool)!=pointer(array),
                        "copy_pointer_distinct":pointer(c_copy.get_input(f"edge{i}"))!=pointer(array),
                        "original_consumer_pool_zeroed":True,
                        "copy_content_equal":bool(np.array_equal(c_copy.get_input(f"edge{i}").numpy(),array.numpy()))})
                assert all(all(c[k] for k in ("producer_pointer_stable","original_consumer_pool_distinct","copy_pointer_distinct","copy_content_equal")) for c in checks)
                c_shared.run(); c_copy.run()
                outputs["C_"+label]=output(c_shared)
                outputs["D_"+label]=output(c_copy)
                transport_rows[label]=checks
            comparisons={name:{"elements":value.size,"mismatches":int(np.count_nonzero(value!=expected)),
                "max_abs_error":float(np.max(np.abs(value-expected)))} for name,value in outputs.items() if name!="A"}
            np.savez(local/f"outputs_{case}.npz",**outputs)
            row["cases"].append({"case":case,"comparisons":comparisons,"transport":transport_rows})
            save(local/"result.json",row)
            assert all(c["mismatches"]==0 for c in comparisons.values())
        results.append(row)
        print("[PASS]",layer,"existing barrier",row["existing_stop_fusion_on_cut"],"A/B equal",fa==fb,flush=True)
    save(args.output/"summary.json",{"complete":True,"rows":results,
        "comparison_elements":sum(c["elements"] for r in results for case in r["cases"] for c in case["comparisons"].values()),
        "all_exact":True,"full_e3_complete":False,"board_qualified":False,
        "semantics":"preserves legacy int8 wrapping, not a policy repair"})


if __name__=="__main__":
    main()
