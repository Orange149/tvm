#!/usr/bin/env python3
"""Census logical LOAD/STORE descriptors in complete archived segment functions.

This is an E2 input, not runtime qualification or physical DDR traffic.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import tvm
import vta
from extract_static_vta_dma import extract_module_dma


def uop_scope_calls(function):
    """Static push-scope invocations, not hardware ALU operations/uop cache fills."""
    analyzer,loops,counts=tvm.arith.Analyzer(),[],Counter()
    def before(node):
        if isinstance(node,(tvm.tir.IfThenElse,tvm.tir.While)):
            raise ValueError("Conditional uop scope counting unsupported")
        if isinstance(node,tvm.tir.For):
            extent=analyzer.simplify(node.extent)
            if not isinstance(extent,tvm.tir.IntImm):
                raise ValueError("Dynamic uop scope loop")
            loops.append(int(extent))
        if isinstance(node,tvm.tir.AttrStmt) and node.attr_key=="coproc_uop_scope":
            assert isinstance(node.value,tvm.tir.StringImm)
            name=node.value.value
            assert name in {"VTAPushGEMMOp","VTAPushALUOp"}
            count=1
            for extent in loops:
                count*=extent
            counts[name]+=count
    def after(node):
        if isinstance(node,tvm.tir.For):
            loops.pop()
    tvm.tir.stmt_functor.ir_transform(function.body,before,after,None)
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    read = lambda path: json.loads(path.read_text())
    env = vta.get_env()
    assert read(args.archive/"provenance.json")["vta_config"] == env.cfg_dict
    summary = read(args.archive/"summary.json")
    assert summary["complete"]
    rows = []
    for stage in summary["rows"]:
        local = args.archive/stage["segment_id"].replace(":", "_")
        calls = Counter(n["attrs"]["func_name"] for n in read(local/"graph.json")["nodes"]
                        if n["op"] == "tvm_op")
        functions, totals, pushes = {}, Counter(), Counter()
        for snapshot in stage["snapshots"]:
            raw = (local/snapshot["file"]).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == snapshot["sha256"]
            mod = tvm.ir.load_json(raw.decode())
            for gv, function in mod.functions.items():
                name = gv.name_hint
                assert name not in functions and name in calls
                externs, unsafe = [], []
                def inspect(node):
                    if isinstance(node, (tvm.tir.IfThenElse, tvm.tir.While)):
                        unsafe.append(type(node).__name__)
                    if (isinstance(node, tvm.tir.Call) and isinstance(node.op, tvm.ir.Op)
                            and node.op.name == "tir.call_extern"):
                        externs.append(node.args[0].value)
                tvm.tir.stmt_functor.post_order_visit(function.body, inspect)
                dma = any(n in {"VTALoadBuffer2D", "VTAStoreBuffer2D"} for n in externs)
                if dma:
                    assert not unsafe, (name, "Control-flow unsupported by static extractor", unsafe)
                    counted = extract_module_dma(tvm.IRModule({"main":function}),env)
                    scopes=uop_scope_calls(function)
                else:
                    counted = {"totals":{},"request_descriptors":[]}
                    scopes={}
                functions[name] = {"invocations":calls[name],"static_dma":counted,
                                   "uop_scope_calls":scopes,
                                   "extern_call_sites":dict(Counter(externs))}
                for key,count in scopes.items():
                    pushes[key]+=count*calls[name]
                for key, count in counted["totals"].items():
                    if key.endswith(("_calls", "_bytes")) and "average" not in key:
                        totals[key] += count*calls[name]
        assert set(functions) == set(calls)
        rows.append({"segment_id":stage["segment_id"],"functions":functions,
                     "logical_descriptor_totals":dict(totals),"graph_function_coverage_complete":True,
                     "uop_scope_calls":dict(pushes)})
    result = {"scope":"Static logical LOAD/STORE API descriptors at CPUAccessRewrite, weighted by GraphExecutor invocations",
              "runtime_qualified":False, "e2_complete":False,
              "excluded":"CPU adapter memory accesses, runtime uop/instruction traffic, ALU execution counts, physical DDR bursts, transfer time and stall",
              "source_summary_sha256":hashlib.sha256((args.archive/"summary.json").read_bytes()).hexdigest(),
              "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "extractor_sha256":hashlib.sha256(Path(__file__).with_name("extract_static_vta_dma.py").read_bytes()).hexdigest(),
              "uop_scope_interpretation":"coproc_uop_scope -> LLVM CreateStaticInit push function; static prediction of host push calls, not hardware operations or stall cycles",
              "rows":rows}
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps([{k:v for k,v in r.items() if k != "functions"} for r in rows],indent=2))


if __name__ == "__main__":
    main()
