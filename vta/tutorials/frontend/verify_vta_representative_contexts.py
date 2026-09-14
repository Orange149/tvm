#!/usr/bin/env python3
"""Link Relay -> compiler hash -> graph function -> TIR; audit representative coverage."""
import argparse
from collections import Counter, defaultdict
import json
import itertools
import math
from pathlib import Path

import tvm
import vta
from audit_vta_compile_context import sha, save
from compare_vta_segment_tir import normalize


def cpu_accesses(function):
    """Count logical BufferLoad/Store bytes, not cache/DRAM transactions or time."""
    analyzer, loops, predicates, totals = tvm.arith.Analyzer(), [], [], Counter()
    def integer(expr):
        result = analyzer.simplify(expr)
        if not isinstance(result, tvm.tir.IntImm):
            raise ValueError("Nonconstant CPU loop extent")
        return int(result)
    def before(node):
        if isinstance(node,tvm.tir.While):
            raise ValueError("Dynamic CPU while loop not supported")
        if isinstance(node,tvm.tir.IfThenElse):
            predicates.append(node.condition)
            visit(node.then_case)
            predicates.pop()
            if node.else_case is not None:
                predicates.append(tvm.tir.Not(node.condition))
                visit(node.else_case)
                predicates.pop()
            return node  # Children were explicitly visited with branch predicates.
        if isinstance(node, tvm.tir.For):
            loops.append((node.loop_var,integer(node.min),integer(node.extent)))
        if (isinstance(node,tvm.tir.Call) and isinstance(node.op,tvm.ir.Op)
                and node.op.name=="tir.if_then_else"):
            raise ValueError("Conditional expression not supported")
        if isinstance(node, (tvm.tir.BufferLoad, tvm.tir.BufferStore)):
            used=set()
            for predicate in predicates:
                tvm.tir.stmt_functor.post_order_visit(predicate,
                    lambda n: used.add(n) if isinstance(n,tvm.tir.Var) else None)
            if not used <= {v for v,_,_ in loops}:
                raise ValueError("Conditional access depends on runtime variables")
            relevant=[entry for entry in loops if entry[0] in used]
            if math.prod(e for _,_,e in relevant)>100000:
                raise ValueError("Predicate enumeration budget exceeded")
            valid=0
            for values in itertools.product(*(range(m,m+e) for _,m,e in relevant)):
                bindings={entry[0]:tvm.tir.IntImm(entry[0].dtype,value) for entry,value in zip(relevant,values)}
                valid+=all(integer(tvm.tir.stmt_functor.substitute(p,bindings)) for p in predicates)
            count=valid*math.prod(e for v,_,e in loops if v not in used)
            is_load = isinstance(node, tvm.tir.BufferLoad)
            dtype = tvm.DataType(node.dtype if is_load else node.value.dtype)
            totals["read_bytes" if is_load else "write_bytes"] += count*dtype.bits*dtype.lanes//8
    def after(node):
        if isinstance(node, tvm.tir.For):
            loops.pop()
    def visit(body):
        tvm.tir.stmt_functor.ir_transform(body,before,after,None)
    visit(function.body)
    return dict(totals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive",type=Path,required=True)
    parser.add_argument("--scan",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    read = lambda path: json.loads(path.read_text())
    archive, scan = read(args.archive/"summary.json"), read(args.scan/"summary.json")
    assert archive["complete"] and scan["complete"]
    prior = {r["segment_id"]:r for r in scan["rows"]}
    expected_contexts = set()
    for sid in prior:
        expected_contexts |= {p["signature"] for p in read(args.scan/sid.replace(":","_")/"primitives.json")}
    classes, semantic, rows = defaultdict(list),defaultdict(list),[]
    for row in archive["rows"]:
        sid = row["segment_id"]
        local = args.archive/sid.replace(":","_")
        assert sha((local/"graph.json").read_bytes()) == row["graph_sha256"]
        assert sha((local/"primitive_inventory.json").read_bytes()) == row["primitive_inventory_sha256"]
        inventory = read(local/"primitive_inventory.json")
        nodes = [n for n in read(local/"graph.json")["nodes"] if n["op"]=="tvm_op"]
        assert Counter(p["compiler_hash"] for p in inventory) == Counter(n["attrs"]["hash"] for n in nodes)
        hashes, symbols = {},defaultdict(set)
        for p in inventory:
            h = p["compiler_hash"]
            assert h not in hashes or hashes[h]["signature"]==p["signature"], "Hash collision"
            hashes[h] = p
        for n in nodes:
            symbols[n["attrs"]["hash"]].add(n["attrs"]["func_name"])
        assert all(len(s)==1 for s in symbols.values()), "Ambiguous hash/function association"
        funcs = {}
        for snapshot in row["snapshots"]:
            raw = (local/snapshot["file"]).read_bytes()
            assert sha(raw)==snapshot["sha256"]
            for gv, func in tvm.ir.load_json(raw.decode()).functions.items():
                assert gv.name_hint not in funcs
                funcs[gv.name_hint]=func
        assert set(funcs)=={n["attrs"]["func_name"] for n in nodes}
        per_stage_cpu = Counter()
        cpu_rows, dma_functions = [],[]
        calls = Counter(n["attrs"]["func_name"] for n in nodes)
        for h, symbol_set in symbols.items():
            name = next(iter(symbol_set))
            p, function = hashes[h], funcs[name]
            normalized = normalize(function)
            classes[p["signature"]].append((sid,name,normalized))
            externs=[]
            def inspect(node):
                if (isinstance(node,tvm.tir.Call) and isinstance(node.op,tvm.ir.Op)
                        and node.op.name=="tir.call_extern"):
                    externs.append(node.args[0].value)
            tvm.tir.stmt_functor.post_order_visit(function.body,inspect)
            if any(n in {"VTALoadBuffer2D","VTAStoreBuffer2D"} for n in externs):
                dma_functions.append(name)
            else:
                try:
                    counted=cpu_accesses(function)
                    status="static_logical_accesses"
                    for key,value in counted.items():
                        per_stage_cpu[key]+=value*calls[name]
                except ValueError as error:
                    counted={"error":str(error)}
                    status="unsupported"
                cpu_rows.append({"function":name,"ops":p["ops"],"invocations":calls[name],
                                 "status":status,"accesses_per_invocation":counted})
        weights={json.dumps(c["weight"],sort_keys=True):c for c in prior[sid]["convolutions"]}
        mapped=[]
        for p in inventory:
            name=next(iter(symbols[p["compiler_hash"]]))
            for weight in p["weights"]:
                c=weights[json.dumps(weight,sort_keys=True)]
                assert c["primitive_signature"]==p["signature"]
                identity=c["semantic_identity"]["occurrence"]
                semantic[identity].append((sid,name,normalize(funcs[name])))
                mapped.append(identity)
        assert len(mapped)==len(weights)==len(set(mapped))
        rows.append({"segment_id":sid,"graph_invocations":len(nodes),"unique_functions":len(funcs),
                     "mapped_occurrences":mapped,"dma_functions":dma_functions,
                     "cpu_helpers":cpu_rows,"cpu_logical_access_totals":dict(per_stage_cpu),
                     "cpu_logical_access_totals_complete":all(r["status"]!="unsupported" for r in cpu_rows)})
    def comparisons(groups):
        result=[]
        for key, entries in sorted(groups.items()):
            representative=entries[0][2]
            same=[tvm.ir.structural_equal(representative, e[2],map_free_vars=True) for e in entries]
            result.append({"identity":key,"placements":len(entries),"all_equal":all(same),
                           "equality_checks_excluding_reference":len(entries)-1,
                           "members":[{"segment":e[0],"function":e[1],"equal_to_reference":eq}
                                      for e,eq in zip(entries,same)]})
        return result
    class_results, semantic_results = comparisons(classes),comparisons(semantic)
    result={"scope":"Complete representative builds linked through compiler hash; no all-87 TIR extrapolation",
            "script_sha256":sha(Path(__file__).read_bytes()),
            "archive_summary_sha256":sha((args.archive/"summary.json").read_bytes()),
            "scan_summary_sha256":sha((args.scan/"summary.json").read_bytes()),
            "representative_segments":len(rows),"covered_contexts":len(classes),"expected_contexts":len(expected_contexts),
            "missing_contexts":sorted(expected_contexts-set(classes)),
            "unexpected_contexts":sorted(set(classes)-expected_contexts),
            "semantic_occurrences":len(semantic),
            "varying_primitive_classes":[r["identity"] for r in class_results if not r["all_equal"]],
            "varying_semantic_occurrences":[r["identity"] for r in semantic_results if not r["all_equal"]],
            "cpu_access_caveat":"Logical TIR BufferLoad/Store bytes before later host optimization; not physical traffic or cache misses",
            "classes":class_results,"semantic":semantic_results,"rows":rows}
    save(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in {"classes","semantic","rows"}},indent=2))


if __name__=="__main__":
    main()
