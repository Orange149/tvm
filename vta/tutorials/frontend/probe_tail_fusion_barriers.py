#!/usr/bin/env python3
"""E3 host counterfactual: remove only the two pre-tail stop_fusion annotations.

This is not the production schedule or a proposed VTA-compatible optimization.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import tvm

from qualify_once_quantized_tail_split import Replace,build,executor,output,save,sha,tail_edges,typed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit",type=Path,default=Path(
        "vta/tutorials/frontend/report_out/stage_tile_cotuning/compile_context_audit"))
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    baseline=args.audit/"e3_once_quantized_run2"
    sources=[Path(__file__),Path(__file__).with_name("qualify_once_quantized_tail_split.py")]
    for layer,start in ((2,5),(3,10),(4,15)):
        sources.extend(baseline/f"layer{layer}"/"A"/name for name in ("relay.json","fusion.json"))
        sources.extend(baseline/f"layer{layer}"/f"outputs_{i}.npz" for i in range(3))
        sources.extend(args.audit/"board_e2_run1"/f"vta_{start:02d}_{start+1:02d}"/f"inputs_{i}.npz" for i in range(3))
    save(args.output/"preregistered.json",{
        "scope":"LLVM-only counterfactual, not frozen VTA pipeline or runtime tuning",
        "change":"remove exactly two existing pre-tail stop_fusion annotations; preserve all numerical ops/constants",
        "cases":"all nine previously frozen E2 inputs",
        "checks":"FuseOps primitive signature multiset versus A; outputs bit-exact versus A",
        "sources_sha256":{str(p):sha(p) for p in sources}})
    rows=[]
    for layer,start in ((2,5),(3,10),(4,15)):
        mod=tvm.ir.load_json((baseline/f"layer{layer}"/"A/relay.json").read_text())
        edges=tail_edges(mod)
        assert all(e.op.name=="annotation.stop_fusion" for e in edges)
        changed=typed(Replace({e:e.args[0] for e in edges}).visit(mod["main"].body),list(mod["main"].params))
        factory,fusion=build(changed,args.output/f"layer{layer}")
        exe=executor(factory)
        original=json.loads((baseline/f"layer{layer}"/"A/fusion.json").read_text())
        row={"layer":layer,"primitive_signatures_changed":fusion!=original,
             "original_primitive_counts":[sum(s.values()) for s in original],
             "released_primitive_counts":[sum(s.values()) for s in fusion],"cases":[]}
        for case in range(3):
            with np.load(args.audit/"board_e2_run1"/f"vta_{start:02d}_{start+1:02d}"/f"inputs_{case}.npz") as values:
                exe.set_input(**{k:values[k] for k in values.files})
            exe.run()
            actual=output(exe)
            with np.load(baseline/f"layer{layer}"/f"outputs_{case}.npz") as values:
                expected=values["A"]
            np.savez(args.output/f"layer{layer}"/f"output_{case}.npz",output=actual)
            row["cases"].append({"case":case,"elements":actual.size,
                "mismatches":int(np.count_nonzero(actual!=expected))})
        rows.append(row)
        save(args.output/"summary.json",{"complete":len(rows)==3,"rows":rows,
            "board_qualified":False,"production_changes":False})
        assert all(c["mismatches"]==0 for c in row["cases"])
        print(json.dumps(row),flush=True)


if __name__=="__main__":
    main()
