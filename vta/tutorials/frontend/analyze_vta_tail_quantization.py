#!/usr/bin/env python3
"""Post-hoc E3 pilot audit: verify terminal quantization chain and int8 wrapping."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tvm
from tvm import relay


def terminal_scale(mod):
    def call(expr,name,dtype=None):
        assert isinstance(expr,relay.Call) and expr.op.name==name,(name,expr)
        if dtype:
            assert str(expr.attrs.dtype)==dtype
        return expr.args[0]

    body=mod["main"].body
    value=call(body,"multiply")
    scale=float(body.args[1].data.numpy())
    assert scale==0.0625
    value=call(value,"cast","float32")
    value=call(value,"annotation.stop_fusion")
    value=call(value,"cast","int8")
    value=call(value,"nn.relu")
    call(value,"add")
    for branch in value.args:
        call(branch,"cast","int32")
    return scale


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot",type=Path,required=True)
    args=parser.parse_args()
    result_path=args.pilot/"wrap_analysis.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    rows=[]
    sources=[Path(__file__),args.pilot/"preregistered.json",args.pilot/"summary.json"]
    sources.extend(args.pilot.glob("vta_*/params_identity.json"))
    for layer,start in ((2,5),(3,10),(4,15)):
        qpath=args.pilot/f"vta_{start:02d}_{start+2:02d}"/"quantized.json"
        scale=terminal_scale(tvm.ir.load_json(qpath.read_text()))
        sources.append(qpath)
        for case in range(3):
            path=args.pilot/f"layer{layer}_case{case}.npz"
            sources.append(path)
            with np.load(path) as data:
                outside=data["tail_outside"]
                inside=data["tail_inside"]
                grid=outside/scale
                assert np.array_equal(grid,np.rint(grid))
                wrapped=((grid.astype("int32")+128)%256-128).astype("float32")*scale
                mismatch=np.flatnonzero(outside!=inside)
                rows.append({"layer":layer,"case":case,"elements":outside.size,
                    "mismatches":int(mismatch.size),"quant_grid_exact":True,
                    "predicted_wrap_matches_joined":bool(np.array_equal(wrapped,inside)),
                    "outside_gt_int8_max":int(np.count_nonzero(grid>127)),
                    "changed_values":[{"flat_index":int(i),"outside":float(outside.flat[i]),
                        "inside":float(inside.flat[i]),"wrapped_prediction":float(wrapped.flat[i])}
                        for i in mismatch]})
    result={"scope":"Post-hoc mechanism check, not preregistered accuracy or performance evidence",
        "terminal_chain_verified":"int32 add -> relu -> int8 cast (no clip) -> stop_fusion -> float32 cast -> scale 1/16",
        "rows":rows,"all_wrap_predictions_exact":all(r["predicted_wrap_matches_joined"] for r in rows),
        "elements":sum(r["elements"] for r in rows),"mismatches":sum(r["mismatches"] for r in rows),
        "sources_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    result_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    assert result["all_wrap_predictions_exact"]
    print(json.dumps({k:v for k,v in result.items() if k not in {"rows","sources_sha256"}},indent=2))


if __name__=="__main__":
    main()
