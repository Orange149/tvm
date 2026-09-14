#!/usr/bin/env python3
"""E3 pilot: frozen E2 inputs, independent-stage quantization at three tail cuts.

CPU-only correctness diagnostic; not a fusion ablation or pipeline performance run.
"""
import argparse
import hashlib
import json
from pathlib import Path

from mxnet.gluon.model_zoo import vision
import numpy as np
import tvm
from tvm import relay
from tvm.contrib import graph_executor
import vta

from audit_vta_compile_context import constant
from split_resnet18_stages import (build_resnet18_unit_blocks, build_resnet18_unit_metadata,
    make_stage_block, relay_inputs_for_stage, lower_stage_to_relay)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, default=Path(
        "vta/tutorials/frontend/report_out/stage_tile_cotuning/compile_context_audit"))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    profile=args.audit.parent.parent/"resource_aware_maxplus/v1_profile_manifest.json"
    domain={s["segment_id"]:s for s in json.loads(profile.read_text())["segments"]}
    inputs=[args.audit/"board_e2_run1"/f"vta_{s:02d}_{s+1:02d}"/f"inputs_{i}.npz"
            for s in (5,10,15) for i in range(3)]
    save(args.output/"preregistered.json",{
        "scope":"CPU-only local boundary pilot, no new board measurements or tile search",
        "pairs":[[s,s+1,s+2] for s in (5,10,15)],
        "inputs":"all three frozen E2 cases per pair; no selection based on outputs",
        "comparison":"independently quantized main+projection then float32 add/relu versus quantized joined stage",
        "threshold":"bit exact; report every mismatch, not ImageNet accuracy",
        "qconfig":{"global_scale":8.0,"skip_conv_layers":[]},
        "sources_sha256":{str(p):sha(p) for p in [Path(__file__),profile,*inputs]}})
    model=vision.get_model("resnet18_v1",pretrained=True)
    features,head=list(model.features),model.output
    blocks=build_resnet18_unit_blocks(features,head)
    metadata=build_resnet18_unit_metadata(blocks,vta.get_env().BATCH,224)
    rows=[]
    for start in (5,10,15):
        executors={}
        for end in (start+1,start+2):
            sid=f"vta:{start:02d}:{end:02d}"
            segment=domain[sid]
            stage={"name":"stage_audit","device":"vta","unit_names":segment["unit_names"]}
            relay_inputs=relay_inputs_for_stage(stage,metadata)
            mod,params=lower_stage_to_relay(make_stage_block(features,head,stage,unit_blocks=blocks),relay_inputs)
            identity={k:constant(relay.const(v)) for k,v in params.items()}
            prior=args.audit/"precodegen87_run1"/sid.replace(":","_")/"params_identity.json"
            assert identity==json.loads(prior.read_text()),"Model parameters changed"
            local=args.output/sid.replace(":","_")
            local.mkdir()
            save(local/"params_identity.json",identity)
            with tvm.transform.PassContext(opt_level=3):
                with relay.quantize.qconfig(global_scale=8.0,skip_conv_layers=[]):
                    quantized=relay.quantize.quantize(mod,params=params)
            (local/"quantized.json").write_text(tvm.ir.save_json(quantized))
            (local/"quantized.relay").write_text(quantized.astext(show_meta_data=False))
            with tvm.transform.PassContext(opt_level=3,disabled_pass=["AlterOpLayout"]):
                factory=relay.build(quantized,target="llvm",params=params)
            executors[end]=graph_executor.GraphModule(factory["default"](tvm.cpu()))
            print("[BUILD]",sid,flush=True)
        for case in range(3):
            input_path=args.audit/"board_e2_run1"/f"vta_{start:02d}_{start+1:02d}"/f"inputs_{case}.npz"
            with np.load(input_path) as data:
                values={k:data[k] for k in data.files}
            joined_input=args.audit/"board_e2_run1"/f"vta_{start:02d}_{start+2:02d}"/f"inputs_{case}.npz"
            with np.load(joined_input) as data:
                assert set(values)==set(data.files)
                assert all(np.array_equal(v,data[k]) for k,v in values.items())
            outputs={}
            for end,executor in executors.items():
                executor.set_input(**values)
                executor.run()
                outputs[end]=[executor.get_output(i).numpy() for i in range(executor.get_num_outputs())]
            assert len(outputs[start+1])==2 and len(outputs[start+2])==1
            main,projection=outputs[start+1]
            outside=np.maximum(main+projection,np.float32(0))
            inside=outputs[start+2][0]
            assert outside.shape==inside.shape
            np.savez(args.output/f"layer{start//5+1}_case{case}.npz",
                     main=main,projection=projection,tail_outside=outside,tail_inside=inside)
            row={"layer":start//5+1,"case":case,"elements":inside.size,
                 "mismatched":int(np.count_nonzero(outside!=inside)),
                 "max_abs_error":float(np.max(np.abs(outside-inside))),
                 "inside_negative":int(np.count_nonzero(inside<0)),
                 "outside_max":float(outside.max()),"inside_max":float(inside.max()),
                 "input_sha256":sha(input_path),"joined_input_sha256":sha(joined_input)}
            rows.append(row)
            print("[RESULT]",json.dumps(row),flush=True)
    save(args.output/"summary.json",{"complete":True,"rows":rows,
        "all_exact":all(r["mismatched"]==0 for r in rows),
        "full_e3_complete":False,"scope":"LLVM local-tail numerical pilot; no whole-pipeline or dataset accuracy claim"})


if __name__=="__main__":
    main()
