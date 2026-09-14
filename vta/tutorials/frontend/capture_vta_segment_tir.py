#!/usr/bin/env python3
"""Capture complete Relay and lowered VTA TIR for frozen E1 representatives."""
import argparse
import json
import os
from pathlib import Path

import tvm
from tvm import relay
from mxnet.gluon.model_zoo import vision
import vta

from audit_vta_compile_context import sha, save, REPORT, ROOT, signature, constant
from collections import Counter
from profile_split_resnet18_stages import build_vta_stage
from split_resnet18_stages import (build_resnet18_unit_blocks, build_resnet18_unit_metadata,
    relay_inputs_for_stage, make_stage_block, lower_stage_to_relay)


@tvm.instrument.pass_instrument
class Archive:
    def __init__(self, directory):
        self.directory = directory
        self.snapshots = []
        self.inventory = None

    def run_after_pass(self, mod, info):
        if info.name == "FuseOps":
            (self.directory / "build_fused.relay").write_text(mod.astext(show_meta_data=False))
        if info.name == "AnnotateMemoryScope":
            inventory = []
            def collect(node):
                if not (isinstance(node, relay.Call) and isinstance(node.op, relay.Function)
                        and node.op.attrs is not None and "Primitive" in node.op.attrs.keys()):
                    return
                function = node.op
                digest, _ = signature(function)
                weights, ops = [], []
                bindings = dict(zip(function.params, node.args))
                def inside(call):
                    if isinstance(call, relay.Call) and isinstance(call.op, tvm.ir.Op):
                        ops.append(call.op.name)
                        if call.op.name == "nn.conv2d":
                            weight = bindings.get(call.args[1], call.args[1])
                            assert isinstance(weight, relay.Constant)
                            weights.append(constant(weight))
                relay.analysis.post_order_visit(function.body, inside)
                inventory.append({"signature":digest, "compiler_hash":str(function.attrs["hash"]),
                                  "ops":ops, "weights":weights})
            relay.analysis.post_order_visit(mod["main"].body, collect)
            assert self.inventory is None, "Multiple final Relay snapshots require disambiguation"
            self.inventory = inventory
            save(self.directory/"primitive_inventory.json", inventory)
        if info.name != "tir.vta.CPUAccessRewrite":
            return
        name = f"lowered_{len(self.snapshots):03d}"
        data = tvm.ir.save_json(mod)
        (self.directory / (name+".json")).write_text(data)
        (self.directory / (name+".tir")).write_text(mod.script())
        self.snapshots.append({"file":name+".json", "sha256":sha(data.encode()),
                               "pass":str(info.name), "functions":[gv.name_hint for gv in mod.functions]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path, help="Frozen coverage_and_representatives.json")
    parser.add_argument("--scan", type=Path, help="Matching completed level-1 directory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    env = vta.get_env()
    model = vision.get_model("resnet18_v1",pretrained=True)
    features,head = list(model.features),model.output
    units = build_resnet18_unit_blocks(features,head)
    metadata = build_resnet18_unit_metadata(units,env.BATCH,224)
    profile = json.loads((REPORT/"resource_aware_maxplus/v1_profile_manifest.json").read_text())
    ids = {"vta:15:15","vta:15:16","vta:15:17"}
    if args.selection:
        assert args.scan is not None
        selection = json.loads(args.selection.read_text())
        assert sha((args.scan/"summary.json").read_bytes()) == selection["source_summary_sha256"]
        ids = set(selection["level2_proposal"]["selected_segments"])
        prior = json.loads((args.scan/"provenance.json").read_text())
        for path,digest in prior["sources"].items():
            assert sha((ROOT/path).read_bytes()) == digest, ("Frozen source changed",path)
    selected = [s for s in profile["segments"] if s["segment_id"] in ids]
    assert len(selected)==len(ids)
    save(args.output/"provenance.json",{"script_sha256":sha(Path(__file__).read_bytes()),
         "libtvm_sha256":sha(Path(tvm._ffi.base._LIB._name).read_bytes()),
         "vta_config":env.cfg_dict,"target":str(env.target),"host":str(env.target_host),
         "compile_host_environment":{k:os.environ.get(k) for k in
             ["TVM_NUM_THREADS","TVM_THREAD_POOL_SPIN_COUNT","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS"]},
         "profile_builder_sha256":sha(Path(__file__).with_name("profile_split_resnet18_stages.py").read_bytes()),
         "selection":sorted(ids), "selection_sha256":sha(args.selection.read_bytes()) if args.selection else None,
         "scan_provenance_sha256":sha((args.scan/"provenance.json").read_bytes()) if args.scan else None,
         "scope":"compile-only; no board measurement"})
    rows=[]
    for segment in selected:
        sid=segment["segment_id"]
        local=args.output/sid.replace(":","_")
        local.mkdir()
        stage={"name":"stage_audit","device":"vta","unit_names":segment["unit_names"]}
        block=make_stage_block(features,head,stage,unit_blocks=units)
        mod,params=lower_stage_to_relay(block,relay_inputs_for_stage(stage,metadata))
        archive,audit=Archive(local),{}
        print("[CAPTURE]",sid,flush=True)
        graph,lib,_=build_vta_stage("stage_audit",mod["main"],params,env,
            tuning_audit=audit,require_tuned=True,compile_instruments=[archive])
        (local/"graph.json").write_text(graph)
        lib.save(str(local/"graphlib.o"))
        assert archive.snapshots
        assert archive.inventory
        if args.scan:
            expected = json.loads((args.scan/sid.replace(":","_")/"primitives.json").read_text())
            assert Counter(p["signature"] for p in expected) == Counter(p["signature"] for p in archive.inventory)
        rows.append({"segment_id":sid,"snapshots":archive.snapshots,"tuning":audit,
                     "graph_sha256":sha(graph.encode()),
                     "primitive_inventory_sha256":sha((local/"primitive_inventory.json").read_bytes()),
                     "level1_inventory_equal":bool(args.scan),
                     "scope":"raw complete lowered modules at fixed pass; exact segment DMA not yet qualified"})
        save(args.output/"summary.json",{"rows":rows,"complete":len(rows)==len(selected)})
        print("[OK]",sid,"functions",len(archive.snapshots),flush=True)


if __name__=="__main__":
    main()
