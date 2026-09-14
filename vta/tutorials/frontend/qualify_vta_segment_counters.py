#!/usr/bin/env python3
"""E2: board correctness and runtime-counter qualification for frozen endpoints."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import tvm
from tvm import relay, rpc
from tvm.contrib import cc
from mxnet.gluon.model_zoo import vision
import vta
from audit_vta_compile_context import ROOT, REPORT, sha, save
from capture_vta_segment_tir import Archive
from compare_vta_segment_tir import normalize
from profile_split_resnet18_stages import build_vta_stage, create_stage_module
from profile_vta_stage_boundary import build_cpu_reference, host_outputs, compare_outputs
from split_resnet18_stages import (build_resnet18_unit_blocks, build_resnet18_unit_metadata,
    make_stage_block, relay_inputs_for_stage, lower_stage_to_relay)

SSH_OPTIONS=["-o","BatchMode=yes","-o","ConnectTimeout=8","-o","HostKeyAlgorithms=+ssh-rsa",
             "-o","PubkeyAcceptedAlgorithms=+ssh-rsa"]


def board_snapshot(host):
    # Delimited key=value output; never inspect credentials or arbitrary environment.
    command="""for pair in 'boot:/proc/sys/kernel/random/boot_id' 'fpga:/sys/class/fpga_manager/fpga0/state' 'udmabuf:/sys/class/u-dma-buf/udmabuf0/size' 'frequency:/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq' 'governor:/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor' 'ps_raw:/sys/bus/iio/devices/iio:device0/in_temp0_ps_temp_raw' 'ps_offset:/sys/bus/iio/devices/iio:device0/in_temp0_ps_temp_offset' 'ps_scale:/sys/bus/iio/devices/iio:device0/in_temp0_ps_temp_scale'; do key=${pair%%:*}; path=${pair#*:}; printf '%s=' "$key"; cat "$path"; done
sha256sum /lib/firmware/vta_hpc.bit /mnt/sd/tvm_deploy/hpc/libvta.so /mnt/sd/tvm_deploy/hpc/libtvm_runtime.so
"""
    raw=subprocess.check_output(["ssh",*SSH_OPTIONS,"root@"+host,command],text=True,timeout=20)
    result={"host_utc":datetime.now(timezone.utc).isoformat(),"raw":raw}
    for line in raw.splitlines():
        if "=" in line:
            key,value=line.split("=",1)
            result[key]=value
    result["temperature_ps_c"]=(float(result["ps_raw"])+float(result["ps_offset"]))*float(result["ps_scale"])/1000
    assert result["fpga"]=="operating" and result["udmabuf"]=="201326592"
    for digest in ["7bf1ac95b1182c670cd25241111f33725b4ea37ed6d66f2df37b0b2e7df528d6",
                   "eedfabb0630d58bf2eabaef9b4f2d2e4bfcdfa52a70503090f5608e79404ee5d",
                   "04e894baf305311315fd0c79d03ec2a6375b9591916e9c6f429cf3697c4457b2"]:
        assert digest in raw,"Frozen board artifact changed"
    return result


def counter_differences(actual, expected):
    return {key:{"expected":value,"observed":actual.get(key)}
            for key,value in expected.items() if actual.get(key)!=value}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host",required=True)
    parser.add_argument("--port",type=int,default=9091)
    parser.add_argument("--archive",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--segments",nargs="*",help="Explicit qualification subset, logged before any build")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    read=lambda p:json.loads(p.read_text())
    ids={f"vta:{s:02d}:{e:02d}" for s in (5,10,15) for e in range(s,s+3)}|{"vta:01:02","vta:18:19"}
    if args.segments:
        assert set(args.segments)<={r["segment_id"] for r in read(args.archive/"summary.json")["rows"]}
        ids=set(args.segments)
    snapshot=board_snapshot(args.host)
    archive=read(args.archive/"summary.json")
    assert archive["complete"]
    priors={r["segment_id"]:r for r in archive["rows"]}
    census={r["segment_id"]:r for r in read(args.archive/"logical_dma_census_v2.json")["rows"]}
    env=vta.get_env()
    save(args.output/"preregistered.json",{"segments":sorted(ids),"input_seeds":[260908,260909,260910],
        "input_distributions":["uniform[-1,1]","normal(0,2)","zeros"],"warmup_per_input":2,
        "profile_runs_per_input":3,"correctness":"bit-exact to independent same-qconfig LLVM",
        "timing_claim":"none: profiled single-stage diagnostics only, no pipeline FPS",
        "threadpool":"1 thread, default big-core affinity mode 1; no explicit host pinning claim",
        "script_sha256":sha(Path(__file__).read_bytes()),"archive_summary_sha256":sha((args.archive/"summary.json").read_bytes()),
        "census_sha256":sha((args.archive/"logical_dma_census_v2.json").read_bytes()),
        "board_before":snapshot})
    model=vision.get_model("resnet18_v1",pretrained=True)
    features,head=list(model.features),model.output
    units=build_resnet18_unit_blocks(features,head)
    metadata=build_resnet18_unit_metadata(units,env.BATCH,224)
    domain=read(REPORT/"resource_aware_maxplus/v1_profile_manifest.json")["segments"]
    results=[]
    for segment in domain:
        sid=segment["segment_id"]
        if sid not in ids:
            continue
        local=args.output/sid.replace(":","_")
        local.mkdir()
        print("[BUILD]",sid,flush=True)
        stage={"name":"stage_audit","device":"vta","unit_names":segment["unit_names"]}
        inputs=relay_inputs_for_stage(stage,metadata)
        mod,params=lower_stage_to_relay(make_stage_block(features,head,stage,unit_blocks=units),inputs)
        reference=build_cpu_reference({"relay_mod":mod,"params":params})
        capture,audit=Archive(local),{}
        graph,lib,lowered=build_vta_stage("stage_audit",mod["main"],params,env,tuning_audit=audit,
                                        require_tuned=True,compile_instruments=[capture])
        assert sha(graph.encode())==priors[sid]["graph_sha256"],"Graph changed since static census"
        assert len(capture.snapshots)==len(priors[sid]["snapshots"])
        for new,old in zip(capture.snapshots,priors[sid]["snapshots"]):
            lhs=tvm.ir.load_json((local/new["file"]).read_text())
            rhs=tvm.ir.load_json((args.archive/sid.replace(":","_")/old["file"]).read_text())
            assert set(g.name_hint for g in lhs.functions)==set(g.name_hint for g in rhs.functions)
            for gv,func in lhs.functions.items():
                assert tvm.ir.structural_equal(normalize(func),normalize(rhs[gv.name_hint]),map_free_vars=True)
        (local/"graph.json").write_text(graph)
        (local/"graph.params").write_bytes(relay.save_param_dict(lowered))
        binary=local/"stage.so"
        sysroot=os.environ["SDKTARGETSYSROOT"]
        lib.export_library(str(binary),fcompile=cc.cross_compiler("aarch64-xilinx-linux-g++"),
                           options=["--sysroot="+sysroot,"-Wl,-rpath-link,"+sysroot+"/lib",
                                    "-Wl,-rpath-link,"+sysroot+"/usr/lib"])
        row={"segment_id":sid,"build_tir_equal":True,"tuning":audit,"so_sha256":sha(binary.read_bytes()),
             "params_sha256":sha((local/"graph.params").read_bytes()),"samples":[],"before":board_snapshot(args.host)}
        assert row["before"]["boot"]==snapshot["boot"]
        # Each child RPC process owns its allocator; no accumulation across stages.
        remote=rpc.connect(args.host,args.port,session_timeout=180)
        remote.get_function("runtime.config_threadpool")(1,1)
        assert remote.get_function("runtime.NumThreads")()==1
        remote.upload(str(binary),target="e2_stage.so")
        executor,ctx=create_stage_module("stage_audit","vta",graph,remote.load_module("e2_stage.so"),remote)
        executor.set_input(**lowered)
        clear=remote.get_function("vta.runtime.profiler_clear")
        status=remote.get_function("vta.runtime.profiler_status")
        expected=dict(census[sid]["logical_descriptor_totals"])
        expected.update(push_alu_op_calls=census[sid]["uop_scope_calls"].get("VTAPushALUOp",0),
                        push_gemm_op_calls=census[sid]["uop_scope_calls"].get("VTAPushGEMMOp",0))
        for case,seed in enumerate([260908,260909,260910]):
            rng=np.random.default_rng(seed)
            values={name:(rng.uniform(-1,1,shape) if case==0 else rng.normal(0,2,shape)
                          if case==1 else np.zeros(shape)).astype("float32") for name,shape in inputs}
            np.savez(local/f"inputs_{case}.npz",**values)
            executor.set_input(**values)
            reference.set_input(**values)
            reference.run()
            wanted=[reference.get_output(i).numpy() for i in range(reference.get_num_outputs())]
            for _ in range(2):
                executor.run()
            for repeat in range(3):
                clear()
                executor.run()
                observed=json.loads(status())
                diffs=counter_differences(observed,expected)
                correctness=compare_outputs(host_outputs(executor,remote),wanted)
                row["samples"].append({"input_case":case,"repeat":repeat,"runtime_profile":observed,
                                       "counter_differences":diffs,"correctness":correctness})
                save(local/"qualification.json",row)
                assert correctness["correct"],("Output differs",sid,case,correctness)
                assert not diffs,("Counters differ",sid,case,diffs)
        row["after"]=board_snapshot(args.host)
        assert row["after"]["boot"]==snapshot["boot"]
        row["status"]="passed"
        save(local/"qualification.json",row)
        results.append(row)
        save(args.output/"summary.json",{"complete":len(results)==len(ids),"results":results,
                                         "scope":"single-boot functional/counter qualification, not FPS or DDR stall"})
        # Close explicitly: otherwise references can keep the remote child alive.
        remote._sess.get_function("CloseRPCConnection")()
        print("[PASS]",sid,"samples",len(row["samples"]),flush=True)
    print("[COMPLETE]",len(results),flush=True)


if __name__=="__main__":
    main()
