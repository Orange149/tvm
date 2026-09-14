#!/usr/bin/env python3
"""E3 board numerical/counter qualification using frozen VTA binaries and CPU tails."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import tvm
from tvm import relay,rpc
from tvm.contrib import cc,graph_executor

from qualify_vta_segment_counters import board_snapshot,counter_differences,SSH_OPTIONS
from profile_split_resnet18_stages import create_stage_module
from profile_vta_stage_boundary import host_outputs,compare_outputs
from qualify_once_quantized_tail_split import save,sha


def check_ranges(rows,base,size):
    for row in rows:
        assert row["virtual_address"]%256==0 and row["physical_address"]%256==0
        assert base<=row["physical_address"] and row["physical_address"]+row["bytes"]<=base+size
    ordered=sorted(rows,key=lambda r:r["physical_address"])
    assert all(a["physical_address"]+a["bytes"]<=b["physical_address"] for a,b in zip(ordered,ordered[1:]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host",required=True)
    parser.add_argument("--port",type=int,default=9091)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--audit",type=Path,default=Path(
        "vta/tutorials/frontend/report_out/stage_tile_cotuning/compile_context_audit"))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    prior=args.audit/"board_e2_run1"
    host=args.audit/"e3_once_quantized_run2"
    snapshot=board_snapshot(args.host)
    raw=subprocess.check_output(["ssh",*SSH_OPTIONS,"root@"+args.host,
        "cat /sys/class/u-dma-buf/udmabuf0/phys_addr"],text=True,timeout=20)
    base=int(raw.strip(),0)
    helper=Path("vta/apps/native_deploy/vta_e3_buffer_probe.cc")
    census_path=args.audit/"representatives23_run1/logical_dma_census_v2.json"
    sources=[Path(__file__),helper,census_path,Path(__file__).with_name("qualify_vta_segment_counters.py"),
             Path(__file__).with_name("profile_vta_stage_boundary.py"),Path(__file__).with_name("profile_split_resnet18_stages.py")]
    for start in (5,10,15):
        for end in (start+1,start+2):
            local=prior/f"vta_{start:02d}_{end:02d}"
            qual=json.loads((local/"qualification.json").read_text())
            assert qual["status"]=="passed"
            assert sha(local/"stage.so")==qual["so_sha256"] and sha(local/"graph.params")==qual["params_sha256"]
            sources.extend(local/name for name in ("stage.so","graph.params","graph.json","qualification.json"))
        sources.append(host/f"layer{start//5+1}/float32_consumer/relay.json")
        sources.extend(host/f"layer{start//5+1}/outputs_{case}.npz" for case in range(3))
        sources.extend(prior/f"vta_{start:02d}_{start+1:02d}/inputs_{case}.npz" for case in range(3))
        sources.extend(args.audit/"e3_tail_pilot_run1"/f"layer{start//5+1}_case{case}.npz" for case in range(3))
    save(args.output/"preregistered.json",{
        "scope":"single-boot single-inflight numerical/counter qualification, not FPS or C2 K2 safety proof",
        "arms":["frozen joined VTA","frozen VTA producer + explicit float32 quant-semantics CPU tail/shared",
                "same producer + same CPU tail/copy"],
        "input_cases":[0,1,2],"repeats_per_case_arm":3,"initial_warmup_per_arm":2,
        "order":"rotate mono/shared/copy by case and repeat; not performance statistics",
        "correctness":"bit-exact to frozen once-quantized host A; preserves legacy wrapping",
        "copy_scope":"two CPU set_input copies from CPU views; counted by API ledger, not VTA software profiler",
        "shared_scope":"two ext_dev owners with CPU views; producer set_output_zero_copy and consumer set_input_zero_copy",
        "board_before":snapshot,"udmabuf_base":base,
        "sources_sha256":{str(p):sha(p) for p in sources}})
    sysroot=os.environ["SDKTARGETSYSROOT"]
    options=["--sysroot="+sysroot,"-Wl,-rpath-link,"+sysroot+"/lib","-Wl,-rpath-link,"+sysroot+"/usr/lib"]
    probe=args.output/"buffer_probe.so"
    subprocess.run(["aarch64-xilinx-linux-g++","-std=c++17","-shared","-fPIC","-O2",
        *options,"-Iinclude","-I3rdparty/dlpack/include","-I3rdparty/dmlc-core/include",
        "-I3rdparty/vta-hw/include",str(helper),"-o",str(probe)],check=True)
    expected={r["segment_id"]:r for r in json.loads(census_path.read_text())["rows"]}
    rows=[]
    for start in (5,10,15):
        layer=start//5+1
        local=args.output/f"layer{layer}"
        local.mkdir()
        mod_path=host/f"layer{layer}/float32_consumer/relay.json"
        mod=tvm.ir.load_json(mod_path.read_text())
        with tvm.transform.PassContext(opt_level=3,disabled_pass=["AlterOpLayout"]):
            factory=relay.build(mod,target="llvm -mtriple=aarch64-linux-gnu")
        binary=local/"cpu_tail.so"
        factory.get_lib().export_library(str(binary),fcompile=cc.cross_compiler("aarch64-xilinx-linux-g++"),options=options)
        (local/"graph.json").write_text(factory.get_graph_json())
        (local/"graph.params").write_bytes(relay.save_param_dict(factory.get_params()))
        row={"layer":layer,"cpu_tail_sha256":sha(binary),"buffer_probe_sha256":sha(probe),
             "before":board_snapshot(args.host),"samples":[]}
        assert row["before"]["boot"]==snapshot["boot"]
        remote=rpc.connect(args.host,args.port,session_timeout=300)
        try:
            remote.get_function("runtime.config_threadpool")(1,1)
            assert remote.get_function("runtime.NumThreads")()==1
            remote.upload(str(probe),target="e3_probe.so")
            probe_module=remote.load_module("e3_probe.so")
            describe=remote.get_function("vta.e3.describe_cpu_view")
            cpu_view=remote.get_function("vta.e3.shared_cpu_view")
            sync=remote.get_function("vta.e3.sync_host_read")
            clear=remote.get_function("vta.runtime.profiler_clear")
            status=remote.get_function("vta.runtime.profiler_status")
            stages={}
            for arm,end in (("producer",start+1),("mono",start+2)):
                archived=prior/f"vta_{start:02d}_{end:02d}"
                remote.upload(str(archived/"stage.so"),target=arm+".so")
                exe,_=create_stage_module(arm,"vta",(archived/"graph.json").read_text(),remote.load_module(arm+".so"),remote)
                exe.load_params((archived/"graph.params").read_bytes())
                stages[arm]=exe
            remote.upload(str(binary),target="cpu_tail.so")
            module=remote.load_module("cpu_tail.so")
            consumers={arm:graph_executor.create(factory.get_graph_json(),module,remote.cpu(0)) for arm in ("shared","copy")}
            for c in consumers.values():
                c.load_params(relay.save_param_dict(factory.get_params()))
            producer=stages["producer"]
            owners=[tvm.nd.empty(tuple(producer.get_output(i).shape),"float32",remote.ext_dev(0)) for i in range(2)]
            views=[cpu_view(owner) for owner in owners]
            ranges=[json.loads(describe(view)) for view in views]
            check_ranges(ranges,base,int(snapshot["udmabuf"]))
            row["shared_ranges"]=ranges
            payload=sum(r["bytes"] for r in ranges)
            for i,(owner,view) in enumerate(zip(owners,views)):
                producer.set_output_zero_copy(i,owner)
                consumers["shared"].set_input_zero_copy(f"edge{i}",view)
            def run(arm,diagnostic):
                clear()
                if arm=="mono":
                    stages["mono"].run()
                    after=json.loads(status())
                    return after,after
                producer.run()
                after_producer=json.loads(status())
                for owner in owners:
                    sync(owner)
                if arm=="copy":
                    for i,view in enumerate(views):
                        consumers[arm].set_input(f"edge{i}",view)
                consumers[arm].run()
                return after_producer,json.loads(status())
            for case in range(3):
                with np.load(prior/f"vta_{start:02d}_{start+1:02d}/inputs_{case}.npz") as data:
                    values={k:data[k] for k in data.files}
                with np.load(host/f"layer{layer}/outputs_{case}.npz") as data:
                    wanted=data["A"]
                with np.load(args.audit/"e3_tail_pilot_run1"/f"layer{layer}_case{case}.npz") as data:
                    producer_wanted=[data["main"],data["projection"]]
                for stage in stages.values():
                    stage.set_input(**values)
                if case==0:
                    for arm in ("mono","shared","copy"):
                        for _ in range(2):
                            run(arm,False)
                for repeat in range(3):
                    arms=["mono","shared","copy"]
                    offset=(case+repeat)%3
                    for arm in arms[offset:]+arms[:offset]:
                        first,last=run(arm,True)
                        sid=f"vta:{start:02d}:{start+(2 if arm=='mono' else 1):02d}"
                        target=expected[sid]
                        counters=dict(target["logical_descriptor_totals"])
                        counters.update(push_alu_op_calls=target["uop_scope_calls"].get("VTAPushALUOp",0),
                                        push_gemm_op_calls=target["uop_scope_calls"].get("VTAPushGEMMOp",0))
                        differences=counter_differences(first,counters)
                        tail_diff=counter_differences(last,{k:first[k] for k in counters})
                        exe=stages["mono"] if arm=="mono" else consumers[arm]
                        got=host_outputs(exe,remote)
                        correctness=compare_outputs(got,[wanted])
                        bridge=None
                        if arm!="mono":
                            # After profiler snapshots; host retrieval is diagnostic only.
                            bridge=compare_outputs([v.numpy() for v in views],producer_wanted)
                            assert bridge["correct"]
                        sample={"case":case,"repeat":repeat,"arm":arm,"correctness":correctness,
                            "producer_correctness":bridge,"profile_after_producer":first,"profile_after_tail":last,
                            "counter_differences":differences,"tail_counter_differences":tail_diff,
                            "boundary_copy_api_calls":2 if arm=="copy" else 0,
                            "boundary_copy_payload_bytes":payload if arm=="copy" else 0}
                        row["samples"].append(sample)
                        save(local/"result.json",row)
                        np.savez(local/f"output_{case}_{repeat}_{arm}.npz",output=got[0])
                        assert correctness["correct"] and not differences and not tail_diff
            assert [json.loads(describe(v)) for v in views]==ranges
            row["after"]=board_snapshot(args.host)
            assert row["after"]["boot"]==snapshot["boot"]
            row["passed"]=True
            save(local/"result.json",row)
            rows.append(row)
            save(args.output/"summary.json",{"complete":len(rows)==3,"rows":rows,
                "full_e3_complete":False,"pipeline_fps_measured":False,"preserves_legacy_wrapping":True})
            print("[PASS]",layer,"samples",len(row["samples"]),flush=True)
            # Returned remote NDArray views must release their server-side
            # containers while the RPC session is still live.
            del views,owners,view,owner
        finally:
            remote._sess.get_function("CloseRPCConnection")()


if __name__=="__main__":
    main()
