"""Reuse the repaired task-switch recorder without per-call TASKSTATS reads.

One known ResNet layout, one new frozen observer executable, 3 sessions / 57
network inferences. The sole compiled adaptation is 12 -> 13 target roles.
No scheduling-policy, runtime, module, scoring, or DP changes.
"""
from copy import deepcopy
from pathlib import Path
from types import FunctionType
import argparse
import inspect
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from cpu_vta_partitioning.common import read,write as original_write,sha256,digest,verify_inventory
from finite_service.common import freeze as immutable_freeze
from phase_profile.prepare import link_checked as original_link
from phase_profile import runtime as phase
from phase_profile.campaign import accept as phase_accept
from scheduler_gap_audit import runtime as counters
from run_three_thread_worker_gap_compat import validate_control_tids
from yolo_scheduler_trace import events as old_events
from yolo_scheduler_trace_repair import events as repaired_events
import finite_service_holdout_board as parent
import run_resnet_middle_gap as target

ROOT=parent.ROOT;OUTPUT=parent.OUTPUT;CONFIG=parent.CONFIG
POOL=ROOT/"artifacts/resnet_middle_switch_01"
RUN=ROOT/"runs/resnet_middle_switch_01"
BOOT=target.BOOT;CID=target.CID
FROZEN=ROOT/"artifacts/yolo_scheduler_trace_repair_01/build"


def protocol():
    p=parent.protocol();p.update(experiment="resnet_middle_switch_01",no_observer=False,
        purpose="known CPU5-7 queue and return-tail attribution with existing event recorder",
        active_roles=13,queue_counter_reads=False,PERF_RECORD_SWITCH=True,periodic_sampling=False,
        no_new_sampling_thread=True,recorder_revision=2,probe_quality_fraction=0.2,
        native_sessions=3,DNN_inferences=57,no_new_predictor=True)
    return p


def sources():
    return {**parent.sources(),**{str(ROOT/p):sha256(ROOT/p) for p in
        ("run_resnet_middle_switch.py","tests/test_resnet_middle_switch.py","run_resnet_middle_gap.py",
         "yolo_scheduler_trace/events.py","yolo_scheduler_trace_repair/events.py",
         "scheduler_gap_audit/runtime.py","run_three_thread_worker_gap_compat.py")}}


def port_header(text):
    before="targets_.size()!=12";after="targets_.size()!=13"
    if text.count(before)!=1:raise ValueError("frozen target-count anchor changed")
    return text.replace(before,after),dict(before=before,after=after,other_native_recorder_code_unchanged=True)


def build():
    dest=POOL/"build";src=dest/"sources";info=dest/"BUILD.json"
    if info.exists():
        b=read(info)
        for p,h in b["source_files"].items():
            if sha256(p)!=h:raise ValueError("recorder build source changed")
        if sha256(dest/"vta_stage_pipeline_runner_phase")!=b["runner_sha256"]:raise ValueError("recorder binary changed")
        return b
    old=read(FROZEN/"BUILD.json");src.mkdir(parents=True,exist_ok=True)
    for p,h in old["source_files"].items():
        if sha256(p)!=h:raise ValueError("repaired recorder source changed")
        shutil.copy2(p,src/Path(p).name)
    header=src/"vta_session_profile.h";text,proof=port_header(header.read_text());header.write_text(text)
    tc=old["toolchain"]
    if sha256(tc["sdk_setup"])!=tc["sdk_setup_sha256"]:raise ValueError("SDK changed")
    for t in tc["tools"].values():
        if sha256(t["path"])!=t["sha256"]:raise ValueError("compiler changed")
    recipe=read(ROOT/"artifacts/cross_network_model_01/phase_profile/build/BUILD.json")["compiler_recipe"]
    env=dict(os.environ);env.pop("LD_LIBRARY_PATH",None);started=time.monotonic()
    done=subprocess.run(["bash","-c",'source "$1" || exit $?; shift; exec "$@"',"middle-switch-sdk",tc["sdk_setup"],
        "bash","-c",recipe,"middle-switch-compile",str(src/"vta_stage_pipeline_runner_phase.cc"),
        str(dest/"vta_stage_pipeline_runner_phase"),str(ROOT.parents[4]),
        str(ROOT/"artifacts/scheduler_gap_audit_01/resnet18/capsule/runtime")],
        env=env,capture_output=True,text=True,timeout=180)
    (dest/"compile.log").write_text(done.stdout+done.stderr)
    if done.returncode:raise ValueError("optional recorder build failed; see compile.log")
    b=dict(runner_sha256=sha256(dest/"vta_stage_pipeline_runner_phase"),
        source_files={str(p):sha256(p) for p in src.iterdir()},toolchain=tc,header_port=proof,
        parent_binary_sha256=old["runner_sha256"],parent_build_sha256=sha256(FROZEN/"BUILD.json"),
        native_source_only_target_count_changed=True,runtime_rebuilt=False,modules_recompiled=False,
        seconds=time.monotonic()-started)
    original_write(info,b);return b


def selection():
    s=target.selection()
    return dict(selected_configurations=s["selected_configurations"],expected_boot_id=BOOT,DNN_budget=57)


def specs():
    return [dict(network="resnet18",scope="pipeline",case=CID,phase=p,probe=p=="observed",runner="phase")
        for p in ("qualify","clean","observed")]


def freeze(path,value):
    if Path(path).name=="BOARD_PLAN.json" and Path(path).is_relative_to(POOL):
        value={**value,"DNN_inferences":57,"native_sessions":3}
    return immutable_freeze(path,value)


def link_checked(source,destination,expected):
    if Path(destination).name=="vta_stage_pipeline_runner_phase" and Path(destination).is_relative_to(POOL):
        source=POOL/"build/vta_stage_pipeline_runner_phase";expected=read(POOL/"build/BUILD.json")["runner_sha256"]
    return original_link(source,destination,expected)


def write(path,value,**kwargs):
    if Path(path).name=="manifest.json" and Path(path).is_relative_to(POOL):
        value={k:v for k,v in value.items() if k!="bundle_id"}
        value["phase_runner_sha256"]=read(POOL/"build/BUILD.json")["runner_sha256"]
        value["bundle_id"]=digest(value)[:24]
    if Path(path).name=="preparation.json" and Path(path).is_relative_to(POOL):
        value={**value,"recorder_build_sha256":sha256(POOL/"build/BUILD.json"),
            "native_change":"existing repaired recorder, exact 13-role target count; computation/runtime/modules unchanged"}
    return original_write(path,value,**kwargs)


def adapt(name):
    f=getattr(parent,name);g=FunctionType(f.__code__,{**f.__globals__,**globals()},f.__name__,f.__defaults__)
    g.__kwdefaults__=f.__kwdefaults__;return g


def prepare():
    b=build()
    freeze(POOL/"PLAN.json",dict(sources=sources(),protocol=protocol(),selection=selection(),
        build_sha256=sha256(POOL/"build/BUILD.json"),taskstats_disabled_in_observation=True,
        predeclared_attribution="all originally scored CPU5-7 calls whose complete launch/phase intervals are inside valid event coverage; no favorable-frame selection",
        no_FPS_or_pipeline_values_as_static_inputs=True,
        success="assess probe perturbation and observed competing actors; do not change predictor or extend measurement automatically"))
    freeze(POOL/"SELECTION_FREEZE.json",dict(**selection(),sources=sources(),plan_sha256=sha256(POOL/"PLAN.json")))
    result=adapt("prepare")()
    bundle=POOL/"resnet18/capsule"
    freeze(POOL/"COMMAND_AUDIT.json",dict(commands=[dict(task=make_task(bundle,s),
        argv=native_command(make_task(bundle,s),bundle,"/bundle","/job")[0]) for s in specs()],
        one_binary_for_clean_and_observed=True,active_roles=13,queue_counter_queries=0,
        no_CPU_affinity_or_waiting_policy_change=True,DNN_inferences=57))
    return result


def validate(bundle):
    build()
    if read(POOL/"PLAN.json")["sources"]!=sources():raise ValueError("diagnostic controller changed")
    return adapt("validate")(bundle)


def make_task(bundle,spec):
    if spec not in specs():raise ValueError("unplanned thread-event session")
    b=Path(bundle);m=read(b/"manifest.json");cfg=read(b/"plans/configurations.json")["pipelines"][CID]
    q=spec["phase"]=="qualify"
    doc=dict(kind="qualify_pipeline" if q else "performance",configuration=deepcopy(cfg),stages=deepcopy(cfg["stages"]),
        spec=deepcopy(spec),environment=protocol()["environment"],**protocol()["qualification" if q else "pipeline"],
        input_indices=[0],executor_sha256=m["phase_runner_sha256"],adapter_sha256=m["adapter_sha256"] if spec["probe"] else None,
        runtime_sha256=m["runtime_sha256"],bundle_id=m["bundle_id"],experiment=protocol()["experiment"],
        startup_affinity_probe=True,scheduler_events=spec["probe"],scheduler_recording_revision=2,
        scheduler_recording_policy="collector_completion",queue_counter_reads=False)
    doc["task_id"]=digest(doc)[:24];return doc


def native_command(doc,bundle,remote_bundle,remote_job):
    f=phase.native_command
    argv,blobs=FunctionType(f.__code__,{**f.__globals__,"make_task":make_task})(doc,bundle,remote_bundle,remote_job)
    if doc["spec"]["probe"]:
        at=argv.index(remote_bundle+"/runtime/ld-linux-aarch64.so.1")
        argv[at:at]=["VTA_QUEUE_COUNTERS=0","VTA_QUEUE_STATUS="+remote_job+"/queue_status.json"]
        argv += ["--session-profile-json",remote_job+"/profile.json","--session-profile-window","8,16"]
    return argv,blobs


def disabled_queue_rows(rows,*,scope):
    keys=("queue_before_ns","queue_after_ns","queue_read_start0_ns","queue_read_end0_ns","queue_read_start1_ns","queue_read_end1_ns") if scope=="worker" else (
        "queue_before_ns","queue_after_ns","read_start0_ns","read_end0_ns","read_start1_ns","read_end1_ns")
    if not rows or any(type(r.get(k)) is not int or r[k]!=0 for r in rows for k in keys):
        raise ValueError("disabled TASKSTATS fields must be explicitly zero, never treated as no waiting")
    return dict(records=len(rows),TASKSTATS_disabled=True,zero_fields_are_not_queue_measurements=True)


def phase_summary(raw,doc,affinity):
    f=counters.phase_summary
    result=FunctionType(f.__code__,{**f.__globals__,"validate_queue_rows":disabled_queue_rows})(raw,doc,affinity)
    if doc["spec"]["probe"]:
        rows=[json.loads(s) for s in (Path(raw)/"phase_trace.jsonl").read_text().splitlines()]
        validate_control_tids(rows,affinity)
    return result


def validate_events(profile,doc,affinity):
    if (profile.get("ring_trace_revision")!=2 or profile.get("drain_policy")!="collector_completion"
        or not isinstance(profile.get("retention_end_ns"),int) or profile["retention_end_ns"]<profile["stop_end_ns"]):
        raise ValueError("repaired recorder envelope absent")
    if any(c["end_ns"]>profile["retention_end_ns"] for s in profile["streams"] for c in s["drain_chunks"]):
        raise ValueError("ring drain escaped retention")
    text=inspect.getsource(old_events.validate_profile)
    if text.count("len(roles)!=12")!=1:raise ValueError("frozen role-count check changed")
    text=text.replace("len(roles)!=12","len(roles)!=13")
    ns={**old_events.validate_profile.__globals__,"decode_stream":repaired_events.decode_stream}
    exec(compile(text,"<same strict switch decoder, thirteen roles>","exec"),ns)
    result=ns["validate_profile"](profile,doc,affinity)
    result.update(physical_drain_evidence_valid=True,expected_roles=13)
    return result


def summary(raw,doc,bundle):
    f=phase.summary
    result=FunctionType(f.__code__,{**f.__globals__,"phase_summary":phase_summary})(raw,doc,bundle)
    path=Path(raw)/"profile.json"
    if doc["scheduler_events"]:
        status=read(Path(raw)/"queue_status.json")
        if status["enabled"] or status["errors"] or status["reads"]:raise ValueError("unexpected TASKSTATS query")
        events=[json.loads(s) for s in (Path(raw)/"wait_trace.jsonl").read_text().splitlines()]
        disabled_queue_rows(events,scope="worker")
        result["scheduler"]=validate_events(read(path),doc,read(Path(raw)/"affinity.json"))
        result["queue_status"]=status
    elif path.exists():raise ValueError("unexpected scheduler events in clean session")
    return result


def accept(*args,**kwargs):return FunctionType(phase_accept.__code__,{**phase_accept.__globals__,"summary":summary})(*args,**kwargs)
def matched_ssh(config,*,execute_board=False):return adapt("matched_ssh")(config,execute_board=execute_board)
def qemu():return adapt("qemu")()


def board(*,execute_board=False):return adapt("board")(execute_board=execute_board)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("command",choices=("prepare","qemu","validate","board","status"));p.add_argument("--execute-board",action="store_true");a=p.parse_args()
    if a.command=="board":r=board(execute_board=a.execute_board)
    elif a.command=="validate":r=validate(POOL/"resnet18/capsule")
    elif a.command=="status":r=read(RUN/"state.json") if (RUN/"state.json").exists() else dict(status="NOT_STARTED")
    else:r=globals()[a.command]()
    print(json.dumps(r,ensure_ascii=False,indent=2))
