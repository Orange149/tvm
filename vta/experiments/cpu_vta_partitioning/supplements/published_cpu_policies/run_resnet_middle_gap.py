"""Bounded CPU5-7 diagnosis: matched isolation plus one observed pipeline.

Reuse accepted clean pipeline/qualification; do not repeat their inference.
Four new sessions: 21 isolated calls and 28 complete network inferences.
No new compiled binaries, prediction rules, complete search or reboot.
"""
from copy import deepcopy
from pathlib import Path
from types import FunctionType
import argparse
import json
import shlex
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from cpu_vta_partitioning.common import read, write as base_write, sha256, digest, verify_inventory, locked, now, sync_tree
from cpu_vta_partitioning.transport import SSH
from cpu_policies.open_resources import domain
from finite_service.common import freeze as immutable_freeze
from phase_profile import runtime as phase
from phase_profile.campaign import accept as phase_accept
from run_three_thread_worker_gap_compat import summary as observed_summary
from yolo_validation.components import hardware_check
import finite_service_holdout_board as parent
import run_compiled_gap_validation as clean

ROOT=parent.ROOT
OUTPUT=parent.OUTPUT
CONFIG=parent.CONFIG
POOL=ROOT/"artifacts/resnet_middle_gap_01"
RUN=ROOT/"runs/resnet_middle_gap_01"
CID="cfg_2ac037fa511b37a62f7c43a4"
BOOT="6018b3b7-bb29-4f80-92c1-0bfd8c81bb99"


def freeze(path,value):
    if Path(path).name=="BOARD_PLAN.json" and Path(path).is_relative_to(POOL):
        value={**value,"DNN_inferences":28,"CPU_stage_calls":21,"native_sessions":4,
               "pipeline_qualification_reused":True}
    return immutable_freeze(path,value)


def protocol():
    p=parent.protocol()
    p.update(experiment="resnet_middle_gap_01", component=dict(warmup=2,scored=8,tail=0),
        target_program="cpu_05_07",target_stage_index=2,target_mapping=[1,2,3],
        DNN_inferences=28,CPU_stage_calls=21,native_sessions=4,
        no_observer=False,only_known_layout_diagnosis=True,
        pipeline_clean_and_qualification_reused_without_inference=True,
        no_new_prediction_or_model=True,probe_quality_fraction=0.2,
        no_automatic_repeats=True,no_reboot=True)
    return p


def sources():
    return {**parent.sources(), **{str(ROOT/p):sha256(ROOT/p) for p in
        ("run_resnet_middle_gap.py","tests/test_resnet_middle_gap.py",
         "run_three_thread_worker_gap_compat.py","phase_profile/worker.py",
         "scheduler_gap_audit/runtime.py","scheduler_gap_audit/analysis.py",
         "phase_model/extract.py")}}


def selection():
    if read(clean.RUN/"state.json")["status"]!="COMPLETE":raise ValueError("clean cohort incomplete")
    s=clean.selection();cfg=next(c for c in s["selected_configurations"]["resnet18"] if c["id"]==CID)
    row=cfg["stages"][2]
    if row!=domain.stage("cpu",5,8,3,[1,2,3]) or cfg["execution_mode"]!="off":
        raise ValueError("target resource allocation changed")
    refs={};bundle=clean.POOL/"resnet18/capsule"
    for spec in clean.specs():
        if spec["network"]!="resnet18" or spec["case"]!=CID:continue
        task=clean.make_task(bundle,spec);p=clean.RUN/"accepted"/BOOT/(task["task_id"]+".json")
        r=read(p)
        if r["task"]!=task or r["boot_id"]!=BOOT:raise ValueError("clean reference identity differs")
        verify_inventory(r["raw"],r["files"])
        refs[spec["phase"]]=dict(receipt=str(p),sha256=sha256(p))
    if set(refs)!={"qualify","clean"}:raise ValueError("missing accepted clean/qualification reference")
    return dict(selected_configurations={"resnet18":[cfg]},
        components={"middle_t3":domain.component(row)}, references=refs,
        expected_boot_id=BOOT,DNN_budget=28)


def specs():
    component=[dict(network="resnet18",case="middle_t3",scope="component",phase=p,
                    probe=p!="clean",runner="phase") for p in ("qualify","clean","observed")]
    return component+[dict(network="resnet18",case=CID,scope="pipeline",phase="observed",probe=True,runner="phase")]


def write(path,value,**kwargs):
    # Add the new component before the reused packager hashes its capsule.
    if Path(path).name=="configurations.json" and Path(path).is_relative_to(POOL):
        value={**value,"components":selection()["components"]}
    return base_write(path,value,**kwargs)


def adapt(name):
    f=getattr(parent,name);g=FunctionType(f.__code__,{**f.__globals__,**globals()},f.__name__,f.__defaults__)
    g.__kwdefaults__=f.__kwdefaults__;return g


def prepare():
    s=selection()
    freeze(POOL/"PLAN.json",dict(selection=s,sources=sources(),protocol=protocol(),
        hypothesis="CPU5-7 residual reflects compute CPU growth, critical-worker CPU queue, or caller/runtime work; distinguish before changing model",
        selection_is_known_gap_diagnosis_not_unseen_validation=True,
        no_pipeline_values_used_as_prediction_inputs=True,
        no_FPGA_runtime_or_executor_rebuild=True,
        remaining_gap_ms=50.3639319375-39.01803,
        stop="report probe perturbation and unknowns; no automatic repeat or new intervention"))
    freeze(POOL/"SELECTION_FREEZE.json",dict(**s,sources=sources(),plan_sha256=sha256(POOL/"PLAN.json")))
    result=adapt("prepare")()
    if read(POOL/"BOARD_PLAN.json")["expected_boot_id"]!=BOOT:raise ValueError("packager boot differs")
    bundle=POOL/"resnet18/capsule";parent_bundle=clean.POOL/"resnet18/capsule"
    m=read(bundle/"manifest.json");old=read(parent_bundle/"manifest.json")
    identity_keys=("phase_runner_sha256","runtime_sha256","adapter_sha256")
    if any(m[k]!=old[k] for k in identity_keys):raise ValueError("diagnostic programs differ from clean reference")
    for name,item in read(parent_bundle/"plans/programs.json").items():
        if read(bundle/"plans/programs.json")[name]!=item:raise ValueError("program input/frontier identity differs")
    freeze(POOL/"COMMAND_AUDIT.json",dict(commands=[dict(spec=s,task_id=make_task(bundle,s)["task_id"],
        argv=native_command(make_task(bundle,s),bundle,"/bundle","/job")[0]) for s in specs()],
        same_executor_runtime_and_modules_as_clean=True,reused_qualification_has_identical_full_output_contract=True,
        observer_original_wait_only=True,taskstats_explicit=True,DNN_inferences=28,CPU_stage_calls=21))
    return result


def validate(bundle):
    if read(POOL/"PLAN.json")["sources"]!=sources():raise ValueError("diagnostic code changed")
    return adapt("validate")(bundle)


def make_task(bundle,spec):
    b=Path(bundle);m=read(b/"manifest.json")
    if spec not in specs():raise ValueError("unplanned diagnostic session")
    cfg=read(b/"plans/configurations.json")["components" if spec["scope"]=="component" else "pipelines"][spec["case"]]
    q=spec["phase"]=="qualify";component=spec["scope"]=="component"
    doc=dict(kind=("qualify_component" if q else "calibration") if component else "performance",
        configuration=deepcopy(cfg),stages=deepcopy(cfg["stages"]),spec=deepcopy(spec),
        environment=protocol()["environment"],**protocol()["qualification" if q else "component" if component else "pipeline"],
        input_indices=[0],executor_sha256=m["phase_runner_sha256"],
        adapter_sha256=m["adapter_sha256"] if spec["probe"] else None,
        runtime_sha256=m["runtime_sha256"],bundle_id=m["bundle_id"],experiment=protocol()["experiment"],
        startup_affinity_probe=True,queue_counter_scope="per-TID TASKSTATS with read envelopes" if spec["probe"] else None)
    doc["task_id"]=digest(doc)[:24];return doc


def native_command(doc,bundle,remote_bundle,remote_job):
    f=phase.native_command
    argv,blobs=FunctionType(f.__code__,{**f.__globals__,"make_task":make_task})(doc,bundle,remote_bundle,remote_job)
    if doc["spec"]["probe"]:
        at=argv.index(remote_bundle+"/runtime/ld-linux-aarch64.so.1")
        argv[at:at]=["VTA_QUEUE_COUNTERS=1","VTA_QUEUE_STATUS="+remote_job+"/queue_status.json"]
    return argv,blobs


def summary(raw,doc,bundle):
    if not doc["spec"]["probe"]:return phase.summary(raw,doc,bundle)
    result=observed_summary(raw,doc,bundle)
    q=read(Path(raw)/"queue_status.json")
    if not q["enabled"] or q["errors"] or not q["reads"]:raise ValueError("TASKSTATS observer unavailable")
    return dict(**result,queue_status=q)


def accept(*args,**kwargs):
    return FunctionType(phase_accept.__code__,{**phase_accept.__globals__,"summary":summary})(*args,**kwargs)


def matched_ssh(config,*,execute_board=False):return adapt("matched_ssh")(config,execute_board=execute_board)
def qemu():return adapt("qemu")()


def verified_receipts():
    state=read(RUN/"state.json")
    if state["status"]!="COMPLETE" or state["accepted_sessions"]!=4:raise ValueError("diagnostic cohort incomplete")
    bundle=POOL/"resnet18/capsule";validate(bundle);result=[]
    for spec in specs():
        t=make_task(bundle,spec);p=RUN/"accepted"/BOOT/(t["task_id"]+".json");r=read(p)
        if r["task"]!=t or r["boot_id"]!=BOOT:raise ValueError("diagnostic identity differs")
        verify_inventory(r["raw"],r["files"])
        if summary(r["raw"],t,bundle)!=r["summary"]:raise ValueError("diagnostic evidence no longer replays")
        result.append((p,r))
    return result


def board(*,execute_board=False):
    if not execute_board:raise ValueError("--execute-board is required")
    plan=read(POOL/"BOARD_PLAN.json")
    if plan["sources"]!=sources() or plan["selection_sha256"]!=sha256(POOL/"SELECTION_FREEZE.json"):
        raise ValueError("frozen diagnostic controller differs")
    if not all(v["passed"] for v in read(POOL/"QEMU.json").values()):raise ValueError("QEMU required")
    s=selection();reference=read(s["references"]["qualify"]["receipt"])
    ssh=matched_ssh(read(CONFIG)["board"],execute_board=True);boot=ssh.boot_id()
    if boot!=BOOT:raise ValueError("boot changed; stop without mixing clean reference")
    b=POOL/"resnet18/capsule";validate(b);started=time.monotonic()
    with locked(RUN):
        freeze(RUN/"run.json",dict(boot_id=boot,plan_sha256=sha256(POOL/"PLAN.json"),protocol=protocol()))
        def state(status,**kw):write(RUN/"state.json",dict(status=status,boot_id=boot,updated_at=now(),**kw),replace=True)
        try:
            (RUN/"resnet18").mkdir(parents=True,exist_ok=True)
            remote=ssh.deploy(b,RUN/"resnet18");ssh.qualify_environment(remote,boot,RUN/"resnet18",b)
            write(RUN/"hardware.json",hardware_check(ssh,b,boot),replace=True)
            expected={"pipeline":reference["summary"]["output_fingerprints"]};cpu=dnn=0
            for i,spec in enumerate(specs(),1):
                t=make_task(b,spec);scope=spec["scope"]
                state("RUNNING",accepted_sessions=i-1,total_sessions=4,current_spec=spec)
                r=accept(ssh,t,b,remote,RUN,boot,expected.get(scope))
                if spec["phase"]=="qualify":expected[scope]=r["summary"]["output_fingerprints"]
                calls=sum(t[k] for k in ("warmup","scored","tail"))
                if scope=="pipeline":dnn+=calls
                else:cpu+=calls
                print("ResNet中段误差诊断",i,"/4",scope,spec["phase"],flush=True)
            if (dnn,cpu)!=(28,21):raise ValueError("diagnostic call budget differs")
            ssh.idle_command("cd "+shlex.quote(remote)+" && sha256sum -c board.sha256 > middle_gap_post_verify.log && sync",boot=boot,timeout=300)
            if ssh.boot_id()!=BOOT:raise ValueError("boot changed at completion")
            state("COMPLETE",accepted_sessions=4,DNN_inferences=dnn,CPU_stage_calls=cpu,
                seconds=time.monotonic()-started,no_reboot=True,no_automatic_extra_measurement=True)
            sync_tree(RUN)
        except Exception as e:
            state("STOPPED_ON_ERROR",error_type=type(e).__name__,error=str(e));raise
    return read(RUN/"state.json")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("command",choices=("prepare","qemu","validate","board","status"))
    p.add_argument("--execute-board",action="store_true");a=p.parse_args()
    if a.command=="board":r=board(execute_board=a.execute_board)
    elif a.command=="validate":r=dict(passed=True,receipts=len(verified_receipts()))
    elif a.command=="status":r=read(RUN/"state.json") if (RUN/"state.json").exists() else dict(status="NOT_STARTED")
    else:r=globals()[a.command]()
    print(json.dumps(r,ensure_ascii=False,indent=2))
