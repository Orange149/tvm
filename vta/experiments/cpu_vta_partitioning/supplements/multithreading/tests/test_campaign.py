import copy
import json
from pathlib import Path
import numpy as np
import pytest

from cpu_vta_partitioning.common import read,write
from cpu_vta_partitioning.domain import Stage
from cpu_vta_partitioning.transport import TransportError
from multithreading.domain import ENVIRONMENTS,protocol,task,row
from multithreading.campaign import create,run_current,validate_state
from multithreading.evidence import Evidence,check_gate,identified
from multithreading.transport import native_command


class FakeBoard:
    def __init__(self):
        self.boot="00000000-0000-0000-0000-000000000000"
        self.launches=set();self.disconnect=False;self.synced=[]
    def boot_id(self): return self.boot
    def wait_idle(self,boot): assert boot==self.boot
    def deploy(self,bundle,root): return "/bundle"
    def qualify_environment(self,*a): pass
    def recover_abandoned(self,*a): pass
    def idle_command(self,script,**kwargs): self.synced.append((self.boot,script));return ""
    def acknowledge(self,*args): pass
    def measure(self,document,bundle,remote,root,boot):
        result=Path(root)/"raw"/boot/document["task_id"]/"collected"
        result.mkdir(parents=True,exist_ok=True)
        write(result.parent/"task.json",document,replace=True)
        if (boot,document["task_id"]) not in self.launches:
            self.launches.add((boot,document["task_id"]))
            af=[]
            for i,s in enumerate(document["stages"]):
                tids=list(range(10+i*10,10+i*10+s["threads"]))
                af.append({"requested_cpus":s["mask"],"thread_ids":tids,"active_runtime_threads":s["threads"],
                           "effective_affinities":[{"tid":t,"cpus":s["mask"]} for t in tids]})
            write(result/"affinity.json",{"stages":af})
            n=sum(document[k] for k in ("warmup","scored","tail"))
            rows=[]
            for i in range(n):
                item={"frame_id":i,"completion_index":i,"input_index":i%len(document["input_indices"]),
                      "completion_ms":float(i+1)*30,"total_latency_ms":40,
                      "stage_cpu_time_scope":"exclusive_process_cpu_time"}
                for j,s in enumerate(document["stages"]):
                    wall=10+(s["end"]-s["begin"])*10/s["threads"]
                    item.update({f"stage{j}_set_ms":.25,f"stage{j}_get_ms":.25,
                                 f"stage{j}_run_ms":wall,
                                 f"stage{j}_run_process_cpu_ms":wall*s["threads"],
                                 f"stage{j}_set_process_cpu_ms":.125,f"stage{j}_get_process_cpu_ms":.125})
                if document["execution_mode"]=="resource_group":
                    item["cpu_stage_gate_mode"]="resource_group"
                    for j,s in enumerate(document["stages"]):
                        for delta,key in enumerate(("request","acquire","release")):
                            item[f"stage{j}_cpu_gate_{key}_ms"]=i*30+j*.1+delta*.01 if s["device"]=="cpu" else None
                if document["kind"].startswith("qualify"):
                    dumps=[]
                    for j,s in enumerate(document["stages"]):
                        out=result/"outputs"/f"run_{i}"/f"stage{j}.bin"
                        out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(bytes([1,2]))
                        dumps.append({"output_files":[{"path":str(out),"dtype":"int8","shape":[2]}]})
                    item["stage_raw_outputs"]=dumps
                rows.append(item)
            (result/"result.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            if document.get("detailed"):
                wait=100 if document["diagnostic_placement"]=="overlap" else 50
                write(result/"profile.json",{"threads":[{"tid":tid,"runtime_ns":1000,"runqueue_wait_ns":wait}
                      for a in af for tid in a["thread_ids"]]})
        if self.disconnect:
            self.disconnect=False
            raise TransportError("simulated SSH disconnect after native completion")
        return result


@pytest.fixture
def environment(tmp_path,monkeypatch):
    import multithreading.prepare as prepare
    import multithreading.campaign as campaign
    import multithreading.analysis as analysis
    monkeypatch.setattr(prepare,"identity",lambda:"test-controller")
    monkeypatch.setattr(prepare,"validate",lambda b:read(Path(b)/"manifest.json"))
    p=protocol();p["expected"]["shared4_configurations"]=225;p["expected"]["dedicated3_configurations"]=49
    monkeypatch.setattr(campaign,"protocol",lambda:p)
    bundle=tmp_path/"bundle";bundle.mkdir()
    write(bundle/"manifest.json",{"bundle_id":"test-bundle","controller_identity":"test-controller"})
    programs={}
    for device,begin,end in (("cpu",0,3),("vta",3,18),("cpu",18,21)):
        name=f"{device}_{begin:02}_{end-1:02}"
        programs[name]={"name":name,"device":device,"begin":begin,"end":end,
                        "input_names":["data"],"input_node_ids":[0],"output_node_ids":[0],
                        "output_types":[{"shape":[2],"dtype":"int8"}]}
    # Historical three-island programs are fixture-only aliases needed when the
    # confirmation pool adds the frozen historical compositions.
    for device,begin,end in (("vta",3,8),("cpu",8,10),("vta",10,13),("cpu",13,15),("vta",15,18)):
        name=f"{device}_{begin:02}_{end-1:02}"
        programs[name]={"name":name,"device":device,"begin":begin,"end":end,
                        "input_names":["data"],"input_node_ids":[0],"output_node_ids":[0],
                        "output_types":[{"shape":[2],"dtype":"int8"}]}
    # Restrict search to the one-island topology while still qualifying the
    # compiled historical programs. Both independent counts are explicit.
    from cpu_vta_partitioning.domain import topologies as real_topologies
    def only_one(ps):
        return (t for t in real_topologies(ps) if len(t)==3)
    import multithreading.search as search
    monkeypatch.setattr(search,"topologies",only_one)
    write(bundle/"plans/programs.json",programs)
    records=[]
    for i in range(67):
        ref=bundle/"references"/f"{i}.npy";ref.parent.mkdir(exist_ok=True);np.save(ref,np.array([1,2],dtype="int8"))
        records.append({"input":f"inputs/{i}.bin","frontiers":{"0":{"npy":f"references/{i}.npy","raw":f"inputs/{i}.bin"}}})
    write(bundle/"plans/inputs.json",{"records":records})
    root=tmp_path/"run"
    create(bundle,root,{"host":"test","user":"root","remote_base":"/tmp/test"})
    return root,bundle,FakeBoard()


def test_ten_boots_resume_independent_confirmation_and_analysis(environment,recwarn):
    root,bundle,board=environment
    for i in range(10):
        board.boot=f"00000000-0000-0000-0000-{i+1:012d}"
        state=run_current(root,board)
        assert state["boot_index"]==i+1
        count=len(board.launches)
        run_current(root,board)
        assert len(board.launches)==count
    assert state["status"]=="COMPLETE"
    analysis=read(root/"analysis/analysis.json")
    assert len(analysis["primary"])==12
    assert all(v["n_boots"]==6 for v in analysis["primary"].values())
    assert not analysis["performance_gate"]
    assert len(read(root/"costs.json")["boot_ids"])==2
    assert not any("Glyph" in str(w.message) for w in recwarn)


def test_disconnect_resume_and_unexpected_reboot_preserve_failed_boot(environment):
    root,bundle,board=environment
    board.disconnect=True
    with pytest.raises(TransportError): run_current(root,board)
    assert validate_state(root)["boot_index"]==0
    launched=len(board.launches)
    state=run_current(root,board)
    assert state["boot_index"]==1
    assert len(board.launches)>launched
    old=board.boot
    board.boot="00000000-0000-0000-0000-000000000002"
    board.disconnect=True
    with pytest.raises(TransportError): run_current(root,board)
    board.boot="00000000-0000-0000-0000-000000000003"
    run_current(root,board)
    assert (root/"accepted"/old).is_dir()
    assert len(validate_state(root)["completed_boots"])==2


def test_execution_mode_is_native_argument_and_part_of_identity(environment):
    root,bundle,board=environment
    s=[Stage("cpu",0,3,1,(0,)),Stage("vta",3,18,1,(0,1,2,3)),Stage("cpu",18,21,1,(0,))]
    docs=[task("confirmation",s,ENVIRONMENTS[0],execution_mode=m) for m in ("off","resource_group")]
    assert docs[0]["task_id"]!=docs[1]["task_id"]
    for doc in docs:
        args,inputs=native_command(doc,bundle,"/bundle","/job")
        assert args[args.index("--cpu-stage-gate")+1]==doc["execution_mode"]
    with pytest.raises(ValueError,match="mode"):
        native_command({**docs[0],"execution_mode":"exclusive"},bundle,"/bundle","/job")


def test_wrong_mode_overlapping_gate_intervals_and_corrupted_receipts(environment):
    root,bundle,board=environment
    from cpu_vta_partitioning.campaign import measured
    doc=task("confirmation",[Stage("cpu",0,3,1,(0,))],ENVIRONMENTS[0])
    value=measured(doc,bundle,"/bundle",root,board.boot,board)
    evidence=Evidence(root,"test-bundle")
    evidence.summary(value,doc,board.boot)
    with pytest.raises(ValueError,match="expected boot"):
        evidence.summary(value,doc,"00000000-0000-0000-0000-000000000099")
    receipt=read(root/value["_receipt"]["path"])
    result=next(root/n for n in receipt["files"] if n.endswith("result.jsonl"))
    result.write_text(result.read_text()+"\n")
    with pytest.raises(ValueError,match="missing or changed"):
        Evidence(root,"test-bundle").summary(value,doc,board.boot)
    with pytest.raises(ValueError,match="mode"):
        check_gate([{"cpu_stage_gate_mode":"resource_group"}],doc)


def test_duplicate_frames_rejected(environment):
    root,bundle,board=environment
    from cpu_vta_partitioning.measurement import frames
    path=root/"duplicate.jsonl";path.write_text('\n'.join(json.dumps({"frame_id":0,"completion_index":i}) for i in range(2)))
    with pytest.raises(ValueError,match="duplicate"):
        frames(path,2)


def test_preflight_boot_is_excluded_and_frozen_artifacts_are_checked(environment):
    root,bundle,board=environment
    state=read(root/"state.json");state["excluded_preflight_boot"]=board.boot
    write(root/"state.json",state,replace=True)
    assert run_current(root,board)["boot_index"]==0
    assert not board.launches
    from multithreading.campaign import pin
    artifact=root/"frozen-test.json";write(artifact,{"value":1})
    pin(root,state,[artifact])
    artifact.write_text('{}')
    with pytest.raises(ValueError,match="missing or changed"):
        validate_state(root)


def test_complete_rounds_from_abandoned_boot_do_not_block_replacement(environment):
    root,bundle,board=environment
    calls=[]
    def integrity(*args):
        calls.append(1)
        if len(calls)==2: raise TransportError("post-run integrity failed")
    board.verify_payload=integrity
    with pytest.raises(TransportError): run_current(root,board)
    old=board.boot
    assert read(root/"state.json")["boot_index"]==0
    board.boot="00000000-0000-0000-0000-000000000002"
    state=run_current(root,board)
    assert state["completed_boots"]==[board.boot]
    assert (root/"shared4/calibration"/("boot0-abandoned-"+old)/"round0.json").is_file()


def test_remote_completed_job_is_collected_without_relaunch(environment,monkeypatch):
    from multithreading.transport import SSH
    from cpu_vta_partitioning.common import digest
    root,bundle,board=environment
    document=identified(task("confirmation",[Stage("cpu",0,3,1,(0,))],ENVIRONMENTS[0]),"test-bundle")
    collected=board.measure(document,bundle,"/bundle",root,board.boot)
    ssh=SSH({"host":"test","user":"root","remote_base":"/tmp/test"},execute_board=True)
    remote=ssh.job_path(document,root,board.boot)
    args,inputs=native_command(document,bundle,"/bundle",remote)
    (collected/"boot_id").write_text(board.boot)
    (collected/"task_hash").write_text(digest(document))
    (collected/"inputs.txt").write_text(inputs)
    monkeypatch.setattr(ssh,"boot_id",lambda:board.boot)
    monkeypatch.setattr(ssh,"collect",lambda *args:collected)
    def command(script,**kwargs):
        if "/task_hash" in script: return digest(document)
        if "/exit_code" in script: return "0"
        return ""
    monkeypatch.setattr(ssh,"command",command)
    monkeypatch.setattr(ssh,"upload",lambda *args:pytest.fail("a completed job must not be relaunched"))
    assert ssh.measure(document,bundle,"/bundle",root,board.boot)==collected


def test_process_exit_after_round_checkpoint_resumes_without_recounting(environment,monkeypatch):
    import multiprocessing
    import os
    import multithreading.campaign as campaign
    root,bundle,board=environment
    original=campaign.write
    def child():
        def crash(path,*args,**kwargs):
            original(path,*args,**kwargs)
            if Path(path).name=="round0.json": os._exit(23)
        campaign.write=crash
        run_current(root,board)
    process=multiprocessing.get_context("fork").Process(target=child)
    process.start();process.join(45)
    if process.is_alive(): process.kill();process.join();pytest.fail("checkpoint process hung")
    assert process.exitcode==23
    checkpoint=read(root/"shared4/calibration/boot0/round0.json")
    accepted_tasks={v["_receipt"]["path"].split("/")[-1][:-5] for v in checkpoint["measurements"].values()}
    assert run_current(root,board)["boot_index"]==1
    assert not {task_id for boot,task_id in board.launches}&accepted_tasks
