#!/usr/bin/env python3
"""Independent local-only mock review; never execute SSH, SCP, RPC or ARM code."""
import argparse
from collections import Counter
from contextlib import contextmanager
import ctypes
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
from types import MethodType, SimpleNamespace

import numpy as np
import c3_valid_window_board as board


def h(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    with Path(path).open("x") as stream:
        json.dump(value,stream,indent=2,sort_keys=True)
        stream.write("\n")


def run(template,output):
    output.mkdir(parents=True,exist_ok=False)
    source_sha = h(board.__file__)
    results,unresolved = {},[]
    base = board._base()
    prepared = board.read(template/"board_prepared.json")
    protocol = board.read(template/"board_protocol.json")
    protocol["source_sha256"] = board.execution_sources()
    protocol["helpers"] = base.helper_certificate()
    fake_root = Path(tempfile.mkdtemp(prefix="c3_valid_window_mock_",dir="/tmp"))
    write(fake_root/"board_protocol.json",protocol)
    prepared["protocol_sha256"] = h(fake_root/"board_protocol.json")
    write(fake_root/"board_prepared.json",prepared)
    args = SimpleNamespace(execute_board=True,phase="qualify",output_dir=fake_root,
        port=9090,host="MOCK_NO_NETWORK",timeout=900,known_hosts="MOCK_NOT_READ")
    events=[]
    class FakeBoard:
        instances=[]
        deploy_fail=False
        def __init__(self,args,runtime,port):
            self.runtime,self.port,self.pid=runtime,port,None
            self.commands,self.stops=[],0
            self.instances.append(self)
        def state(self):
            return {"boot_id":protocol["required_boot_id"],"fpga_state":"operating",
                "udmabuf_size":201326592,"bitstream_sha256":base.reuse.BITSTREAM,
                "raw":"MOCK ONLY\ntmpfs on /var/volatile type tmpfs\n"}
        def ssh(self,command,timeout=25):
            self.commands.append(command)
            events.append({"backend":"MOCK_STRING_RETURN_ONLY","command":command})
            if "stat -f -c %T /tmp" in command:return "tmpfs\n0x0000000068500000\n"
            if "printf 'BOOT '" in command:return "BOOT "+protocol["required_boot_id"]+"\nOWNED 111\nLISTEN 111 4242\n"
            if "/proc/$c3_pid/maps" in command:return self.runtime+"/libvta.so\n"+self.runtime+"/libtvm_runtime.so\n"
            return "MOCK no process or filesystem was touched\n"
        def deploy(self,paths):
            if self.deploy_fail:raise ValueError("MOCK_PREEXISTING_UNOWNED_DEPLOY_REJECTED")
            return dict(protocol["runtime_sha256"])
        def reload(self):return {"sha256":base.reuse.BITSTREAM,"state":"operating","mock":True}
        def start(self):self.pid=111;return self.pid
        def stop(self):self.stops+=1
        def environment(self,require_affinity=False):
            return {"boot_id":protocol["required_boot_id"],"masks":["3"],"governor":"MOCK_RECORD_ONLY","raw":"MOCK"}
    class FakeRemote:
        def __init__(self):self.count=0
        def get_function(self,name):
            if name.endswith("reset_counts"):return lambda:setattr(self,"count",0)
            if name.endswith("counts"):return lambda:self.count
            if name.endswith("fused_timed_once"):
                def timer(*args):self.count+=1;return .125
                return timer
            if name.endswith("fused_diagnostic_once"):return lambda *args:"MOCK"
            raise ValueError("unexpected mock helper "+name)
    fake_functions={key:(lambda:"{}") for key in ("vta.runtime.profiler_clear","vta.runtime.profiler_status",
        "vta.runtime.queue_capacity_status","vta.runtime.replay_status")}
    inventory={"queue_capacity_initial":{"insn_capacity_bytes":33554432,"uop_capacity_bytes":33554432,
        "insn_submit_threshold_bytes":33554432,"insn_submit_threshold_configured":True,
        "command_manifest_id":"","replay_policy":"disabled","submissions":0},
        "replay_initial":{"mode":"disabled","template_count":0},"profiler_initial":{"driver_run_calls":0}}
    base.ExperimentBoard=FakeBoard
    base.runtime_inventory=lambda remote:(inventory,fake_functions)
    base.load_ephemeral=lambda remote,binary:SimpleNamespace(mock=True,path=str(binary))
    board._connect_once=lambda host,port,timeout:FakeRemote()
    runner=board.Runner(args)
    results["Runner_init"]={"passed":True,"real_reference_math_rechecked":True,"four_buffer_ABIs_checked":True,
        "narrow_clone_compiled":True,"fake_preflight_objects":len(FakeBoard.instances),"board_contacted":False}
    # Actual cloned session, with every command intercepted as a local string.
    runner.start_phase("mock_session")
    with runner.session("Q1",runner.libraries["Q1"]) as values:
        assert len(values)==5
    good=runner.sessions[-1]
    assert good["status"]=="PASS" and good["rpc_readiness"]["connect_attempts_so_far"]==0
    results["cloned_session_positive"]={"passed":True,"cleanup_status":good["cleanup_status"],
        "readiness":good["rpc_readiness"],"connect_calls":1,"target_calls":0}
    FakeBoard.deploy_fail=True
    try:
        with runner.session("Q1",runner.libraries["Q1"]):
            raise AssertionError("failed deployment unexpectedly entered body")
    except ValueError as error:
        assert "MOCK_PREEXISTING" in str(error)
    failed_board=FakeBoard.instances[-1]
    assert failed_board.stops==0 and not any("rm -r" in command or "rpc.log" in command for command in failed_board.commands)
    assert runner.sessions[-1]["cleanup_status"].startswith("SKIPPED_UNPROVEN")
    FakeBoard.deploy_fail=False
    results["unowned_deployment_cleanup"]={"passed":True,"stop_calls":failed_board.stops,
        "rm_calls":0,"log_read_calls":0,"target_calls":0}
    # Current original timing boundary is one packed call and no extra sync.
    timer_source=(Path(board.HERE)/"c3_node34_fused_timed_once.cc").read_text()
    assert timer_source.count("function(args[2], args[3], args[4], args[5]);")==1
    assert "VTASynchronize" not in timer_source
    results["timer_boundary"]={"passed":True,"packed_calls_in_source":1,"adds_synchronize":False,
        "source_sha256":h(Path(board.HERE)/"c3_node34_fused_timed_once.cc")}
    # Real deferred wrapper: late native rejection must commit FAIL once.
    import c3_valid_window_uop_values as uops
    original_cert=uops.certify_uop_values
    runner.start_phase("mock_native_reject")
    (runner.out/"fake_dump.txt").write_text("MOCK")
    original_raw=runner._raw_call
    def rejecting_raw(*args,**kwargs):
        return runner.record({"program":"Q1","seed":0,"call_kind":"warmed_diagnostic",
            "session_id":"MOCK_NATIVE_REJECT","correct":True,"arguments_unchanged":True,
            "dump_file":"fake_dump.txt","dump_sha256":h(runner.out/"fake_dump.txt"),"call_position":4,
            "mock":True,"board_contacted":False})
    runner._raw_call=rejecting_raw
    def reject(*args,**kwargs):raise ValueError("MOCK_LATE_UOP_CERTIFICATE_REJECTION")
    uops.certify_uop_values=reject
    before=len(runner.rows)
    try:runner.call()
    except ValueError as error:assert "MOCK_LATE_UOP" in str(error)
    else:raise AssertionError("native rejection did not stop")
    assert len(runner.rows)==before+1 and runner.rows[-1]["status"]=="FAIL_NATIVE_OR_EXECUTION"
    assert not runner._defer_record and runner._pending_record is None
    results["late_native_rejection"]={"passed":True,"committed_status":runner.rows[-1]["status"],
        "rows_added":1,"exception_propagates":True,"pending_cleared":True}
    runner._raw_call=original_raw
    uops.certify_uop_values=original_cert
    # Verify startup policy failures before any target call.
    invalid=dict(inventory,queue_capacity_initial=dict(inventory["queue_capacity_initial"],replay_policy="enabled"))
    try:board._verify_inventory(invalid,fake_functions)
    except ValueError:pass
    else:raise AssertionError("wrong replay startup policy accepted")
    try:board._verify_maps("/different/libvta.so\n/different/libtvm_runtime.so", "/var/volatile/owned")
    except ValueError:pass
    else:raise AssertionError("different actual runtime maps accepted")
    results["startup_negative_guards"]={"passed":True,"wrong_replay_rejected":True,"wrong_runtime_maps_rejected":True}
    # Exercise the exact qualify/diagnose/formal loop methods against a mock
    # call backend. This validates labels, seeds, ordering and counter closures;
    # it deliberately makes no claim about FPGA numeric correctness.
    flow_rows={}
    def reset_phase(phase):
        runner.rows=[];runner.sessions=[];runner.out=None
        args.phase=phase
    @contextmanager
    def fake_session(self,name,binary):
        remote=FakeRemote();sid="MOCK_FLOW_%s_%02d" % (self.out.name,len(self.sessions))
        yield remote,None,{},[],sid
        self.sessions.append({"session_id":sid,"program":name,"mock":True})
    runner.session=MethodType(fake_session,runner)
    runner.buffers=MethodType(lambda self,remote,seed:tuple(SimpleNamespace(handle=ctypes.c_void_p(i+1)) for i in range(4)),runner)
    def fake_call(self,lib,funcs,name,buffers,seed,kind,sid,**kwargs):
        timer=kwargs.get("timer")
        latency=timer(lib,base.SYMBOL,*buffers) if timer else None
        signature=dict(self.prepared["programs"][name]["local_native_signature"])
        return self.record({"program":name,"seed":seed,"call_kind":kind,"session_id":sid,
            "correct":True,"arguments_unchanged":True,"latency_ms":latency,
            "runtime_profile":{"push_gemm_op_us":1,"push_alu_op_us":1},"command_signature":signature,
            "mock":True,"board_contacted":False,**{k:v for k,v in kwargs.items() if k not in ("timer","diagnostic","weight","expected")}})
    runner.call=MethodType(fake_call,runner)
    for phase,method in (("qualify","qualify"),("diagnose","diagnose"),("formal","formal")):
        reset_phase(phase)
        getattr(runner,method)()
        flow_rows[phase]=list(runner.rows)
    assert [len(flow_rows[p]) for p in ("qualify","diagnose","formal")]==[27,90,180]
    formal=flow_rows["formal"]
    timing=[r for r in formal if r["call_kind"]=="timing"]
    warms=[r for r in formal if r["call_kind"]=="warmup"]
    expected=[(r+1,name,seed,pos+4) for r,order in enumerate(board.ORDERS) for name in order
        for pos,seed in enumerate(board.SEED_ORDERS[r])]
    assert [(r["round"],r["program"],r["seed"],r["call_position"]) for r in timing]==expected
    assert len(timing)==len(warms)==90
    assert Counter(r["call_kind"] for r in flow_rows["qualify"])=={"correctness":15,"weight_ABA":12}
    assert Counter(r["call_kind"] for r in flow_rows["diagnose"])=={"diagnostic_prefix":45,"warmed_diagnostic":45}
    results["exact_flow_ledger"]={"passed":True,"qualification":27,"diagnostic":90,"formal":180,
        "formal_warmups":90,"formal_timings":90,"total_mock_target_calls":297,
        "formal_order_seed_and_position_exact":True,"scope":"mock loop/control/counter test; not FPGA output or latency evidence"}
    # The invalid marker prevents a later phase even if all row outputs match.
    write(fake_root/"invalid_attempt.json",{"mock":True,"status":"MOCK_ONLY"})
    try:runner.ensure_identity()
    except ValueError as error:assert "no automatic retry" in str(error)
    else:raise AssertionError("retained invalid marker did not stop later work")
    results["invalid_attempt_global_stop"]={"passed":True,"marker_inside_tmp_mock_root_only":True}
    assert h(board.__file__)==source_sha,"reviewed board source changed during mock review"
    write(output/"review.json",{"status":"LOCAL_MOCK_BOARD_REVIEW_PASSED_NO_BOARD_ACCESS",
        "board_source_sha256":source_sha,"review_script_sha256":h(__file__),"results":results,
        "unresolved":unresolved,"fake_root":str(fake_root),"board_contacted":False,
        "SSH_SCP_RPC_executed":False,"ARM_code_executed":False,"numeric_and_latency_claims":False})
    write(output/"fake_command_strings.json",events)
    print(json.dumps({"status":"LOCAL_MOCK_BOARD_REVIEW_PASSED_NO_BOARD_ACCESS","tests":len(results),
        "unresolved":unresolved,"fake_target_calls":297,"board_source_sha256":source_sha}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    run(args.template.resolve(),args.output_dir.resolve())
