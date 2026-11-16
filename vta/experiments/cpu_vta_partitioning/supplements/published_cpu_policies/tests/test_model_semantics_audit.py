"""Accounting and real frozen baselines, rather than implementation mirrors."""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import json
import socket
import subprocess
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT.parents[1]/'src')]
from model_semantics_audit.simulator import AuditReplay
from model_semantics_audit.common import metadata_projection,offline_guard,replay_input_guard
from model_semantics_audit.replay import scenario_specs,validate_isolated,run_isolated
from pool_temporal.simulator import PoolReplay


def serial_cpu(amount,core=0):
    return dict(device='cpu',mask=[core],worker_core_map=[core],set_cpu_ns=0,set_nonrunning_ns=0,
                get_cpu_ns=0,get_nonrunning_ns=0,profile=dict(kernels=[],serial_cpu_segments=[amount]))


def parallel_cpu(work):
    n=len(work)
    return dict(device='cpu',mask=list(range(n)),worker_core_map=list(range(n)),set_cpu_ns=0,set_nonrunning_ns=0,
        get_cpu_ns=0,get_nonrunning_ns=0,profile=dict(serial_cpu_segments=[0,0],kernels=[dict(tasks=n,
            barrier_generations=0,launch_prefix_cpu=0,workers=[dict(compute_segments=[v]) for v in work])]))


def vta(cpu=1,wall=4):
    return dict(device='vta',mask=[0,1,2,3],worker_core_map=[],set_cpu_ns=0,set_nonrunning_ns=0,
                get_cpu_ns=0,get_nonrunning_ns=0,device_profile=dict(host_serial_CPU=[0,0],devices=[dict(CPU=cpu,wall=wall)]))


def controls(p=0,c=0,core=1):
    return {r:dict(CPU_ns=w,mask=[core]) for r,w in [('_producer',p),('_collector',c)]}


def test_zero_control_recovers_every_old_replay_field():
    stages=[parallel_cpu([5,9]),vta(),serial_cpu(2,2)]
    kwargs=dict(frames=12,warmup=2,scored=6,queue_depth=2)
    expected=PoolReplay(stages,**kwargs).run()
    result=AuditReplay(stages,controls=controls(),trace=True,**kwargs).run()
    assert result['replay']==expected
    assert result['accounting']['control_CPU']['numerator']==0


def test_completion_precedes_collection_and_output_backpressure_is_real():
    r=AuditReplay([serial_cpu(10)],controls=controls(c=100),trace=True,frames=8,warmup=0,scored=8).run()
    assert r['replay']['timestamps_ns']==[10,20,30,40,120,220,320,420]
    assert r['accounting']['collected_frames']==8
    assert r['accounting']['control_CPU']==dict(numerator=800,denominator=1)
    assert r['accounting']['graph_CPU']==dict(numerator=80,denominator=1)
    assert max(r['accounting']['queue_max'].values())<=2
    assert r['accounting']['output_blocked_wall']['numerator']>0


def test_producer_CPU_is_work_and_queue_blocking_is_not_CPU():
    r=AuditReplay([serial_cpu(100)],controls=controls(p=1),frames=12,warmup=0,scored=12).run()
    assert r['accounting']['control_CPU']==dict(numerator=12,denominator=1)
    assert r['accounting']['producer_blocked_wall']['numerator']>0
    assert r['accounting']['collected_frames']==12


def test_producer_sets_frame_readiness():
    r=AuditReplay([serial_cpu(10)],controls=controls(p=100),frames=4,warmup=0,scored=4).run()
    assert r['replay']['timestamps_ns']==[110,210,310,410]


def test_rational_CPU_conservation_and_no_sum_of_waits_penalty():
    r=AuditReplay([parallel_cpu([5,10])],controls=controls(p=Fraction(1,2),c=Fraction(3,2),core=3),
                  trace=True,frames=4,warmup=0,scored=4).run()
    assert r['accounting']['graph_CPU']==dict(numerator=60,denominator=1)
    assert r['accounting']['control_CPU']==dict(numerator=8,denominator=1)
    assert r['accounting']['synthetic_wait_CPU']['numerator']>0
    for t in r['tasks']:
        cpu=Fraction(**t['CPU']);wait=Fraction(**t['ideal_service_deprivation'])
        assert cpu>=0 and wait>=0


def test_last_finisher_ties_are_kept_not_highest_slot_only():
    r=AuditReplay([parallel_cpu([5,5])],trace=True,frames=4,warmup=0,scored=4).run()
    assert all(k['last_slots']==[0,1] for k in r['kernel_joins'])
    assert all(k['ready_skew_ns']==k['first_service_skew_ns']==0 for k in r['kernel_joins'])


def test_group_gate_releases_before_blocked_output_push():
    stages=[serial_cpu(5),vta(),serial_cpu(6)]
    r=AuditReplay(stages,controls=controls(c=50),gate='resource_group',frames=6,warmup=0,scored=6).run()
    assert r['replay']['group_acquisitions']==12
    assert r['accounting']['collected_frames']==6


def test_fixed_host_trajectories_respect_original_masks():
    stages=[serial_cpu(1),vta(),serial_cpu(1)]
    with pytest.raises(ValueError):AuditReplay(stages,fixed_hosts={1:4})
    with pytest.raises(ValueError):AuditReplay(stages,fixed_hosts={})
    r=AuditReplay(stages,fixed_hosts={1:2},trace=True,frames=3,warmup=0,scored=3).run()
    host=[t for t in r['tasks'] if t['stage']==1]
    assert host and all(t['allowed_mask']==[2] for t in host)


def test_static_input_rejects_diagnostic_control_and_FPS():
    d=dict(schema='isolated-semantics-input-v1',origin='isolated_projection',contains_pipeline_data=False,cases={})
    validate_isolated(d)
    for key,value in [('diagnostic_only',True),('FPS',12.0),('pipeline_service_ms',42)]:
        with pytest.raises(ValueError):validate_isolated({**d,key:value})
    with pytest.raises(ValueError):validate_isolated(dict(origin='observed_pipeline_control_CPU',diagnostic_only=True))


def test_metadata_decoder_skips_performance_subtrees(tmp_path,monkeypatch):
    p=tmp_path/'receipt.json'
    p.write_text('{"task":{"id":1},"summary":{"fps":999,"nested":["}\\\"",{}]},"raw":"ok"}')
    original=json.JSONDecoder.raw_decode
    def guarded(self,text,index=0):
        if text[index:].startswith('{"fps"'):raise AssertionError('performance subtree was decoded')
        return original(self,text,index)
    monkeypatch.setattr(json.JSONDecoder,'raw_decode',guarded)
    assert metadata_projection(p,{'task','raw'})==dict(task={'id':1},raw='ok')


def test_offline_guard_denies_network_and_board_subprocesses():
    with offline_guard() as ledger:
        with pytest.raises(RuntimeError):socket.create_connection(('127.0.0.1',22))
        with pytest.raises(RuntimeError):subprocess.run(['ssh','board','uname'])
    assert ledger['network_attempts']==1 and ledger['forbidden_process_attempts']==1


def test_replay_cannot_read_performance_or_raw_receipts(tmp_path):
    p=tmp_path/'RESULTS.json';p.write_text('{}')
    with replay_input_guard(tmp_path):
        with pytest.raises(RuntimeError):p.read_text()


def test_scenario_enumeration_is_full_deterministic_198_without_FPS():
    cases={}
    for name,count in [('resnet',3),('yolo_overlap',2),('yolo_split',2)]:
        stages=[serial_cpu(1)]
        for _ in range(count):stages += [vta(),serial_cpu(1)]
        cases[name]=dict(configuration={'execution_mode':'off'},stages=stages,assumptions=[],expected_baseline={})
    iso=dict(schema='isolated-semantics-input-v1',origin='isolated_projection',contains_pipeline_data=False,cases=cases)
    diag=dict(origin='observed_pipeline_control_CPU',diagnostic_only=True,forbidden_for_static_prediction=True,cases={n:controls() for n in cases})
    rows=scenario_specs(iso,diag)
    assert len(rows)==len({r['id'] for r in rows})==198 and rows==scenario_specs(iso,diag)
    assert sum(r['case']=='resnet' for r in rows)==130
    assert sum(r['case']=='yolo_overlap' for r in rows)==34


def test_next_parallel_launch_waits_for_all_previous_workers():
    stage=parallel_cpu([1,10]);stage['profile']['kernels'] += deepcopy(stage['profile']['kernels'])
    stage['profile']['kernels'][1]['workers']=[dict(compute_segments=[10]),dict(compute_segments=[1])]
    stage['profile']['serial_cpu_segments']=[0,0,0]
    r=AuditReplay([stage],trace=True,frames=2,warmup=0,scored=2).run()
    assert r['replay']['stage_service_ns']==[20]  # max(sum work)=11 would lose the two joins
    joins=r['kernel_joins'];assert joins[1]['ready_ns']>=joins[0]['join_ns']


def test_same_logical_caller_has_no_concurrent_actions():
    r=AuditReplay([parallel_cpu([1,5])],trace=True,frames=4,warmup=0,scored=4).run()
    caller=sorted([t for t in r['tasks'] if t['role']=='stage' and t['slot']==0],key=lambda r:r['ready_ns'])
    assert all(a['end_ns']<=b['ready_ns'] for a,b in zip(caller,caller[1:]))


def test_no_CPU_work_is_charged_for_empty_consumer_waiting():
    r=AuditReplay([serial_cpu(10)],controls=controls(p=100,c=2,core=2),frames=4,warmup=0,scored=4).run()
    assert r['accounting']['control_CPU']==dict(numerator=408,denominator=1)
    assert r['accounting']['graph_CPU']==dict(numerator=40,denominator=1)


@pytest.mark.parametrize('network,case',[('resnet18','static_first'),('yolov3_tiny416','overlap'),('yolov3_tiny416','split')])
def test_real_104_frame_frozen_baseline_exact(network,case):
    from cpu_vta_partitioning.common import read
    from cpu_policies.open_resources.costs import Costs
    from policy_branch_eval.ranking import inputs
    from pool_temporal.ranking import model_stages
    from model_semantics_audit.common import PARENT,STUDY
    p=read(PARENT/'PREDICTIONS_FREEZE.json')['cases'][network+'/'+case]
    programs,records,ddr,*_=inputs(network)
    stages,_=model_stages(network,p['configuration'],programs,Costs(records,programs=programs,ddr=ddr,distribution='session_weights'),
                          read(STUDY/'pool_temporal/INTRINSICS.json'))
    r=AuditReplay(stages,trace=True,frames=104,warmup=8,scored=64).run()
    assert r['replay']==p['matched_long_replay']
