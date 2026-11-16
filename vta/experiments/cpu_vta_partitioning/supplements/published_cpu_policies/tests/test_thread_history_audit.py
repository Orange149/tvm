from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import sys,pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT.parents[1]/'src')]
from thread_history_audit.simulator import HistoryReplay
from phase_model.simulator import rational
from model_semantics_audit.simulator import AuditReplay
from model_semantics_audit.replay import validate_isolated
from model_semantics_audit.common import offline_guard,replay_input_guard


def serial_cpu(work=1,core=0):
    return dict(device='cpu',mask=[core],worker_core_map=[core],set_cpu_ns=0,set_nonrunning_ns=0,
        get_cpu_ns=0,get_nonrunning_ns=0,profile=dict(kernels=[],serial_cpu_segments=[work]))


def parallel_cpu(work):
    n=len(work)
    return dict(device='cpu',mask=list(range(n)),worker_core_map=list(range(n)),set_cpu_ns=0,set_nonrunning_ns=0,
        get_cpu_ns=0,get_nonrunning_ns=0,profile=dict(serial_cpu_segments=[0,0],kernels=[dict(tasks=n,barrier_generations=0,
            launch_prefix_cpu=0,workers=[dict(compute_segments=[v]) for v in work])]))


def vta(cpu=2,wall=5,mask=None):
    return dict(device='vta',mask=[0,1,2,3] if mask is None else mask,set_cpu_ns=0,set_nonrunning_ns=0,
        get_cpu_ns=0,get_nonrunning_ns=0,device_profile=dict(host_serial_CPU=[1,1],devices=[dict(CPU=cpu,wall=wall)]))


def test_instant_restores_all_frozen_replay_fields_without_changing_accounting():
    stages=[parallel_cpu([3,7]),vta(),serial_cpu(2,2)]
    kw=dict(frames=6,warmup=1,scored=4,controls={'_producer':dict(CPU_ns=1,mask=[3]),'_collector':dict(CPU_ns=2,mask=[3])})
    old=AuditReplay(stages,**kw).run();new=HistoryReplay(stages,placement='instant',**kw).run()
    assert new['replay']==old['replay'] and new['accounting']==old['accounting'] and new['actor_totals']==old['actor_totals']


@pytest.mark.parametrize('policy',['instant','retain','wake_boundary'])
def test_busy_completion_wait_has_CPU_cost_not_a_second_wall_penalty(policy):
    kw=dict(frames=2,warmup=0,scored=2,placement=policy)
    busy=HistoryReplay([parallel_cpu([1,10])],**kw).run()
    blocked=HistoryReplay([parallel_cpu([1,10])],runnable_completion_wait=False,**kw).run()
    assert busy['replay']['timestamps_ns']==blocked['replay']['timestamps_ns']==[10,20]
    assert rational(busy['accounting']['graph_CPU'])==rational(blocked['accounting']['graph_CPU'])==22
    assert rational(busy['accounting']['synthetic_wait_CPU'])==18
    assert rational(blocked['accounting']['synthetic_wait_CPU'])==0
    assert list(map(rational,busy['window_cpu']['core_totals']))==[10,10,0,0]
    assert list(map(rational,blocked['window_cpu']['core_totals']))==[1,10,0,0]


@pytest.mark.parametrize('policy',['retain','wake_boundary'])
def test_same_TID_retains_core_between_CPU_actions(policy):
    sim=HistoryReplay([serial_cpu()],placement=policy,frames=2,warmup=0,scored=2)
    def new(stage,mask):return sim._new_task(stage,10,mask,lambda:None,dict(role='stage',stage=stage,slot=0,operation='test',frame=0))
    pinned=new(0,[0]);host=new(1,[0,1]);assert sim.active[host]['assigned_core']==1
    sim.active.pop(host);sim.active.pop(pinned);new(2,[1])
    host=new(1,[0,1]);assert sim.active[host]['assigned_core']==1
    sim.active.pop(host);sim.potential_wakes.add(('stage',1,0));host=new(1,[0,1])
    assert sim.active[host]['assigned_core']==(0 if policy=='wake_boundary' else 1)


def test_wakeup_equal_load_prefers_previous_core():
    sim=HistoryReplay([serial_cpu()],placement='wake_boundary',frames=2,warmup=0,scored=2)
    sim.actor_core['stage',1,0]=1;sim.potential_wakes.add(('stage',1,0))
    key=sim._new_task(1,5,[0,1],lambda:None,dict(role='stage',stage=1,slot=0,operation='test',frame=0))
    assert sim.active[key]['assigned_core']==1


def test_allowed_masks_and_fixed_core_inputs_cannot_be_overridden():
    for policy in ('retain','wake_boundary'):
        r=HistoryReplay([serial_cpu(2,2),vta(mask=[1,3]),serial_cpu(1,2)],placement=policy,frames=4,warmup=0,scored=4).run()
        hosts=[row for row in r['window_cpu']['rows'] if row['role']=='stage' and row['stage']==1]
        assert hosts and all(rational(row['core_CPU'][0])==rational(row['core_CPU'][2])==0 for row in hosts)
    with pytest.raises(ValueError):HistoryReplay([serial_cpu()],placement='CFS')
    with pytest.raises(ValueError):HistoryReplay([serial_cpu()],fixed_hosts={})


def test_control_CPU_queue_backpressure_and_whole_window_accounting():
    controls={'_producer':dict(CPU_ns=0,mask=[1]),'_collector':dict(CPU_ns=100,mask=[1])}
    for policy in ('instant','retain','wake_boundary'):
        r=HistoryReplay([serial_cpu(10)],placement=policy,controls=controls,frames=8,warmup=0,scored=8).run()
        assert r['replay']['timestamps_ns']==[10,20,30,40,120,220,320,420]
        assert rational(r['accounting']['control_CPU'])==800
        assert max(r['accounting']['queue_max'].values())<=2
        assert rational(r['window_cpu']['duration'])==410
        assert sum(map(rational,r['window_cpu']['core_totals']))<=4*410


def test_completion_dependencies_and_gate_remain_intact():
    stage=parallel_cpu([1,10]);stage['profile']['kernels']+=deepcopy(stage['profile']['kernels'])
    stage['profile']['kernels'][1]['workers']=[dict(compute_segments=[10]),dict(compute_segments=[1])]
    stage['profile']['serial_cpu_segments']=[0,0,0]
    for policy in ('instant','retain','wake_boundary'):
        r=HistoryReplay([stage],placement=policy,frames=2,warmup=0,scored=2).run()
        assert r['replay']['stage_service_ns']==[20]
        assert r['kernel_joins'][1]['ready_ns']>=r['kernel_joins'][0]['join_ns']
    r=HistoryReplay([serial_cpu(3),vta(),serial_cpu(4)],placement='retain',gate='resource_group',frames=3,warmup=0,scored=3).run()
    assert r['replay']['group_acquisitions']==6


def test_offline_and_prediction_guards_reject_board_and_pipeline_inputs(tmp_path):
    import socket,subprocess
    with offline_guard():
        with pytest.raises(RuntimeError):socket.create_connection(('127.0.0.1',22))
        with pytest.raises(RuntimeError):subprocess.run(['ssh','board'])
    p=tmp_path/'RESULTS.json';p.write_text('{}')
    with replay_input_guard(tmp_path):
        with pytest.raises(RuntimeError):p.read_text()
    with pytest.raises(ValueError):validate_isolated(dict(origin='observed_pipeline_control_CPU',diagnostic_only=True))
