from pathlib import Path
from fractions import Fraction
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT.parents[1]/'src')]
from static_revision.trace_replay import TraceReplay
from static_revision.workflow import intrinsic_summary
from static_revision.common import freeze,protocol
from thread_history_audit.simulator import HistoryReplay
from model_semantics_audit.replay import validate_isolated


def cpu():
    return dict(device='cpu',mask=[0,1],worker_core_map=[0,1],set_cpu_ns=1,get_cpu_ns=1,
        profile=dict(serial_cpu_segments=[1,1],kernels=[dict(tasks=2,barrier_generations=0,
            launch_prefix_cpu=1,workers=[dict(compute_segments=[3]),dict(compute_segments=[7])])]))


@pytest.mark.parametrize('placement',('instant','retain','wake_boundary'))
def test_restoring_task_trace_keeps_all_frozen_simulation_outputs(placement):
    kwargs=dict(placement=placement,frames=8,warmup=2,scored=4,gate='off')
    observed=TraceReplay([cpu()],**kwargs).run();old=HistoryReplay([cpu()],**kwargs).run()
    assert observed['replay']==old['replay'] and observed['actor_totals']==old['actor_totals']
    tasks=[t for t in observed['tasks'] if t['operation']=='worker_compute']
    assert len(tasks)==8 and all(t['first_service_ns']==t['ready_ns'] for t in tasks)
    assert all(k['last_slots']==[1] for k in observed['kernel_joins'])


def test_intrinsic_summary_keeps_per_launch_completion_information():
    a=cpu();second=dict(a['profile']['kernels'][0]);second['workers']=[dict(compute_segments=[7]),dict(compute_segments=[3])]
    a['profile']=dict(serial_cpu_segments=[1,1,1],kernels=[a['profile']['kernels'][0],second])
    x=intrinsic_summary('test',dict(stages=[a]))[0]
    assert x['sum_launch_max_worker_ms']==14/1e6
    assert x['max_thread_summed_CPU_ms']==10/1e6
    assert x['dependency_vs_total_difference_ms']==4/1e6


def test_projection_rejects_pipeline_cpu_or_fps():
    for key in ('pipeline_fps','cpu_delay_total','observed_service_ms'):
        case=dict(configuration={},stages=[{**cpu(),key:3}],assumptions=[],expected_baseline={})
        with pytest.raises(ValueError):validate_isolated(dict(schema='isolated-semantics-input-v1',origin='isolated_projection',contains_pipeline_data=False,cases={'x':case}))


def test_frozen_protocol_never_silently_changes(tmp_path):
    p=tmp_path/'plan.json';freeze(p,protocol());freeze(p,protocol())
    with pytest.raises(ValueError):freeze(p,{**protocol(),'new_native_sessions':1})
