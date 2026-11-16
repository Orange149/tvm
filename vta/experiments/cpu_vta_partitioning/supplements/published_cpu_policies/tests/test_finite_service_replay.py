from pathlib import Path
import sys,pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT.parents[1]/'src')]
from finite_service_replay import FiniteReplay
from thread_history_audit.simulator import HistoryReplay
from phase_model.simulator import rational

Q={str(n):12 for n in range(2,7)}

def cpu():
    return dict(device='cpu',mask=[0,1],worker_core_map=[0,1],set_cpu_ns=1,get_cpu_ns=1,
        profile=dict(serial_cpu_segments=[1,1],kernels=[dict(tasks=2,barrier_generations=0,
            launch_prefix_cpu=1,workers=[dict(compute_segments=[3]),dict(compute_segments=[7])])]))


def test_finite_DAG_preserves_work_join_and_queue_accounting():
    args=dict(placement='wake_boundary',frames=8,warmup=2,scored=4,gate='off')
    second=cpu();second['worker_core_map']=[1,0]
    r=FiniteReplay([cpu(),second],quanta=Q,**args).run()
    old=HistoryReplay([cpu(),second],**args).run()
    assert r['replay']['useful_CPU_ns']==old['replay']['useful_CPU_ns']==240
    assert r['replay']['processed_frames']==8
    assert all(not k['model_assumes_immediate_positive_fluid_service'] for k in r['kernel_joins'])
    assert any(k['first_service_skew_ns']>0 for k in r['kernel_joins'])
    for t in r['tasks']:
        assert rational(t['CPU'])+rational(t['ready_not_running'])==t['end_ns']-t['ready_ns']


def test_fixed_group_mutex_and_no_observed_controls():
    args=dict(placement='wake_boundary',frames=8,warmup=2,scored=4,gate='resource_group')
    r=FiniteReplay([cpu(),cpu()],quanta=Q,**args).run()
    assert r['replay']['group_acquisitions']==16
    assert r['replay']['useful_CPU_ns']==240
    with pytest.raises(ValueError,match='observed controls'):
        FiniteReplay([cpu()],quanta=Q,controls={'_collector':{'CPU_ns':1}})
