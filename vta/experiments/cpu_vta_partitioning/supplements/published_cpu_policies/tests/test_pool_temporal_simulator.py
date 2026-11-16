"""Real group mutex semantics, finite queues and all-participant barriers."""
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT.parents[1]/'src')]
from pool_temporal.simulator import PoolReplay


def cpu(mask,work):
    return dict(device='cpu',mask=mask,worker_core_map=mask,set_cpu_ns=0,get_cpu_ns=0,
        profile=dict(serial_cpu_segments=[work],kernels=[]))


def parallel(mask,work,barriers=0):
    return dict(device='cpu',mask=mask,worker_core_map=mask,set_cpu_ns=0,get_cpu_ns=0,
        profile=dict(serial_cpu_segments=[0,0],kernels=[dict(tasks=len(work),barrier_generations=barriers,
            launch_prefix_cpu=0,workers=[dict(compute_segments=w) for w in work])]))


def run(stages,**kwargs):return PoolReplay(stages,**kwargs).run()


def test_same_group_serial_and_different_groups_parallel():
    a=run([cpu([0],10),cpu([0],10)],gate='resource_group')
    b=run([cpu([0],10),cpu([1],10)],gate='resource_group')
    assert a['interval_ns']==20 and b['interval_ns']==10
    assert a['group_acquisitions']==b['group_acquisitions']==56


def test_nonadjacent_reuse_releases_before_output_backpressure():
    for depth in (1,2,3):
        result=run([cpu([0],10),cpu([1],100),cpu([0],10)],gate='resource_group',queue_depth=depth)
        assert result['processed_frames']==28 and result['useful_CPU_ns']==120*28
        assert result['group_acquisitions']==84


def test_partially_overlapping_groups_are_allowed_only_in_open_mode():
    stages=[cpu([0,1],10),cpu([1,2],10)]
    assert run(stages)['processed_frames']==28
    with pytest.raises(ValueError,match='partially overlapping'):run(stages,gate='resource_group')


def test_worker_work_may_be_a_subset_of_configured_slots():
    result=run([parallel([0,1,2,3],[[7]])])
    assert result['interval_ns']==7 and result['useful_CPU_ns']==7*28
    assert result['synthetic_wait_CPU_ns']==0


def test_internal_barrier_preserves_completion_dependency_without_double_CPU():
    stage=parallel([0,1],[[1,10],[10,1]],barriers=1)
    a=run([stage],runnable_completion_wait=True)
    b=run([stage],runnable_completion_wait=False)
    assert a['interval_ns']==b['interval_ns']==20
    assert a['useful_CPU_ns']==b['useful_CPU_ns']==22*28
    assert a['synthetic_wait_CPU_ns']>=9*28


def test_zero_and_duplicate_boundary_events_drain_every_frame():
    stage=parallel([0,1,2],[[0,2],[1,0]],barriers=1)
    result=run([stage,cpu([3],0)],gate='resource_group')
    assert result['processed_frames']==28 and result['useful_CPU_ns']==3*28
