"""Physical-core accounting, joins, bounded flow and device exclusivity."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT), str(ROOT.parents[1]/'src')]
from phase_model.simulator import Replay


def cpu(core, work=10):
    return dict(device='cpu',mask=[core],worker_core_map=[core],set_cpu_ns=0,get_cpu_ns=0,
                profile=dict(kernels=[],serial_cpu_segments=[work]))


def fork(mapping, work):
    return dict(device='cpu',mask=sorted(mapping),worker_core_map=mapping,set_cpu_ns=0,get_cpu_ns=0,
                profile=dict(serial_cpu_segments=[0,0],kernels=[dict(barrier_generations=0,
                    launch_prefix_cpu=0,workers=[dict(compute_segments=[n]) for n in work])]))


def run(stages, **kw):
    return Replay(stages,frames=28,warmup=8,scored=16,**kw).run()


def test_single_stage_exact_work_and_frame_window():
    x=run([cpu(0)])
    assert x['interval_ns']==10 and x['useful_CPU_ns']==280
    assert x['processed_frames']==28 and len(x['timestamps_ns'])==16
    assert x['synthetic_wait_CPU_ns']==0


def test_same_core_accumulates_and_disjoint_pipeline_overlaps():
    shared=run([cpu(0),cpu(0)])
    split=run([cpu(0),cpu(1)])
    assert shared['interval_ns']==20 and split['interval_ns']==10
    assert shared['useful_CPU_ns']==split['useful_CPU_ns']==560


def test_one_physical_device_is_exclusive_across_islands():
    vta=dict(device='vta',mask=[0,1,2,3],set_cpu_ns=0,run_cpu_ns=0,get_cpu_ns=0,
             run_nonrunning_ns=10)
    x=run([vta,cpu(0,1),vta])
    assert x['interval_ns']>=20 and x['useful_CPU_ns']==28


def test_join_prevents_next_launch_before_slowest_worker():
    x=run([fork([0,1],[1,10])])
    assert x['interval_ns']==10 and x['stage_service_ns']==[10]
    assert x['synthetic_wait_CPU_ns']==9*28


def test_waiting_changes_shared_availability_but_never_intrinsic_work():
    stages=[fork([0,1],[1,10]),cpu(0,10)]
    a=run(stages,runnable_completion_wait=True)
    b=run(stages,runnable_completion_wait=False)
    assert a['useful_CPU_ns']==b['useful_CPU_ns']==21*28
    assert a['synthetic_wait_CPU_ns']>0 and b['synthetic_wait_CPU_ns']==0
    assert a['interval_ns']>=b['interval_ns']


def test_queue_depth_does_not_duplicate_or_drop_frames():
    for depth in (1,2,3):
        x=run([cpu(0,20),cpu(1,1),cpu(0,5)],queue_depth=depth)
        assert x['processed_frames']==28 and x['useful_CPU_ns']==26*28
