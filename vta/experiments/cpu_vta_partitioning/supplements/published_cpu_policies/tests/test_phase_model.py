"""Isolation CPU accounting and actual completion/barrier boundaries."""
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT.parents[1]/'src')]
from phase_model.extract import project_call,average_calls,worker_segments


def sample():
    phase=dict(tid=1,wall_start_ns=0,wall_end_ns=200,cpu_start_ns=10,cpu_end_ns=100)
    launch=dict(caller=1,launch=5,wall_start_ns=30,wall_end_ns=170,cpu_start_ns=20,cpu_end_ns=80,
                tasks=2,module='programs/cpu/module.so',function_offset=12)
    workers=[dict(task=0,tid=1,wall_start_ns=40,wall_end_ns=90,cpu_start_ns=30,cpu_end_ns=60),
             dict(task=1,tid=2,wall_start_ns=50,wall_end_ns=160,cpu_start_ns=100,cpu_end_ns=125)]
    return phase,launch,workers


def test_caller_is_not_counted_twice_with_other_workers():
    p,l,w=sample();value=project_call(p,[l],{5:w},{})
    assert value['known_CPU_ns']==115==value['caller_CPU_ns']+value['noncaller_callback_CPU_ns']
    assert value['serial_cpu_segments_ns']==[10,20]
    assert value['kernels'][0]['launch_prefix_cpu_ns']==10
    assert value['kernels'][0]['caller_completion_wait_and_return_cpu_ns']==20
    assert average_calls([value,value])['known_CPU']['numerator']==115


def test_no_parallel_launch_is_explicit_serial_work():
    p,_,_=sample();v=project_call(p,[],{},{});
    assert v['kernels']==[] and v['serial_cpu_segments_ns']==[90]


def test_barrier_CPU_does_not_become_compute_demand():
    cb=dict(cpu_start_ns=10,cpu_end_ns=110,wall_start_ns=0,wall_end_ns=200)
    barrier=dict(cpu_start_ns=40,cpu_end_ns=50,wall_start_ns=60,wall_end_ns=150)
    assert worker_segments(cb,[barrier])==[30,60]
    with pytest.raises(ValueError):worker_segments(cb,[{**barrier,'wall_end_ns':201}])


def test_caller_and_worker_identity_or_missing_callback_rejected():
    p,l,w=sample()
    with pytest.raises(ValueError):project_call(p,[l],{5:w[:1]},{})
    with pytest.raises(ValueError):project_call(p,[l],{5:[{**w[0],'tid':8},w[1]]},{})


def test_kernel_pattern_changes_are_not_averaged_together():
    p,l,w=sample();v=project_call(p,[l],{5:w},{})
    u=project_call(p,[{**l,'function_offset':99}],{5:w},{})
    with pytest.raises(ValueError,match='sequence'):average_calls([v,u])
