from pathlib import Path
import sys
from copy import deepcopy
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT.parents[1]/'src')]
from placement_alignment.analysis import overlap_bounds,align_kernel
from placement_alignment.counter import check,board_self_test


def count(value):return dict(value=value,enabled=10**9+value,running=value)
def readings(total,a,b):return list(map(count,(total,a,b,0,0)))


def native_result():
    return dict(status='MEASURED',cpus=[0,1,2,3],rows=[
        dict(case='busy_first',requested_core=0,CPU_ns=150000000,wall_ns=160000000,read_envelope_ns=30000,
            before=readings(0,0,0),after=readings(150000000,150000000,0)),
        dict(case='busy_migrated',requested_core=1,CPU_ns=150000000,wall_ns=160000000,read_envelope_ns=30000,
            before=readings(150000000,150000000,0),after=readings(300000000,150000000,150000000)),
        dict(case='sleep',requested_core=1,CPU_ns=10000,wall_ns=100000000,read_envelope_ns=30000,
            before=readings(300000000,150000000,150000000),after=readings(300010000,150000000,150010000))])


def test_counting_interface_needs_conservation_and_correct_migration():
    r=check(native_result());assert r['passed']
    assert r['cases'][0]['by_CPU_ns']['0']==150000000
    assert r['cases'][1]['by_CPU_ns']['1']==150000000
    assert all(c['no_enabled_running_rescaling'] for c in r['cases'])


def test_wrong_task_clock_cpu_filter_is_detected_not_scaled():
    d=native_result();d['rows'][0]['after']=list(map(count,[150000000]*5))
    assert not check(d)['passed']


def test_zero_or_unsupported_counters_do_not_mean_no_migration():
    assert not check(dict(status='UNAVAILABLE',errno=1))['passed']
    d=native_result()
    for r in d['rows']:r['after']=deepcopy(r['before'])
    assert not check(d)['passed']


def test_sleep_wall_not_charged_as_cpu():
    d=native_result();d['rows'][2]['after']=readings(400000000,150000000,250000000)
    assert not check(d)['passed']


def test_counter_backward_or_missing_cpu_is_rejected():
    d=native_result();d['rows'][1]['after'][0]['value']=0
    with pytest.raises(ValueError):check(d)


def test_counter_rejects_impossible_timing_or_wrong_requested_core():
    d=native_result();d['rows'][0]['after'][0]['running']=2*10**9
    with pytest.raises(ValueError):check(d)
    d=native_result();d['rows'][0]['requested_core']=1
    with pytest.raises(ValueError):check(d)
    d=native_result();d['rows'][0]['read_envelope_ns']=-1
    with pytest.raises(ValueError):check(d)
    d=native_result();d['rows'][0]['after'].pop()
    with pytest.raises(ValueError):check(d)


def test_no_board_command_without_explicit_gate(tmp_path):
    with pytest.raises(ValueError,match='execute-board'):board_self_test(tmp_path)


def test_overlap_is_temporal_budget_not_percore_or_additive():
    assert overlap_bounds(20,80,0,100,80)==(60,40,60)
    assert overlap_bounds(101,120,0,100,80)==(0,0,0)
    with pytest.raises(ValueError):overlap_bounds(2,1,0,100,50)


def worker(slot,cpu,wall,queue=0):
    return dict(task=slot,cpu_start_ns=0,cpu_end_ns=cpu,barrier_cpu_ns=0,
        wall_start_ns=10,wall_end_ns=10+wall,queue_before_ns=0,queue_after_ns=queue,
        queue_read_start0_ns=0,queue_read_end0_ns=9,queue_read_start1_ns=11+wall,queue_read_end1_ns=20+wall)


def kernel(workers):
    return dict(kernel=0,module='cpu_00_02/module.so',function_offset=100,tasks=2,workers=workers,
                last_slot=max(workers,key=lambda w:w['wall_end_ns'])['task'],launch_wall_ns=max(w['wall_end_ns'] for w in workers),launch_start_skew_ns=0)


def test_kernel_alignment_preserves_logical_slots_when_finisher_changes():
    a=kernel([worker(0,90,100),worker(1,80,90)])
    b=kernel([worker(0,91,110),worker(1,81,170,80)])
    d=align_kernel([a],[b])
    assert d['matched_exact_function_and_logical_slot']
    assert d['isolated_last_counts']=={0:1} and d['pipeline_last_counts']=={1:1}
    assert d['slots'][1]['pipeline_CPU_ms']==81/1e6
    with pytest.raises(ValueError):align_kernel([a],[{**b,'function_offset':200}])
