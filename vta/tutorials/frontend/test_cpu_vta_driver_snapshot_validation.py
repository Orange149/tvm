import copy
import pytest
from test_cpu_vta_all_stage_gate_validation import rows as gate_rows
from analyze_cpu_vta_driver_snapshot_validation import analyze
from run_cpu_vta_driver_snapshot_validation import derive_runner, SOURCE, adapters
import run_cpu_vta_residual_validation as native


def fixture():
    rows=gate_rows()
    for r in rows:
        for i in range(3):r[f'stage{i}_set_ms']=r[f'stage{i}_get_ms']=1
        r['benchmark_monotonic_origin_ms']=10000.
        r['stage0_driver_snapshot']=r['stage2_driver_snapshot']=None
        r['stage1_driver_snapshot']={'tid':100,'bracket_start_monotonic_ms':10000+r['stage1_run_start_ms']-.1,
            'bracket_end_monotonic_ms':10000+r['stage1_run_end_ms']+.1,'probe_before_ms':.08,'probe_after_ms':.1,
            'thread_cpu_ms':.4,'sched_runqueue_ms':.7,'sched_dispatches':100,
            'driver_run_calls':15,'driver_run_insns':3397,'driver_timeout_calls':0,'driver_poll_iters':100,
            'driver_total_ms':1.5,'driver_mmio_ms':.03,'driver_post_start_sleep_ms':.12,'driver_poll_wait_ms':1.33}
    return rows


def test_counter_partitions_and_clock_window():
    m=analyze(fixture(),True,'all')
    assert m['stable_frame_ids']==[7,8,9] and m['sched_counter_observable']
    assert m['means']['driver_other_ms']==pytest.approx(.02)
    assert m['means']['host_outside_driver_ms']==pytest.approx(.5)
    assert m['means']['thread_non_cpu_not_runqueue_ms']==pytest.approx(1.1)


@pytest.mark.parametrize('fault',['frame','origin','tid','cpu','missing','calls','insns','timeout','nan','order','run','nested','outside','budget','process','service'])
def test_bad_snapshot_rejected(fault):
    rows=fixture();r=rows[0];x=r['stage1_driver_snapshot']
    if fault=='frame':r['frame_id']=9
    elif fault=='origin':r['benchmark_monotonic_origin_ms']=9999.
    elif fault=='tid':x['tid']=101
    elif fault=='cpu':r['stage0_driver_snapshot']=copy.deepcopy(x)
    elif fault=='missing':del r['stage1_driver_snapshot']
    elif fault=='calls':x['driver_run_calls']=14
    elif fault=='insns':x['driver_run_insns']=123
    elif fault=='timeout':x['driver_timeout_calls']=1
    elif fault=='nan':x['thread_cpu_ms']=float('nan')
    elif fault=='order':x['bracket_start_monotonic_ms']+=1
    elif fault=='run':r['stage1_run_ms']=1
    elif fault=='nested':x['driver_poll_wait_ms']=9
    elif fault=='outside':x['sched_runqueue_ms']=9
    elif fault=='budget':x['probe_after_ms']=9
    elif fault=='process':r['stage1_process_cpu_ms']=1
    else:r['stage1_set_ms']=0
    with pytest.raises(ValueError):analyze(rows,True,'all')


def test_default_schema_and_unobservable_schedstat():
    assert not analyze(gate_rows(),False,'all')['enabled']
    with pytest.raises(ValueError):analyze(fixture(),False,'all')
    rows=fixture()
    for r in rows:r['stage1_driver_snapshot']['sched_runqueue_ms']=0
    assert not analyze(rows,True,'all')['sched_counter_observable']


def test_patch_runner_and_restore_adapters():
    source=(SOURCE/'raw/runner_sources/vta_stage_pipeline_runner.cc').read_text()
    derived=derive_runner(source)
    assert 'bool driver_snapshots = false' in derived
    assert derived.index('timing->driver_snapshot.Begin();')<derived.index('  stage->run();')
    assert 'timing->driver_snapshot.End(timing->get_start_ms)' in derived
    assert 'timing->driver_snapshot.Begin();\n    timing->set_end_ms = NowMillis();' in derived
    with pytest.raises(ValueError):derive_runner(source.replace('#include "vta_cpu_stage_gate.h"',''))
    build,argv=native.build_runner,native.runner_argv
    with pytest.raises(RuntimeError):
        with adapters():raise RuntimeError('injected')
    assert native.build_runner is build and native.runner_argv is argv
