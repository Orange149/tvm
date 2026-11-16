import copy
import pytest
from analyze_cpu_vta_host_priority_validation import verify_priority, summarize
from run_cpu_vta_host_priority_validation import derive_runner, SOURCE, adapters
import run_cpu_vta_residual_validation as native


def identities():
    affinity={'stages':[{'caller_tid':tid,'thread_ids':[tid,tid+10]} for tid in (101,102,103)]}
    init={'vta_tid':102,'requested_nice':-10,'threads':[{'tid':tid,'nice':-10 if tid==102 else 0} for tid in (100,101,102,103,111,112,113)]}
    rows=[{'frame_id':f,**{f'stage{i}_host_priority':{'tid':tid,'nice':-10 if i==1 else 0} for i,tid in enumerate((101,102,103))},'stage1_driver_snapshot':{'tid':102}} for f in range(10)]
    return rows,init,affinity


def test_worker_and_pool_priorities():
    rows,init,affinity=identities()
    assert verify_priority(rows,-10,init,affinity)['frame_checks']==30


@pytest.mark.parametrize('fault',['frame','init_nice','CPU_pool','missing_pool','caller','frame_nice','frame_tid','snapshot_tid','duplicate','missing'])
def test_actual_identity_drift_rejected(fault):
    rows,init,affinity=identities()
    if fault=='frame':rows.pop(0)
    elif fault=='init_nice':init['requested_nice']=0
    elif fault=='CPU_pool':init['threads'][-1]['nice']=-10
    elif fault=='missing_pool':init['threads'].pop()
    elif fault=='caller':affinity['stages'][1]['caller_tid']=999
    elif fault=='frame_nice':rows[3]['stage1_host_priority']['nice']=0
    elif fault=='frame_tid':rows[3]['stage0_host_priority']['tid']=102
    elif fault=='snapshot_tid':rows[3]['stage1_driver_snapshot']['tid']=999
    elif fault=='duplicate':init['threads'].append(copy.deepcopy(init['threads'][0]))
    else:del rows[3]['stage0_host_priority']
    with pytest.raises(ValueError):verify_priority(rows,-10,init,affinity)


def perf_fixture():
    xs=[]
    for boot in (1,2,3):
        for block in range(2):
            for name in ('N0','N1'):
                for on in (False,True):
                    run=80 if name=='N0' else 70+boot
                    xs.append({'mode':'pipeline','boot_index':boot,'block':f'pair{block}_'+('on' if on else 'off'),
                        'condition':name,'driver_snapshots_enabled':on,'metrics':{'ii_ms':105+(0.5 if on else 0)},
                        'gate_metrics':{'stages':[{'run_mean_ms':55},{'run_mean_ms':run+(0.5 if on else 0)},{'run_mean_ms':45}]},
                        'snapshot_metrics':{'means':{'sched_runqueue_ms':16 if name=='N0' else 2,'driver_total_ms':run-7,'thread_cpu_ms':14}}})
    return xs


def test_boot_unit_t_interval_and_qualified_signal():
    r=summarize(perf_fixture())
    assert r['boots']==3 and r['calibration_passed'] and r['qualified_pilot_signal']
    ci=r['boot_level_CI']['off']
    assert ci['df']==2 and ci['mean_reduction_ms']==8
    assert ci['95_percent_CI_ms']==pytest.approx([8-4.302652729911275/(3**.5),8+4.302652729911275/(3**.5)])


def test_calibration_failure_no_signal_and_single_boot_no_CI():
    xs=[r for r in perf_fixture() if r['boot_index']==1]
    assert not summarize(xs)['boot_level_CI']
    xs[1]['metrics']['ii_ms']=130
    assert not summarize(xs)['qualified_pilot_signal']
    assert not summarize(xs[:-1])['calibration_passed']


def test_patch_default_and_setter_scope_restore():
    source=(SOURCE/'raw/runner_sources/vta_stage_pipeline_runner.cc').read_text()
    derived=derive_runner(source)
    assert 'bool host_priority_enabled = false;' in derived
    assert 'if (args.host_priority_enabled && stages[stage_index].device == "vta")\n              vta_deploy::ApplyOwnedHostNice' in derived
    assert derived.index('ApplyOwnedHostNice(args.vta_host_nice)')<derived.index('std::shared_ptr<Frame> frame;',derived.index('ApplyOwnedHostNice(args.vta_host_nice)'))
    with pytest.raises(ValueError):derive_runner(source.replace('#include "vta_driver_snapshot.h"',''))
    build,argv=native.build_runner,native.runner_argv
    with pytest.raises(RuntimeError):
        with adapters():raise RuntimeError('failure')
    assert native.build_runner is build and native.runner_argv is argv
