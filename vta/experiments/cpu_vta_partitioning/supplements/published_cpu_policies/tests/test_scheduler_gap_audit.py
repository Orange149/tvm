"""Scope/accounting/identity tests for the bounded diagnostic, not FPS fitting."""
from copy import deepcopy
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT.parents[1]/'src')]

from scheduler_gap_audit.capability import check_self_test
from scheduler_gap_audit.curves import summarize_group
from scheduler_gap_audit.runtime import validate_queue_rows
from scheduler_gap_audit.analysis import perturbation,queue_bounds
from scheduler_gap_audit.prepare import configurations,specs


def test_all_tasks_bounded_and_qualification_precedes_performance():
    rows=specs();assert len(rows)==18
    seen=set();DNN=component=0
    from scheduler_gap_audit.common import protocol
    for r in rows:
        key=(r['network'],r['scope'],r['case'])
        if r['phase']=='qualify':seen.add(key)
        else:assert key in seen
        w=protocol()['qualification' if r['phase']=='qualify' else r['scope']]
        if r['scope']=='pipeline':DNN+=sum(w.values())
        else:component+=sum(w.values())
    assert (DNN,component)==(627,63)
    assert len({str(r) for r in rows})==18
    assert specs()==rows


def test_configurations_preserve_static_first_and_YOLO_match():
    from cpu_vta_partitioning.common import read
    from scheduler_gap_audit.common import PARENT
    res=configurations('resnet18')['pipelines']['static_first']
    first=read(PARENT/'common_top20/resnet18/RANKINGS.json')['lists']['current'][0]
    assert res['id']==first['id']
    y=configurations('yolov3_tiny416')['pipelines']
    assert [[s['begin'],s['end']] for s in y['overlap']['stages']]==[[s['begin'],s['end']] for s in y['split']['stages']]
    assert all(c['host_mask']==c['control_mask']==[0,1,2,3] and c['execution_mode']=='off' for c in (res,*y.values()))
    assert y['overlap']['stages'][-1]['mask']==[0] and y['split']['stages'][-1]['mask']==[3]


def test_curve_keeps_noisy_nonmonotonic_and_saturated_options():
    rows=[dict(program='cpu',threads=n,service_ms=t,run_core_ms=u) for n,t,u in [(1,10,10),(2,5,10),(3,5.1,12),(4,5,14)]]
    options,group=summarize_group(rows)
    assert len(options)==4 and all(r['retained_in_local_table'] for r in options)
    assert options[2]['nonmonotonic_vs_previous'] and options[2]['marginal_service_reduction']<0
    assert group['best_observed_threads']==2
    assert options[-1]['core_growth_vs_one']==1.4
    with pytest.raises(ValueError):summarize_group(rows[:3])


def scenarios():
    return [dict(case=name,tid=i+1,wall_ns=10**9,cpu_ns=cpu,queue_ns=q,read_max_ns=20000)
            for i,(name,cpu,q) in enumerate([('alone',990000000,1000000),('contended',500000000,490000000),('contended',500000000,490000000),('sleep',10000,1000)])]


def test_counter_capability_requires_positive_controls_not_zero_claims():
    rows=scenarios();assert check_self_test(rows)['passed']
    zero=[{**r,'queue_ns':0} for r in rows];assert not check_self_test(zero)['passed']
    wrong=deepcopy(rows);wrong[-1]['queue_ns']=10**9;assert not check_self_test(wrong)['passed']
    with pytest.raises(ValueError):check_self_test(rows[:-1])


def trace():
    return dict(queue_before_ns=100,queue_after_ns=180,queue_read_start0_ns=10,queue_read_end0_ns=20,
                wall_start_ns=25,wall_end_ns=125,queue_read_start1_ns=130,queue_read_end1_ns=140)


def test_queue_envelopes_bracket_body_and_bound_unknown_read_point():
    r=trace();assert validate_queue_rows([r],scope='worker')['records']==1
    assert queue_bounds(r)==(50,80)
    bad={**r,'queue_after_ns':10}
    with pytest.raises(ValueError):validate_queue_rows([bad],scope='worker')
    bad={**r,'queue_read_end0_ns':30}
    with pytest.raises(ValueError):validate_queue_rows([bad],scope='worker')


def test_probe_quality_is_not_performance_drift_rejection():
    a=perturbation(145,146,220,221)
    assert a['quantitative_attribution_allowed']
    b=perturbation(145,170,220,220)
    assert not b['quantitative_attribution_allowed'] and b['no_performance_result_rejected']
    assert not perturbation(145,145,145,145)['quantitative_attribution_allowed']


def test_observer_keeps_original_wait_and_disables_in_clean_path():
    from scheduler_gap_audit.observer_patch import adapter_source,phase_header,control_patch
    from phase_profile.patch import patch
    source=adapter_source();header=phase_header()
    assert 'r.result=original_launch(call_task,&c,tasks)' in source
    assert 'if(!blocking)r.result=original_barrier(task,env)' in source
    assert 'worker_queue_baselines' in source and 'known_worker_tids' in source
    assert 'if (!active_)return;' in header
    original=(ROOT/'artifacts/bundles/abbcccc62af230ee844745ac/sources/vta/apps/native_deploy/vta_stage_pipeline_runner.cc').read_text()
    observed=control_patch(patch(original))
    assert observed.count('stage->run();')==original.count('stage->run();')
    assert observed.count('std::thread producer_thread')==original.count('std::thread producer_thread')
    assert 'collector_phase.End()' in observed


def test_no_board_access_without_explicit_switch(tmp_path):
    from scheduler_gap_audit.campaign import run
    with pytest.raises(ValueError,match='execute-board'):run(tmp_path,tmp_path,tmp_path)


def test_acceptance_is_idempotent_and_retains_hash_guard(tmp_path,monkeypatch):
    from scheduler_gap_audit import campaign
    from types import FunctionType
    from cpu_vta_partitioning.common import digest
    raw=tmp_path/'raw';task=dict(task_id='task',kind='calibration')
    result=dict(output_fingerprints=['correct'])
    function=FunctionType(campaign.accept.__code__,{**campaign.accept.__globals__,'summary':lambda *a:result},argdefs=campaign.accept.__defaults__)
    class Fake:
        count=0
        def measure(self,*a):
            self.count+=1;raw.mkdir(exist_ok=True);(raw/'boot_id').write_text('boot');(raw/'task_hash').write_text(digest(task));return raw
        def acknowledge(self,*a):assert (tmp_path/'accepted/boot/task.json').exists()
        def idle_command(self,*a,**kw):return ''
        def job_path(self,*a):return '/fake'
    fake=Fake();function(fake,task,tmp_path,'/bundle',tmp_path,'boot');function(fake,task,tmp_path,'/bundle',tmp_path,'boot')
    assert fake.count==1
    (raw/'boot_id').write_text('wrong')
    with pytest.raises(ValueError):function(fake,task,tmp_path,'/bundle',tmp_path,'boot')
