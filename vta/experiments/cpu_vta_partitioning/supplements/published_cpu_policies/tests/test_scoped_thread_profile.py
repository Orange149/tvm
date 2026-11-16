from copy import deepcopy
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_scoped_thread_profile as w
from cpu_vta_partitioning.common import read,write


def fixture():
    cfg=w.configs('resnet18')['cpu_00_02_t3']
    doc=dict(stages=cfg['stages'],warmup=0,scored=1,tail=0)
    slots=[dict(slot=i,tid=10+i,clock_id=-100-i,cpus=[i]) for i in range(3)]
    row=dict(stage='cpu_00_02',call=0,graph_begin_ns=1000,graph_end_ns=2000,
        process_CPU_begin_ns=200,process_CPU_end_ns=500,
        threads=[dict(slot=i,tid=10+i,before_read_begin_ns=900+10*i,before_read_end_ns=905+10*i,
            CPU_begin_ns=100,after_read_begin_ns=2010+10*i,after_read_end_ns=2015+10*i,CPU_end_ns=200) for i in range(3)])
    data=dict(schema='scoped_thread_budget_v1',stage='cpu_00_02',slots=slots,records=[row],
        registration_launches=1,registration_callbacks=3,registration_outside_stage_phases=True,
        registration_begin_ns=1,registration_end_ns=10)
    affinity=dict(stages=[dict(worker_slots=slots,caller_tid=10)])
    return data,doc,affinity


def test_thread_clocks_include_all_run_work_once_and_preserve_read_envelopes():
    data,doc,affinity=fixture();r=w.budget_summary(data,doc,affinity)
    assert r['run_thread_weights']==[1/3]*3
    assert r['mean_process_minus_thread_sum_ms']==0
    assert r['all_process_budgets_inside_thread_envelope']
    assert r['max_thread_read_window_widening_ms']>0


def test_counter_scope_disagreement_is_reported_not_fabricated_as_worker_work():
    data,doc,affinity=fixture();data['records'][0]['process_CPU_end_ns']=600
    r=w.budget_summary(data,doc,affinity)
    assert not r['all_process_budgets_inside_thread_envelope']
    assert r['run_thread_weights']==[1/3]*3
    assert r['mean_process_minus_thread_sum_ms']==100/1e6


def test_missing_duplicate_frames_and_wrong_thread_affinity_or_clock_are_rejected():
    for change in ('frame','tid','core','clock','order','counter'):
        data,doc,affinity=fixture()
        if change=='frame':data['records'].append(deepcopy(data['records'][0]))
        if change=='tid':data['slots'][0]['tid']=99
        if change=='core':data['slots'][0]['cpus']=[3]
        if change=='clock':data['slots'][1]['clock_id']=data['slots'][0]['clock_id']
        if change=='order':data['records'][0]['threads'][0]['before_read_end_ns']=1100
        if change=='counter':data['records'][0]['threads'][0]['CPU_end_ns']=0
        with pytest.raises(ValueError):w.budget_summary(data,doc,affinity)


def test_exactly_three_components_six_sessions_and_33_calls():
    assert len(w.specs())==6
    p=w.protocol()
    assert sum(sum(p['qualification' if s['phase']=='qualify' else 'calibration'].values()) for s in w.specs())==33
    with pytest.raises(ValueError,match='execute-board'):w.board()


def test_optional_patch_does_not_replace_wait_policy_or_runtime():
    source=(w.SOURCE/'vta_stage_pipeline_runner_phase.cc').read_text();patched=w.patch(source)
    assert patched.count('scoped_thread_budget::Ensure(stage->name)')==1
    assert patched.count('scoped_thread_budget::Scope thread_budget')==1
    assert 'stage->run();\n  thread_budget.End();\n  phase_run.End();' in patched
    assert source.count('stage->run();')==patched.count('stage->run();')
    assert 'requires one serial CPU stage' in patched


def test_native_parameters_keep_pinning_and_only_add_run_clock_instrumentation(tmp_path):
    network='resnet18';b=tmp_path/'capsule';parent=w.PARENTS[network]
    for name in ('programs.json','inputs.json'):write(b/'plans'/name,read(parent/'plans'/name))
    write(b/'manifest.json',read(parent/'manifest.json'))
    write(b/'plans/configurations.json',dict(components=w.configs(network),pipelines={}))
    spec=next(s for s in w.specs() if s['network']==network and s['phase']=='clean')
    doc=w.make_task(b,spec);argv,_=w.native_command(doc,b,'/bundle','/job')
    assert 'CPU_RUN_BUDGET_JSON=/job/thread_budget.json' in argv
    assert '--cpu-worker-pinning' in argv
    assert argv[argv.index('--phase-event-json')+1]=='/job/phase_trace.jsonl'
    assert not any('libwait_adapter' in a for a in argv)
