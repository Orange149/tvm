"""Decision-complete baseline, factor accounting, frozen execution and recovery."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from staged_ablation.common import MODELS,settings,api,PARENT,SOURCE
from staged_ablation.model import score
from staged_ablation.search import engine,method,applicable_score
from staged_ablation.analysis import evaluate
from cpu_policies.open_resources.costs import Costs,quantize_ns
from cpu_policies.open_resources import domain
from test_host_work_ablation import synthetic

@pytest.mark.parametrize('key',[r[0] for r in MODELS])
def test_each_model_matches_independent_small_domain_CPP_and_scalar(tmp_path,key):
    programs,records=synthetic()
    costs=Costs(records,distribution='session_weights' if settings(key)['thread_weights'] else 'balanced')
    module=engine(tmp_path/'sources',key)
    ranked=module.rank_all(programs,costs,unit_count=7,max_vta_islands=3,
        contexts=[domain.context('shared4'),domain.context('host_shared_2'),domain.context('host_reserved_3')],
        output_dir=tmp_path/'checkpoints',verify=True)
    assert ranked['verification']['passed']
    for row in ranked['methods'][method(key)]['rankings']:
        assert row['score_ns']==score(row,costs,key)['score_ns']
    if key=='tarnawski':
        assert ranked['counts']['single_core']==4**4+4**4+3**4
        assert all(s['threads']==1 for r in ranked['methods']['single_core']['rankings'] for s in r['stages'] if s['device']=='cpu')

def test_device_baseline_uses_service_not_hidden_CPU_host_capacity():
    programs,records=synthetic()
    for r in records.values():
        cpu=r['configuration']['stages'][0]['device']=='cpu'
        # Deliberately distinguish service accounting from CPU work so the old
        # inherited enhanced score would fail this hand-computed contract.
        r['cost'].update(service_ms=3. if cpu else 2.,run_core_ms=100.,boundary_core_ms=25.)
    costs=Costs(records,programs=programs)
    rows=[domain.stage(p['device'],p['begin'],p['end'],1,[0] if p['device']=='cpu' else [0,1,2,3]) for p in programs.values()]
    cfg=domain.candidate(rows,'shared4','resource_group')
    result=score(cfg,costs,'tarnawski')
    assert result['score_ns']==12_000_000
    assert result['VTA_service_sum_ns']==6_000_000 and result['scored_host_cpu_ns']==0
    assert set(result['capacity_terms_ns'])=={'VTA_sum','DDR_sum','CPU_device_0','CPU_device_1','CPU_device_2','CPU_device_3'}

def test_host_toggle_and_CPU_work_conservation_without_rewriting_costs():
    programs,records=synthetic();before=deepcopy(records)
    rows=[domain.stage(p['device'],p['begin'],p['end'],2 if p['device']=='cpu' else 1,
        [0,1] if p['device']=='cpu' else [2]) for p in programs.values()]
    cfg=domain.candidate(rows,'host_shared_2','off')
    for a,b in [('coarse_no_host','coarse_host'),('fine_no_host','fine_host')]:
        costs=Costs(records,programs=programs,distribution='session_weights' if settings(a)['thread_weights'] else 'balanced')
        aa,bb=score(cfg,costs,a),score(cfg,costs,b)
        assert aa['score_ns']<=bb['score_ns']
        assert aa['CPU_core_demand_ns']==bb['CPU_core_demand_ns']
        expected=sum(quantize_ns(costs.get(r,cfg)['run_core_ms'])+quantize_ns(costs.get(r,cfg)['boundary_core_ms']) for r in rows if r['device']=='cpu')
        assert sum(aa['CPU_core_demand_ns'])==expected
        assert aa['VTA_service_sum_ns']==bb['VTA_service_sum_ns']
        assert aa['DDR_service_ns']==bb['DDR_service_ns']
        assert aa['scored_host_cpu_ns']==0 and bb['scored_host_cpu_ns']>0
    assert records==before

def test_team_guard_is_independent_and_host_singleton_not_double_counted():
    programs,records=synthetic()
    for r in records.values():
        if r['configuration']['stages'][0]['device']=='cpu':
            n=r['configuration']['stages'][0]['threads'];r['session_thread_weights']=[1/n]*n
    rows=[domain.stage(p['device'],p['begin'],p['end'],2 if p['device']=='cpu' else 1,
        [0,1] if p['device']=='cpu' else [2]) for p in programs.values()]
    cfg=domain.candidate(rows,'host_shared_2','off')
    balanced=Costs(records,programs=programs);weighted=Costs(records,programs=programs,distribution='session_weights')
    assert score(cfg,balanced,'coarse_host')['score_ns']<=score(cfg,balanced,'team_only')['score_ns']
    assert score(cfg,weighted,'weights_only')['score_ns']<=score(cfg,weighted,'fine_host')['score_ns']
    assert score(cfg,weighted,'fine_host')['score_ns']==score(cfg,weighted,'concentrated')['score_ns']

def test_scoring_cannot_read_files_and_inapplicable_baseline_is_not_faked(monkeypatch):
    programs,records=synthetic()
    rows=[domain.stage(p['device'],p['begin'],p['end'],2 if p['device']=='cpu' else 1,
        [0,1] if p['device']=='cpu' else [0,1,2,3]) for p in programs.values()]
    cfg=domain.candidate(rows,'shared4','off')
    monkeypatch.setattr('builtins.open',lambda *a,**k:(_ for _ in ()).throw(AssertionError('score opened FPS file')))
    assert not applicable_score(cfg,Costs(records),'tarnawski')['applicable']
    assert score(cfg,Costs(records),'coarse_no_host')['pipeline_FPS_used'] is False

def test_budget_lengths_interaction_and_shared_identity():
    ids=[str(i) for i in range(20)]
    lists={k:ids[:settings(k)['measurement_budget']] for k,*_ in MODELS}
    result=evaluate(lists,{i:dict(ii_ms=100.) for i in ids})
    assert '20' not in result['methods']['weights_only']['budgets']
    assert all(x['fps_improvement_percent']==0 and x['selected_configuration_same'] for x in result['comparisons'])
    assert all(x['CPU_host_log_interaction']==pytest.approx(0) for x in result['interactions'])

@pytest.fixture(scope='module')
def capsule(tmp_path_factory):
    from staged_ablation.prepare import _controller
    c=api();b=tmp_path_factory.mktemp('staged_capsule')/'bundle';b.mkdir()
    for folder in ('sources/base_controller/src','sources/cpu_strategy'):(b/folder).mkdir(parents=True)
    controller=_controller(b)
    rows=c.read(SOURCE/'static_rankings.json')['pure_static20']
    docs={'selection_rankings.json':{k:rows[:settings(k)['measurement_budget']] for k,*_ in MODELS},
          'static_pool.json':{r['id']:r for r in rows},'programs.json':c.read(PARENT/'plans/programs.json'),
          'inputs.json':c.read(PARENT/'plans/inputs.json'),'executors.json':c.read(PARENT/'plans/executors.json'),
          'board_execution.json':dict(storage_margin_bytes=256*1024**2)}
    for name,value in docs.items():c.write(b/'plans'/name,value)
    c.write(b/'manifest.json',{'bundle_id':'fixture'})
    name='frozen_staged_ablation_test'
    spec=importlib.util.spec_from_file_location(name,controller/'staged_ablation/__init__.py',submodule_search_locations=[str(controller/'staged_ablation')])
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return b,controller,__import__(name+'.campaign',fromlist=['tasks']),__import__(name+'.runtime',fromlist=['pipeline_task'])

def test_frozen_tasks_preserve_native_modes_and_deduplicate(capsule):
    b,controller,campaign,runtime=capsule
    tasks=campaign.tasks(b);assert len(tasks)==40 and len({d['task_id'] for d in tasks})==40
    for d in tasks:
        assert d['campaign']=='resnet18_staged_ablation'
        assert (d['warmup'],d['scored'],d['tail'])==((0,1,0) if d['kind']=='qualify_pipeline' else (8,16,4))
        argv,_=runtime.native_command(d,b,'/bundle','/job')
        assert '--process-cpu-affinity' not in argv
        assert argv[argv.index('--cpu-stage-gate')+1]==d['configuration']['execution_mode']
        bad=deepcopy(d);bad['executor_sha256']='wrong'
        with pytest.raises(ValueError):runtime.validate_document(bad,b)
    result=subprocess.run([sys.executable,'-c','import sys; from staged_ablation.common import identity; '
        'from staged_ablation.analysis import evaluate; assert "tvm" not in sys.modules; assert len(identity())==64'],
        cwd=controller,capture_output=True,text=True)
    assert result.returncode==0,result.stderr

def test_resume_accepts_existing_raw_once_and_rejects_hash_tamper(capsule,tmp_path,monkeypatch):
    b,_,campaign,_=capsule;root=tmp_path/'run';root.mkdir();doc=campaign.tasks(b)[0]
    state={'bundle':str(b),'boot_id':'boot-A','frozen_files':{}}
    monkeypatch.setattr(campaign,'validate_summary',lambda *a:{'correct':True})
    class Transport:
        calls=0
        def measure(self,d,bundle,remote,root,boot):
            self.calls+=1;p=root/'raw'/boot/d['task_id']/'collected';p.mkdir(parents=True)
            (p/'result.jsonl').write_text('original');return p
        def acknowledge(self,*a):pass
    t=Transport();r=campaign.Recorder(root,state,t,'/remote')
    assert r.saved(doc)==r.saved(doc) and t.calls==1
    next((root/'raw').rglob('result.jsonl')).write_text('tampered')
    with pytest.raises(ValueError):r.saved(doc)

def test_wrong_boot_stops_before_deployment(capsule,tmp_path,monkeypatch):
    b,_,campaign,_=capsule
    prep=__import__(campaign.__package__+'.prepare',fromlist=['validate_bundle'])
    state=dict(bundle=str(b),boot_id='old',frozen_files={},status='INTERRUPTED')
    monkeypatch.setattr(campaign,'validate_state',lambda *a:state)
    monkeypatch.setattr(prep,'validate_bundle',lambda *a:None)
    class Transport:
        def boot_id(self):return 'new'
        def wait_idle(self,*a):raise AssertionError('must reject before deployment')
    with pytest.raises(ValueError,match='boot'):campaign.run_current(tmp_path,Transport())
    assert api().read(tmp_path/'state.json')['status']=='STOPPED_INVALID_RESULT'

def test_legacy_identities_unchanged():
    from model_ablation.common import identity as model_identity
    from host_work_ablation.common import identity as host_identity
    from frame_stream_top20.common import identity as frame_identity
    old=api().read(ROOT/'artifacts/staged_ablation_01/PRE_IMPLEMENTATION_IDENTITIES.json')
    assert old==dict(model_ablation=model_identity(),host_work_ablation=host_identity(),frame_stream_top20=frame_identity())


import shutil

@pytest.mark.parametrize('bad',['missing_frame','duplicate_frame','wrong_output','wrong_worker','restricted_host'])
def test_new_capsule_rejects_invalid_raw_evidence(capsule,tmp_path,bad):
    import numpy as np
    b,_,campaign,runtime=capsule;c=api(b)
    if not (b/'references').exists():(b/'references').symlink_to(PARENT/'references',target_is_directory=True)
    source=ROOT/'runs/resnet18_thread_work_combined_top20_01';state=c.read(source/'state.json')
    kind='qualify_pipeline' if bad=='wrong_output' else 'performance'
    pool=c.read(b/'plans/static_pool.json')
    doc=next(d for d in c.read(source/'tasks.json') if d['kind']==kind and d['candidate_id'] in pool)
    receipt=c.read(source/'accepted'/state['boot_id']/(doc['task_id']+'.json'))
    raw=tmp_path/'raw';shutil.copytree(source/receipt['raw'],raw)
    task=runtime.pipeline_task(b,doc['candidate_id'],kind=='qualify_pipeline')
    runtime.validate_summary(raw,task,b)
    rows=[json.loads(s) for s in (raw/'result.jsonl').read_text().splitlines()]
    if bad=='missing_frame':rows.pop()
    elif bad=='duplicate_frame':rows[1]=rows[0]
    elif bad=='wrong_output':
        meta=rows[0]['stage_raw_outputs'][0]['output_files'][0]
        output=raw/'outputs/run_0'/Path(meta['path']).name
        values=np.fromfile(output,dtype=meta['dtype']);values[0]=123 if values[0]!=123 else 124;values.tofile(output)
    else:
        report=c.read(raw/'affinity.json')
        if bad=='wrong_worker':report['stages'][0]['effective_affinities'][0]['cpus']=[3]
        else:
            stage=next(s for s,r in zip(report['stages'],task['configuration']['stages']) if r['device']=='vta')
            caller=next(e for e in stage['effective_affinities'] if e['tid']==stage['caller_tid']);caller['cpus']=[0]
        c.write(raw/'affinity.json',report,replace=True)
    (raw/'result.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError):runtime.validate_summary(raw,task,b)



def test_disconnect_recollects_without_relaunch(capsule,tmp_path):
    b,_,campaign,runtime=capsule
    from cpu_vta_partitioning.transport import TransportError
    doc=campaign.tasks(b)[0];root=tmp_path/'run';root.mkdir();c=api(b)
    t=object.__new__(runtime._ssh_class(str(b)));t.config={'remote_base':'/board'};t.boot_id=lambda:'boot-A'
    remote_hash='';launched=collected=0
    def command(script):
        nonlocal remote_hash,launched
        if 'nohup sh execute.sh' in script:launched+=1;remote_hash=c.digest(doc)
        if 'test -f ' in script and '/task_hash' in script:return remote_hash
        if 'test -f ' in script and '/exit_code' in script:return '0'
        return ''
    t.command=command;t.check_space=lambda *a:None;t.upload=lambda *a:None;t.failure_snapshot=lambda *a:'retained'
    def collect(job,raw):
        nonlocal collected
        collected+=1
        if collected==1:raise TransportError('disconnect')
        out=raw/'collected';out.mkdir()
        for name,text in [('task_hash',c.digest(doc)),('boot_id','boot-A'),('inputs.txt',(raw/'inputs.txt').read_text())]:
            (out/name).write_text(text)
        return out
    t.collect=collect
    with pytest.raises(TransportError):t.measure(doc,b,'/bundle',root,'boot-A')
    t.measure(doc,b,'/bundle',root,'boot-A')
    assert launched==1 and collected==2



def test_checkpoint_replay_never_reads_old_pipeline_FPS(monkeypatch,tmp_path):
    from staged_ablation.search import run_one,verify_one
    original=Path.open
    def guard(path,*a,**kw):
        if path.name in ('analysis.json','screening.json','performance.json','configurations.csv') or '/accepted/' in str(path):
            raise AssertionError('ranking attempted to read pipeline results: '+str(path))
        return original(path,*a,**kw)
    monkeypatch.setattr(Path,'open',guard)
    run_one(tmp_path/'coarse_host','coarse_host')
    ranked,proof=verify_one(tmp_path/'coarse_host','coarse_host')
    assert ranked['verification']['passed'] and proof['cache_replayed']


def test_complete_analysis_reconstructs_archived_raw_timestamps(capsule,tmp_path,monkeypatch):
    b,_,campaign,_=capsule;c=api(b)
    analysis=__import__(campaign.__package__+'.analysis',fromlist=['analyze'])
    prepare=__import__(campaign.__package__+'.prepare',fromlist=['validate_bundle'])
    pool=c.read(b/'plans/static_pool.json');programs=c.read(PARENT/'plans/programs.json')
    records=c.read(SOURCE/'isolated_costs.json')['costs'];ddr=c.read(SOURCE/'ddr.json');scores={}
    for key,*_ in MODELS:
        costs=Costs(records,programs=programs,ddr=ddr,distribution='session_weights' if settings(key)['thread_weights'] else 'balanced')
        scores[key]={i:applicable_score(cfg,costs,key) for i,cfg in pool.items()}
    # This fixture isolates analysis of already-recorded multi-thread raw windows.
    # The physical-single-core scorer/domain is independently tested above.
    scores['tarnawski']=deepcopy(scores['coarse_no_host'])
    c.write(b/'plans/model_scores.json',scores,replace=True)
    c.write(b/'plans/isolated_costs.json',{'costs':records},replace=True)
    c.write(b/'plans/ddr.json',ddr,replace=True)
    c.write(b/'plans/search_proofs.json',{r[0]:dict(elapsed_seconds=1.,cache_replayed=True,selected_count=1) for r in MODELS},replace=True)
    c.write(b/'plans/preparation_cost.json',dict(fixture=True),replace=True)
    two=list(pool)[:2]
    c.write(b/'plans/mechanism_audit.json',{'required_new_configurations':[
        {'label':label,'configuration':pool[i]} for label,i in zip(('two_thread_overlap','two_thread_split'),two)]},replace=True)
    source=ROOT/'runs/resnet18_thread_work_combined_top20_01';old=c.read(source/'state.json');lookup={}
    for p in (source/'accepted'/old['boot_id']).glob('*.json'):
        receipt=c.read(p);doc=receipt['task']
        if doc['candidate_id'] in pool:lookup[doc['candidate_id'],doc['kind']]=receipt
    root=tmp_path/'analysis_run';root.mkdir();state=dict(bundle=str(b),boot_id=old['boot_id'])
    c.write(root/'selection_freeze.json',campaign.freeze_document(b))
    for doc in campaign.tasks(b):
        value=deepcopy(lookup[doc['candidate_id'],doc['kind']]);value['task']=doc;value['raw']=str(source/value['raw'])
        c.write(root/'accepted'/old['boot_id']/(doc['task_id']+'.json'),value)
    monkeypatch.setattr(campaign,'validate_state',lambda *a:state)
    monkeypatch.setattr(prepare,'validate_bundle',lambda *a:None)
    class Recorder:
        def __init__(self,*a):pass
        def saved(self,doc):return lookup[doc['candidate_id'],doc['kind']]['summary']
    monkeypatch.setattr(campaign,'Recorder',Recorder)
    result=analysis.analyze(root)
    assert result['pool_size']==20 and result['accepted_sessions']==40
    assert (root/'analysis/method_comparison.png').is_file()
    assert (root/'analysis/CPU_host_ablation.png').is_file()
    data=c.read(root/'analysis/analysis.json')
    assert data['costs']['complete_DNN_inferences']==580
    assert all(r['selected_configuration_same'] for r in data['decision']['comparisons'])
    assert len((root/'analysis/stage_profile_comparison.csv').read_text().splitlines())>20
