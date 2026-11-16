"""Host-demand semantics, independent oracle, capsule and recovery checks."""
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from host_work_ablation.common import MODELS,settings,protocol,api,PARENT,SOURCE
from host_work_ablation.model import score
from host_work_ablation.analysis import evaluate,tied_metrics
from cpu_policies.open_resources.costs import Costs,score_configuration,allocate_core_ns,quantize_ns
from cpu_policies.open_resources import domain

def synthetic():
    programs={}
    for device,b,e in [('cpu',0,1),('vta',1,2),('cpu',2,3),('vta',3,4),('cpu',4,5),('vta',5,6),('cpu',6,7)]:
        n=f'{device}_{b:02}_{e-1:02}';programs[n]=dict(device=device,begin=b,end=e,name=n)
    records={}
    for key,cfg in domain.requirements(programs).items():
        row=cfg['stages'][0];n=row['threads'];cpu=row['device']=='cpu'
        records[key]=dict(configuration=cfg,kind='synthetic',cost=dict(
            service_ms=(35+row['begin'])/n+.2*n if cpu else 4.,
            run_core_ms=34.+row['begin'] if cpu else 2.,boundary_core_ms=.25))
        if cpu:records[key]['session_thread_weights']=[(i+1)/(n*(n+1)/2) for i in range(n)]
    return programs,records

@pytest.mark.parametrize('key',[r[0] for r in MODELS])
def test_eight_independent_small_domains_match_scalar_and_cpp(tmp_path,key):
    from host_work_ablation.search import engine
    programs,records=synthetic();config=settings(key)
    costs=Costs(records,distribution='session_weights' if config['thread_weights'] else 'balanced')
    module=engine(tmp_path/'source',key)
    ranked=module.rank_all(programs,costs,unit_count=7,max_vta_islands=3,
        contexts=[domain.context('shared4'),domain.context('host_shared_2'),domain.context('host_reserved_3')],
        output_dir=tmp_path/'search',verify=True)
    assert ranked['verification']['passed'] and len(ranked['pure_static20'])==20
    for row in ranked['pure_static20']:assert row['score_ns']==score(row,costs,key)['score_ns']
    assert ranked['counts']['topologies']==1

def test_host_exclusion_conserves_CPU_work_VTA_wall_and_DDR():
    c=api();programs=c.read(PARENT/'plans/programs.json');records=c.read(SOURCE/'isolated_costs.json')['costs']
    original=deepcopy(records);ddr=c.read(SOURCE/'ddr.json')
    costs=Costs(records,programs=programs,ddr=ddr,distribution='session_weights')
    base=c.read(SOURCE/'static_rankings.json')['pure_static20'][0]
    from cpu_policies.open_resources.domain import candidate
    for ctx in ('shared4','host_shared_2'):
        cfg=candidate(base['stages'],ctx,base['execution_mode']) if ctx=='shared4' else None
        if cfg is None:
            rows=deepcopy(base['stages'])
            for row in rows:
                if row['device']=='vta':row['mask']=[2]
            cfg=candidate(rows,ctx,base['execution_mode'])
        values={k:score(cfg,costs,k) for k,*_ in MODELS}
        assert values['ignored']['score_ns']<=values['included']['score_ns']<=values['complete']['score_ns']
        h=sum(quantize_ns(costs.get(row,cfg)['run_core_ms'])+quantize_ns(costs.get(row,cfg)['boundary_core_ms'])
              for row in cfg['stages'] if row['device']=='vta')
        assert h>0 and values['ignored']['observed_host_cpu_ns']==h
        assert values['ignored']['scored_host_cpu_ns']==0 and values['ignored']['excluded_host_cpu_ns']==h
        assert values['included']['scored_host_cpu_ns']==h and values['included']['excluded_host_cpu_ns']==0
        for field in ('CPU_core_demand_ns','VTA_service_sum_ns','DDR_service_ns'):
            assert values['ignored'][field]==values['included'][field]==values['complete'][field]
        assert all(values[k]['terms_ns']['team_service']==values['ignored']['terms_ns']['team_service'] for k in values)
        total=sum(values['ignored']['CPU_core_demand_ns'])
        expected=sum(quantize_ns(costs.get(r,cfg)['run_core_ms'])+quantize_ns(costs.get(r,cfg)['boundary_core_ms'])
                     for r in cfg['stages'] if r['device']=='cpu')
        assert total==expected
        if ctx=='host_shared_2':assert values['included']['score_ns']==values['complete']['score_ns']
        assert values['included']['score_ns']==__import__('model_ablation.model',fromlist=['score']).score(cfg,costs,'weights_team')['score_ns']
        assert values['complete']['score_ns']==__import__('model_ablation.model',fromlist=['score']).score(cfg,costs,'complete')['score_ns']
    assert records==original


def test_zero_host_work_three_scores_equal_and_scoring_cannot_open_files(monkeypatch):
    programs,records=synthetic()
    for record in records.values():
        if record['configuration']['stages'][0]['device']=='vta':
            record['cost']['run_core_ms']=0.;record['cost']['boundary_core_ms']=0.
    costs=Costs(records,programs=programs,distribution='session_weights')
    specs=[dict(program=n,device=p['device'],begin=p['begin'],end=p['end'],
                threads=2 if p['device']=='cpu' else 1,mask=[0,1] if p['device']=='cpu' else [0,1,2,3])
           for n,p in programs.items()]
    cfg=domain.candidate(specs,'shared4','off')
    monkeypatch.setattr('builtins.open',lambda *a,**k:(_ for _ in ()).throw(AssertionError('scorer opened file')))
    results=[score(cfg,costs,k) for k,*_ in MODELS]
    assert len({v['score_ns'] for v in results})==1 and all(v['observed_host_cpu_ns']==0 for v in results)
    with pytest.raises(ValueError,match='thread shares'):
        score(cfg,Costs(records),'ignored')


def test_selection_effect_denominators_and_shared_configuration_zero():
    lists={k:[k+str(i) for i in range(20)] for k,*_ in MODELS}
    measured={i:dict(ii_ms=v) for k,v in [('ignored',100.),('included',90.),('complete',80.)] for i in lists[k]}
    result=evaluate(lists,measured)
    first=result['comparisons'][0]
    assert first['fps_improvement_percent']==pytest.approx((100/90-1)*100)
    assert first['ii_reduction_percent']==pytest.approx(10.)
    assert len(result['comparisons'])==9 and len(result['overlaps'])==9
    ids=[str(i) for i in range(20)]
    same=evaluate({k:ids for k,*_ in MODELS},{i:dict(ii_ms=100.) for i in ids})
    assert all(r['selected_configuration_same'] and r['fps_improvement_percent']==0 for r in same['comparisons'])


def test_hand_computed_run_and_copy_work_are_both_excluded():
    programs={n:dict(device=d,begin=b,end=e,name=n) for n,d,b,e in
              [('cpu_00_00','cpu',0,1),('vta_01_01','vta',1,2),('cpu_02_02','cpu',2,3)]}
    records={}
    for key,cfg in domain.requirements(programs).items():
        row=cfg['stages'][0]
        if row['device']=='vta':cost=dict(service_ms=20.,run_core_ms=10.,boundary_core_ms=6.)
        elif row['begin']==0:cost=dict(service_ms=15.,run_core_ms=14.,boundary_core_ms=1.)
        else:cost=dict(service_ms=1.,run_core_ms=.8,boundary_core_ms=.2)
        records[key]=dict(configuration=cfg,kind='synthetic',cost=cost)
        if row['device']=='cpu':records[key]['session_thread_weights']=[1/row['threads']]*row['threads']
    costs=Costs(records,programs=programs,distribution='session_weights')
    stages=[domain.stage('cpu',0,1,1,[0]),domain.stage('vta',1,2,1,[0,1,2,3]),domain.stage('cpu',2,3,1,[1])]
    cfg=domain.candidate(stages,'shared4','off')
    assert [score(cfg,costs,k)['score_ns'] for k,*_ in MODELS]==[20_000_000,20_000_000,31_000_000]
    assert score(cfg,costs,'ignored')['CPU_core_demand_ns']==[15_000_000,1_000_000,0,0]
    assert score(cfg,costs,'ignored')['excluded_host_cpu_ns']==16_000_000
    stages[1]=domain.stage('vta',1,2,1,[0]);pinned=domain.candidate(stages,'host_shared_0','off')
    assert [score(pinned,costs,k)['score_ns'] for k,*_ in MODELS]==[20_000_000,31_000_000,31_000_000]

@pytest.fixture(scope='module')
def capsule(tmp_path_factory):
    from host_work_ablation.prepare import _controller
    c=api();b=tmp_path_factory.mktemp('ablation_capsule')/'bundle';b.mkdir()
    for d in ('sources/base_controller/src','sources/cpu_strategy'):(b/d).mkdir(parents=True)
    controller=_controller(b)
    old=c.read(SOURCE/'static_rankings.json')['pure_static20']
    ranked={r[0]:old for r in MODELS};pool={r['id']:r for r in old}
    docs={'selection_rankings.json':ranked,'static_pool.json':pool,'programs.json':c.read(PARENT/'plans/programs.json'),
          'inputs.json':c.read(PARENT/'plans/inputs.json'),'executors.json':c.read(PARENT/'plans/executors.json'),
          'board_execution.json':dict(storage_margin_bytes=256*1024**2)}
    for name,value in docs.items():c.write(b/'plans'/name,value)
    c.write(b/'manifest.json',{'bundle_id':'fixture'})
    name='frozen_host_work_ablation_test';spec=importlib.util.spec_from_file_location(name,controller/'host_work_ablation/__init__.py',
        submodule_search_locations=[str(controller/'host_work_ablation')])
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    campaign=__import__(name+'.campaign',fromlist=['tasks']);runtime=__import__(name+'.runtime',fromlist=['pipeline_task'])
    return b,controller,campaign,runtime

def test_capsule_deduplicates_models_and_uses_exact_native_contract(capsule):
    b,controller,campaign,runtime=capsule
    tasks=campaign.tasks(b);assert len(tasks)==40 and len({r['task_id'] for r in tasks})==40
    for doc in tasks:
        argv,_=runtime.native_command(doc,b,'/bundle','/job')
        assert doc['campaign']=='resnet18_host_work_ablation' and doc['input_indices']==[0]
        assert (doc['warmup'],doc['scored'],doc['tail'])==((0,1,0) if doc['kind']=='qualify_pipeline' else (8,16,4))
        assert '--process-cpu-affinity' not in argv
        assert argv[argv.index('--cpu-stage-gate')+1]==doc['configuration']['execution_mode']
        bad=deepcopy(doc);bad['executor_sha256']='bad'
        with pytest.raises(ValueError):runtime.validate_document(bad,b)
    result=subprocess.run([sys.executable,'-c','import sys; from host_work_ablation.common import identity; '
        'from host_work_ablation.analysis import evaluate; assert "tvm" not in sys.modules; assert len(identity())==64'],
        cwd=controller,capture_output=True,text=True)
    assert result.returncode==0,result.stderr

def test_receipt_replay_does_not_remeasure_and_rejects_hash_error(capsule,tmp_path,monkeypatch):
    b,_,campaign,_=capsule;root=tmp_path/'run';root.mkdir();doc=campaign.tasks(b)[0]
    state={'bundle':str(b),'boot_id':'boot-A','frozen_files':{}}
    monkeypatch.setattr(campaign,'validate_summary',lambda *a:{'correct':True})
    class Transport:
        calls=0
        def measure(self,d,bundle,remote,root,boot):
            self.calls+=1;p=root/'raw'/boot/d['task_id']/'collected';p.mkdir(parents=True)
            (p/'result.jsonl').write_text('evidence\n');return p
        def acknowledge(self,*a):pass
    t=Transport();recorder=campaign.Recorder(root,state,t,'/remote')
    assert recorder.saved(doc)==recorder.saved(doc) and t.calls==1
    next((root/'raw').rglob('result.jsonl')).write_text('changed')
    with pytest.raises(ValueError):recorder.saved(doc)

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

def test_wrong_boot_rejected_before_deployment(capsule,tmp_path,monkeypatch):
    b,_,campaign,_=capsule
    frozen_prepare=__import__(campaign.__package__+'.prepare',fromlist=['validate_bundle'])
    state=dict(bundle=str(b),boot_id='old',frozen_files={},status='INTERRUPTED')
    monkeypatch.setattr(campaign,'validate_state',lambda *a:state)
    monkeypatch.setattr(frozen_prepare,'validate_bundle',lambda *a:None)
    class Transport:
        def boot_id(self):return 'new'
        def wait_idle(self,*a):raise AssertionError('must reject before deployment')
    with pytest.raises(ValueError,match='boot'):campaign.run_current(tmp_path,Transport())
    assert api().read(tmp_path/'state.json')['status']=='STOPPED_INVALID_RESULT'

def test_no_old_source_identity_changes():
    from frame_stream_top20.common import identity as frame
    from model_ablation.common import identity as old
    baseline=api().read(ROOT/'artifacts/host_work_ablation_01/PRE_IMPLEMENTATION_IDENTITIES.json')
    assert frame()==baseline['frame_stream_top20_identity']
    assert old()==baseline['model_ablation_identity']

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

def test_checkpoint_replay_cannot_read_pipeline_fps(monkeypatch,tmp_path):
    from host_work_ablation.search import run_one,verify_one
    original=Path.open
    def guard(path,*a,**kw):
        if path.name in ('analysis.json','screening.json','performance.json','configurations.csv') or '/accepted/' in str(path):
            raise AssertionError('ranking attempted to read pipeline results: '+str(path))
        return original(path,*a,**kw)
    monkeypatch.setattr(Path,'open',guard)
    run_one(tmp_path/'included','included')
    ranked,proof=verify_one(tmp_path/'included','included')
    assert ranked['verification']['passed'] and proof['cache_replayed']


def test_complete_analysis_reconstructs_raw_intervals_and_writes_artifacts(capsule,tmp_path,monkeypatch):
    b,_,campaign,_=capsule;c=api(b)
    analysis=__import__(campaign.__package__+'.analysis',fromlist=['analyze'])
    prepare=__import__(campaign.__package__+'.prepare',fromlist=['validate_bundle'])
    pool=c.read(b/'plans/static_pool.json');programs=c.read(PARENT/'plans/programs.json')
    records=c.read(SOURCE/'isolated_costs.json')['costs'];ddr=c.read(SOURCE/'ddr.json')
    scores={}
    for key,*_ in MODELS:
        costs=Costs(records,programs=programs,ddr=ddr,distribution='session_weights' if settings(key)['thread_weights'] else 'balanced')
        scores[key]={i:score(cfg,costs,key) for i,cfg in pool.items()}
    c.write(b/'plans/model_scores.json',scores,replace=True)
    c.write(b/'plans/isolated_costs.json',{'costs':records},replace=True)
    c.write(b/'plans/ddr.json',ddr,replace=True)
    c.write(b/'plans/search_proofs.json',{r[0]:dict(elapsed_seconds=1.,cache_replayed=False) for r in MODELS},replace=True)
    c.write(b/'plans/preparation_cost.json',dict(fixture=True),replace=True)
    source=ROOT/'runs/resnet18_thread_work_combined_top20_01';old=c.read(source/'state.json')
    lookup={}
    for p in (source/'accepted'/old['boot_id']).glob('*.json'):
        receipt=c.read(p);doc=receipt['task']
        if doc['candidate_id'] in pool:lookup[(doc['candidate_id'],doc['kind'])]=receipt
    root=tmp_path/'analysis_run';root.mkdir();state=dict(bundle=str(b),boot_id=old['boot_id'])
    c.write(root/'selection_freeze.json',campaign.freeze_document(b))
    for doc in campaign.tasks(b):
        value=deepcopy(lookup[(doc['candidate_id'],doc['kind'])]);value['task']=doc
        value['raw']=str(source/value['raw'])
        c.write(root/'accepted'/old['boot_id']/(doc['task_id']+'.json'),value)
    monkeypatch.setattr(campaign,'validate_state',lambda *a:state)
    monkeypatch.setattr(prepare,'validate_bundle',lambda *a:None)
    class Recorder:
        def __init__(self,*a):pass
        def saved(self,doc):return lookup[(doc['candidate_id'],doc['kind'])]['summary']
    monkeypatch.setattr(campaign,'Recorder',Recorder)
    result=analysis.analyze(root)
    assert result['pool_size']==20 and result['accepted_sessions']==40
    assert (root/'analysis/host_work_summary.png').is_file()
    data=c.read(root/'analysis/analysis.json')
    assert data['costs']['complete_DNN_inferences']==580
    assert all(r['selected_configuration_same'] for r in data['decision']['comparisons'])
