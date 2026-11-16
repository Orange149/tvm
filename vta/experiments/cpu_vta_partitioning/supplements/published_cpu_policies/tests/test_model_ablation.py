"""Three-factor semantics, independent oracle, capsule and recovery checks."""
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
from model_ablation.common import MODELS,settings,protocol,api,PARENT,SOURCE
from model_ablation.model import score
from model_ablation.analysis import evaluate,tied_metrics
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
    from model_ablation.search import engine
    programs,records=synthetic();config=settings(key)
    costs=Costs(records,distribution='session_weights' if config['thread_weights'] else 'balanced')
    module=engine(tmp_path/'source',key)
    ranked=module.rank_all(programs,costs,unit_count=7,max_vta_islands=3,
        contexts=[domain.context('shared4'),domain.context('host_shared_2'),domain.context('host_reserved_3')],
        output_dir=tmp_path/'search',verify=True)
    assert ranked['verification']['passed'] and len(ranked['pure_static20'])==20
    for row in ranked['pure_static20']:assert row['score_ns']==score(row,costs,key)['score_ns']
    assert ranked['counts']['topologies']==1

def test_switches_conserve_work_and_single_core_host_is_redundant(monkeypatch):
    c=api();programs=c.read(PARENT/'plans/programs.json');records=c.read(SOURCE/'isolated_costs.json')['costs']
    ddr=c.read(SOURCE/'ddr.json');base=c.read(SOURCE/'static_rankings.json')['pure_static20'][0]
    for weighted in (False,True):
        costs=Costs(records,programs=programs,ddr=ddr,distribution='session_weights' if weighted else 'balanced')
        keys=('weights','weights_host','weights_team','complete') if weighted else ('uniform','host','team','host_team')
        values=[score(base,costs,key)['score_ns'] for key in keys]
        assert values[1]>=values[0] and values[2]>=values[0] and values[3]==max(values[1:3])
        fixed=deepcopy(base);fixed['context_id']='host_shared_2';fixed['host_mask']=[2]
        for s in fixed['stages']:
            if s['device']=='vta':s['mask']=[2]
        fixed=domain.candidate(fixed['stages'],'host_shared_2','off')
        assert score(fixed,costs,keys[0])['score_ns']==score(fixed,costs,keys[1])['score_ns']
        details=score_configuration(base,costs)
        total=sum(details['cpu_core_demand_ns'])
        expected=sum(allocate_core_ns(101,[.4,.35,.25]))
        assert expected==101 and total==details['cpu_compute_core_ns']+sum(
            quantize_ns(costs.get(s,base)['boundary_core_ms']) for s in base['stages'] if s['device']=='cpu')
    with pytest.raises(ValueError,match='distribution'):score(base,costs,'uniform')
    monkeypatch.setattr('builtins.open',lambda *a,**k:(_ for _ in ()).throw(AssertionError('score opened a file')))
    assert score(base,costs,'complete')['score_ns']>0

def test_budget_deletion_and_exact_factorial_coefficients():
    lists={};measurements={};coeff=[.1,.2,-.3,.04,0,.02,-.01]
    for key,_,w,h,t in MODELS:
        signs=[1 if v else -1 for v in (w,h,t)]
        logfps=2.+sum(coeff[m-1]*math.prod(signs[i] for i in range(3) if m&(1<<i)) for m in range(1,8))
        lists[key]=[key+str(i) for i in range(20)]
        for i in lists[key]:measurements[i]={'ii_ms':1000/math.exp(logfps)}
    result=evaluate(lists,measurements)
    for row in result['factorial_effects']:
        index=[r['factor_combination'] for r in result['factorial_effects'][:7]].index(row['factor_combination'])
        assert row['coefficient_log_fps']==pytest.approx(coeff[index])
    assert len(result['deletion'])==9 and len(result['overlaps'])==84
    ids=[str(i) for i in range(20)];same=evaluate({r[0]:ids for r in MODELS},{i:{'ii_ms':100.} for i in ids})
    assert all(r['removal_cost_percent']==0 and r['selected_configuration_same'] for r in same['deletion'])
    assert tied_metrics(dict(a=1,b=1),dict(a=2,b=3))['prediction_tied_pairs']==1

@pytest.fixture(scope='module')
def capsule(tmp_path_factory):
    from model_ablation.prepare import _controller
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
    name='frozen_ablation_test';spec=importlib.util.spec_from_file_location(name,controller/'model_ablation/__init__.py',
        submodule_search_locations=[str(controller/'model_ablation')])
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    campaign=__import__(name+'.campaign',fromlist=['tasks']);runtime=__import__(name+'.runtime',fromlist=['pipeline_task'])
    return b,controller,campaign,runtime

def test_capsule_deduplicates_models_and_uses_exact_native_contract(capsule):
    b,controller,campaign,runtime=capsule
    tasks=campaign.tasks(b);assert len(tasks)==40 and len({r['task_id'] for r in tasks})==40
    for doc in tasks:
        argv,_=runtime.native_command(doc,b,'/bundle','/job')
        assert doc['campaign']=='resnet18_model_ablation' and doc['input_indices']==[0]
        assert (doc['warmup'],doc['scored'],doc['tail'])==((0,1,0) if doc['kind']=='qualify_pipeline' else (8,16,4))
        assert '--process-cpu-affinity' not in argv
        assert argv[argv.index('--cpu-stage-gate')+1]==doc['configuration']['execution_mode']
        bad=deepcopy(doc);bad['executor_sha256']='bad'
        with pytest.raises(ValueError):runtime.validate_document(bad,b)
    result=subprocess.run([sys.executable,'-c','import sys; from model_ablation.common import identity; '
        'from model_ablation.analysis import evaluate; assert "tvm" not in sys.modules; assert len(identity())==64'],
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
    from frame_stream_top20.common import identity
    assert identity()=='79ec84711558d87c6cacee21c79e59d16242fa577f6573f9d1d787431219bca9'

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

def test_checkpoint_replay_cannot_read_pipeline_fps(monkeypatch):
    from model_ablation.common import DEFAULT_OUTPUT
    from model_ablation.search import verify_one
    original=Path.open
    def guard(path,*a,**kw):
        if path.name in ('analysis.json','screening.json','performance.json','configurations.csv') or '/accepted/' in str(path):
            raise AssertionError('ranking attempted to read pipeline results: '+str(path))
        return original(path,*a,**kw)
    monkeypatch.setattr(Path,'open',guard)
    ranked,proof=verify_one(DEFAULT_OUTPUT/'uniform','uniform')
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
    assert (root/'analysis/ablation_summary.png').is_file()
    data=c.read(root/'analysis/analysis.json')
    assert data['costs']['complete_DNN_inferences']==580
    assert all(r['selected_configuration_same'] for r in data['decision']['deletion'])
