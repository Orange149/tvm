from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from selection_top20.common import api,FULL_PARENT,STATIC_SOURCE
from selection_top20.selector import execution_class,features,seeded_pick,choose
from selection_top20 import static

@lru_cache(maxsize=1)
def complete():
    programs,checkpoints=static.verify_checkpoints(FULL_PARENT,STATIC_SOURCE)
    records,expand=static.compact_records(programs,checkpoints)
    records.sort(key=lambda r:(r['score_ns'],r['canonical']))
    from cpu_policies.open_resources.domain import contexts
    new,coverage=choose(records,records[:5],[c['id'] for c in contexts()],expand,fallback=records[:20])
    return programs,records,expand,new,coverage

def fixture_bundle(tmp_path):
    c=api(FULL_PARENT);b=tmp_path/'bundle';(b/'plans').mkdir(parents=True)
    for p in ('sources/base_controller/src','sources/cpu_strategy'):(b/p).mkdir(parents=True)
    programs,records,expand,new,coverage=complete()
    old=c.read(STATIC_SOURCE/'static_rankings.json')
    ranked={k:old[k] for k in ('pure_static20','diverse20')};ranked['coverage20']=new
    pool={r['id']:r for key in ranked for r in ranked[key]}
    values={'static_pool.json':pool,'selection_rankings.json':ranked,'programs.json':programs,
        'inputs.json':c.read(FULL_PARENT/'plans/inputs.json'),
        'board_execution.json':{'storage_margin_bytes':256*1024**2},
        'executors.json':c.read(FULL_PARENT/'plans/executors.json')}
    for name,value in values.items():c.write(b/'plans'/name,value)
    c.write(b/'manifest.json',{'bundle_id':'fixture'})
    return b

def test_upstream_function_is_unmodified_and_has_no_TVM_dependency():
    from selection_top20.audit import verify_upstream
    origin=verify_upstream()
    assert origin['function_start_line']==312 and origin['modifications'].startswith('none')
    code='import sys; from selection_top20.selector import seeded_pick; assert "tvm" not in sys.modules; assert seeded_pick([(1,),(2,)],[(1,)],1)==[1]'
    result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr

def test_seed_adapter_matches_independent_greedy_for_random_cases():
    rng=random.Random(20261108)
    for _ in range(100):
        vectors=[tuple(rng.randrange(5) for _ in range(7)) for _ in range(30)]
        seeds=[tuple(rng.randrange(5) for _ in range(7)) for _ in range(4)]
        covered=[{r[j] for r in seeds} for j in range(7)]
        remain=list(range(30));expected=[]
        for step in range(12):
            best=max(remain,key=lambda i:sum(vectors[i][j] not in covered[j] for j in range(7)))
            expected.append(best);remain.remove(best)
            for j in range(7):covered[j].add(vectors[best][j])
        assert seeded_pick(vectors,seeds,12)==expected
    assert seeded_pick([],[],0)==[]
    with pytest.raises(ValueError):seeded_pick([(1,)],[],2)

@pytest.mark.parametrize('masks,expected',[
    ([[0,1],[2,3]],'disjoint_open'),([[0,1],[0,1]],'overlap_open'),
    ([[0,1],[1,2]],'overlap_open'),([[0],[0],[1]],'overlap_open')])
def test_exclusive_execution_classes(masks,expected):
    assert execution_class('off',masks)==expected
    if masks==[[0,1],[1,2]]:
        with pytest.raises(ValueError):execution_class('resource_group',masks)
    else:assert execution_class('resource_group',masks)=='fixed_groups_serial'

def test_complete_domain_replay_selection_threshold_and_identity():
    from cpu_policies.open_resources.domain import contexts,canonical_tuple
    p,records,expand,new,coverage=complete()
    assert len(records)==201433 and coverage['eligible_count']==31889
    assert len(new)==len({r['id'] for r in new})==20
    old=api(FULL_PARENT).read(STATIC_SOURCE/'static_rankings.json')
    assert [r['id'] for r in new[:5]]==[r['id'] for r in old['pure_static20'][:5]]
    assert coverage['missing_contexts']==[] and len(coverage['context_coverage'])==9
    assert len(coverage['class_coverage'])==3 and coverage['feature_dimensions']==33
    for row in new:
        label=row['context_id']+':'+row['CPU_execution_class']
        assert row['score_ns']*100<=coverage['class_minima_ns'][label]*115
    again,_=choose(list(reversed(records)),records[:5],[c['id'] for c in contexts()],expand,fallback=records[:20])
    assert new==again
    for r in records[:20]:assert r['canonical']==canonical_tuple(expand(r))
    assert len({r['id'] for r in new+old['pure_static20']+old['diverse20']})==40

def test_fallback_uses_original_static_order_without_invented_candidates():
    p,records,expand,new,coverage=complete()
    anchors=records[:5]
    selected,proof=choose(anchors,anchors,['shared4'],expand,fallback=records[:20])
    assert [r['id'] for r in selected]==[expand(r)['id'] for r in records[:20]]
    assert all(r['selection_reason']=='pure_static_fallback' for r in selected[5:])

def test_features_have_fixed_slots_and_no_hash_or_FPS():
    ctx={'id':'shared4','host_mask':[0,1,2,3],'control_mask':[0,1,2,3]}
    value=features((('cpu',0,3),('vta',3,20),('cpu',20,21)),ctx,'off',
        [(0,3,2,(0,1),0),(20,21,1,(2,),2)])
    assert len(value)==33 and value[3]=='disjoint_open'
    assert value[15:25]==(-1,)*10

def test_ranker_performs_no_IO(monkeypatch):
    p,records,expand,new,coverage=complete()
    def forbidden(*a,**kw):raise AssertionError('selector attempted IO')
    monkeypatch.setattr('builtins.open',forbidden)
    selected,_=choose(records,records[:5],coverage['context_coverage'],expand,fallback=records[:20])
    assert selected==new

def test_component_replay_never_reads_pipeline_performance(tmp_path,monkeypatch):
    # This fast check exercises checkpoint replay and both original selectors.
    original=Path.read_text
    def guard(path,*args,**kwargs):
        if path.name in ('screening.json','analysis.json','performance.json') or '/accepted/' in str(path):
            raise AssertionError('ranking read pipeline observations '+str(path))
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',guard)
    result=static.build(output=tmp_path/'static',replay_components=False)
    provenance=json.loads((result/'source_provenance.json').read_text())
    assert provenance['same_original_lists'] and provenance['measurement_union']==40

def test_tasks_all_configurations_frontiers_modes_and_strict_windows(tmp_path):
    from selection_top20.runtime import pipeline_task,native_command,validate_document
    from selection_top20.campaign import tasks
    b=fixture_bundle(tmp_path);plan=tasks(b)
    assert len(plan)==80 and len({d['candidate_id'] for d in plan})==40
    for d in plan:
        argv,_=native_command(d,b,'/bundle','/job');cfg=d['configuration']
        assert '--process-cpu-affinity' not in argv and '--cpu-worker-pinning' in argv
        assert argv[argv.index('--control-cpu-affinity')+1]==','.join(map(str,cfg['control_mask']))
        assert argv[argv.index('--cpu-stage-gate')+1]==cfg['execution_mode']
        assert d['input_indices']==[0] and d['campaign']=='resnet18_autotvm_diverse_top20'
        assert (d['warmup'],d['scored'],d['tail'])==((0,1,0) if d['kind']=='qualify_pipeline' else (8,16,4))
        bad=deepcopy(d);bad['scored']+=1
        with pytest.raises(ValueError):validate_document(bad,b)
        bad=deepcopy(d);bad['executor_sha256']='wrong'
        with pytest.raises(ValueError):validate_document(bad,b)

def test_reserved_core_and_worker_mapping_rejected(tmp_path):
    from cpu_policies.open_resources.domain import validate_candidate
    _,_,_,new,_=complete();cfg=deepcopy(next(r for r in new if r['reserved_cores']))
    cpu=next(s for s in cfg['stages'] if s['device']=='cpu');cpu['mask']=[cfg['reserved_cores'][0]]
    with pytest.raises(ValueError):validate_candidate(cfg)

@pytest.mark.parametrize('bad',['missing_frame','duplicate_frame','wrong_output'])
def test_pipeline_raw_evidence_rejects_invalid_frames_and_outputs(tmp_path,bad):
    import numpy as np
    from selection_top20.runtime import pipeline_task,validate_summary
    c=api(FULL_PARENT);b=fixture_bundle(tmp_path)
    old_run=ROOT/'runs/resnet18_dp_frame_stream_top20_01'
    old_state=c.read(old_run/'state.json');old_tasks=c.read(old_run/'tasks.json')['pipeline']
    kind='qualify_pipeline' if bad=='wrong_output' else 'performance'
    pool=c.read(b/'plans/static_pool.json')
    old=next(d for d in old_tasks if d['kind']==kind and d['candidate_id'] in pool)
    receipt=c.read(old_run/'accepted'/old_state['boot_id']/(old['task_id']+'.json'))
    raw=tmp_path/'evidence';shutil.copytree(old_run/receipt['raw'],raw)
    shutil.copytree(FULL_PARENT/'references',b/'references')
    new=pipeline_task(b,old['candidate_id'],kind=='qualify_pipeline')
    validate_summary(raw,new,b)
    lines=(raw/'result.jsonl').read_text().splitlines()
    if bad=='missing_frame':lines.pop()
    elif bad=='duplicate_frame':lines[1]=lines[0]
    else:
        row=json.loads(lines[0]);meta=row['stage_raw_outputs'][0]['output_files'][0]
        output=raw/'outputs'/('run_'+str(row['completion_index']))/Path(meta['path']).name
        values=np.fromfile(output,dtype=meta['dtype']);values[0]=123 if values[0]!=123 else 124;values.tofile(output)
    (raw/'result.jsonl').write_text('\n'.join(lines)+'\n')
    with pytest.raises(ValueError):validate_summary(raw,new,b)

def test_receipt_resume_hash_replay_and_no_remeasurement(tmp_path,monkeypatch):
    from selection_top20 import campaign
    b=fixture_bundle(tmp_path);root=tmp_path/'run';root.mkdir();doc=campaign.tasks(b)[0]
    state={'bundle':str(b),'boot_id':'boot-A','frozen_files':{}}
    monkeypatch.setattr(campaign,'validate_summary',lambda *a:{'correct':True})
    class Transport:
        calls=0
        def measure(self,d,bundle,remote,root,boot):
            self.calls+=1;p=root/'raw'/boot/d['task_id']/'collected';p.mkdir(parents=True)
            (p/'result.jsonl').write_text('raw\n');return p
        def acknowledge(self,*a):pass
    t=Transport();rec=campaign.Recorder(root,state,t,'/board')
    assert rec.saved(doc)==rec.saved(doc) and t.calls==1
    next((root/'raw').rglob('result.jsonl')).write_text('tampered')
    with pytest.raises(ValueError):rec.saved(doc)

def test_disconnect_recollects_without_relaunch(tmp_path):
    from selection_top20.runtime import _ssh_class
    from selection_top20.campaign import tasks
    from cpu_vta_partitioning.transport import TransportError
    b=fixture_bundle(tmp_path);doc=tasks(b)[0];root=tmp_path/'run';root.mkdir();c=api(b)
    t=object.__new__(_ssh_class(str(b)));t.config={'remote_base':'/board'};t.boot_id=lambda:'boot-A'
    remote_hash='';launched=0;collected=0
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

def test_wrong_boot_rejected_before_deployment(tmp_path,monkeypatch):
    from selection_top20 import campaign,prepare
    state={'bundle':str(FULL_PARENT),'boot_id':'old','frozen_files':{},'status':'INTERRUPTED'}
    monkeypatch.setattr(campaign,'validate_state',lambda *a:state)
    monkeypatch.setattr(prepare,'validate_bundle',lambda *a:None)
    class Transport:
        def boot_id(self):return 'new'
        def wait_idle(self,*a):raise AssertionError('must reject before deployment')
    with pytest.raises(ValueError,match='boot'):campaign.run_current(tmp_path,Transport())
    assert api().read(tmp_path/'state.json')['status']=='STOPPED_INVALID_RESULT'

def test_first_five_budget_equality_and_twenty_budget_decisions():
    from selection_top20.analysis import evaluate
    old=[str(i) for i in range(20)];new=old[:5]+[str(i) for i in range(20,35)]
    values={str(i):{'ii_ms':100+i,'fps':1000/(100+i)} for i in range(35)}
    lists={'pure_static20':old,'diverse20':old,'coverage20':new}
    result=evaluate(lists,values);assert result['outcome']=='same_as_best_baseline'
    values['34']={'ii_ms':80,'fps':12.5}
    assert evaluate(lists,values)['outcome']=='improved_over_both'
    values['34']={'ii_ms':200,'fps':5};values['19']={'ii_ms':80,'fps':12.5}
    assert evaluate(lists,values)['outcome']=='no_additional_value'

def test_audit_uses_actual_binary_without_modifying_it(tmp_path):
    from selection_top20.audit import audit
    c=api(FULL_PARENT);path=FULL_PARENT/'runtime/libtvm_runtime.so';before=c.sha256(path)
    result=audit(FULL_PARENT,tmp_path/'audit')
    assert result['runtime_sha256']==before==c.sha256(path)
    assert result['binary_yield_path_present'] and not result['hotspot_or_causation_proven']

def test_new_protocol_does_not_change_old_controller_identity():
    from frame_stream_top20.common import identity
    assert identity()=='79ec84711558d87c6cacee21c79e59d16242fa577f6573f9d1d787431219bca9'
