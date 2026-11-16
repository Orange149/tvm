"""Three-method transfer, exact windows and immutable selection/recovery."""
from copy import deepcopy
from pathlib import Path
import json
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parents[1]/'src'))
from yolo_validation.common import METHODS,protocol,frozen,PROFILE
from yolo_validation.pipeline import make_task,native_command,work_boundaries,select_winners,accept
from cpu_vta_partitioning.common import digest,read,write
from cpu_policies.open_resources import domain

@pytest.fixture
def capsule(tmp_path):
    cfg=domain.candidate([domain.stage('cpu',0,1,2,[0,1]),domain.stage('vta',1,13,1,[2]),
        domain.stage('cpu',13,14,2,[2,3])],'host_shared_2','off')
    for name,value in [('programs.json',read(PROFILE/'plans/programs.json')),
                       ('inputs.json',read(PROFILE/'plans/inputs.json')),('pool.json',{cfg['id']:cfg}),('protocol.json',protocol())]:
        write(tmp_path/'plans'/name,value)
    write(tmp_path/'manifest.json',dict(bundle_id='fixture',runner_sha256='runner'))
    return tmp_path,cfg

@pytest.mark.parametrize('phase,total,scored',[('qualify',1,1),('screen',28,16),('confirm',104,64)])
def test_command_has_exact_window_full_frontier_and_no_new_probes(capsule,phase,total,scored):
    b,cfg=capsule;doc=make_task(b,cfg['id'],phase)
    argv,inputs=native_command(doc,b,'/bundle','/job')
    assert doc['scored']==scored and argv[argv.index('--runs')+1]==str(total)
    assert '--queue-timing' not in argv and '--process-cpu-affinity' not in argv
    assert ('--resource-affinity-probe' in argv)==(phase=='qualify')
    assert argv[argv.index('--stage2-input-sources')+1].count(',')>=1
    assert inputs=='/bundle/inputs/000.bin\n'
    assert len(read(b/'plans/programs.json')[cfg['stages'][-1]['program']]['output_node_ids'])==8
    bad=deepcopy(doc);bad['executor_sha256']='tampered'
    with pytest.raises(ValueError):native_command(bad,b,'/bundle','/job')

def test_task_window_and_phase_separate_identity_without_changing_execution(capsule):
    b,cfg=capsule;docs=[make_task(b,cfg['id'],p) for p in ('qualify','screen','confirm')]
    assert len({d['task_id'] for d in docs})==3
    assert docs[0]['configuration']==docs[1]['configuration']==docs[2]['configuration']

def test_window_work_counts_seven_or_fifteen_gaps_and_no_interval_deletion():
    doc=dict(warmup=8,scored=16,stages=[{},{}])
    rows=[dict(completion_ms=100*(i+1),stage0_end_ms=100*(i+1)-1,stage1_end_ms=100*(i+1)) for i in range(28)]
    row=work_boundaries(rows,doc)
    assert row['interval_count']==15 and len(row['output_intervals_ms'])==15
    assert row['stages'][0]['completed_between']==15
    rows[14]['stage0_end_ms']=0
    with pytest.raises(ValueError):work_boundaries(rows,doc)

def test_winners_fixed_by_screen_not_later_performance_and_no_extra_points():
    lists={key:[dict(id=str(i)) for i in range(20)] for key in METHODS}
    screen={str(i):dict(ii_ms=100-i,fps=1000/(100-i)) for i in range(20)}
    before=deepcopy(screen);result=select_winners(lists,screen)
    assert result['candidates']==['0','19','4'] and screen==before
    assert result['methods']['fine_host']['1']['candidate_id']=='0'
    screen.pop('18')
    with pytest.raises(ValueError):select_winners(lists,screen)

def test_screen_ties_follow_frozen_static_order():
    lists={key:[dict(id=str(i)) for i in range(20)] for key in METHODS}
    screen={str(i):dict(ii_ms=1.,fps=1000.) for i in range(20)}
    assert select_winners(lists,screen)['candidates']==['0']

def test_acceptance_resume_no_repeat_and_hash_boot_checks(tmp_path,monkeypatch):
    import yolo_validation.pipeline as p
    monkeypatch.setattr(p,'summary',lambda *args:{'correct':True})
    doc=dict(task_id='one',kind='performance');boot='boot-A'
    class Transport:
        calls=0
        def measure(self,*args):
            self.calls+=1;raw=tmp_path/'raw';raw.mkdir()
            (raw/'boot_id').write_text(boot);(raw/'task_hash').write_text(digest(doc));(raw/'result.jsonl').write_text('valid')
            return raw
        def acknowledge(self,*args):pass
    t=Transport()
    assert accept(t,doc,'bundle','remote',tmp_path,boot)==accept(t,doc,'bundle','remote',tmp_path,boot)
    assert t.calls==1
    (tmp_path/'raw/result.jsonl').write_text('tampered')
    with pytest.raises(ValueError):accept(t,doc,'bundle','remote',tmp_path,boot)

def test_immutable_freeze_rejects_new_winner_or_boot(tmp_path):
    p=tmp_path/'seal.json';frozen(p,{'boot_id':'A','winners':['a']});frozen(p,{'boot_id':'A','winners':['a']})
    with pytest.raises(ValueError):frozen(p,{'boot_id':'B','winners':['a']})
    with pytest.raises(ValueError):frozen(p,{'boot_id':'A','winners':['b']})

@pytest.mark.parametrize('key',METHODS)
def test_frozen_formula_scalar_vector_and_independent_CPP_small_domain(tmp_path,key):
    from staged_ablation.search import engine,method
    from staged_ablation.model import score
    from cpu_policies.open_resources.costs import Costs
    from test_host_work_ablation import synthetic
    programs,records=synthetic()
    costs=Costs(records,programs=programs,distribution='session_weights' if key=='fine_host' else 'balanced')
    module=engine(tmp_path,key)
    ranked=module.rank_all(programs,costs,unit_count=7,max_vta_islands=3,
        contexts=[domain.context('shared4'),domain.context('host_reserved_3')],output_dir=tmp_path/'search',verify=True)
    assert ranked['verification']['passed']
    for cfg in ranked['methods'][method(key)]['rankings']:
        result=score(cfg,costs,key)
        assert result['score_ns']==cfg['score_ns']
        assert 'host_concentration' not in result['terms_ns']
        assert ('team_service' in result['terms_ns'])==(key=='fine_host')
        assert (result['scored_host_cpu_ns']==0)==(key=='tarnawski')

def test_nominal_MACs_count_all_units_not_resnet_constant():
    from yolo_validation.analysis import nominal_macs
    macs=nominal_macs()
    assert len(macs)==14 and sum(v>0 for v in macs)==13 and macs[-1]==0
    assert sum(macs)!=1814073344

def test_raw_outputs_hash_guard_covers_eight_and_duplicate_frames():
    from yolo_transfer.pipeline import check_outputs
    from cpu_vta_partitioning.measurement import frames
    expected={str(i):dict(shape=[1],dtype='float32',nbytes=4,fnv1a64=str(i)) for i in range(8)}
    actual=[dict(index=i,**expected[str(i)]) for i in range(8)]
    check_outputs(actual,dict(output_node_ids=list(range(8))),expected)
    actual[7]['fnv1a64']='bad'
    with pytest.raises(ValueError):check_outputs(actual,dict(output_node_ids=list(range(8))),expected)

def test_import_preserves_original_boot_and_rejects_repaired_module_or_hash_change(tmp_path,monkeypatch):
    import yolo_validation.evidence as evidence
    import yolo_transfer.profile as profile
    from cpu_vta_partitioning.common import inventory,sha256
    assets=tmp_path/'assets';old=assets/'profile_bundle_bridge02';current=tmp_path/'current'
    monkeypatch.setattr(evidence,'ASSETS',assets)
    doc=dict(kind='calibration',component_key='key',bundle_id='old',configuration={'stages':[{'program':'cpu_00_00'}]},
        stages=[{'program':'cpu_00_00'}],environment={'poll_sleep_ns':1000},warmup=2,scored=8,tail=0,
        input_indices=[0],executor_sha256='runner',affinity_probe='startup',expected_outputs='ARM')
    doc['task_id']=digest(doc)[:24]
    monkeypatch.setattr(profile,'make_task',lambda *a:{**doc,'bundle_id':'current'})
    for b in (old,current):
        for leaf in ('graph.json','module.so','params.bin'):
            p=b/'programs/cpu_00_00'/leaf;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(leaf)
        write(b/'plans/inputs.json',{'input':0});write(b/'plans/expected_outputs.json',{'all':8})
        write(b/'manifest.json',dict(bundle_id='old' if b==old else 'current',files={}))
    source=tmp_path/'old_run';boot='original-boot'
    write(source/'run.json',dict(boot_id=boot,bundle_id='old'))
    raw=source/'raw';raw.mkdir();(raw/'boot_id').write_text(boot)
    result={'cost':{'service_ms':1.}}
    monkeypatch.setattr(profile,'summary',lambda *a:result)
    path=source/'accepted'/boot/(doc['task_id']+'.json')
    write(path,dict(task=doc,boot_id=boot,raw=str(raw),files=inventory(raw),summary=result))
    rec=evidence.receipt_record(current,'key',path)
    assert rec['source']['boot_id']==boot and rec['source']['task_id']==doc['task_id']
    assert evidence.receipt_record(current,'key',path,expected_record=rec)==rec
    bad=deepcopy(rec);bad['source']['receipt_sha256']='bad'
    with pytest.raises(ValueError,match='hash'):evidence.receipt_record(current,'key',path,expected_record=bad)
    (current/'programs/cpu_00_00/module.so').write_text('superseded')
    with pytest.raises(ValueError,match='module'):evidence.receipt_record(current,'key',path)
