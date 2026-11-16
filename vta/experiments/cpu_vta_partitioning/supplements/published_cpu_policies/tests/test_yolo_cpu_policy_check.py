"""Family coverage, matched identities, unchanged scoring and exact recovery."""
from copy import deepcopy
import itertools
import json
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT.parents[1]/'src')]
from cpu_vta_partitioning.common import digest,read,write
from cpu_policies.open_resources import domain
from cpu_policies.open_resources.costs import Costs
from staged_ablation.model import score
from yolo_cpu_policy_check.ranking import load_engine,fixed_eligible,fixed_values,scan,gate_twin,block_key
from yolo_cpu_policy_check.common import protocol


@pytest.mark.parametrize('ctx',['shared4','host_shared_2','host_reserved_3'])
@pytest.mark.parametrize('family',['fixed','open'])
def test_full_small_family_vector_scalar_and_CPP_agree(tmp_path,ctx,family):
    from test_host_work_ablation import synthetic
    programs,records=synthetic();costs=Costs(records,programs=programs,distribution='session_weights')
    engine=load_engine();graphs=engine._topologies(programs,7,3);context=domain.context(ctx)
    best,proof=scan(engine,programs,costs,None,graphs,[context],tmp_path,family=family)
    masks=engine._masks(tuple(context['compute_pool']))
    eligible=[a for a in itertools.product(masks,repeat=4)
              if family=='open' or (len(set(a))>=2 and domain.compatible_groups(a))]
    assert proof['count']==len(eligible) and proof['verification_passed']
    assert best['score_ns']==score(best,costs,'fine_host')['score_ns']
    assert scan(engine,programs,costs,None,graphs,[context],tmp_path,family=family)[0]==best
    if family=='fixed':
        a=[tuple(s['mask']) for s in best['stages'] if s['device']=='cpu']
        assert len(set(a))>=2 and domain.compatible_groups(a)


def test_fixed_filter_excludes_one_group_and_partial_overlap():
    engine=load_engine();alloc=engine._allocation_domain((0,1,2,3),3);eligible=fixed_eligible(alloc)
    mapping={tuple(a):bool(v) for a,v in zip(alloc['allocations'],eligible)}
    index={m:i for i,m in enumerate(alloc['options'])}
    assert not mapping[(index[(0,1)],)*3]
    assert not mapping[(index[(0,1)],index[(1,2)],index[(3,)])]
    assert mapping[(index[(0,1,2)],index[(3,)],index[(0,1,2)])]


def test_every_vector_fixed_cost_matches_original_join_and_sample_scalar():
    from test_host_work_ablation import synthetic
    engine=load_engine();programs,records=synthetic();costs=Costs(records,programs=programs,distribution='session_weights')
    ctx=domain.context('shared4');graph=engine._topologies(programs,7,3)[0]
    table=engine._prepare(programs,costs,ctx,None);alloc=engine._allocation_domain((0,1,2,3),4)
    indices,values=fixed_values(engine,graph,table,ctx,alloc)
    full=engine._join_values(tuple(n for n in graph if table[n]['cpu']),graph,table,ctx,alloc)['serial']
    assert np.array_equal(values,full[indices])
    for pos in np.linspace(0,len(indices)-1,12,dtype=int):
        t=engine.Terminal(int(values[pos]),0,tuple(map(int,alloc['allocations'][indices[pos]])),ctx['id'],'resource_group')
        cfg=engine._configuration(t,[graph],programs,{ctx['id']:ctx})
        assert score(cfg,costs,'fine_host')['score_ns']==int(values[pos])


def test_gate_twin_preserves_every_placement_and_rebuilds_identity():
    cfg=domain.candidate([domain.stage('cpu',0,1,3,[0,1,2]),domain.stage('vta',1,2,1,[0,1,2,3]),
        domain.stage('cpu',2,3,1,[3]),domain.stage('vta',3,4,1,[0,1,2,3]),
        domain.stage('cpu',4,14,3,[0,1,2])],'shared4','resource_group')
    twin=gate_twin(cfg)
    assert cfg['stages']==twin['stages'] and block_key(cfg)==block_key(twin)
    assert cfg['id']!=twin['id'] and twin['core_partition']==[]
    domain.validate_candidate(twin,unit_count=14,max_vta_islands=3)


@pytest.fixture
def capsule(tmp_path):
    from yolo_validation.common import PROFILE
    cfg=domain.candidate([domain.stage('cpu',0,1,1,[0]),domain.stage('vta',1,13,1,[2]),
        domain.stage('cpu',13,14,2,[1,3])],'host_shared_2','resource_group')
    for leaf,value in [('pool.json',{cfg['id']:cfg}),('protocol.json',protocol()),
        ('programs.json',read(PROFILE/'plans/programs.json')),('inputs.json',read(PROFILE/'plans/inputs.json'))]:
        write(tmp_path/'plans'/leaf,value)
    write(tmp_path/'manifest.json',dict(bundle_id='fixture',runner_sha256='runner'))
    return tmp_path,cfg


@pytest.mark.parametrize('phase,runs,scored',[('qualify',1,1),('performance',104,64)])
def test_exact_native_windows_multiinput_and_no_extra_probes(capsule,phase,runs,scored):
    from yolo_cpu_policy_check.campaign import make_task,native_command
    bundle,cfg=capsule;task=make_task(bundle,cfg['id'],phase)
    argv,inputs=native_command(task,bundle,'/bundle','/job')
    assert task['scored']==scored and argv[argv.index('--runs')+1]==str(runs)
    assert argv[argv.index('--cpu-stage-gate')+1]=='resource_group'
    assert argv[argv.index('--stage2-input-sources')+1].count(',')>=1
    assert ('--resource-affinity-probe' in argv)==(phase=='qualify')
    assert '--queue-timing' not in argv and '--process-cpu-affinity' not in argv
    assert inputs=='/bundle/inputs/000.bin\n'
    bad=deepcopy(task);bad['configuration']['host_mask']=[0]
    with pytest.raises(ValueError):native_command(bad,bundle,'/bundle','/job')


def test_resume_never_relaunches_accepted_and_rejects_hash_boot(tmp_path,monkeypatch):
    from yolo_cpu_policy_check.campaign import accept
    monkeypatch.setitem(accept.__globals__,'summary',lambda *a:{'correct':True})
    doc={'task_id':'one','kind':'performance'};boot='boot-A'
    class Transport:
        calls=0
        def measure(self,*args):
            self.calls+=1;raw=tmp_path/'raw';raw.mkdir()
            (raw/'boot_id').write_text(boot);(raw/'task_hash').write_text(digest(doc))
            (raw/'result.jsonl').write_text('output');return raw
        def acknowledge(self,*args):pass
    t=Transport();a=accept(t,doc,'bundle','remote',tmp_path,boot)
    assert a==accept(t,doc,'bundle','remote',tmp_path,boot) and t.calls==1
    raw=Path(a['raw']);(raw/'result.jsonl').write_text('wrong')
    with pytest.raises(ValueError):accept(t,doc,'bundle','remote',tmp_path,boot)
    (raw/'result.jsonl').write_text('output');receipt=tmp_path/'accepted'/boot/'one.json'
    item=read(receipt);item['boot_id']='boot-B';write(receipt,item,replace=True)
    with pytest.raises(ValueError):accept(t,doc,'bundle','remote',tmp_path,boot)


def test_64_timestamps_are_63_intervals_and_no_reselection():
    from yolo_validation.pipeline import work_boundaries
    doc=dict(warmup=8,scored=64,tail=32,stages=[{},{}])
    rows=[dict(completion_ms=float((i+1)*10),stage0_end_ms=float((i+1)*10-1),stage1_end_ms=float((i+1)*10)) for i in range(104)]
    measured=work_boundaries(rows,doc)
    assert measured['interval_count']==63 and len(measured['output_intervals_ms'])==63
    assert protocol()['max_candidates']==10 and not protocol()['automatic_remeasurement']


def test_scoring_does_not_read_performance_files(monkeypatch):
    from test_host_work_ablation import synthetic
    engine=load_engine();programs,records=synthetic();costs=Costs(records,programs=programs,distribution='session_weights')
    ctx=domain.context('shared4');graph=engine._topologies(programs,7,3)[0]
    t=engine.Terminal(0,0,(0,1,0,1),ctx['id'],'resource_group')
    cfg=engine._configuration(t,[graph],programs,{ctx['id']:ctx})
    monkeypatch.setattr(Path,'open',lambda *a,**k:(_ for _ in ()).throw(AssertionError('unexpected performance file read')))
    assert score(cfg,costs,'fine_host')['pipeline_FPS_used'] is False


def test_cli_refuses_implicit_board_access():
    from run_yolo_cpu_policy_check import main
    with pytest.raises(SystemExit):main(['board','run'])


def test_interpretation_does_not_claim_serial_effect_without_group_reuse():
    from yolo_cpu_policy_check.analysis import compare
    r=compare(dict(ii_ms=200,fps=5),dict(ii_ms=180,fps=1000/180),dict(ii_ms=181,fps=1000/181),False)
    assert r['fixed_observed_faster'] and not r['gate_has_shared_group_object']
