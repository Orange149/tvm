"""Frozen selection budgets, bridge semantics and board gating/recovery."""
from copy import deepcopy
from pathlib import Path
import pytest

from cpu_vta_partitioning.common import write
from cpu_policies.open_resources.domain import candidate, canonical_tuple, stage
from resnet50_ablation.common import MODELS
from resnet50_ablation.workflow import validate_lists
from resnet50_ablation.analysis import selection_budgets,effects
from resnet50_ablation import campaign


def fixture_lists():
    rows=[]
    for ctx in ('shared4','host_shared_0'):
        mask=[0,1,2,3] if ctx=='shared4' else [0]
        for a in range(4):
            for b in range(4):
                rows.append(candidate([stage('cpu',0,1,1,[a]),stage('vta',1,2,1,mask),
                                       stage('cpu',2,38,1,[b])],ctx,'resource_group'))
    rows=sorted(rows,key=canonical_tuple)[:20]
    lists={k:[{**r,'score_ns':100,'static_rank':i+1} for i,r in enumerate(rows)] for k in MODELS}
    bridges=[dict(previous_budget=b,original_static_rank=r,configuration=rows[r-1],
                  not_a_new_baseline_Top20=True) for b,r in ((1,1),(5,5),(20,11))]
    return lists,{r['id']:r for r in rows},bridges


def test_same_execution_only_once_and_bridges_are_separate():
    lists,pool,bridges=fixture_lists()
    assert validate_lists(lists,pool,bridges=bridges)
    assert len(pool)==20
    bad=deepcopy(lists);bad['fine_host'][0]['id']='changed'
    with pytest.raises(ValueError,match='identity'):validate_lists(bad,pool,bridges=bridges)
    with pytest.raises(ValueError,match='three fixed'):validate_lists(lists,pool,bridges=bridges[:2])
    bad=deepcopy(lists);bad['fine_host'][1]['score_ns']=99
    with pytest.raises(ValueError,match='canonical'):validate_lists(bad,pool,bridges=bridges)


def test_budget_prefixes_and_equal_selected_results_are_zero_effect():
    lists,pool,_=fixture_lists()
    ids=[r['id'] for r in lists['complete']]
    results={name:dict(II_ms=100.-i,FPS=1000/(100-i)) for i,name in enumerate(ids)}
    budgets=selection_budgets(lists,results)
    for r in budgets:
        assert r['static_rank']==r['budget']
    contribution,interaction=effects(budgets)
    assert all(r['FPS_gain_percent']==0 for r in contribution)
    assert all(abs(r['CPU_host_log_interaction'])<1e-14 for r in interaction)


def test_board_flag_rejected_before_any_io(tmp_path,monkeypatch):
    monkeypatch.setattr(campaign.runtime,'transport',lambda *a,**kw:pytest.fail('SSH is forbidden'))
    with pytest.raises(ValueError,match='execute-board'):
        campaign.run(tmp_path,tmp_path,tmp_path/'config')


def test_new_boot_is_rejected_before_deployment(tmp_path,monkeypatch):
    output=tmp_path/'output';run=tmp_path/'run';config=tmp_path/'local.json'
    write(config,{'board':{}});write(output/'CONTROLLERS.json',{'same':True})
    write(output/'ranking/FREEZE.json',{'same':True})
    write(run/'run.json',dict(boot_id='old',bundle_id='fixed',
        controller_sha256=campaign.sha256(output/'CONTROLLERS.json'),
        ranking_freeze_sha256=campaign.sha256(output/'ranking/FREEZE.json')))
    monkeypatch.setattr(campaign,'assert_controller',lambda *a:None)
    monkeypatch.setattr(campaign,'validate_bundle',lambda *a:{'bundle_id':'fixed'})
    class FakeSSH:
        def boot_id(self):return 'different'
    monkeypatch.setattr(campaign.runtime,'transport',lambda *a,**k:FakeSSH())
    monkeypatch.setattr(campaign,'deploy',lambda *a,**kw:pytest.fail('no deployment after boot change'))
    with pytest.raises(ValueError,match='immutable'):
        campaign.run(output,run,config,execute_board=True)


def test_actual_portable_capsule_adapter_no_inference(tmp_path,monkeypatch):
    """Real frozen native assets, synthetic five lists; --help only, zero DNN."""
    import shutil
    from cpu_vta_partitioning.common import read,sha256
    from cpu_policies.open_resources.domain import configuration
    from resnet50_ablation import workflow
    from resnet50_ablation.common import SOURCE,protocol
    from resnet50_ablation.ranking import INPUT_NAMES
    old=read(SOURCE/'ranking/rankings.json')['lists']
    lists={k:deepcopy(old['complete']) for k in MODELS}
    pool={r['id']:{**configuration(r),'id':r['id']} for r in old['complete']}
    bridges=[]
    for budget,rank in ((1,1),(5,5),(20,11)):
        r=old['tarnawski'][rank-1];cfg={**configuration(r),'id':r['id']};pool[r['id']]=cfg
        bridges.append(dict(previous_budget=budget,original_static_rank=rank,configuration=cfg,
                            not_a_new_baseline_Top20=True))
    output=tmp_path/'new';ranking=output/'ranking';docs={
        'rankings.json':{'lists':lists},'pool.json':pool,'scores.json':{},'bridges.json':bridges,
        'protocol.json':protocol(),'execution_order.json':sorted(pool),
        'proof.json':dict(exact_DP=True,pipeline_FPS_used=False,independent_global_enumeration=False,
                          small_domain_and_ResNet18_acceptance={'passed':True})}
    for name,value in docs.items():write(ranking/name,value)
    write(ranking/'FREEZE.json',dict(files={n:sha256(ranking/n) for n in docs},
        candidate_count=len(pool),pipeline_FPS_used=False))
    for name in INPUT_NAMES:
        target=ranking/'inputs'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(SOURCE/'ranking/inputs'/name,target)
    monkeypatch.setattr(workflow,'audit',lambda *a:None)
    monkeypatch.setattr(workflow,'require_acceptance',lambda *a:None)
    result=workflow.prepare(output)
    assert result['new_native_builds']==0 and result['candidates']==len(pool)
    bundle=output/'pipeline_bundle'
    assert workflow.validate_bundle(bundle)['candidate_count']==len(pool)
    relocated=tmp_path/'relocated';bundle.rename(relocated)
    assert workflow.validate_bundle(relocated)['bundle_id']==result['bundle_id']
    copied=next((relocated/'programs').glob('*/params.bin'))
    copied.write_bytes(copied.read_bytes()+b'corruption')
    with pytest.raises(Exception,match='hash|inventory|size|SHA|changed artifact'):
        workflow.validate_bundle(relocated)
