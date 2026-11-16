from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_scope_rank import ROOT,copy_matching_contexts,payload_for,checked_checkpoint
from run_scope_inputs import RunScopeCosts,SCOPE
from cpu_policies.open_resources.costs import Costs
from cpu_policies.open_resources.domain import stage,component,component_key,contexts
from cpu_vta_partitioning.common import read,write,digest
from policy_branch_eval.ranking import engine_from
from policy_branch_eval.common import isolated_inputs_only,offline_guard
from model_ablation.model import score


def fixture():
    programs={};records={};patches={}
    for device,begin,end in [('cpu',0,1),('vta',1,2),('cpu',2,3)]:
        name=f'{device}_{begin:02}_{end-1:02}'
        programs[name]=dict(name=name,device=device,begin=begin,end=end)
        if device=='cpu':
            items=[(n,list(range(n)),'shared4') for n in range(1,5)]
        else:
            items=[(1,ctx['host_mask'],ctx) for ctx in contexts()[:5]]
        for n,mask,ctx in items:
            cfg=component(stage(device,begin,end,n,mask),ctx);key=component_key(cfg)
            run=(8 if begin==0 else 6) if device=='cpu' else 1
            weights=([1] if n==1 else [.7]+[.3/(n-1)]*(n-1))
            records[key]=dict(configuration=cfg,kind='synthetic',
                cost=dict(service_ms=run/n if device=='cpu' else 1,run_core_ms=run,boundary_core_ms=0),
                session_thread_weights=weights)
            if device=='cpu' and n>1:
                patches[key]=dict(status='reconstruction_available',original_weights=weights,
                    estimated_thread_CPU_ns=[1]*n,all_recorded_run_CPU_ns=n,threads=n,
                    source_receipt='isolated.synthetic',source_boot='synthetic',scope=SCOPE)
    patch=dict(original_records_digest=digest(records),CPU_multithread_entries=patches,scope=SCOPE)
    return programs,records,patch


def test_full_search_oracle_and_scalar_consume_scoped_weights(tmp_path):
    proof=read(ROOT/'artifacts/model_ablation_01/complete/proof.json')
    module=engine_from(proof,'scope_rank_fixture')
    programs,records,patch=fixture();ctx=contexts()[0]
    old=Costs(records,programs=programs,distribution='session_weights')
    new=RunScopeCosts(records,patch,programs=programs)
    results={}
    with offline_guard(tmp_path),isolated_inputs_only():
        for name,costs in [('old',old),('new',new)]:
            value=module.rank_all(programs,costs,output_dir=tmp_path/name,
                unit_count=3,max_vta_islands=1,contexts=[ctx],verify=True)
            assert value['verification']['passed']
            for row in value['pure_static20']:
                assert score(row,costs,'complete')['score_ns']==row['score_ns']
            results[name]=value
    assert results['old']['counts']==results['new']['counts']
    assert results['old']['pure_static20'][0]['score_ns']>results['new']['pure_static20'][0]['score_ns']
    graphs=module._topologies(programs,3,1)
    assert len(copy_matching_contexts(module,programs,old,None,graphs,tmp_path/'old',tmp_path/'copy'))==1
    assert not copy_matching_contexts(module,programs,new,None,graphs,tmp_path/'old',tmp_path/'miss')


def test_corrupted_checkpoint_cannot_be_reused(tmp_path):
    proof=read(ROOT/'artifacts/model_ablation_01/complete/proof.json')
    module=engine_from(proof,'scope_rank_fixture_corruption')
    path=tmp_path/'bad.json'
    cp=dict(checkpoint_sha256='wrong',input_sha256='not-a-match')
    write(path,cp)
    with pytest.raises(ValueError,match='content hash'):
        checked_checkpoint(module,path,{'unused':True})
