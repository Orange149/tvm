from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from run_resource_behavior import api,PARENT,SOURCE
from resource_envelope.selection import profile_scales,behavior_key,select
from cpu_policies.open_resources.costs import Costs
from cpu_policies.open_resources.domain import candidate,stage

@lru_cache(maxsize=1)
def facts():
    c=api();p=c.read(PARENT/'plans/programs.json')
    costs=Costs(c.read(SOURCE/'isolated_costs.json')['costs'],programs=p,ddr=c.read(SOURCE/'ddr.json'))
    old=c.read(SOURCE/'static_rankings.json')
    return p,costs,profile_scales(p,costs),old

def test_fifteen_light_tail_variants_are_one_profile_family():
    _,_,scales,old=facts()
    assert len({behavior_key(r,scales) for r in old['pure_static20'][:15]})==1

def test_main_stage_thread_budget_and_overlap_remain_distinct():
    _,_,scales,old=facts();base=old['pure_static20'][0]
    rows=deepcopy(base['stages']);rows[0]=stage('cpu',0,3,3,[0,1,2]);rows[2]=stage('cpu',5,8,1,[3])
    separate=candidate(rows,'shared4','off')
    assert behavior_key(base,scales)!=behavior_key(separate,scales)
    serial=candidate(rows,'shared4','resource_group')
    assert behavior_key(serial,scales)!=behavior_key(separate,scales)

def test_host_caller_helper_and_reserved_positions_preserved():
    _,_,scales,old=facts();rows=deepcopy(old['pure_static20'][0]['stages'])
    rows[0]=stage('cpu',0,3,3,[0,1,2]);rows[2]=stage('cpu',5,8,1,[3])
    def placed(h):
        r=deepcopy(rows)
        for s in r:
            if s['device']=='vta':s['mask']=[h]
        return candidate(r,'host_shared_'+str(h),'resource_group')
    assert behavior_key(placed(1),scales)==behavior_key(placed(2),scales)
    assert behavior_key(placed(0),scales)!=behavior_key(placed(2),scales)
    assert behavior_key(placed(3),scales)!=behavior_key(placed(2),scales)

def test_ranker_no_observation_file_access(monkeypatch):
    _,_,scales,old=facts()
    def forbidden(*a,**kw):raise AssertionError('behavior scoring accessed files')
    monkeypatch.setattr('builtins.open',forbidden)
    value=behavior_key(old['pure_static20'][0],scales)
    assert isinstance(value,tuple)

def test_source_adapter_and_scalar_envelope_agree_on_small_complete_domain(tmp_path):
    from resource_envelope.search_adapter import generated
    from resource_envelope.model import score
    from cpu_policies.open_resources.domain import contexts
    engine=generated(tmp_path);programs={}
    # Exercise generated vector joins without requiring physical runtime files.
    from cpu_policies.open_resources.search import _allocation_domain,_half_vectors
    table={'cpu_a':{'cpu':True,'costs':[[10,12,1,2,1,0,0,0],[7,13,1,2,1,1,0,0]]},
           'cpu_b':{'cpu':True,'costs':[[8,9,1,1,1,0,0,0],[5,10,1,1,1,1,0,0]]},
           'vta':{'cpu':False,'costs':[[6,3,2,4]]}}
    ctx={'compute_pool':[0,1],'host_mask':[0,1]};alloc=engine._allocation_domain((0,1),2)
    # One thread has two physical options and two threads one option.
    for name in ('cpu_a','cpu_b'):
        t=table[name]['costs'];table[name]['costs']=[t[0],t[0],t[1]]
    values=engine._join_values(('cpu_a','cpu_b'),('cpu_a','vta','cpu_b'),table,ctx,alloc)
    options=alloc['options']
    for index,allocation in enumerate(alloc['allocations']):
        core=[0]*4;reservation=[0]*4;service=6;DDR=4
        for name,choice in zip(('cpu_a','cpu_b'),allocation):
            mask=options[choice];s,u,b,d,*weights=table[name]['costs'][choice]
            parts=engine._allocate_integer(u,weights[:len(mask)])
            for c,part in zip(mask,parts):core[c]+=part;reservation[c]+=s
            core[mask[0]]+=b;service=max(service,s);DDR+=d
        expected=max(service,DDR,max(core[c]+(5 if c in (0,1) else 0) for c in range(4)),max(reservation))
        assert int(values['enhanced'][index])==expected

def test_generated_controller_uses_external_run_directory(tmp_path):
    from resource_envelope.prepare_campaign import _controller
    p=_controller(tmp_path)
    text=(p/'envelope_campaign/cli.py').read_text()
    assert "root=args.run_root.resolve()/args.run_id" in text
    assert "root=ROOT/'runs'/args.run_id" not in text
    common=(p/'envelope_campaign/common.py').read_text()
    assert "'resource_envelope'" in common and 'RUN_ROOT=' in common
    import subprocess
    result=subprocess.run([sys.executable,'-c',
        'from envelope_campaign.common import identity; assert len(identity())==64; '
        'from envelope_campaign.analysis import evaluate; '
        'ids=[str(i) for i in range(20)]; '
        'result=evaluate({"pure_static20":ids,"diverse20":ids[::-1],"coverage20":ids[::-1]},'
        '{i:{"ii_ms":int(i)+1,"fps":1000/(int(i)+1)} for i in ids}); '
        'assert result["methods"]["diverse20"]["budgets"]["1"]["selected"]=="19"'],
        cwd=p,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr

def test_history_mirror_preserves_caller_host_relation():
    from resource_envelope.history_labels import mirror_key
    _,_,_,old=facts();base=old['pure_static20'][0]
    assert mirror_key(base)==mirror_key(deepcopy(base))
    cfg=deepcopy(base);cfg['host_mask']=[0]
    assert mirror_key(base)!=mirror_key(cfg)
