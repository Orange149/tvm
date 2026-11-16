"""Branch selection, evidence scope, and independent fixed-family scoring."""
import io
from pathlib import Path
import socket
import subprocess
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT.parents[1]/'src'))

from cpu_policies.open_resources import domain
from policy_branch_eval.selection import fixed_family,mixed
from policy_branch_eval.analysis import budgets,validate_rows
from policy_branch_eval.common import offline_guard,isolated_inputs_only


def cfg(masks,gate='resource_group'):
    stages=[]
    for i,mask in enumerate(masks):
        if i: stages.append(domain.stage('vta',2*i-1,2*i,1,[0,1,2,3]))
        stages.append(domain.stage('cpu',2*i,2*i+1,len(mask),mask))
    return domain.candidate(stages,'shared4',gate)


@pytest.mark.parametrize('masks,expected',[([[0,1],[2,3]],True),
    ([[0,1],[2,3],[0,1]],True),([[0],[1],[1]],True),
    ([[0,1],[0,1]],False),([[0,1],[1,2]],False)])
def test_fixed_family_is_disjoint_partition_with_reuse(masks,expected):
    if expected or domain.compatible_groups(masks): value=cfg(masks)
    else: value=cfg(masks,'off');value['execution_mode']='resource_group'
    assert fixed_family(value)==expected
    assert not fixed_family(cfg(masks,'off'))


def rows(prefix,n): return [dict(id=prefix+str(i),score_ns=i) for i in range(n)]


def test_mix_order_and_quota():
    selected,reasons=mixed(rows('a',20),rows('b',20))
    assert [r['id'] for r in selected[:6]]==['a0','b0','a1','b1','a2','b2']
    assert sum(r['branch']=='fixed' for r in reasons)==10
    assert len({r['id'] for r in selected})==20


def test_identical_lists_keep_original_order():
    original=rows('a',20)
    assert mixed(original,original)[0]==original


def test_short_fixed_branch_fills_without_fabrication():
    result,_=mixed(rows('a',22),rows('b',2))
    assert len(result)==20
    assert sum(r['id'].startswith('b') for r in result)==2
    assert mixed([],[])[0]==[]


def test_overlap_deduplicates_without_hash_sorting():
    result,_=mixed(rows('a',20),list(reversed(rows('a',20))))
    assert len(result)==len({r['id'] for r in result})==20
    assert result[0]['id']=='a0' and result[1]['id']=='a19'


def test_incomplete_prefix_cannot_report_budget_fps():
    measured={'a0':dict(ii_ms=10,fps=100),'a2':dict(ii_ms=5,fps=200)}
    result=budgets(rows('a',20),measured)
    assert result['1']['best_FPS']==100
    assert result['5']['best_FPS'] is None
    assert result['5']['observed_subset_best_FPS']==200
    assert result['5']['covered']==2 and result['20']['missing'][-1]=='a19'


def test_cohorts_do_not_share_measurements():
    a=budgets(rows('a',20),{'a0':dict(ii_ms=10,fps=100)})
    b=budgets(rows('a',20),{'a1':dict(ii_ms=1,fps=1000)})
    assert a['1']['best_FPS']==100 and b['1']['best_FPS'] is None


def test_timestamp_count_is_one_less_than_retained_frames():
    task=dict(warmup=8,scored=16,tail=4)
    frames=[dict(frame_id=i,completion_index=i,input_index=0,completion_ms=i*4.) for i in range(28)]
    assert validate_rows(frames,task)==4.
    frames[10]['frame_id']=9
    with pytest.raises(ValueError,match='frame'): validate_rows(frames,task)


@pytest.mark.parametrize('which',['socket','ssh','shell'])
def test_offline_guard_blocks_remote_actions(tmp_path,which):
    with offline_guard(tmp_path) as ledger:
        with pytest.raises(RuntimeError):
            if which=='socket': socket.create_connection(('192.168.1.247',22))
            elif which=='ssh': subprocess.run(['ssh','192.168.1.247','true'])
            else: subprocess.run('echo forbidden',shell=True)
    assert ledger['network_attempts']+ledger['forbidden_process_attempts']==1


@pytest.mark.parametrize('leaf',['runs/old/result.jsonl','paper/my_paper/analysis/a.json','candidates.csv'])
def test_ranking_rejects_fps_files(tmp_path,leaf):
    p=tmp_path/leaf;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('old FPS')
    with isolated_inputs_only(),pytest.raises(RuntimeError,match='performance'):
        p.read_text()


def test_ranking_permits_isolated_components(tmp_path):
    p=tmp_path/'isolated_costs.json';p.write_text('{}')
    with isolated_inputs_only(): assert p.read_text()=='{}'


def test_fixed_vector_matches_frozen_full_formula_and_canonical_topk():
    from policy_branch_eval.ranking import inputs,engine_from
    from cpu_vta_partitioning.common import read
    from cpu_policies.open_resources.costs import Costs
    from yolo_cpu_policy_check.ranking import fixed_values
    programs,records,ddr,_,root,_=inputs('yolov3_tiny416')
    engine=engine_from(read(root/'proof.json'),'yolov3_tiny416')
    costs=Costs(records,programs=programs,ddr=ddr,distribution='session_weights')
    for ctx in (domain.context('shared4'),domain.context('host_shared_2'),domain.context('host_reserved_3')):
        table=engine._prepare(programs,costs,ctx,ddr)
        graph=('cpu_00_02','vta_03_06','cpu_07_08','vta_09_12','cpu_13_13')
        cpus=tuple(p for p in graph if table[p]['cpu'])
        alloc=engine._allocation_domain(tuple(ctx['compute_pool']),len(cpus))
        indices,values=fixed_values(engine,graph,table,ctx,alloc)
        full=engine._join_values(cpus,graph,table,ctx,alloc)
        np.testing.assert_array_equal(values,full['serial'][indices])
        chosen=engine._first_k(values,np.ones(len(values),dtype=bool),20)
        reference=sorted(range(len(values)),key=lambda i:(int(values[i]),int(indices[i])))[:20]
        assert list(chosen)==reference
