from copy import deepcopy
from itertools import product
import pytest

from cpu_strategy.policy.domain import *
from cpu_strategy.policy.search import Costs,evaluate,policy_selection,rank_dp,enumerate_reference,freeze_rule


def programs():
    p={}
    for a,b in CUTS.values():
        for d,lo,hi in [('cpu',0,a),('vta',a,b),('cpu',b,21)]:
            s=stage(d,lo,hi,1,(0,));p[s['program']]={k:s[k] for k in ('device','begin','end')}
    return p


def synthetic(config):
    s=config['stages'][0];n=s['threads'];pin=config['affinity_mode']=='pinned'
    base=180. if s['begin']==0 else 3. if s['begin']==20 else 120.
    wall=(base/n)*(.88 if pin else 1.) if s['device']=='cpu' else 66.
    work=base*(.9 if pin else 1.) if s['device']=='cpu' else 14.
    return {'configuration':config,'cost':{'service_ms':wall+.75,'run_ms':wall,'run_core_ms':work,
             'boundary_core_ms':.45,'set_ms':.5,'get_ms':.25,'set_core_ms':.3,'get_core_ms':.15}}


def records():
    return {k:synthetic(c) for k,c in policy_components().items()}


def test_partitions_complete_and_configuration_identity_tracks_execution():
    assert len(list(partitions(CORES)))==15 and len(list(partitions(CORES[:3])))==5
    assert len(list(partitions(CORES,True)))==8
    c=cut_candidate('current',(2,2),((0,1),(2,3)))
    pin=cut_candidate('current',(2,2),((0,1),(2,3)),'pinned')
    gate=cut_candidate('current',(2,2),((0,1),(2,3)),'group','resource_group')
    assert len({c['id'],pin['id'],gate['id']})==3
    with pytest.raises(ValueError,match='slot'):cut_candidate('current',(1,2),(CORES,(0,1)),'pinned')
    with pytest.raises(ValueError,match='overlap'):cut_candidate('current',(3,2),((0,1,2),(0,1)),'group','resource_group')


def test_policy_pool_uses_three_workloads_and_caps_real_measurements():
    s=policy_selection(records())
    assert len(s['pool'])<=26 and set(s['families'])==set(CUTS)
    assert all(set(FAMILIES)<=set(x) for x in s['families'].values())
    assert all('pinning'in p and 'host_placement'in p for p in s['pairs'].values())
    for c in s['pool'].values():validate_candidate(c)


@pytest.mark.parametrize('method',['single_core','prefix','enhanced'])
@pytest.mark.parametrize('k',[1,5,20])
def test_full_21_unit_three_topology_dp_matches_independent_mapping_enumeration(method,k):
    p=programs();r=records();rule={'affinity_mode':'group','reserved_host':False,'homogeneous':True}
    a=rank_dp(p,r,rule,method,k);b=enumerate_reference(p,r,rule,method,k)
    assert [(v['id'],v['score_ms']) for v in a['rankings']]==[(v['id'],v['score_ms']) for v in b['rankings']]
    assert not a['ranking_uses_pipeline_feedback']


@pytest.mark.parametrize('method',['single_core','prefix','enhanced'])
@pytest.mark.parametrize('reserved,homogeneous',[(True,True),(False,False)])
def test_pinned_reserved_and_nonhomogeneous_domains_match(method,reserved,homogeneous):
    p=programs();r=records()
    for key,c in complete_components(p,'pinned',reserved,homogeneous).items():r[key]=synthetic(c)
    rule={'affinity_mode':'pinned','reserved_host':reserved,'homogeneous':homogeneous}
    a=rank_dp(p,r,rule,method);b=enumerate_reference(p,r,rule,method)
    assert [(v['id'],v['score_ms']) for v in a['rankings']]==[(v['id'],v['score_ms']) for v in b['rankings']]


def test_identical_core_groups_accumulate_service_and_vta_is_single_resource():
    r=records();costs=Costs(r)
    c=cut_candidate('current',(4,4),(CORES,CORES),'group','resource_group')
    v=evaluate(c,costs)
    assert v['group_services_ms']['0,1,2,3']==pytest.approx(180/4+.75+120/4+.75)
    assert v['cpu_core_ms']==300.
    assert v['host_core_ms']==14.
    assert v['boundary_core_ms']==pytest.approx(1.35)
    assert v['score_ms']>=v['group_services_ms']['0,1,2,3']


def test_multiple_vta_islands_and_cpu_devices_sum_instead_of_max():
    rows=[stage('cpu',0,3,1,(0,)),stage('vta',3,8,1,CORES),stage('cpu',8,12,1,(0,)),
          stage('vta',12,20,1,CORES),stage('cpu',20,21,1,(1,))]
    c=candidate(rows,'group','resource_group');r={}
    for s in rows:
        x=component(s);r[component_key(x)]={'configuration':x,'cost':{'service_ms':10.,'run_core_ms':4.,'boundary_core_ms':1.}}
    value=evaluate(c,Costs(r))
    assert value['vta_service_sum_ms']==20 and value['group_services_ms']['0']==20
    assert value['boundary_core_ms']==5 and value['score_ms']==20


def test_pinned_boundary_cpu_work_belongs_to_the_caller_core():
    c=cut_candidate('current',(2,2),((0,1),(2,3)),'pinned','resource_group');r={}
    for s in c['stages']:
        x=component(s,'pinned');r[component_key(x)]={'configuration':x,'cost':{
            'service_ms':10.,'run_core_ms':8. if s['device']=='cpu' else 0.,'boundary_core_ms':4. if s['device']=='cpu' else 0.}}
    v=evaluate(c,Costs(r))
    assert v['subset_capacity_ms'][0]==4.  # core0: first-stage boundary caller
    assert v['subset_capacity_ms'][3]==4.  # core2: tail boundary caller
    assert v['subset_capacity_ms'][2]==6.  # cores0,1: computation plus one boundary
    assert v['boundary_core_ms']==8.


def test_wrong_signature_missing_profile_and_changed_source_are_not_filled():
    r=records();c=next(iter(r));r.pop(c)
    with pytest.raises(ValueError,match='missing'):policy_selection(r)
    c=cut_candidate('current',(1,1),((1,),(0,)))
    c['affinity_mode']='pinned'
    with pytest.raises(ValueError):validate_candidate(c)


def test_policy_choice_requires_material_increment_over_simple_rule():
    s=policy_selection(records());base={k:100. for k in s['pool']}
    for cut,names in s['families'].items():
        base[names['current']]=120.;base[names['fixed_groups']]=100.
        base[names['same_groups_pinned']]=98.
        base[names['pinned_reallocated']]=97.;base[names['reserved_host']]=102.
    chosen=freeze_rule(s,base,base)
    assert chosen['family']=='fixed_groups'
    for names in s['families'].values():base[names['reserved_host']]=85.
    assert freeze_rule(s,base,base)['reserved_host']
