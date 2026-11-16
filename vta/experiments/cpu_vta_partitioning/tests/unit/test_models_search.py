import pytest
from cpu_vta_partitioning.domain import Stage, signatures, topologies, configurations
from cpu_vta_partitioning.models import Cost, score, load_costs
from cpu_vta_partitioning.search import rank, enumerate_reference, screening_pool, freeze_selection


def test_host_local_boundary_and_single_vta():
    stages=[Stage('cpu',0,1,1,(0,)), Stage('vta',1,2,1,(0,1,2,3)),
            Stage('cpu',2,3,1,(0,)), Stage('vta',3,4,1,(0,1,2,3)),Stage('cpu',4,21,1,(0,))]
    costs={s.signature:Cost(1,8 if s.device=='vta' else 2,1) for s in stages}
    result=score(stages,costs)
    assert result['R']==2  # both VTA stages accumulate on one device
    assert result['C']==pytest.approx(11/4)
    assert result['H']==pytest.approx(27/4)
    assert result['A']==6
    assert result['O']==pytest.approx(27/4)
    assert result['G']==9  # three CPU tasks own the same core including boundaries
    assert result['components']['boundary_core_ms']==5
    assert len(result['components']['capacity'])==15
    zero={k:Cost(v.service_ms,0 if s.device=='vta' else v.run_core_ms,v.boundary_core_ms)
          for s in stages for k,v in [(s.signature,costs[s.signature])]}
    assert score(stages,zero)['H']==result['C']
    assert score(stages,zero)['O']==result['A']


def test_dedicated_capacity():
    stages=[Stage('cpu',0,1,3,(0,1,2)),Stage('vta',1,20,1,(3,)),Stage('cpu',20,21,1,(0,))]
    costs={s.signature:Cost(1,6 if s.device=='vta' else 1,1) for s in stages}
    assert score(stages,costs)['D']==7


def test_rank_matches_independent_scalar_enumerator(programs,costs):
    actual=rank(programs,costs)
    expected=enumerate_reference(programs,costs)
    for g,rows in actual['rankings'].items():
        assert [(r['score_ms'],r['id']) for r in rows]==expected[g]
    assert len(screening_pool(actual))<=47
    values={key:{'boot1':1+index/100,'boot2':1+index/100} for index,key in enumerate(screening_pool(actual))}
    selections=freeze_selection(actual,values)
    assert len({c for budgets in selections.values() for c in budgets.values()})<=14


def test_costs_fail_closed(programs,costs):
    with pytest.raises(ValueError,match='fresh'):
        load_costs({'source':'synthetic'})
    with pytest.raises(ValueError,match='missing calibration'):
        rank(programs,{})
    with pytest.raises(ValueError):
        Cost.from_dict({'service_ms':float('nan'),'run_core_ms':1,'boundary_core_ms':0})


def test_no_tvm_or_ssh_import_dependency():
    import subprocess,sys,os
    code="import cpu_vta_partitioning.models, cpu_vta_partitioning.search, cpu_vta_partitioning.analysis; import sys; assert 'tvm' not in sys.modules; assert 'cpu_vta_partitioning.transport' not in sys.modules"
    subprocess.run([sys.executable,'-c',code],check=True,env={**os.environ,'PYTHONPATH':'src'})


def test_random_control_can_select_every_eligible_candidate_in_small_stratum():
    from cpu_vta_partitioning.common import PROJECT, read
    from cpu_vta_partitioning.domain import candidate_id
    programs = {p['name']:p for p in read(PROJECT/'tests/fixtures/small_domain.json')[:3]}
    costs = {key:Cost(1,1,.1) for key in signatures(programs)}
    first = rank(programs, costs, seed=0)
    excluded = {r['id'] for rows in first['rankings'].values() for r in rows[:5]}
    eligible = {candidate_id(c) for t in topologies(programs) for c in configurations(t,4,False)} - excluded
    assert len(eligible) == 5
    outcomes = []
    for seed in range(25):
        result = rank(programs, costs, seed=seed)
        assert result['rankings'] == first['rankings']
        sample = frozenset(r['id'] for r in result['stratified_random'])
        assert len(sample) == 4 and sample <= eligible
        outcomes.append(sample)
    assert set.union(*(set(x) for x in outcomes)) == eligible
    assert set.intersection(*(set(x) for x in outcomes)) == set()
    assert len(set(outcomes)) > 1
    assert first['stratified_random'] == rank(programs, costs, seed=0)['stratified_random']
