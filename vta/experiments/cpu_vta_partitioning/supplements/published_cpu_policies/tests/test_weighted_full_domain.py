from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from run_resource_behavior import api,PARENT,SOURCE
from cpu_policies.open_resources.domain import stage,candidate
from cpu_policies.open_resources.costs import Costs
from resource_envelope.model import score
import pytest

def test_isolated_thread_shares_change_host_helper_choice_without_pipeline_input(monkeypatch):
    c=api();records=c.read(SOURCE/'isolated_costs.json')['costs'];programs=c.read(PARENT/'plans/programs.json')
    ddr=c.read(SOURCE/'ddr.json');costs={k:Costs(records,programs=programs,ddr=ddr,distribution=k)
        for k in ('balanced','session_weights')}
    def forbid(*args,**kwargs):raise AssertionError('score read a file')
    monkeypatch.setattr('builtins.open',forbid)
    values={}
    for host in (1,2):
        rows=[stage('cpu',0,3,3,[0,1,2]),stage('vta',3,5,1,[host]),
            stage('cpu',5,8,1,[3]),stage('vta',8,20,1,[host]),stage('cpu',20,21,1,[0])]
        cfg=candidate(rows,'host_shared_'+str(host),'off')
        for mode,table in costs.items():values[(mode,host)]=score(cfg,table)['combined']
    assert values['balanced',1]<values['balanced',2]
    assert values['session_weights',2]<values['session_weights',1]

def test_weighted_original_reuses_unchanged_independent_scalar_oracle(tmp_path):
    from resource_envelope.weighted_search import original_engine
    engine=original_engine(tmp_path);folder=Path(engine.__file__).parent
    assert (folder/'oracle.cc').read_bytes()==(ROOT/'cpu_policies/open_resources/oracle.cc').read_bytes()
    text=(folder/'search.py').read_text()
    assert 'compact terminal strata remain available' in text
    assert 'team_core' not in text and 'reservation' not in text

def test_weighted_family_retains_distinct_helper_roles():
    from resource_envelope.selection import behavior_key,profile_scales
    c=api();programs=c.read(PARENT/'plans/programs.json')
    costs=Costs(c.read(SOURCE/'isolated_costs.json')['costs'],programs=programs,ddr=c.read(SOURCE/'ddr.json'))
    scales=profile_scales(programs,costs);configs=[]
    for host in (1,2):
        rows=[stage('cpu',0,3,3,[0,1,2]),stage('vta',3,20,1,[host]),stage('cpu',20,21,1,[3])]
        configs.append(candidate(rows,'host_shared_'+str(host),'off'))
    a,b=configs
    assert behavior_key(a,scales)==behavior_key(b,scales)
    assert behavior_key(a,scales,preserve_worker_roles=True)!=behavior_key(b,scales,preserve_worker_roles=True)

def test_search_model_settings_and_labels_are_explicit():
    from resource_envelope.prepare_campaign import model_settings,campaign_protocol
    for variant in ('original','combined'):
        settings=model_settings({'cpu_distribution':'session_weights','variant':variant})
        protocol=campaign_protocol(settings)
        assert protocol['cpu_distribution']=='session_weights'
        assert protocol['score_variant']==variant
        assert '逐线程' in protocol['method_labels']['diverse20']
        assert ('包络' in protocol['method_labels']['diverse20'])==(variant=='combined')
    with pytest.raises(ValueError):model_settings({'cpu_distribution':'pipeline_weights'})
