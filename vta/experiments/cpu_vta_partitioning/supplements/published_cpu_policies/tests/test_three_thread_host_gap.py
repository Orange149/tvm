from copy import deepcopy
from audit_three_thread_host_gap import target_pipeline,target_component
from run_three_thread_host_gap import RefreshedCosts
from cpu_policies.open_resources.domain import stage,component,component_key


def test_invalid_metadata_not_a_pipeline_match():
    assert not target_pipeline(None)
    assert not target_pipeline({'stages':[None]})
    assert not target_component({'configuration':None,'stages':[]})


def test_component_match_requires_exact_fixed_prefix_mapping():
    c=component(stage('cpu',0,3,3,[0,1,2]));t=dict(configuration=c,stages=c['stages'])
    assert target_component(t)
    t=deepcopy(t);t['stages'][0]['worker_core_map']=[2,0,1]
    assert not target_component(t)


def test_cost_overlay_keeps_original_and_separates_uniform_counterfactual():
    s=stage('cpu',0,3,3,[0,1,2]);key=component_key(component(s))
    original={key:dict(cost=dict(service_ms=5.,run_core_ms=9.,boundary_core_ms=0.),
                      session_thread_weights=[.5,.3,.2])}
    saved=deepcopy(original)
    a=RefreshedCosts(original,distribution='session_weights',overrides={'3':dict(service_ms=4.,run_core_ms=8.,boundary_core_ms=0.)})
    b=RefreshedCosts(original,distribution='session_weights',overrides={'3':dict(service_ms=4.,run_core_ms=8.,boundary_core_ms=0.)},uniform=True)
    assert a.get(s)['service_ms']==b.get(s)['service_ms']==4.
    assert a.get(s)['cpu_core_weights']==[.5,.3,.2]
    assert b.get(s)['cpu_core_weights']==[1/3]*3
    assert original==saved
