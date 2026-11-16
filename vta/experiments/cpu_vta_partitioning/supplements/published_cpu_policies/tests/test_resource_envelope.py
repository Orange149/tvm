from copy import deepcopy
from itertools import product
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from run_resource_behavior import api,PARENT,SOURCE,configurations
from resource_envelope.model import score
from cpu_policies.open_resources.costs import Costs

def test_host_simplex_guard_is_exact_maximum_of_extreme_points():
    for demand in ([1,2,3,4],[10,0,5,7]):
        for allowed in ([0],[0,1,2,3],[1,3]):
            H=11
            robust=max(demand[c]+(H if c in allowed else 0) for c in range(4))
            vertices=max(max(demand[c]+(H if c==h else 0) for c in range(4)) for h in allowed)
            assert robust==vertices

def test_costs_and_fields_not_mutated_fixed_host_gated_matches_original():
    c=api();records=c.read(SOURCE/'isolated_costs.json')['costs']
    costs=Costs(records,programs=c.read(PARENT/'plans/programs.json'),ddr=c.read(SOURCE/'ddr.json'))
    for name,cfg in configurations().items():
        old=deepcopy(cfg);value=score(cfg,costs)
        assert cfg==old and value['combined']==max(value[k] for k in ('original','host_concentration','team_occupancy'))
        if len(cfg['host_mask'])==1:assert value['host_concentration']==value['original']
        if cfg['execution_mode']=='resource_group':assert value['team_occupancy']==value['original']
        assert value['pipeline_FPS_used'] is False and value['planning_proxy_not_a_latency_bound']

def test_prediction_never_opens_pipeline_observations(monkeypatch):
    c=api();cfg=configurations()['static_first']
    costs=Costs(c.read(SOURCE/'isolated_costs.json')['costs'],
        programs=c.read(PARENT/'plans/programs.json'),ddr=c.read(SOURCE/'ddr.json'))
    def forbidden(*a,**kw):raise AssertionError('scoring must not perform file IO')
    monkeypatch.setattr('builtins.open',forbidden)
    result=score(cfg,costs)
    assert result['original']==68320857 and result['host_concentration']>result['original']
