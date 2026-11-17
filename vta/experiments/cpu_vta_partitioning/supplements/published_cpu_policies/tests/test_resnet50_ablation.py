"""Policy bounds, unchanged execution identities and independent small domains."""
from copy import deepcopy
import pytest

from cpu_policies.open_resources import domain
from resnet50_validation import search as original
from resnet50_validation.search_validation import _small
from resnet50_ablation.common import MODELS, settings
from resnet50_ablation.model import costs_for, score
from resnet50_ablation.search import rank, value
from resnet50_ablation.verification import selected_cpp, keys


@pytest.mark.parametrize('key', MODELS)
def test_label_scalar_subsets_and_suffix_bounds(key):
    n, programs, records, ddr = _small('unequal_work')
    costs = costs_for(key, records, programs, ddr)
    for ctx in domain.contexts():
        opts = original._options(programs, costs, ctx, 'complete')
        first = [o for o in opts if o.row['begin']==0 and o.row['end']==1 and o.row['device']=='cpu']
        middle = next(o for o in opts if o.row['begin']==1 and o.row['end']==4 and o.row['device']=='vta')
        last = [o for o in opts if o.row['begin']==4 and o.row['end']==5 and o.row['device']=='cpu']
        for a in first:
            for b in last:
                for gate in ('off', 'resource_group'):
                    if gate=='resource_group' and a.bits!=b.bits and a.bits & b.bits:
                        continue
                    label=original.ZERO_COMPLETE
                    prefix_bounds=[]
                    for opt in (a,middle,b):
                        label=original._advance(label,opt,'complete',gate)
                        prefix_bounds.append(value(label,ctx,gate,settings(key)))
                    cfg=domain.candidate([a.row,middle.row,b.row],ctx,gate)
                    expected=score(cfg,costs,key)['score_ns']
                    assert prefix_bounds[-1]==expected
                    assert max(prefix_bounds)<=expected


@pytest.mark.parametrize('key', MODELS)
def test_pruned_unpruned_cpp_and_original_module_unchanged(tmp_path,key):
    n,programs,records,ddr=_small('unequal_work')
    costs=costs_for(key,records,programs,ddr)
    contexts=[domain.context('host_reserved_3')]
    old_value=original._value
    new=rank(programs,costs,key=key,unit_count=n,contexts=contexts,output_dir=tmp_path/'new')
    unpruned=rank(programs,costs,key=key,unit_count=n,contexts=contexts,prune=False)
    assert keys(new['methods']['complete']['rankings'])==keys(unpruned['methods']['complete']['rankings'])
    selected_cpp({key:new['methods']['complete']['rankings']},programs,records,ddr,tmp_path/'cpp',unit_count=n)
    assert original._value is old_value


def test_checkpoint_rejects_policy_change_and_resumes(tmp_path):
    n,programs,records,ddr=_small('unequal_work')
    args=dict(unit_count=n,contexts=[domain.context('shared4')],output_dir=tmp_path)
    costs=costs_for('coarse_no_host',records,programs,ddr)
    partial=rank(programs,costs,key='coarse_no_host',max_expansions=5,**args)
    assert partial['status']=='SEARCH_BUDGET_EXCEEDED'
    with pytest.raises(ValueError,match='input or implementation'):
        rank(programs,costs_for('coarse_host',records,programs,ddr),key='coarse_host',resume=True,**args)
    resumed=rank(programs,costs,key='coarse_no_host',resume=True,**args)
    fresh=rank(programs,costs,key='coarse_no_host',unit_count=n,contexts=args['contexts'])
    assert keys(resumed['methods']['complete']['rankings'])==keys(fresh['methods']['complete']['rankings'])


def test_host_demand_zero_and_constraint_monotonicity():
    label=list(original.ZERO_COMPLETE);label[:4]=[10,20,30,40];label[4]=17;label[7]=11
    label[8:12]=[15,25,35,45]
    for ctx in domain.contexts():
        a=value(label,ctx,'off',settings('coarse_no_host'))
        b=value(label,ctx,'off',settings('coarse_host'))
        assert b>=a
        c=value(label,ctx,'off',settings('fine_host'))
        d=value(label,ctx,'off',settings('complete'))
        assert d>=c>=b
        zero=deepcopy(label);zero[4]=0
        assert value(zero,ctx,'off',settings('coarse_no_host'))==value(zero,ctx,'off',settings('coarse_host'))
