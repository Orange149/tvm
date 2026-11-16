from copy import deepcopy
import pytest
from cpu_policies.open_resources.domain import candidate,stage
from run_host_gap_intervention import placed


def base():
    return candidate([stage('cpu',0,3,4,[0,1,2,3]),stage('vta',3,7,1,[0,1,2,3]),
        stage('cpu',7,9,3,[1,2,3]),stage('vta',9,13,1,[0,1,2,3]),stage('cpu',13,14,1,[0])])


def test_only_host_stage_masks_change_with_distinct_identity():
    a=base();old=deepcopy(a);b=placed(a,[3,2]);c=placed(a,[2,1])
    assert a==old and placed(a,None)==a
    assert [s for s in a['stages'] if s['device']=='cpu']==[s for s in b['stages'] if s['device']=='cpu']
    assert b['control_mask']==a['control_mask'] and b['execution_mode']==a['execution_mode']
    assert [s['mask'] for s in b['stages'] if s['device']=='vta']==[[3],[2]]
    assert len({a['id'],b['id'],c['id']})==3
    assert placed(a,[3,2])==b


def test_invalid_host_placement_rejected():
    for value in ([1],[1,4],[True,1]):
        with pytest.raises(ValueError,match='host placement'):placed(base(),value)
