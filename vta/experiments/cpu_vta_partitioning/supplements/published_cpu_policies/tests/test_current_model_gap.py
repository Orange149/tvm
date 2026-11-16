from copy import deepcopy
import pytest
from match_current_model_gap import project


def cases():
    original=dict(configuration=dict(execution_mode='off',stages=[
        dict(device='cpu',program='cpu',mask=[0,1]),dict(device='vta',program='vta',mask=[0,1,2,3])]),
        stages=[dict(device='cpu',mask=[0,1],work=20),dict(device='vta',mask=[0,1,2,3],work=30)])
    donor=deepcopy(original);donor['stages'][0]['work']=100;donor['stages'][1]['work']=50
    return original,donor


def test_only_matching_vta_isolated_input_changes():
    a,b=cases();before=deepcopy(a);out=project(a,b)
    assert out['configuration']==a['configuration']
    assert out['stages'][0]['work']==20 and out['stages'][1]['work']==50
    assert a==before


def test_host_profile_placement_mismatch_rejected():
    a,b=cases();b['stages'][1]['mask']=[3]
    with pytest.raises(ValueError,match='host placement'):project(a,b)
