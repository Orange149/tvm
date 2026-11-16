from pathlib import Path
from fractions import Fraction
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src'))
from recent_load.model import RecentLoad,choose_core,HALF_LIFE_NS,METRIC_SCALE,RecentLoadReplay


def test_half_life_and_idle_decay():
    load=RecentLoad();load.update({0:2},HALF_LIFE_NS)
    assert load.metric(0)==METRIC_SCALE
    load.update({},HALF_LIFE_NS)
    assert load.metric(0)==METRIC_SCALE//2
    with pytest.raises(ValueError,match='negative'):load.update({},-1)


def test_past_load_breaks_equal_current_counts_but_idle_core_is_preferred():
    load=RecentLoad();load.update({0:3,1:1},HALF_LIFE_NS)
    assert choose_core([0,1],{0:1,1:1},load,0)==1
    assert choose_core([0,1],{1:1},load,1)==0
    with pytest.raises(ValueError,match='mask'):choose_core([0,0],{},load,None)


def test_same_group_retains_original_affinity_and_actor_history():
    # Direct action construction, no runtime or native data are involved.
    r=RecentLoadReplay([dict(device='vta',mask=[1,2],worker_core_map=[])],
        quanta={n:10 for n in range(2,7)},frames=2,warmup=0,scored=2)
    meta=dict(role='stage',stage=0,slot=0,frame=0,operation='host_serial')
    r.recent_load.update({1:2,2:1},HALF_LIFE_NS)
    key=r._new_task(0,Fraction(2),[1,2],lambda:None,meta)
    assert r.active[key]['assigned_core']==2
    assert r.active[key]['mask']==(1,2)
    # Recent load flips while actor remains ready: no invented migration.
    r.recent_load.update({2:6},HALF_LIFE_NS)
    r.active.pop(key)
    key=r._new_task(0,Fraction(2),[1,2],lambda:None,meta)
    assert r.active[key]['assigned_core']==2
