from fractions import Fraction
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src'))
from dispatch_service.model import DispatchScheduler
from dispatch_service.workflow import synthetic
from finite_service.scheduler import Scheduler

Q={2:10,3:8,4:6,5:4,6:4}


def descriptor(actor,remaining=100):
    return dict(actor=actor,core=0,remaining=Fraction(remaining))


def test_waking_peer_does_not_grant_fresh_full_budget():
    old=Scheduler(Q);new=DispatchScheduler(Q)
    one={0:descriptor('A')};two={**one,1:descriptor('B')}
    for model in (old,new):
        model.select(one);model.consume(one,Fraction(4))
        model.vruntime['B']=Fraction(20)
    assert old.select(two)[1]==10
    assert new.select(two)[1]==6
    assert new.dispatch_used[0]==4


def test_count_changes_preserve_used_CPU_until_dispatch_ends():
    model=DispatchScheduler(Q);tasks={0:descriptor('A'),1:descriptor('B')}
    model.vruntime['B']=100
    model.select(tasks);model.consume(tasks,Fraction(2))
    alone={0:tasks[0]};model.select(alone);model.consume(alone,Fraction(5))
    assert model.select(tasks)[1]==3
    assert model.dispatch_used[0]==7


def test_task_boundary_does_not_reset_same_actor_and_core_capacity():
    model=DispatchScheduler(Q);first={0:descriptor('A'),1:descriptor('B')}
    model.vruntime['B']=100
    model.select(first);model.consume(first,Fraction(3))
    second={5:descriptor('A'),1:descriptor('B')}
    assert model.select(second)[1]==7
    model.consume(second,Fraction(7));model.select(second)
    assert model.dispatch_used[0]==0
    with pytest.raises(ValueError,match='negative'):model.consume(second,Fraction(-1))


def test_synthetic_work_and_sleeps_are_not_changed():
    p=synthetic([[('cpu',40)],[('sleep',5),('cpu',8)]],Q)
    assert p['CPU_ns']==48 and len(p['wall_ns'])==2
    assert all(x>=0 for x in p['first_service_ns'].values())
