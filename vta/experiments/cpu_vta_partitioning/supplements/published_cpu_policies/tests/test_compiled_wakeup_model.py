from copy import deepcopy
from fractions import Fraction
import pytest
from finite_service.scheduler import Scheduler
from compiled_wakeup_model import KernelWakeScheduler,adapted_select


def quanta():return {2:12000000,3:8000000,4:8000000,5:4000000,6:4000000}


def prepared(cls):
    x=cls(quanta(),3000000) if cls is KernelWakeScheduler else cls(quanta())
    x.vruntime={'running':Fraction(4500000),'waking':Fraction(0)}
    x.last_core={'running':0,'waking':0};x.current={0:'running'}
    x.left={0:Fraction(1000000)};x.last_count={0:2};x.previous={'running'}
    x.minimum[0]=Fraction(4500000);return x


def test_independent_wakeup_threshold_changes_only_the_known_preemption_decision():
    tasks={'a':dict(actor='running',core=0),'b':dict(actor='waking',core=0)}
    assert prepared(Scheduler).select(tasks)[0][0]=='a'
    assert prepared(KernelWakeScheduler).select(tasks)[0][0]=='b'


def test_continuous_busy_tasks_keep_the_calibrated_service_budget():
    old=prepared(Scheduler);new=prepared(KernelWakeScheduler)
    old.previous=new.previous={'running','waking'}
    tasks={'a':dict(actor='running',core=0),'b':dict(actor='waking',core=0)}
    assert old.select(tasks)==new.select(tasks)
    old.consume(tasks,Fraction(500000));new.consume(tasks,Fraction(500000))
    assert old.vruntime==new.vruntime and old.left==new.left
    assert adapted_select()[1]['other_scheduler_rules_unchanged']


def test_kernel_parameter_requires_explicit_positive_integer():
    for v in (None,0,-1,3.0,True):
        with pytest.raises(ValueError):KernelWakeScheduler(quanta(),v)
