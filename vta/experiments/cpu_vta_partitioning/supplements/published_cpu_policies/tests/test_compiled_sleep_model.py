from fractions import Fraction
import pytest
from compiled_sleep_model import CompiledScheduler,make_select

Q={2:12000000,3:8000000,4:8000000,5:4000000,6:4000000}


def test_compiled_sleep_credit_independent_of_runnable_count():
    for n in (1,2,3,4):
        s=CompiledScheduler(Q,sleeper=True,sleeper_credit_ns=9000000)
        s.vruntime={'wake':Fraction(-100000000)};s.minimum[0]=Fraction(60000000)
        tasks={str(i):dict(actor='wake' if i==0 else 'busy'+str(i),core=0) for i in range(n)}
        s.previous={'busy'+str(i) for i in range(1,n)}
        s.select(tasks)
        assert s.vruntime['wake']==51000000


def test_original_credit_and_both_switches_keep_separate_proofs():
    old=CompiledScheduler(Q);old.vruntime={'wake':Fraction(-100000000)};old.minimum[0]=Fraction(60000000)
    old.select({'0':dict(actor='wake',core=0),'1':dict(actor='busy',core=0)})
    assert old.vruntime['wake']==48000000
    assert len(make_select(wake=True,sleeper=True)[1]['replacements'])==2
    assert len(make_select(sleeper=True)[1]['replacements'])==1


def test_no_wakeup_original_and_corrected_busy_dispatch_match():
    a=CompiledScheduler(Q);b=CompiledScheduler(Q,wake=True,sleeper=True,wakeup_granularity_ns=3000000,sleeper_credit_ns=9000000)
    tasks={'0':dict(actor='a',core=0),'1':dict(actor='b',core=0)}
    a.vruntime={'a':Fraction(0),'b':Fraction(1000000)};b.vruntime=dict(a.vruntime)
    a.previous=b.previous={'a','b'}
    assert a.select(tasks)==b.select(tasks)
    a.consume(tasks,Fraction(1000));b.consume(tasks,Fraction(1000))
    assert a.vruntime==b.vruntime and a.left==b.left


def test_no_silent_parameter_defaults():
    with pytest.raises(ValueError):CompiledScheduler(Q,sleeper=True)
    with pytest.raises(ValueError):CompiledScheduler(Q,wake=True,wakeup_granularity_ns=True)
