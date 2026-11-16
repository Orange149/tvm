import pytest
from audit_worker_dispatch_gap import entry_gap


def example():
    worker=dict(tid=2,slot=1,launch=3,frame=12,kernel=0,last=True,callback_start_ns=30)
    caller=dict(callback_start_ns=10)
    stream=dict(on=[dict(start_ns=0,end_ns=12),dict(start_ns=28,end_ns=40)],
                off=[dict(start_ns=12,end_ns=28,preempt=False)])
    raw=dict(tid=2,task=1,launch=3,wall_start_ns=30,queue_read_start0_ns=29,queue_read_end0_ns=30)
    return worker,caller,stream,raw


def test_posted_interval_and_read_envelope_not_double_counted():
    r=entry_gap(*example())
    assert r['post_caller_entry_gap_ns']==20
    assert r['on_CPU_ns']==4 and r['nonpreempt_off_ns']==16
    assert r['entry_counter_probe_overlap_ns']==1 and r['probe_on_CPU_ns']==1
    assert r['exact_wakeup_time_known'] is False


def test_worker_already_started_has_zero_post_caller_gap():
    w,c,s,r=example();c['callback_start_ns']=35
    x=entry_gap(w,c,s,r)
    assert x['post_caller_entry_gap_ns']==0 and x['began_before_caller']


def test_wrong_callback_and_missing_coverage_rejected():
    w,c,s,r=example();r['tid']=4
    with pytest.raises(ValueError,match='identity'):entry_gap(w,c,s,r)
    w,c,s,r=example();s['off']=[]
    with pytest.raises(ValueError,match='fully captured'):entry_gap(w,c,s,r)
