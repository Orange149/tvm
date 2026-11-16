from copy import deepcopy
import struct
import pytest
from run_clock_scope_audit import analyze_rows


def examples():
    result=[]
    for kind,enabled in [('busy',False),('busy',True),('poll',False),('poll',True)]:
        r=dict(case=kind,trace=enabled,pid=10,tid=10,allowed=[3],
            wall_start_ns=30,wall_end_ns=70,cpu_start_ns=0,cpu_end_ns=20,
            read0_start_ns=25,read0_end_ns=28,read1_start_ns=72,read1_end_ns=75,
            counter_before=dict(value=0,enabled=0,running=0),
            counter_after=dict(value=20,enabled=60,running=60),
            enable_start_ns=10,enable_end_ns=11,disable_start_ns=90,disable_end_ns=91,
            head=0,tail=0,data_size=4096,overflow=False,records_hex='')
        if enabled:
            raw=b''.join(struct.pack('<IHHIIQII',14,misc,32,10,10,t,3,0)
                         for misc,t in [(0,20),(1<<13,40),(0,60),(1<<13,80)])
            r.update(records_hex=raw.hex(),head=len(raw))
        result.append(r)
    return result


def test_switch_span_and_clock_envelope_stay_distinct():
    rows=analyze_rows(examples())
    assert rows[1]['switch_span_CPU_ms']==20/1e6
    assert rows[1]['switch_outs']==1
    assert rows[1]['thread_clock_minus_switch_ms']==0
    assert rows[1]['counter_external_wall_envelope_ns']==10


def test_loss_and_disabled_records_rejected():
    rows=examples();rows[1]['overflow']=True
    with pytest.raises(ValueError,match='overflow'):analyze_rows(rows)
    rows=examples();rows[0]['records_hex']='aabb'
    with pytest.raises(ValueError,match='disabled'):analyze_rows(rows)


def test_uncovered_window_cannot_be_zero_wait():
    rows=examples();rows[1]['wall_start_ns']=15
    rows[1]['read0_start_ns']=12;rows[1]['read0_end_ns']=14
    with pytest.raises(ValueError,match='fully captured'):analyze_rows(rows)
