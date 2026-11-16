from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit_vta_callback_competition import callback_addresses, IntervalIndex, split_on_CPU, match_frame


def test_callback_address_does_not_cross_an_intervening_call_or_clobber():
    s='''
    5f44: 90000000 adrp x0, 5000 <helper>
    5f48: 913d9000 add x0, x0, #0xf64
    5f50: d63f0100 blr x8
    6190: 90000000 adrp x0, 6000 <helper>
    6194: 97ff0000 bl f90 <function>
    6198: d63f0100 blr x8
    6200: 90000000 adrp x0, 6000 <helper>
    6204: aa0103e0 mov x0, x1
    6208: d63f0100 blr x8
    '''
    assert callback_addresses(s)==[(0x5f64,0x5f50)]


def test_only_on_CPU_intersections_count_and_the_split_conserves_time():
    hosts=IntervalIndex([dict(start_ns=3,end_ns=9),dict(start_ns=12,end_ns=18)])
    callbacks=IntervalIndex([dict(start_ns=1,end_ns=5,family='activation'),
        dict(start_ns=8,end_ns=15,family='layout')])
    r=split_on_CPU(4,16,hosts,callbacks)
    assert r==dict(total_ns=9,CPU_callback_ns=5,outside_CPU_callback_ns=4,
        families={'activation':1,'layout':4})
    with pytest.raises(ValueError,match='overlapping'):
        IntervalIndex([dict(start_ns=1,end_ns=5),dict(start_ns=4,end_ns=7)])


def test_callback_identity_and_missing_or_duplicate_events_are_rejected():
    phase=dict(tid=7,wall_start_ns=0,wall_end_ns=100)
    event=dict(launch=2,caller=7,tasks=1,tid=7,wall_start_ns=10,wall_end_ns=60,
        module='/module.so',function_offset=24,result=0)
    worker={**event,'task':0,'wall_start_ns':20,'wall_end_ns':50,'cpu_start_ns':0,
        'cpu_end_ns':10,'barrier_calls':0,'barrier_cpu_ns':0,'barrier_wall_ns':0}
    assert len(match_frame(phase,[event],{2:[worker]},{24:{}},1))==1
    for values in ([],[worker,worker],[{**worker,'function_offset':25}]):
        with pytest.raises(ValueError):
            match_frame(phase,[event],{2:values},{24:{}},1)
    with pytest.raises(ValueError,match='missing or extra'):
        match_frame(phase,[event,event],{2:[worker]},{24:{}},1)
