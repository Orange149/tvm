from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
from yield_semantics.audit import check, protocol, summarize


def rows(fractions=(.5, .2, .9)):
    result=[]
    for name, fraction in zip(protocol()['cases'],fractions):
        for i in (0,1):
            CPU=int(1000*(fraction if i else .5))
            result.append(dict(case=name,index=i,segments=[dict(wall_start_ns=100,
                wall_end_ns=1100,CPU_clock_ns=CPU)],on_CPU=[dict(start_ns=100,end_ns=100+CPU)],
                switches=4,nonpreempt_out=1))
    return result


def test_compare_common_window_and_negative_outcome():
    report=summarize(rows())
    assert report['same_core_Yield_reduces_waiter_fraction']
    assert report['cases'][1]['waiter_CPU_fraction']==.2
    assert not summarize(rows((.3,.5,.9)))['same_core_Yield_reduces_waiter_fraction']


def test_missing_pair_and_no_common_window_rejected():
    with pytest.raises(ValueError,match='pair incomplete'):summarize(rows()[:-1])
    data=rows();data[1]['segments'][0].update(wall_start_ns=1200,wall_end_ns=1300)
    with pytest.raises(ValueError,match='no common work window'):summarize(data)


def test_unavailable_and_incomplete_raw_rejected():
    with pytest.raises(ValueError,match='unavailable'):check(dict(status='UNAVAILABLE'))
    with pytest.raises(ValueError,match='incomplete'):check(dict(status='MEASURED',
        DNN_inferences=0,periodic_sampling=False,streams=[]))


def test_bounded_zero_inference_scope():
    p=protocol()
    assert p['DNN_inferences']==0 and p['max_native_sessions']==1
    assert p['target_busy_CPU_ns_total']==240_000_000
    assert not p['FPGA_access'] and not p['automatic_repeat']
