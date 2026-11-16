from pathlib import Path
from fractions import Fraction
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src'))
from polling_replay import driver_segments


def device(checks):
    return dict(CPU=20,wall=100,poll_checks=checks,submit_mmio_ns=2,post_start_sleep_ns=10)


@pytest.mark.parametrize('checks',[1,2,10,Fraction(9,2)])
def test_same_CPU_and_off_CPU_totals(checks):
    for periodic in (False,True):
        parts=driver_segments(device(checks),periodic)
        assert sum(v for k,v in parts if k=='cpu')==20
        assert sum(v for k,v in parts if k=='delay')==80
        assert all(v>=0 for _,v in parts)


def test_polling_repeats_without_inventing_more_work():
    parts=driver_segments(device(10),True)
    assert sum(k=='cpu' for k,_ in parts)==11  # submission + ten checks
    assert sum(k=='delay' for k,_ in parts)==10
    assert driver_segments(device(10),False)==[('cpu',20),('delay',80)]


def test_invalid_poll_count_and_negative_work_rejected():
    with pytest.raises(ValueError,match='count'):driver_segments(device(0),True)
    bad=device(2);bad['wall']=1
    with pytest.raises(ValueError,match='invalid driver'):driver_segments(bad,True)
