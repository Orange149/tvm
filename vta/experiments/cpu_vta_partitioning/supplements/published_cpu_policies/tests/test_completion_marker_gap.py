"""Reject incorrect timing claims at the callback/notification boundary."""
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit_completion_marker_gap import marker_interval


def test_late_bookkeeping_can_belong_to_an_earlier_finishing_worker():
    # Worker 0 finishes computing first, but its counter read is delayed. Using
    # only the last computing worker's epilogue would miss this mandatory work.
    workers=[dict(wall_end_ns=3,queue_read_end1_ns=15),dict(wall_end_ns=9,queue_read_end1_ns=10)]
    r=marker_interval(dict(wall_end_ns=20),workers,True)
    assert r['later_observed_end_ns']==15
    assert r['unresolved_to_return_ns']==5
    assert r['later_marker_is_runtime_signal'] is False


@pytest.mark.parametrize('workers,enabled',[
    ([dict(wall_end_ns=9,queue_read_end1_ns=8)],True),
    ([dict(wall_end_ns=9,queue_read_end1_ns=21)],True),
    ([dict(wall_end_ns=9,queue_read_end1_ns=1)],False),
    ([dict(wall_end_ns=22,queue_read_end1_ns=0)],False)])
def test_inconsistent_or_disabled_interface_markers_are_rejected(workers,enabled):
    with pytest.raises(ValueError):marker_interval(dict(wall_end_ns=20),workers,enabled)
