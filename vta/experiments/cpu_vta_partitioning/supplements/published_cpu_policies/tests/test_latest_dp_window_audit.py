from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit_latest_dp_window import window_inventory


def example():
    return [dict(frame_id=f,completion_ms=20+10*f,
                 stage0_start_ms=1+5*f,stage0_end_ms=5+5*f,
                 stage1_start_ms=16+10*f,stage1_end_ms=19+10*f) for f in range(8)]


def test_drain_is_conserved_as_a_change_in_work_in_flight():
    result=window_inventory(example(),[dict(program='a',device='cpu'),dict(program='b',device='cpu')],1,4)
    first=result['stages'][0]
    assert result['gaps']==3
    assert first['completed_in_window']==2
    assert first['inventory_change']==-1
    assert first['stage_exhausted_before_window_end']
    assert first['window_after_last_stage_call_ns']==20_000_000


def test_last_stage_does_not_exhaust_in_scored_window_and_no_negative_inventory():
    result=window_inventory(example(),[dict(program='a',device='cpu'),dict(program='b',device='cpu')],1,4)
    last=result['stages'][-1]
    assert last['completed_in_window']==3
    assert last['inventory_change']==0
    assert not last['stage_exhausted_before_window_end']


def test_duplicate_frame_or_impossible_single_stage_overlap_is_rejected():
    data=example();data[1]['frame_id']=0
    with pytest.raises(ValueError,match='frame'):
        window_inventory(data,[dict(program='a',device='cpu')],1,4)
    data=example();data[1]['stage0_start_ms']=4
    with pytest.raises(ValueError,match='order'):
        window_inventory(data,[dict(program='a',device='cpu')],1,4)
