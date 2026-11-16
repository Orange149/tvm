from copy import deepcopy
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analyze_phase_cycles import decompose


def rows():
    return [dict(frame_id=i,stage_count=1,stage0_start_ms=i*10,
        stage0_end_ms=i*10+8,completion_ms=i*13+10) for i in range(5)]


def test_completion_gap_is_not_mistaken_for_between_frame_idle_or_CPU_time():
    r=decompose(rows(),warmup=0,scored=5)
    assert r["stage_service_ms_per_cycle"]==8
    assert r["between_frame_gap_ms"]==2
    assert r["stage_start_cycle_ms"]==10
    assert r["downstream_latency_change_per_gap_ms"]==3
    assert r["completion_interval_ms"]==13
    assert r["latency_change_across_window_ms"]==12


def test_missing_duplicate_and_overlapping_stage_frames_rejected():
    for mutate in (lambda rs:rs.pop(1),lambda rs:rs.append(deepcopy(rs[0])),
                   lambda rs:rs[2].update(stage0_start_ms=17)):
        rs=rows();mutate(rs)
        with pytest.raises(ValueError):decompose(rs,warmup=0,scored=5)
