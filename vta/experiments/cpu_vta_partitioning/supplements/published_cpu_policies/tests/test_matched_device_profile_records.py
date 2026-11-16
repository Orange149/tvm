from copy import deepcopy
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src'))
from matched_device_profile.records import frame_alignment,intersect_intervals,validate_events,device_projection


def runs(pairs,stage=0):
    return [{f'stage{stage}_run_start_ms':a/1e6,f'stage{stage}_run_end_ms':b/1e6} for a,b in pairs]


def test_unique_frame_association_preserves_unknown_absolute_origin():
    events=[dict(wall_start_ns=x,wall_end_ns=x+2) for x in [1010,1040,1070]]
    result=frame_alignment({0:events},runs([(0,20),(30,50),(60,80)]))
    assert result['per_frame_mapping_reliable']
    assert result['mapping']==[(0,0,0),(0,1,1),(0,2,2)]
    assert result['relative_origin_intervals_ns']==[(992,1010)]
    assert result['no_origin_selected_to_fit_performance']


def test_ambiguous_middle_event_is_not_projected_to_a_guessed_frame():
    events=[dict(wall_start_ns=a,wall_end_ns=b) for a,b in [(1010,1020),(1100,1110),(1150,1160)]]
    result=frame_alignment({0:events},runs([(0,100),(110,210)]))
    assert not result['per_frame_mapping_reliable']
    assert result['mapping'] is None


def test_missing_invocation_and_inconsistent_clock_origins_are_rejected():
    with pytest.raises(ValueError,match='coverage'):
        frame_alignment({0:[dict(wall_start_ns=1010,wall_end_ns=1012),dict(wall_start_ns=1070,wall_end_ns=1072)]},
                        runs([(0,20),(30,50),(60,80)]))
    frame={**runs([(0,20)],0)[0],**runs([(5,25)],1)[0]}
    with pytest.raises(ValueError):
        frame_alignment({0:[dict(wall_start_ns=1010,wall_end_ns=1012)],
                         1:[dict(wall_start_ns=2015,wall_end_ns=2017)]},[frame])


def test_adjacent_integer_origins_remain_separate():
    assert intersect_intervals([(0,5),(6,10)],[(0,10)])==[(0,5),(6,10)]


def sample():
    e=dict(kind='device',tid=42,result=0,instructions=5,driver_run_calls=1,
        driver_timeout_calls=0,counters_monotonic=True,MMIO_error=False,
        start_writes=1,auto_writes=3,true_reads=1,false_reads=4,
        observed_poll_reads=5,driver_poll_iters=5,
        wall_start_ns=1001,launch_before_ns=1010,launch_after_ns=1020,
        last_false_before_ns=1030,last_false_after_ns=1031,
        first_true_before_ns=1050,first_true_after_ns=1051,wall_end_ns=1060,
        cpu_start_ns=1,cpu_end_ns=11,poll_read_envelope_ns=5)
    status=dict(kind='status',overflow=False,events=1,stats_bytes=64,extra_MMIO_reads=0,
        per_poll_probe=True,completion_brackets=True,provider='/x/runtime/libvta.so')
    doc=dict(stages=[dict(device='vta',program='vta_test')],warmup=0,scored=1,tail=0)
    affinity=dict(stages=[dict(caller_tid=42)])
    return [status,e],doc,affinity,runs([(0,80)])


def test_projection_keeps_driver_cpu_separate_from_nonrunning_wall():
    events,doc,affinity,frames=sample()
    valid=validate_events(events,doc,affinity,frames)
    projected=device_projection(valid,frames,doc)['stages']['vta_test']['frames'][0]
    assert projected['driver_wall_ns']==59
    assert projected['driver_CPU_ns']==10
    assert projected['outside_driver_run_wall_ns']==21
    assert projected['completion_lower_sum_ns']==10
    assert projected['completion_upper_sum_ns']==41


@pytest.mark.parametrize('change',[dict(overflow=True),dict(extra_MMIO_reads=1),dict(stats_bytes=32),dict(events=2)])
def test_trace_loss_extra_reads_and_wrong_abi_are_never_zero_cost(change):
    events,doc,affinity,frames=sample();events[0].update(change)
    with pytest.raises(ValueError):validate_events(events,doc,affinity,frames)
