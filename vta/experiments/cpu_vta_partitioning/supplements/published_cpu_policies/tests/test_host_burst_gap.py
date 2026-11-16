import pytest
from audit_host_burst_gap import Phases, critical_overlap


def test_cpu_span_crosses_phase_boundaries_and_unobserved_gaps():
    phases = Phases([dict(operation='set', wall_start_ns=3, wall_end_ns=7),
                     dict(operation='run', wall_start_ns=9, wall_end_ns=13)])
    assert phases.split(5, 15) == [('set', 5, 7), ('outside_recorded_phases', 7, 9),
                                    ('run', 9, 13), ('outside_recorded_phases', 13, 15)]
    assert phases.split(0, 3) == [('outside_recorded_phases', 0, 3)]
    assert phases.split(7, 9) == [('outside_recorded_phases', 7, 9)]
    assert phases.split(8, 8) == []


def test_overlapping_or_backwards_phases_rejected():
    with pytest.raises(ValueError, match='overlap'):
        Phases([dict(operation='set', wall_start_ns=3, wall_end_ns=7),
                dict(operation='run', wall_start_ns=6, wall_end_ns=10)])
    with pytest.raises(ValueError, match='backwards'):
        Phases([dict(operation='set', wall_start_ns=7, wall_end_ns=3)])


def test_only_same_core_preempted_overlap_is_partitioned_once():
    worker = dict(on=[dict(start_ns=0, end_ns=10), dict(start_ns=20, end_ns=30)],
                  off=[dict(start_ns=10, end_ns=20, preempt=True, from_cpu=1, to_cpu=1)])
    host = dict(tid=2, role=dict(role='vta/host'),
                on=[dict(start_ns=12, end_ns=18, cpu=1), dict(start_ns=21, end_ns=25, cpu=1)])
    trace = dict(critical_rows=[dict(tid=1, callback_start_ns=0, callback_end_ns=30, kernel=0)],
                 observed=dict(scheduler=dict(streams={'1': worker})),
                 coincident_CPU_by_role_ns={'vta/host': 6}, complete_front_frames=[0])
    rows = critical_overlap(trace, [host], {2: Phases([
        dict(operation='run', wall_start_ns=14, wall_end_ns=16)])}, threshold=5)
    assert sum(r['coincident_CPU_ns'] for r in rows) == 6
    assert sum(r['coincident_CPU_ns'] for r in rows if r['operation']=='run') == 2
    assert sum(r['coincident_CPU_ns'] for r in rows if r['operation']=='outside_recorded_phases') == 4
    assert {r['span_type'] for r in rows} == {'long'}
