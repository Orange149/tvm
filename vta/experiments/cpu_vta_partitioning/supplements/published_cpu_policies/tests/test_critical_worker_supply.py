from fractions import Fraction
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_critical_worker_supply import cpu_crossing, deadline_supply


def test_service_crossing_retains_fraction_and_does_not_interpolate_off_cpu():
    spans = [dict(start_ns=2, end_ns=6), dict(start_ns=10, end_ns=16)]
    assert cpu_crossing(spans, 0, 20, Fraction(9, 2)) == Fraction(21, 2)
    assert cpu_crossing(spans, 0, 20, 11) is None
    assert cpu_crossing(spans, 3, 13, 5) == 12


def test_relative_deadline_counts_real_cpu_not_all_callback_wall():
    stream = dict(on=[dict(start_ns=100, end_ns=104), dict(start_ns=110, end_ns=120)],
                  off=[dict(start_ns=104, end_ns=110, preempt=True)])
    row = deadline_supply(stream, 100, 120, 10, Fraction(8))
    assert row['native_CPU_by_relative_deadline_ns'] == 4
    assert row['isolated_work_deficit_at_relative_deadline_ns'] == 4
    assert row['isolated_work_crossing_from_native_callback_start_ns'] == 14
    assert row['remaining_native_wall_after_work_crossing_ns'] == 6


def test_late_model_deadline_cannot_create_observed_supply_after_callback():
    stream = dict(on=[dict(start_ns=0, end_ns=4)], off=[])
    row = deadline_supply(stream, 0, 4, 10, Fraction(5))
    assert row['deadline_clipped_to_callback_end']
    assert row['native_CPU_by_relative_deadline_ns'] == 4
    assert row['measured_supply_below_isolated_demand_at_callback_end']
    assert row['isolated_work_crossing_from_native_callback_start_ns'] is None


def test_duplicate_cpu_intervals_or_missing_trace_are_rejected():
    with pytest.raises(ValueError, match='overlapping'):
        cpu_crossing([dict(start_ns=0, end_ns=4), dict(start_ns=3, end_ns=6)], 0, 6, 7)
    with pytest.raises(ValueError, match='coverage'):
        deadline_supply(dict(on=[dict(start_ns=1, end_ns=4)], off=[]), 0, 4, 4, Fraction(2))
