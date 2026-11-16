from pathlib import Path
import sys
from fractions import Fraction
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_caller_gap_scope import caller_intervals, model_interval_budget


def test_real_return_tail_and_serial_work_are_not_counted_twice():
    rows = caller_intervals(0, 100, [
        dict(start=10, last_callback_end=40, end=50),
        dict(start=60, last_callback_end=80, end=90)])
    assert [(r['start'], r['end']) for r in rows] == [(0, 10), (40, 50), (50, 60), (80, 90), (90, 100)]
    assert sum(r['end'] - r['start'] for r in rows) == 50


def test_continuation_queue_is_counted_in_model_serial_interval():
    tasks = [dict(ready_ns=40, end_ns=60, operation='serial',
                  CPU=dict(numerator=7, denominator=1))]
    row = model_interval_budget(40, 60, tasks)
    assert row['CPU_ns'] == Fraction(7)
    assert row['wall_minus_CPU_including_rounding_ns'] == Fraction(13)
    rows = caller_intervals(0, 70, [dict(start=0, last_callback_end=40, end=40),
                                   dict(start=60, last_callback_end=70, end=70)])
    assert any(r['start'] == 40 and r['end'] == 60 for r in rows)


def test_overlapping_launches_and_partial_model_work_are_rejected():
    with pytest.raises(ValueError, match='overlap'):
        caller_intervals(0, 100, [dict(start=10, last_callback_end=40, end=50),
                                    dict(start=45, last_callback_end=70, end=80)])
    with pytest.raises(ValueError, match='crosses'):
        model_interval_budget(10, 20, [dict(ready_ns=0, end_ns=20,
            operation='serial', CPU=dict(numerator=5, denominator=1))])


def test_clock_fraction_is_retained_until_reporting():
    value = model_interval_budget(0, 3, [dict(ready_ns=0, end_ns=3,
        operation='serial', CPU=dict(numerator=7, denominator=3))])
    assert value['CPU_ns'] == Fraction(7, 3)
    assert value['wall_minus_CPU_including_rounding_ns'] == Fraction(2, 3)
