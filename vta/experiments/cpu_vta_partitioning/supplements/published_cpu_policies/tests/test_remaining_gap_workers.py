from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_remaining_gap_workers import match_workers, model_index


def fixture():
    tasks = [dict(stage=0, operation='worker_compute', frame=12, kernel=6, slot=i,
                  CPU=dict(numerator=i+1, denominator=1),
                  end_ns=20 if i==1 else 10, first_service_ns=0) for i in range(4)]
    native = [dict(task=i, tid=100+i, wall_start_ns=0,
                   wall_end_ns=30 if i==3 else 10, cpu_start_ns=0, cpu_end_ns=5)
              for i in range(4)]
    return tasks, native, {i:100+i for i in range(4)}


def test_compare_same_worker_when_model_and_native_last_workers_differ():
    tasks, native, tids = fixture()
    rows = match_workers(12, 6, native, model_index(tasks), tids)
    actual_last = next(r for r in rows if r['native_last'])
    assert actual_last['slot'] == 3
    assert actual_last['model_isolated_compute_CPU']['numerator'] == 4
    assert not actual_last['model_last']
    assert next(r for r in rows if r['model_last'])['slot'] == 1


def test_duplicate_model_task_and_missing_callback_rejected():
    tasks, native, tids = fixture()
    with pytest.raises(ValueError, match='duplicate'):
        model_index(tasks + [tasks[0]])
    with pytest.raises(ValueError, match='incomplete'):
        match_workers(12, 6, native[:-1], model_index(tasks), tids)


def test_same_slot_wrong_tid_is_rejected():
    tasks, native, tids = fixture()
    native[0]['tid'] = tids[1]
    with pytest.raises(ValueError, match='TID'):
        match_workers(12, 6, native, model_index(tasks), tids)
