from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_isolated_run_scope import inspect_scope, logical_counters


def fixture():
    row = dict(device='cpu', threads=3, worker_core_map=[0, 1, 2])
    task = dict(kind='calibration', stages=[row], input_indices=[0], warmup=2,
                scored=8, tail=0, environment=dict(spin_count=0))
    frames = [dict(frame_id=i, completion_index=i, mode='serial', stage_count=1,
                   input_index=0, stage_cpu_time_scope='exclusive_process_cpu_time',
                   stage0_run_process_cpu_ms=12) for i in range(10)]
    affinity = dict(mode='serial', cpu_worker_pinning=True, stages=[dict(
        caller_tid=10, worker_slots=[dict(slot=i, tid=10+i) for i in range(3)],
        effective_affinities=[dict(tid=10+i, cpus=[i], cpu_time_delta_ms=v)
                              for i, v in enumerate([100, 40, 30])])])
    launcher = ('env TVM_THREAD_POOL_SPIN_COUNT=0 TVM_EXCLUDE_WORKER0=1 '
                'TVM_BIND_THREADS=1 runner --serial --cpu-worker-pinning')
    return task, frames, affinity, launcher


def test_reconstruction_removes_caller_outside_run_without_changing_run_budget():
    task, frames, affinity, launcher = fixture()
    before = deepcopy((task, frames, affinity))
    result = inspect_scope(task, frames, affinity, launcher)
    assert result['reconstruction']['estimated_thread_CPU_ns'] == [50_000_000, 40_000_000, 30_000_000]
    assert result['estimated_weights'] == [50/120, 40/120, 30/120]
    assert (task, frames, affinity) == before


def test_old_mapping_requires_unique_singletons_and_correct_caller():
    task, frames, affinity, launcher = fixture()
    affinity['stages'][0]['worker_slots'] = []
    assert logical_counters(affinity, task['stages'][0])[1].startswith('inferred_')
    affinity['stages'][0]['caller_tid'] = 11
    with pytest.raises(ValueError, match='caller'):
        inspect_scope(task, frames, affinity, launcher)


def test_negative_remainder_and_unobservable_worker_preserve_fallback():
    task, frames, affinity, launcher = fixture()
    for f in frames:
        f['stage0_run_process_cpu_ms'] = 1
    result = inspect_scope(task, frames, affinity, launcher)
    assert result['status'] == 'retain_original_incompatible_remainder'
    assert result['estimated_weights'] is None
    affinity['stages'][0]['effective_affinities'][1]['cpu_time_delta_ms'] = 0
    result = inspect_scope(task, frames, affinity, launcher)
    assert result['status'] == 'retain_original_unobservable_noncaller'
    assert result['estimated_weights'] is None


@pytest.mark.parametrize('corruption', ['pipeline', 'duplicate', 'spin', 'affinity'])
def test_incompatible_scope_and_identity_rejected(corruption):
    task, frames, affinity, launcher = fixture()
    if corruption == 'pipeline':
        task['kind'] = 'performance'
    elif corruption == 'duplicate':
        frames[-1]['frame_id'] = 8
    elif corruption == 'spin':
        launcher = launcher.replace('SPIN_COUNT=0', 'SPIN_COUNT=1')
    else:
        affinity['stages'][0]['effective_affinities'][1]['cpus'] = [2]
    with pytest.raises(ValueError):
        inspect_scope(task, frames, affinity, launcher)
