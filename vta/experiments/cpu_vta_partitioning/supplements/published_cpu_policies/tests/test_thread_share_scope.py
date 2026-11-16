from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit_thread_share_scope import run_work,normalize


def q(n):return dict(numerator=n,denominator=1)


def test_caller_phase_includes_own_callback_and_must_not_be_counted_twice():
    profile=dict(caller_CPU=q(20),known_CPU=q(31),kernels=[dict(tasks=2,barrier_generations=0,
        workers=[dict(compute_segments=[q(8)]),dict(compute_segments=[q(11)])])])
    work,barriers=run_work(profile,2)
    assert work==[20,11] and barriers==0
    assert normalize(work)==[20/31,11/31]
    with pytest.raises(ValueError,match='known scoped'):run_work({**profile,'known_CPU':q(39)},2)


def test_a_worker_not_used_by_a_kernel_does_not_acquire_invented_work():
    profile=dict(caller_CPU=q(20),known_CPU=q(31),kernels=[dict(tasks=2,barrier_generations=0,
        workers=[dict(compute_segments=[q(8)]),dict(compute_segments=[q(11)])])])
    work,_=run_work(profile,3)
    assert work==[20,11,0]
