from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import analyze_caller_wait_budget as a


def profile():
    return dict(serial_cpu_segments=[1,2],caller_CPU=12,run_wall=20,kernels=[dict(
        launch_prefix_cpu=2,isolated_completion_wait_and_return_cpu=3,
        workers=[dict(compute_segments=[4]),dict(compute_segments=[7])])])


def test_cpu_partition_does_not_sum_waiting_walls_or_process_cpu():
    r=a.projected_metrics(profile())
    assert r['caller_run_CPU_ms']==12/1e6 and r['caller_off_CPU_ms']==8/1e6
    assert r['recorded_stage_run_CPU_ms']==19/1e6
    assert sum(r[k] for k in ('caller_compute_CPU_ms','caller_other_CPU_ms',
        'completion_and_return_CPU_ms','caller_off_CPU_ms'))==pytest.approx(r['caller_run_wall_ms'],rel=1e-12)


def test_cpu_nonconservation_or_negative_offcpu_rejected():
    p=profile();p['caller_CPU']=13
    with pytest.raises(ValueError,match='partition'):a.projected_metrics(p)
    p=profile();p['run_wall']=11
    with pytest.raises(ValueError,match='exceeds'):a.projected_metrics(p)
