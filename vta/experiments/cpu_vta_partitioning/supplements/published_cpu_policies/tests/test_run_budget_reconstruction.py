from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_budget_reconstruction import reconstruct


def test_caller_result_processing_does_not_change_run_total_or_worker_demand():
    rows=[dict(stage0_run_process_cpu_ms=40),dict(stage0_run_process_cpu_ms=40)]
    value=reconstruct(rows,[26,24])
    assert value['estimated_thread_CPU_ns']==[30_000_000,26_000_000,24_000_000]
    assert value['weights']==[30/80,26/80,24/80]
    assert value['newly_measured_per_thread_profile'] is False


def test_inconsistent_coarse_counters_are_rejected_not_clipped():
    with pytest.raises(ValueError,match='incompatible'):
        reconstruct([dict(stage0_run_process_cpu_ms=10)],[8,8])
    with pytest.raises(ValueError):reconstruct([],[1,1])
