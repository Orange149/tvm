from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from project_current_isolated_phases import changes,VTA_profile
from device_phase.extract import split_run


def test_probe_quality_is_not_a_performance_drift_gate():
    clean=dict(run_ms=100,run_core_ms=50)
    assert changes(clean,dict(run_ms=95,run_core_ms=55))['passed']
    assert not changes(clean,dict(run_ms=100,run_core_ms=65))['passed']


def test_measured_host_CPU_order_is_preserved_without_moving_it_to_fit_FPS():
    span=dict(call=2,cpu_start_ns=10,cpu_end_ns=70,wall_start_ns=0,wall_end_ns=130)
    events=[dict(call=2,cpu_start_ns=20,cpu_end_ns=25,wall_start_ns=20,wall_end_ns=50,instructions=10),
        dict(call=2,cpu_start_ns=40,cpu_end_ns=50,wall_start_ns=70,wall_end_ns=100,instructions=20)]
    result=split_run(span,events)
    assert result['host_serial_CPU_ns']==[10,15,20]
    assert sum(result['host_serial_CPU_ns'])+sum(d['CPU_ns'] for d in result['devices'])==60
    broken=[{**events[0],'cpu_start_ns':9},events[1]]
    with pytest.raises(ValueError,match='backwards'):split_run(span,broken)
