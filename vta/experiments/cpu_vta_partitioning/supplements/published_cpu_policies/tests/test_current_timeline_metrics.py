from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src'))
from current_timeline_metrics import host_segments,module_key


def test_full_pipeline_host_split_filters_both_frame_and_thread():
    run=dict(call=2,tid=11,wall_start_ns=0,wall_end_ns=200,cpu_start_ns=10,cpu_end_ns=80)
    event=dict(call=2,tid=11,wall_start_ns=50,wall_end_ns=150,cpu_start_ns=30,cpu_end_ns=50)
    other=dict(event,tid=99,wall_start_ns=0,wall_end_ns=200)
    value=host_segments(run,[other,event])
    assert [s['CPU_ns'] for s in value['segments']]==[20,30]
    assert [s['wall_ns'] for s in value['segments']]==[50,50]
    assert value['host_nonrunning_ns']==50
    assert value['host_CPU_ns']+value['driver_CPU_ns']==70
    assert value['host_wall_ns']+value['driver_wall_ns']==200


def test_host_time_reversal_or_missing_device_is_rejected():
    r=dict(call=2,tid=11,wall_start_ns=0,wall_end_ns=200,cpu_start_ns=10,cpu_end_ns=80)
    with pytest.raises(ValueError,match='same-TID'):host_segments(r,[])
    e=dict(call=2,tid=11,wall_start_ns=50,wall_end_ns=250,cpu_start_ns=30,cpu_end_ns=50)
    with pytest.raises(ValueError,match='outside run'):host_segments(r,[e])


def test_module_identity_uses_bytes_and_offset_not_deployment_path():
    files={'programs/cpu_02_04/module.so':dict(sha256='frozen')}
    assert module_key('/first/programs/cpu_02_04/module.so',files)==module_key('/second/programs/cpu_02_04/module.so',files)
    with pytest.raises(ValueError,match='unknown'):module_key('/second/programs/other/module.so',files)
