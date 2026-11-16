"""Refinement preserves total isolated work; no second device-wait penalty."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT.parents[1]/'src')]
from phase_model.simulator import Replay
from device_phase.predict_development import DeviceReplay


def stage():
    return dict(device='vta',mask=[0],set_cpu_ns=0,set_nonrunning_ns=0,get_cpu_ns=0,get_nonrunning_ns=0,
        run_cpu_ns=15,run_nonrunning_ns=45,
        device_profile=dict(host_serial_CPU=[3,5,2],
            devices=[dict(CPU=2,wall=20),dict(CPU=3,wall=30)]))


def test_interleaving_and_aggregation_have_identical_work_and_isolation_service():
    for order in ('before','after'):
        a=DeviceReplay([stage()],host_order=order).run()
        b=Replay([stage()],host_order=order).run()
        assert a['interval_ns']==b['interval_ns']==60
        assert a['useful_CPU_ns']==b['useful_CPU_ns']==15*28


def test_device_CPU_is_not_added_again_to_device_wait():
    row=stage();row['device_profile']['devices'][0]['CPU']=20
    row['run_cpu_ns']=33;row['run_nonrunning_ns']=27
    result=DeviceReplay([row]).run()
    assert result['interval_ns']==60
    assert result['useful_CPU_ns']==33*28
