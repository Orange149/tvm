from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src'))
from polling_replay import PollingFinite,PollingFluid


def stage():
    return dict(device='vta',mask=[0,1,2,3],worker_core_map=[],set_cpu_ns=0,get_cpu_ns=0,
        set_nonrunning_ns=0,get_nonrunning_ns=0,
        device_profile=dict(host_serial_CPU=[0,0],devices=[dict(CPU=20,wall=100,
            poll_checks=10,submit_mmio_ns=2,post_start_sleep_ns=10)]))


def test_driver_chronology_has_identical_isolated_service_and_useful_CPU():
    outcomes=[]
    for cls in (PollingFinite,PollingFluid):
        for mode in ('aggregate','periodic'):
            extra=dict(quanta={n:10 for n in range(2,7)}) if cls is PollingFinite else {}
            sim=cls([stage()],driver_timing=mode,placement='wake_boundary',
                frames=4,warmup=0,scored=3,queue_depth=2,**extra)
            result=sim.run()
            outcomes.append(result['replay']['useful_CPU_ns'])
            assert result['replay']['interval_ns']==100
            assert result['replay']['processed_frames']==4
            assert not result['within_device_call_extra_migration_choices']
            assert not sim.switches
            assert {key[-1] for key in sim.assignment_counts}=={0}
    assert outcomes==[80]*4
