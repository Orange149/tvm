from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src'))
from device_completion_replay import CompletionFinite,CompletionFluid
from test_device_completion_replay import device


def test_independent_device_completes_before_host_return_and_drains_pipeline():
    for cls in (CompletionFinite,CompletionFluid):
        stage=dict(device='vta',mask=[0,1,2,3],worker_core_map=[],set_cpu_ns=0,get_cpu_ns=0,
            device_profile=dict(host_serial_CPU=[0,0],devices=[device()]))
        extra=dict(quanta={n:10 for n in range(2,7)}) if cls is CompletionFinite else {}
        sim=cls([stage],placement='wake_boundary',frames=4,warmup=0,scored=3,queue_depth=2,**extra)
        result=sim.run();calls=result['driver_completion']
        assert len(calls)==4 and {r['frame'] for r in calls}=={0,1,2,3}
        assert all(r['checks']==9 for r in calls)
        assert result['replay']['interval_ns']==101 and result['replay']['processed_frames']==4
        assert result['replay']['useful_CPU_ns']==44
        assert all(c['host_return_ns']['numerator']/c['host_return_ns']['denominator']>=
                   c['device_done_ns']['numerator']/c['device_done_ns']['denominator'] for c in calls)
        assert not result['device_completion_is_not_independent_hardware_event']
        assert not sim.active and not sim.delays and not sim.device_owner
