from pathlib import Path
from fractions import Fraction
import sys,pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src'))
from device_completion_replay import parameters,CompletionMixin


def device():
    return dict(CPU=12,wall=112,poll_checks=10,submit_mmio_ns=2,
        post_start_sleep_ns=10,completion_lower_ns=80,completion_upper_ns=100)


def test_profile_separates_device_clock_cpu_work_and_completion_bracket():
    p=parameters(device(),'midpoint')
    assert p['duration']==90 and p['submit_CPU']==2 and p['poll_CPU']==1 and p['sleep']==10
    assert parameters(device(),'lower')['duration']==80
    assert parameters(device(),'upper')['duration']==100
    for change in [dict(CPU=113),dict(poll_checks=0),dict(completion_lower_ns=101)]:
        with pytest.raises(ValueError):parameters({**device(),**change},'midpoint')


class Harness(CompletionMixin):
    """Drive the same generator with a synthetic host deschedule interval."""
    def __init__(self,stall=0):
        self.stage=dict(device='vta',mask=[0],set_cpu_ns=0,get_cpu_ns=0,
            device_profile=dict(host_serial_CPU=[0,0],devices=[device()]))
        self.stages=[self.stage];self.state=[dict(frame=0)];self.time=Fraction(0)
        self.potential_wakes=set();self.completion_model='independent';self.completion_point='midpoint'
        self.device_records=[];self.stall=stall
    def execute(self):
        stalled=False
        for action in self.actions(self.stage):
            if action[0]=='cpu' and action[3]['operation']=='driver_poll' and not stalled:
                self.time+=self.stall;stalled=True
            self.time+=action[1]
        return self.device_records[0]


def test_unscheduled_host_does_not_pause_device_or_force_old_poll_count():
    normal=Harness().execute();stalled=Harness(100).execute()
    assert normal['device_done_ns']==stalled['device_done_ns']==92
    assert normal['checks']==9 and stalled['checks']==1
    assert stalled['host_return_ns']==113 and stalled['CPU']<normal['CPU']
    assert normal['host_return_ns']>=normal['device_done_ns']


def test_bounds_are_observation_uncertainty_not_fitted_pipeline_penalties():
    h=Harness();h.completion_point='upper';u=h.execute()
    h=Harness();h.completion_point='lower';l=h.execute()
    assert l['device_done_ns']<=u['device_done_ns']
    assert l['checks']<=u['checks']


def test_first_poll_completion_and_nonprogressing_invalid_loop():
    h=Harness();h.stage['device_profile']['devices'][0].update(completion_lower_ns=0,completion_upper_ns=0)
    assert h.execute()['checks']==1
    h=Harness();h.stage['device_profile']['devices'][0].update(CPU=0,wall=0,submit_mmio_ns=0,post_start_sleep_ns=0)
    with pytest.raises(ValueError,match='progress'):h.execute()
