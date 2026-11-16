from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit_critical_core_schedule import ServiceRecorder, model_occupancy


class Delegate:
    def __init__(self):
        self.choices={0:7}
        self.received=[]

    def consume(self,tasks,duration):
        self.received.append((tasks,duration))


def test_recorder_delegates_exact_duration_and_merges_only_same_work():
    delegate=Delegate()
    task=dict(meta=dict(role='stage',stage=0,frame=12,slot=0,operation='worker_compute',kernel=6),waiting=False)
    owner=SimpleNamespace(time=Fraction(0),scheduler=delegate,active={7:task})
    recorder=ServiceRecorder(owner,0,20)
    descriptor={7:dict(actor=('stage',0,0),core=0)}
    recorder.consume(descriptor,Fraction(7,2))
    owner.time=Fraction(7,2)
    recorder.consume(descriptor,Fraction(9,2))
    assert delegate.received==[(descriptor,Fraction(7,2)),(descriptor,Fraction(9,2))]
    assert len(recorder.rows[0])==1
    assert (recorder.rows[0][0]['start_ns'],recorder.rows[0][0]['end_ns'])==(0,8)
    assert owner.time==Fraction(7,2)


def test_other_core_work_and_idle_are_not_counted_as_target_cpu():
    rows={0:[dict(start_ns=0,end_ns=4,identity=['stage',0,12,0,'worker_compute',6]),
             dict(start_ns=4,end_ns=9,identity=['stage',1,11,0,'host_serial',-1])]}
    recorder=SimpleNamespace(rows=rows)
    task=dict(first_service_ns=0,end_ns=10,role='stage',stage=0,slot=0)
    cfg=dict(stages=[dict(device='cpu',program='cpu_00_02'),dict(device='vta',program='vta_03_06')])
    value=model_occupancy(recorder,task,0,cfg)
    assert value==dict(model_on_CPU_ns=4,model_other_on_core_ns=5,model_idle_ns=1,
                       model_competing_roles_ns={'vta_03_06/host':5})
