from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src'))
from audit_poll_wakeup_gap import host_cycles


def test_only_complete_nonpreemptive_cycles_inside_same_run_are_counted():
    def e(t,d,c=0,p=False):return dict(time_ns=t,direction=d,cpu=c,preempt=p)
    stream=dict(tid=7,role=dict(role='host'),events=[e(1,'in'),e(2,'out'),e(3,'in',1),
        e(4,'out',1,True),e(5,'in',1),e(6,'out',1),e(9,'in',2)])
    phases=[dict(operation='run',tid=7,wall_start_ns=0,wall_end_ns=8,call=0)]
    rows=host_cycles(stream,phases,0,10)
    assert len(rows)==1 and rows[0]['on_CPU_ns']==1 and rows[0]['nonpreempt_off_ns']==1
    assert rows[0]['migrated'] and not rows[0]['exact_wakeup_time_observed']
