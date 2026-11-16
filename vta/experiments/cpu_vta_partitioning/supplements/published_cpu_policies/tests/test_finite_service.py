from pathlib import Path
from fractions import Fraction
import sys,pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT.parents[1]/'src')]
from finite_service.scheduler import Scheduler
from finite_service.synthetic import run,mixed_actions
from finite_service.native import source_text,board
from finite_service.common import protocol

Q={n:12 for n in range(2,7)}


def test_one_physical_actor_per_core():
    s=Scheduler(Q);tasks={0:dict(actor='a',core=0,remaining=100),1:dict(actor='b',core=0,remaining=100)}
    selected,deadline=s.select(tasks)
    assert selected=={0:0} and deadline==12
    s.consume(tasks,12);selected,deadline=s.select(tasks)
    assert selected=={0:1} and s.vruntime['a']==12


def test_actor_runtime_persists_across_task_boundaries():
    s=Scheduler(Q);tasks={0:dict(actor='a',core=0,remaining=5),1:dict(actor='b',core=0,remaining=100)}
    s.select(tasks);s.consume(tasks,5);tasks[2]=tasks.pop(0);chosen,left=s.select(tasks)
    assert chosen[0]==2 and left==7


def test_competition_conserves_CPU_without_simultaneous_completion():
    a=[[('cpu',30)],[('cpu',30)]];finite=run(a,Q);fluid=run(a,Q,fluid=True)
    assert finite['CPU_ns']==fluid['CPU_ns']==60
    assert finite['wall_ns']!=fluid['wall_ns'] and fluid['wall_ns']=={'0':60,'1':60}


def test_sleeping_work_does_not_consume_CPU():
    x=run([[('cpu',10),('sleep',20),('cpu',5)]],Q)
    assert x['CPU_ns']==15 and x['wall_ns']=={'0':35}


def test_platform_scope_and_patched_probe():
    p=protocol();text=source_text()
    assert p['DNN_inferences']==0 and p['CPU_work_budget_ns']==1_696_480_000
    assert 'rows.size()==23' in text and '{"busy6",6}' in text and 'pin(1);' in text
    assert 'repeat<16' in text and 'i==1?1000000:30000' in text
    assert sum(a for actions in mixed_actions() for op,a in actions if op=='cpu')==96_480_000


def test_reject_unmeasured_count_and_implicit_board(tmp_path):
    s=Scheduler(Q)
    with pytest.raises(ValueError,match='outside measured'):s.select({i:dict(actor=i,core=0,remaining=1) for i in range(7)})
    with pytest.raises(ValueError,match='execute-board'):board(tmp_path)
