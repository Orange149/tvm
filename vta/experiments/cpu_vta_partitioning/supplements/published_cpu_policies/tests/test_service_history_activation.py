from fractions import Fraction
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit_service_history_activation import ObserveHistory
from finite_service.scheduler import Scheduler


def test_logging_does_not_change_choices_budgets_or_runtime_history():
    q={n:12 for n in range(2,7)}
    original=Scheduler(q);control=Scheduler(q);audit=ObserveHistory(original)
    solo={1:dict(actor=('stage',0,0),core=0,remaining=100)}
    competing={**solo,2:dict(actor=('stage',1,0),core=0,remaining=100)}
    for tasks,duration in [(solo,20),(competing,12),(competing,5),(solo,8),(competing,2)]:
        assert audit.select(tasks)==control.select(tasks)
        audit.consume(tasks,Fraction(duration));control.consume(tasks,Fraction(duration))
        assert original.__dict__==control.__dict__
    rows=audit.result()
    assert rows and rows[0]['service_since_observed_dispatch_ns']==20
    assert rows[0]['prior_service_exceeds_new_budget'] is True


def test_idle_core_does_not_inherit_previous_dispatch_service():
    audit=ObserveHistory(Scheduler({n:12 for n in range(2,7)}))
    solo={1:dict(actor=('stage',0,0),core=0,remaining=100)}
    audit.select(solo);audit.consume(solo,Fraction(30));audit.select({})
    audit.select(solo);audit.consume(solo,Fraction(1))
    audit.select({**solo,2:dict(actor=('stage',1,0),core=0,remaining=100)})
    assert audit.result()[-1]['service_since_observed_dispatch_ns']==1
