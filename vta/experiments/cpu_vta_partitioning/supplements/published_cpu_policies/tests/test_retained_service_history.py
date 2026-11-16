from fractions import Fraction
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from finite_service.scheduler import Scheduler
from retained_service_history import RetainedServiceHistory


def scheduler():return RetainedServiceHistory(Scheduler({2:12,3:8,4:6,5:5,6:4}))
def task(actor):return dict(actor=('stage',actor,0),core=0,remaining=100)


def test_uncontended_service_reduces_budget_when_competitor_arrives():
    s=scheduler();a={1:task(0)};s.select(a);s.consume(a,Fraction(5))
    chosen,expiry=s.select({**a,2:task(1)})
    assert chosen[0]==1 and expiry==7 and s.service_used[0]==5


def test_recomputed_capacity_uses_total_dispatch_history():
    s=scheduler();a={1:task(0)};s.select(a);s.consume(a,Fraction(5))
    b={**a,2:task(1)};s.select(b);s.consume(b,Fraction(2))
    chosen,expiry=s.select({**b,3:task(2)})
    assert chosen[0]==1 and expiry==1


def test_expired_history_can_dispatch_an_already_waiting_actor():
    s=scheduler();a={1:task(0)};s.select(a);s.consume(a,Fraction(20))
    s.vruntime[('stage',1,0)]=Fraction(18)
    chosen,expiry=s.select({**a,2:task(1)})
    assert chosen[0]==2 and expiry==12 and s.service_used[0]==0


def test_idle_and_real_dispatch_reset_history():
    s=scheduler();a={1:task(0)};s.select(a);s.consume(a,Fraction(8))
    s.select({});s.select(a);assert s.service_used[0]==0
    b={**a,2:task(1)};_,limit=s.select(b);assert limit==12
