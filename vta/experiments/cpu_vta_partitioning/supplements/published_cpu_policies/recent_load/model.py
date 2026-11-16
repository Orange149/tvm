"""Keep a recent load estimate when movable host work wakes.

Linux v5.4 pelt.c fixes y^32=1/2; a measurement unit is 1024ns and
a period is 1024 units. This continuous EWMA is a declared approximation,
not a bit-exact PELT port or a reproduction of select_task_rq_fair.
"""
from collections import Counter
from fractions import Fraction

from finite_service.scheduler import Scheduler
from finite_service_replay import FiniteReplay
from dispatch_service.model import DispatchScheduler
from model_semantics_audit.simulator import AuditReplay
from thread_history_audit.simulator import actor

HALF_LIFE_NS=32*1024*1024
METRIC_SCALE=1<<24


class RecentLoad:
    def __init__(self):
        self.average={c:0.0 for c in range(4)}

    def update(self,counts,duration):
        if duration<0:raise ValueError('negative load-history interval')
        if any(c not in range(4) or n<0 for c,n in counts.items()):
            raise ValueError('invalid recent load count')
        decay=2.0**(-float(duration)/HALF_LIFE_NS)
        for c in range(4):
            self.average[c]=decay*self.average[c]+(1-decay)*counts.get(c,0)

    def metric(self,core):
        if core not in self.average:raise ValueError('invalid recent-load core')
        return round(self.average[core]*METRIC_SCALE)


def choose_core(mask,counts,history,previous):
    if not mask or len(mask)!=len(set(mask)) or any(c not in range(4) for c in mask):
        raise ValueError('invalid movable core mask')
    # A declared load-balancing approximation: an idle allowed core first;
    # otherwise recent runnable demand, then current count and cache affinity.
    # A task never moves merely because another model event occurs.
    return min(mask,key=lambda c:(counts.get(c,0)!=0,history.metric(c),
                                  counts.get(c,0),c!=previous,c))


class LoadScheduler:
    def __init__(self,base,history):
        self.base=base;self.history=history

    def __getattr__(self,key):return getattr(self.base,key)

    def select(self,tasks):return self.base.select(tasks)

    def consume(self,tasks,duration):
        counts=Counter(t['core'] for t in tasks.values())
        self.base.consume(tasks,duration)
        self.history.update(counts,duration)


class RecentLoadReplay(FiniteReplay):
    def __init__(self,stages,*,quanta,dispatch_elapsed=False,**kwargs):
        if kwargs.get('placement','wake_boundary')!='wake_boundary':
            raise ValueError('recent load has one frozen wake-boundary policy')
        self.recent_load=RecentLoad();self.dispatch_elapsed=dispatch_elapsed
        super().__init__(stages,quanta=quanta,**kwargs)
        base=DispatchScheduler(quanta) if dispatch_elapsed else Scheduler(quanta)
        self.scheduler=LoadScheduler(base,self.recent_load)

    def _new_task(self,owner,work,mask,callback,meta,waiting=False,queue_zero=False):
        if work==0 and not queue_zero:
            return AuditReplay._new_task(self,owner,work,mask,callback,meta,waiting,queue_zero)
        who=actor(meta);previous=self.actor_core.get(who);counts=self._loads()
        if len(mask)==1:core=mask[0]
        elif previous is not None and who not in self.potential_wakes:core=previous
        else:core=choose_core(mask,counts,self.recent_load,previous)
        if core not in mask:raise ValueError('recent-load task escaped original allowed cores')
        if previous is not None and previous!=core:self.switches[who]+=1
        self.actor_core[who]=core;self.potential_wakes.discard(who)
        self.assignment_counts[who+(core,)]+=1
        key=AuditReplay._new_task(self,owner,work,mask,callback,meta,waiting,queue_zero)
        if key is not None:self.active[key]['assigned_core']=core
        return key

    def run(self):
        result=super().run()
        result['recent_load_placement']=True
        result['recent_load_half_life_ns']=HALF_LIFE_NS
        result['recent_load_metric_scale']=METRIC_SCALE
        result['dispatch_elapsed_accounting']=self.dispatch_elapsed
        result['PELT_bit_exact_reproduction']=False
        result['Linux_wake_affine_reproduced']=False
        result['no_observed_core_distribution_as_input']=True
        return result
