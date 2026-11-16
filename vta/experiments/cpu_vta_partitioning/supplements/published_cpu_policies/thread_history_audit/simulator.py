"""Limit arbitrary migration while retaining the original fluid CPU service.

These are unfitted sensitivity hypotheses. Nonrunning profile intervals do
not establish real sleeps, and strict retention is not Linux's full policy.
"""
from collections import Counter,defaultdict
from fractions import Fraction
from model_semantics_audit.simulator import AuditReplay,exact
from phase_model.simulator import rational


def actor(meta):
    return (meta['role'],meta.get('stage',-1),meta.get('slot',0))


class HistoryReplay(AuditReplay):
    def __init__(self,stages,*,placement='instant',**kwargs):
        if placement not in ('instant','retain','wake_boundary'):raise ValueError('unknown placement hypothesis')
        if kwargs.get('fixed_hosts') is not None:raise ValueError('history hypotheses cannot receive fitted host cores')
        self.placement=placement;self.actor_core={};self.potential_wakes=set()
        self.assignment_counts=Counter();self.switches=Counter();self.window_totals={}
        self._running=False
        super().__init__(stages,trace=True,**kwargs)

    def actions(self,stage):
        index=next(i for i,s in enumerate(self.stages) if s is stage)
        for action in super().actions(stage):
            if action[0]=='delay' and rational(action[1])>0:
                self.potential_wakes.add(('stage',index,0))
            yield action

    def settle(self):
        # Model queue/resource waits are known. Positive nonrunning profile
        # spans above are only possible, not observed, sleep/wakeup points.
        for i,s in enumerate(self.stages):
            if self.state[i] is not None:continue
            if not self.queues[i] or self.pending_output[i] is not None or (s['device']=='vta' and self.device_owner is not None):
                self.potential_wakes.add(('stage',i,0))
        if self.control_enabled:
            if not self.collector_active and not self.queues[-1]:self.potential_wakes.add(('_collector',-1,0))
            if self.producer_pending is not None and len(self.queues[0])>=self.depth:self.potential_wakes.add(('_producer',-1,0))
        return super().settle()

    def _loads(self):
        loads=Counter()
        for t in self.active.values():loads[t['assigned_core']]+=1
        return loads

    def _new_task(self,owner,work,mask,callback,meta,waiting=False,queue_zero=False):
        if self.placement=='instant':return super()._new_task(owner,work,mask,callback,meta,waiting,queue_zero)
        if work==0 and not queue_zero:return super()._new_task(owner,work,mask,callback,meta,waiting,queue_zero)
        who=actor(meta);previous=self.actor_core.get(who);loads=self._loads()
        if len(mask)==1:core=mask[0]
        elif previous is None:
            core=min(mask,key=lambda c:(loads[c],c))
        elif self.placement=='retain' or who not in self.potential_wakes:
            core=previous
        else:
            core=min(mask,key=lambda c:(loads[c],c!=previous,c))
        if core not in mask:raise ValueError('logical actor escaped original allowed set')
        if previous is not None and core!=previous:self.switches[who]+=1
        self.actor_core[who]=core;self.potential_wakes.discard(who)
        self.assignment_counts[who+(core,)]+=1
        key=super()._new_task(owner,work,mask,callback,meta,waiting,queue_zero)
        if key is not None:self.active[key]['assigned_core']=core
        return key

    def _close_task(self,key):
        # Keep existing aggregate accounting and joins, not all task traces.
        trace=self.trace;self.trace=False
        try:return super()._close_task(key)
        finally:self.trace=trace

    def rates(self):
        if self.placement=='instant':rates=super().rates()
        else:
            cores={c:[] for c in range(4)};self.core_choices={}
            for key,t in self.active.items():
                core=t['assigned_core']
                if core not in t['mask']:raise ValueError('assignment outside allowed mask')
                cores[core].append(key);self.core_choices[key]=core
            rates={key:Fraction(1,len(keys)) for keys in cores.values() for key in keys}
        # Completion milestones occur at event boundaries. Integrate exactly
        # between C_warmup and C_(warmup+scored-1), including CPU and wait work.
        if self._running and self.warmup in self.completions and self.warmup+self.scored-1 not in self.completions:
            events=[t['remaining']/rates[k] for k,t in self.active.items() if t['remaining'] is not None]
            events += [when-self.time for when,_ in self.delays.values()]
            if events:
                duration=min(events)
                for key,t in self.active.items():
                    m=t['meta'];group=actor(m)+(m['operation'],t['waiting'])
                    row=self.window_totals.setdefault(group,[Fraction(0) for _ in range(4)])
                    row[self.core_choices[key]]+=duration*rates[key]
        return rates

    def run(self):
        self._running=True
        try:result=super().run()
        finally:self._running=False
        duration=self.completions[self.warmup+self.scored-1]-self.completions[self.warmup]
        core_sum=[sum(v[c] for v in self.window_totals.values()) for c in range(4)]
        if any(n<0 or n>duration for n in core_sum):raise ValueError('window CPU exceeds per-core capacity')
        result['placement']=self.placement
        result['window_cpu']=dict(duration=exact(duration),gaps=self.scored-1,
            rows=[dict(role=k[0],stage=k[1],slot=k[2],operation=k[3],synthetic_wait=k[4],core_CPU=[exact(x) for x in v])
                  for k,v in sorted(self.window_totals.items())],
            core_totals=[exact(v) for v in core_sum],
            native_read_envelope_not_modeled=True)
        result['assignment_counts']=[dict(role=k[0],stage=k[1],slot=k[2],core=k[3],count=v) for k,v in sorted(self.assignment_counts.items())]
        result['placement_changes']=[dict(role=k[0],stage=k[1],slot=k[2],count=v) for k,v in sorted(self.switches.items())]
        result['Linux_scheduler_reproduced']=False
        result['instantaneous_positive_fluid_service_unchanged']=True
        return result
