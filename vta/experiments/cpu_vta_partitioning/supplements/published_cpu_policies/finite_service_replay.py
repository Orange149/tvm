"""Finite-service hypothesis over the frozen isolated fork/join/queue model.

No native DNN timing, scheduler traces, fitted slowdown or observed controls
are accepted. Host placement follows an existing unfitted history rule.
"""
from collections import defaultdict
from fractions import Fraction
from model_semantics_audit.simulator import AuditReplay,exact
from phase_model.simulator import ceiling
from static_revision.trace_replay import TraceReplay
from thread_history_audit.simulator import actor
from finite_service.scheduler import Scheduler


class FiniteReplay(TraceReplay):
    def __init__(self,stages,*,quanta,placement='wake_boundary',**kwargs):
        if placement not in ('wake_boundary','retain'):raise ValueError('finite service needs a history placement rule')
        if kwargs.get('controls'):raise ValueError('finite predictor rejects observed controls')
        self.scheduler=Scheduler(quanta);self.dispatches=0
        super().__init__(stages,placement=placement,**kwargs)

    def descriptors(self):
        return {k:dict(actor=actor(t['meta']),core=t['assigned_core'],remaining=t['remaining']) for k,t in self.active.items()}

    def _close_task(self,key):
        before=len(self.task_records)
        AuditReplay._close_task(self,key)
        if len(self.task_records)>before:
            row=self.task_records[-1]
            row['ready_not_running']=row.pop('ideal_service_deprivation')
            row['service_mode']='finite_virtual_runtime'

    def worker_done(self,fork,slot):
        group=self.forks[fork];last=len(group['pending'])==1
        starts=dict(group['starts']);n=len(self.kernel_records)
        AuditReplay.worker_done(self,fork,slot)
        if last and len(self.kernel_records)>n:
            row=self.kernel_records[n]
            if len(starts)!=row['tasks']:raise ValueError('finite worker first-service record missing')
            row['first_service_skew_ns']=ceiling(max(starts.values())-min(starts.values()))
            row['first_service_ns_by_slot']={str(s):ceiling(t) for s,t in starts.items()}
            row['model_assumes_immediate_positive_fluid_service']=False

    def first_service(self,key):
        t=self.active[key]
        if t['first'] is not None:return
        t['first']=self.time;m=t['meta']
        if m['operation']!='worker_compute':return
        matching=[g for g in self.forks.values() if g['owner']==m['stage'] and g['frame']==m['frame'] and
                  g['meta']['kernel']==m['kernel'] and g['meta']['generation']==m['generation']]
        if len(matching)!=1:raise ValueError('finite first-service fork identity ambiguous')
        matching[0]['starts'][m['slot']]=self.time

    def run(self):
        while not self._complete():
            self.settle()
            if self._complete():break
            descriptor=self.descriptors();selected,expiry=self.scheduler.select(descriptor)
            keys=set(selected.values())
            nexts=[self.active[k]['remaining'] for k in keys if self.active[k]['remaining'] is not None]
            nexts += [t-self.time for t,_ in self.delays.values()]
            if expiry is not None:nexts.append(expiry)
            if not nexts:raise ValueError('finite dependency replay stalled')
            delta=min(nexts)
            if delta<0:raise ValueError('finite time moved backwards')
            for k in keys:self.first_service(k)
            self.scheduler.consume(descriptor,delta)
            window=(self.warmup in self.completions and self.warmup+self.scored-1 not in self.completions)
            finished=[]
            for k,t in self.active.items():
                used=delta if k in keys else Fraction(0)
                t['CPU']+=used;t['deprived']+=delta-used;c=t['assigned_core']
                t['core_CPU'][c]=t['core_CPU'].get(c,Fraction(0))+used
                if window and used:
                    m=t['meta'];group=actor(m)+(m['operation'],t['waiting'])
                    self.window_totals.setdefault(group,[Fraction(0) for _ in range(4)])[c]+=used
                if t['waiting']:self.wait_work+=used
                else:
                    self.cpu_work+=used
                    if t['meta']['role']=='stage':self.graph_CPU+=used
                    else:self.control_CPU+=used
                    t['remaining']-=used
                    if t['remaining']==0 and k in keys:finished.append(k)
            self.time+=delta;callbacks=[]
            for k in finished:self._close_task(k);callbacks.append(self.active.pop(k)['callback'])
            due=[(k,i) for k,(t,i) in self.delays.items() if t==self.time]
            for k,_ in due:del self.delays[k]
            for fn in callbacks:fn()
            for _,i in due:self.advance(i)
            self.events+=1
            if self.events>1_000_000:raise ValueError('finite dependency replay unbounded')
        if self.active or self.delays or self.forks or self.group_owners or self.blocked_outputs:
            raise ValueError('finite replay resources not drained')
        if self.graph_CPU+self.control_CPU!=self.cpu_work:raise ValueError('finite CPU work not conserved')
        kept=[self.completions[i] for i in range(self.warmup,self.warmup+self.scored)]
        duration=kept[-1]-kept[0]
        core_sum=[sum(v[c] for v in self.window_totals.values()) for c in range(4)]
        if any(c<0 or c>duration for c in core_sum):raise ValueError('finite core capacity exceeded')
        replay=dict(interval_ns=ceiling(duration/(self.scored-1)),timestamps_ns=list(map(ceiling,kept)),
            stage_service_ns=[ceiling(sum(self.ends[i,f]-self.starts[i,f] for f in range(self.warmup,self.warmup+self.scored))/self.scored) for i in range(len(self.stages))],
            useful_CPU_ns=ceiling(self.cpu_work),synthetic_wait_CPU_ns=ceiling(self.wait_work),
            processed_frames=len(self.completions),events=self.events,control_CPU_unmodeled=not self.control_enabled,
            service_model='finite_virtual_runtime',CPU_gate=self.gate,group_acquisitions=self.group_acquisitions,
            isolated_wait_CPU_not_added_as_constant=True)
        totals=[]
        for key,v in sorted(self.totals.items()):
            role,stage,slot,op,kernel,generation=key
            totals.append(dict(role=role,stage=stage,slot=slot,operation=op,kernel=kernel,generation=generation,
                calls=v['calls'],CPU=exact(v['CPU']),ready_not_running=exact(v['deprived']),
                elapsed=exact(v['elapsed']),core_CPU={str(c):exact(n) for c,n in v['core_CPU'].items()}))
        return dict(replay=replay,actor_totals=totals,tasks=self.task_records,kernel_joins=self.kernel_records,
            window_cpu=dict(duration=exact(duration),gaps=self.scored-1,
                rows=[dict(role=k[0],stage=k[1],slot=k[2],operation=k[3],synthetic_wait=k[4],
                           core_CPU=[exact(x) for x in v]) for k,v in sorted(self.window_totals.items())],
                core_totals=[exact(x) for x in core_sum]),
            placement=self.placement,scheduler_quanta_ns={str(k):int(v) for k,v in self.scheduler.quanta.items()},
            Linux_CFS_reproduced=False,no_native_DNN_values_used=True,
            no_positive_service_for_queued_actors=True)
