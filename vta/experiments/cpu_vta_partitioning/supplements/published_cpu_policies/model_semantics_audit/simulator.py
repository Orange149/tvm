"""Parameter-free, ideal-sharing semantics variants; not a Linux scheduler.

Zero controls use the inherited settling order exactly. Diagnostic controls
have separate actors and bounded queues, and never add time directly to II.
"""
from collections import Counter,defaultdict,deque
from fractions import Fraction
from pool_temporal.simulator import PoolReplay
from phase_model.simulator import rational,ceiling


def exact(value):
    value=Fraction(value);return dict(numerator=value.numerator,denominator=value.denominator)


class AuditReplay(PoolReplay):
    def __init__(self,stages,*,fixed_hosts=None,controls=None,trace=False,**kwargs):
        self.fixed_hosts=None if fixed_hosts is None else {int(k):v for k,v in fixed_hosts.items()}
        if self.fixed_hosts is not None:
            expected={i for i,s in enumerate(stages) if s['device']=='vta'}
            if set(self.fixed_hosts)!=expected or any(self.fixed_hosts[i] not in stages[i]['mask'] for i in expected):
                raise ValueError('fixed host trajectory must cover each VTA actor inside its original allowed set')
        self.controls=controls or {}
        self.control_work={r:rational(self.controls.get(r,{}).get('CPU_ns',0)) for r in ('_producer','_collector')}
        if any(v<0 for v in self.control_work.values()):raise ValueError('negative control work')
        self.control_enabled=any(self.control_work.values());self.trace=trace
        self.task_records=[];self.kernel_records=[];self.totals={};self.core_choices={}
        self.graph_CPU=Fraction(0);self.control_CPU=Fraction(0)
        self.producer_active=False;self.producer_pending=None;self.producer_pending_since=None
        self.collector_active=False;self.collected=set();self.producer_blocked=Fraction(0)
        self.blocked_outputs={};self.output_blocked=Fraction(0);self.queue_max={}
        super().__init__(stages,**kwargs)
        if self.control_enabled:self.queues.append(deque())
        for role in self.control_work:
            mask=self.controls.get(role,{}).get('mask',[0,1,2,3])
            if not mask or len(set(mask))!=len(mask) or any(type(c) is not int or c not in range(4) for c in mask):
                raise ValueError('invalid control allowed cores')

    def actions(self,stage):
        index=next(i for i,s in enumerate(self.stages) if s is stage)
        mask=stage['mask'];mapping=stage.get('worker_core_map',[])
        if stage['device']=='vta':
            if self.fixed_hosts is not None:mask=[self.fixed_hosts[index]]
            def cpu(work,op):return ('cpu',rational(work),mask,dict(operation=op,slot=0))
            yield cpu(stage['set_cpu_ns'],'set')
            yield ('delay',rational(stage.get('set_nonrunning_ns',0)))
            timing=stage['device_profile']
            for i,device in enumerate(timing['devices']):
                yield cpu(timing['host_serial_CPU'][i],'host_serial')
                cpu_work=rational(device['CPU']);off=rational(device['wall'])-cpu_work
                if off<0:raise ValueError('device CPU exceeds wall')
                pair=(cpu(cpu_work,'device_host_CPU'),('delay',off))
                yield from (pair if self.host_order=='before' else pair[::-1])
            yield cpu(timing['host_serial_CPU'][-1],'host_serial')
            yield cpu(stage['get_cpu_ns'],'get')
            yield ('delay',rational(stage.get('get_nonrunning_ns',0)));return
        if sorted(mapping)!=sorted(mask):raise ValueError('CPU thread mapping differs')
        caller=mapping[0];profile=stage['profile']
        def cpu(work,op,kernel=-1):return ('cpu',rational(work),[caller],dict(operation=op,slot=0,kernel=kernel))
        yield cpu(stage['set_cpu_ns'],'set')
        yield ('delay',rational(stage.get('set_nonrunning_ns',0)))
        for i,kernel in enumerate(profile['kernels']):
            tasks=kernel['tasks'];generations=kernel['barrier_generations']
            if not 1<=tasks<=len(mapping) or len(kernel['workers'])!=tasks:raise ValueError('invalid participant count')
            if any(len(w['compute_segments'])!=generations+1 for w in kernel['workers']):raise ValueError('invalid barrier profile')
            yield cpu(profile['serial_cpu_segments'][i],'serial',i)
            yield cpu(kernel['launch_prefix_cpu'],'launch_prefix',i)
            for g in range(generations+1):
                work=[rational(w['compute_segments'][g]) for w in kernel['workers']]
                yield ('fork',work,mapping[:tasks],g<generations,dict(kernel=i,generation=g))
        yield cpu(profile['serial_cpu_segments'][-1],'serial')
        yield cpu(stage['get_cpu_ns'],'get')
        yield ('delay',rational(stage.get('get_nonrunning_ns',0)))

    def _new_task(self,owner,work,mask,callback,meta,waiting=False,queue_zero=False):
        if work is not None and work<0:raise ValueError('negative CPU work')
        if work==0 and not queue_zero:callback();return None
        key=self.new_id()
        self.active[key]=dict(owner=owner,remaining=work,mask=tuple(mask),callback=callback,waiting=waiting,
            meta=meta,ready=self.time,first=None,CPU=Fraction(0),deprived=Fraction(0),core_CPU={})
        return key

    def start_cpu(self,owner,work,mask,callback,*,waiting=False):
        meta=dict(role='stage',stage=owner,frame=self.state[owner]['frame'],slot=0,operation='auxiliary')
        return self._new_task(owner,work,mask,callback,meta,waiting)

    def advance(self,index):
        action=next(self.state[index]['iterator'],None)
        if action is None:self.finish_stage(index)
        elif action[0]=='cpu':
            meta=dict(role='stage',stage=index,frame=self.state[index]['frame'],**action[3])
            self._new_task(index,action[1],action[2],lambda:self.advance(index),meta)
        elif action[0]=='delay':
            if action[1]<0:raise ValueError('negative delay')
            if action[1]==0:self.advance(index)
            else:self.delays[self.new_id()]=(self.time+action[1],index)
        elif action[0]=='fork':
            work,mapping,barrier,meta=action[1:];fork=self.new_id()
            if any(v<0 for v in work):raise ValueError('negative worker work')
            self.forks[fork]=dict(owner=index,pending=set(range(len(work))),finished=set(),waiters={},mapping=mapping,
                barrier=barrier,meta=meta,frame=self.state[index]['frame'],ready=self.time,ends={},starts={})
            for slot,(amount,core) in enumerate(zip(work,mapping)):
                task_meta=dict(role='stage',stage=index,frame=self.state[index]['frame'],operation='worker_compute',slot=slot,**meta)
                self._new_task(index,amount,[core],lambda f=fork,s=slot:self.worker_done(f,s),task_meta,queue_zero=True)
        else:raise ValueError('unknown replay action')

    def _close_task(self,key):
        task=self.active[key];meta=task['meta'];elapsed=self.time-task['ready']
        if task['CPU']+task['deprived']!=elapsed:raise ValueError('per-task ideal service accounting does not conserve wall')
        groupkey=(meta['role'],meta.get('stage',-1),meta.get('slot',0),meta['operation'],meta.get('kernel',-1),meta.get('generation',-1))
        selected=self.warmup<=meta['frame']<self.warmup+self.scored
        if selected:
            total=self.totals.setdefault(groupkey,dict(calls=0,CPU=Fraction(0),deprived=Fraction(0),elapsed=Fraction(0),core_CPU={}))
            total['calls']+=1
            for field in ('CPU','deprived'):total[field]+=task[field]
            total['elapsed']+=elapsed
            for core,cpu in task['core_CPU'].items():total['core_CPU'][core]=total['core_CPU'].get(core,Fraction(0))+cpu
        if self.trace and selected:
            self.task_records.append(dict(**meta,ready_ns=ceiling(task['ready']),first_service_ns=ceiling(task['first']) if task['first'] is not None else None,
                end_ns=ceiling(self.time),CPU=exact(task['CPU']),ideal_service_deprivation=exact(task['deprived']),
                allowed_mask=list(task['mask']),synthetic_wait=task['waiting']))

    def worker_done(self,fork,slot):
        group=self.forks[fork];group['pending'].remove(slot);group['finished'].add(slot);group['ends'][slot]=self.time
        if not group['pending']:
            for waiter in group['waiters'].values():self._close_task(waiter);self.active.pop(waiter)
            if self.warmup<=group['frame']<self.warmup+self.scored:
                last=max(group['ends'].values());ties=sorted(s for s,t in group['ends'].items() if t==last)
                self.kernel_records.append(dict(stage=group['owner'],frame=group['frame'],**group['meta'],
                    tasks=len(group['mapping']),ready_ns=ceiling(group['ready']),join_ns=ceiling(self.time),
                    last_slots=ties,ready_skew_ns=0,first_service_skew_ns=0,
                    model_assumes_immediate_positive_fluid_service=True))
            del self.forks[fork];self.advance(group['owner']);return
        if self.runnable_wait:
            waiting=group['finished'] if group['barrier'] else group['finished']&{0}
            for task in sorted(waiting):
                if task not in group['waiters']:
                    meta=dict(role='stage',stage=group['owner'],frame=group['frame'],slot=task,
                              operation='barrier_wait' if group['barrier'] else 'completion_wait',**group['meta'])
                    group['waiters'][task]=self._new_task(group['owner'],None,[group['mapping'][task]],None,meta,True)

    def finish_stage(self,index):
        frame=self.state[index]['frame'];super().finish_stage(index)
        if self.control_enabled and index==len(self.stages)-1:self.completions[frame]=self.time

    def _start_control(self,role,frame):
        if role=='_producer':self.producer_active=True
        else:self.collector_active=True
        def done():
            if role=='_producer':
                self.producer_active=False;self.producer_pending=frame;self.producer_pending_since=self.time
            else:self.collector_active=False;self.collected.add(frame)
        mask=self.controls[role]['mask'];meta=dict(role=role,stage=-1,slot=0,frame=frame,operation='feed' if role=='_producer' else 'collect')
        self._new_task(-1,self.control_work[role],mask,done,meta)

    def settle(self):
        if not self.control_enabled:return super().settle()
        changed=True
        while changed:
            changed=False
            if self.producer_pending is not None and len(self.queues[0])<self.depth:
                self.queues[0].append(self.producer_pending)
                self.producer_blocked+=self.time-self.producer_pending_since
                self.producer_pending=None;self.producer_pending_since=None;changed=True
            if self.producer_pending is None and not self.producer_active and self.source<self.frames:
                frame=self.source;self.source+=1;self._start_control('_producer',frame);changed=True
            for i in reversed(range(len(self.stages))):
                frame=self.pending_output[i]
                if frame is None:continue
                if len(self.queues[i+1])<self.depth:
                    self.queues[i+1].append(frame);self.pending_output[i]=None;changed=True
                    if i in self.blocked_outputs:self.output_blocked+=self.time-self.blocked_outputs.pop(i)
                else:self.blocked_outputs.setdefault(i,self.time)
            if not self.collector_active and self.queues[-1]:
                frame=self.queues[-1].popleft();self._start_control('_collector',frame);changed=True
            for i,stage in enumerate(self.stages):
                if self.state[i] is not None or self.pending_output[i] is not None or not self.queues[i]:continue
                if stage['device']=='vta' and self.device_owner is not None:continue
                group=tuple(stage['mask'])
                if stage['device']=='cpu' and self.gate=='resource_group' and group in self.group_owners:continue
                frame=self.queues[i].popleft();self.starts[i,frame]=self.time
                self.state[i]=dict(frame=frame,iterator=iter(self.actions(stage)))
                if stage['device']=='vta':self.device_owner=i
                elif self.gate=='resource_group':self.group_owners[group]=i;self.group_acquisitions+=1
                self.advance(i);changed=True
            for i,q in enumerate(self.queues):
                if len(q)>self.depth:raise ValueError('bounded queue capacity exceeded')
                self.queue_max[i]=max(self.queue_max.get(i,0),len(q))

    def rates(self):
        cores={c:[] for c in range(4)};self.core_choices={}
        for key,task in sorted(self.active.items(),key=lambda p:(len(p[1]['mask']),p[0])):
            core=min(task['mask'],key=lambda c:(len(cores[c]),c));cores[core].append(key);self.core_choices[key]=core
        return {key:Fraction(1,len(keys)) for keys in cores.values() for key in keys}

    def _complete(self):
        return len(self.completions)==self.frames and (not self.control_enabled or len(self.collected)==self.frames)

    def run(self):
        while not self._complete():
            self.settle()
            if self._complete():break
            rates=self.rates();nexts=[t['remaining']/rates[k] for k,t in self.active.items() if t['remaining'] is not None]
            nexts += [t-self.time for t,_ in self.delays.values()]
            if not nexts:raise ValueError('replay deadlocked or has only indefinite waiters')
            delta=min(nexts)
            if delta<0:raise ValueError('event time moved backwards')
            finished=[]
            for key,task in self.active.items():
                used=delta*rates[key];task['CPU']+=used;task['deprived']+=delta-used
                core=self.core_choices[key];task['core_CPU'][core]=task['core_CPU'].get(core,Fraction(0))+used
                if task['first'] is None and delta>0:task['first']=self.time
                if task['waiting']:self.wait_work+=used
                else:
                    self.cpu_work+=used
                    if task['meta']['role']=='stage':self.graph_CPU+=used
                    else:self.control_CPU+=used
                    task['remaining']-=used
                    if task['remaining']==0:finished.append(key)
            self.time+=delta
            callbacks=[]
            for key in finished:self._close_task(key);callbacks.append(self.active.pop(key)['callback'])
            due=[(k,i) for k,(t,i) in self.delays.items() if t==self.time]
            for key,_ in due:del self.delays[key]
            for callback in callbacks:callback()
            for _,index in due:self.advance(index)
            self.events+=1
            if self.events>1000000:raise ValueError('unbounded replay')
        if self.active or self.delays or self.forks or self.group_owners or self.blocked_outputs:
            raise ValueError('resources not drained at replay end')
        if self.graph_CPU+self.control_CPU!=self.cpu_work:raise ValueError('finite CPU accounting inconsistent')
        kept=[self.completions[i] for i in range(self.warmup,self.warmup+self.scored)]
        result=dict(interval_ns=ceiling((kept[-1]-kept[0])/(self.scored-1)),timestamps_ns=[ceiling(t) for t in kept],
            stage_service_ns=[ceiling(sum(self.ends[i,f]-self.starts[i,f] for f in range(self.warmup,self.warmup+self.scored))/self.scored) for i in range(len(self.stages))],
            useful_CPU_ns=ceiling(self.cpu_work),synthetic_wait_CPU_ns=ceiling(self.wait_work),processed_frames=len(self.completions),events=self.events,
            control_CPU_unmodeled=not self.control_enabled,scheduling_is_idealized=True,CPU_gate=self.gate,
            group_acquisitions=self.group_acquisitions,isolated_wait_CPU_not_added_as_constant=True)
        totals=[]
        for key,v in sorted(self.totals.items()):
            role,stage,slot,op,kernel,generation=key
            totals.append(dict(role=role,stage=stage,slot=slot,operation=op,kernel=kernel,generation=generation,calls=v['calls'],
                CPU=exact(v['CPU']),ideal_service_deprivation=exact(v['deprived']),elapsed=exact(v['elapsed']),
                core_CPU={str(c):exact(n) for c,n in sorted(v['core_CPU'].items())}))
        return dict(replay=result,accounting=dict(graph_CPU=exact(self.graph_CPU),control_CPU=exact(self.control_CPU),
            synthetic_wait_CPU=exact(self.wait_work),producer_blocked_wall=exact(self.producer_blocked),
            output_blocked_wall=exact(self.output_blocked),collected_frames=len(self.collected) if self.control_enabled else None,
            control_CPU_expected=exact(self.frames*sum(self.control_work.values())),queue_max=self.queue_max,
            completion_not_collection=True),actor_totals=totals,kernel_joins=self.kernel_records if self.trace else [],
            tasks=self.task_records,host_assignment=self.fixed_hosts,
            control_mode='diagnostic_observed' if self.control_enabled else 'zero',
            ideal_service_deprivation_is_not_TASKSTATS_delay=True)
