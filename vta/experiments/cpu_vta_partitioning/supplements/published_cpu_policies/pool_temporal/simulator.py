"""Extend the frozen ideal-sharing replay with real gates and barriers.

This models execution semantics; it does not reconstruct Linux scheduling.
CPU worker participation is per launch, not assumed equal to configured slots.
"""
from phase_model.simulator import Replay,rational


class PoolReplay(Replay):
    def __init__(self,stages,*,gate='off',**kwargs):
        if gate not in ('off','resource_group'):raise ValueError('unknown CPU gate')
        self.gate=gate;self.group_owners={};self.group_acquisitions=0
        groups=[set(s['mask']) for s in stages if s['device']=='cpu']
        if gate=='resource_group' and any(a!=b and a&b for a in groups for b in groups):
            raise ValueError('partially overlapping mutex groups are illegal')
        super().__init__(stages,**kwargs)

    def actions(self,stage):
        mask=stage['mask']
        if stage['device']=='vta':
            timing=stage['device_profile']
            yield ('cpu',rational(stage['set_cpu_ns']),mask)
            yield ('delay',rational(stage.get('set_nonrunning_ns',0)))
            for i,device in enumerate(timing['devices']):
                yield ('cpu',rational(timing['host_serial_CPU'][i]),mask)
                cpu=rational(device['CPU']);off=rational(device['wall'])-cpu
                if off<0:raise ValueError('device thread CPU exceeds device wall')
                pair=(('cpu',cpu,mask),('delay',off))
                yield from (pair if self.host_order=='before' else pair[::-1])
            yield ('cpu',rational(timing['host_serial_CPU'][-1]),mask)
            yield ('cpu',rational(stage['get_cpu_ns']),mask)
            yield ('delay',rational(stage.get('get_nonrunning_ns',0)))
            return
        mapping=stage['worker_core_map'];profile=stage['profile']
        if sorted(mapping)!=sorted(mask):raise ValueError('worker/core group differs')
        caller=mapping[0]
        yield ('cpu',rational(stage['set_cpu_ns']),[caller])
        yield ('delay',rational(stage.get('set_nonrunning_ns',0)))
        for i,kernel in enumerate(profile['kernels']):
            tasks=kernel['tasks'];generations=kernel['barrier_generations']
            if not 1<=tasks<=len(mapping) or len(kernel['workers'])!=tasks:
                raise ValueError('observed launch participants exceed configured CPU slots')
            if any(len(w['compute_segments'])!=generations+1 for w in kernel['workers']):
                raise ValueError('worker barrier generations differ')
            yield ('cpu',rational(profile['serial_cpu_segments'][i]),[caller])
            yield ('cpu',rational(kernel['launch_prefix_cpu']),[caller])
            for generation in range(generations+1):
                work=[rational(w['compute_segments'][generation]) for w in kernel['workers']]
                yield ('fork',work,mapping[:tasks],generation<generations)
        yield ('cpu',rational(profile['serial_cpu_segments'][-1]),[caller])
        yield ('cpu',rational(stage['get_cpu_ns']),[caller])
        yield ('delay',rational(stage.get('get_nonrunning_ns',0)))

    def advance(self,index):
        action=next(self.state[index]['iterator'],None)
        if action is None:self.finish_stage(index)
        elif action[0]=='cpu':
            self.start_cpu(index,action[1],action[2],lambda:self.advance(index))
        elif action[0]=='delay':
            if action[1]<0:raise ValueError('negative delay')
            if action[1]==0:self.advance(index)
            else:self.delays[self.new_id()]=(self.time+action[1],index)
        elif action[0]=='fork':
            work,mapping,barrier=action[1:];fork=self.new_id()
            if any(n<0 for n in work):raise ValueError('negative thread work')
            self.forks[fork]=dict(owner=index,pending=set(range(len(work))),finished=set(),
                                   waiters={},mapping=mapping,barrier=barrier)
            for slot,(amount,core) in enumerate(zip(work,mapping)):
                self.active[self.new_id()]=dict(owner=index,remaining=amount,mask=(core,),waiting=False,
                    callback=lambda f=fork,s=slot:self.worker_done(f,s))
        else:raise ValueError('unknown temporal action')

    def worker_done(self,fork,slot):
        group=self.forks[fork];group['pending'].remove(slot);group['finished'].add(slot)
        if not group['pending']:
            for waiter in group['waiters'].values():self.active.pop(waiter,None)
            del self.forks[fork];self.advance(group['owner']);return
        if self.runnable_wait:
            waiting=group['finished'] if group['barrier'] else group['finished']&{0}
            for task in sorted(waiting):
                if task not in group['waiters']:
                    group['waiters'][task]=self.start_cpu(group['owner'],None,[group['mapping'][task]],None,waiting=True)

    def finish_stage(self,index):
        stage=self.stages[index]
        if self.gate=='resource_group' and stage['device']=='cpu':
            group=tuple(stage['mask'])
            if self.group_owners.pop(group,None)!=index:raise ValueError('CPU gate ownership lost')
        # Release resource before attempting a potentially blocked output push.
        super().finish_stage(index)

    def settle(self):
        changed=True
        while changed:
            changed=False
            while self.source<self.frames and len(self.queues[0])<self.depth:
                self.queues[0].append(self.source);self.source+=1;changed=True
            for i in reversed(range(len(self.stages))):
                frame=self.pending_output[i]
                if frame is not None:
                    if i==len(self.stages)-1:
                        self.completions[frame]=self.time;self.pending_output[i]=None;changed=True
                    elif len(self.queues[i+1])<self.depth:
                        self.queues[i+1].append(frame);self.pending_output[i]=None;changed=True
            for i,stage in enumerate(self.stages):
                if self.state[i] is not None or self.pending_output[i] is not None or not self.queues[i]:continue
                if stage['device']=='vta' and self.device_owner is not None:continue
                group=tuple(stage['mask'])
                if stage['device']=='cpu' and self.gate=='resource_group' and group in self.group_owners:continue
                frame=self.queues[i].popleft();self.starts[i,frame]=self.time
                self.state[i]=dict(frame=frame,iterator=iter(self.actions(stage)))
                if stage['device']=='vta':self.device_owner=i
                elif self.gate=='resource_group':
                    self.group_owners[group]=i;self.group_acquisitions+=1
                self.advance(i);changed=True

    def run(self):
        result=super().run()
        if self.group_owners or self.forks or self.active or self.delays:
            raise ValueError('temporal resources leaked after queue drain')
        return dict(**result,CPU_gate=self.gate,group_acquisitions=self.group_acquisitions,
                    isolated_wait_CPU_not_added_as_constant=True)
