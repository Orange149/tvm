"""Development replay of isolated CPU work, fork/join and bounded queues.

Equal CPU sharing and greedy placement of migratable host work are hypotheses,
not a reconstruction of Linux scheduling. No measured pipeline duration enters.
"""
from collections import deque
from fractions import Fraction


def rational(value):
    if isinstance(value, dict):
        return Fraction(value['numerator'], value['denominator'])
    return Fraction(value)


def ceiling(value):
    return -(-value.numerator // value.denominator)


class Replay:
    def __init__(self, stages, *, frames=28, warmup=8, scored=16, queue_depth=2,
                 runnable_completion_wait=True, host_order='before'):
        if not stages or frames < warmup + scored or scored < 2 or queue_depth < 1:
            raise ValueError('invalid replay window or queue')
        if host_order not in ('before', 'after'):
            raise ValueError('unknown host ordering hypothesis')
        self.stages = stages
        self.frames, self.warmup, self.scored = frames, warmup, scored
        self.depth, self.runnable_wait, self.host_order = queue_depth, runnable_completion_wait, host_order
        self.time = Fraction(0)
        self.queues = [deque() for _ in stages]
        self.state = [None for _ in stages]
        self.pending_output = [None for _ in stages]
        self.starts = {}
        self.ends = {}
        self.completions = {}
        self.source = 0
        self.device_owner = None
        self.active = {}
        self.delays = {}
        self.forks = {}
        self.counter = 0
        self.cpu_work = Fraction(0)
        self.wait_work = Fraction(0)
        self.events = 0
        for stage in stages:
            if not stage['mask'] or len(stage['mask']) != len(set(stage['mask'])):
                raise ValueError('empty or duplicate core mask')
            if any(type(c) is not int or c not in range(4) for c in stage['mask']):
                raise ValueError('invalid physical core')

    def new_id(self):
        self.counter += 1
        return self.counter

    def actions(self, stage):
        mask = stage['mask']
        if stage['device'] == 'cpu':
            caller = stage['worker_core_map'][0]
            if sorted(stage['worker_core_map']) != sorted(mask):
                raise ValueError('worker mapping differs from allowed group')
            profile = stage['profile']
            for op in ('set',):
                yield ('cpu', rational(stage[op+'_cpu_ns']), [caller])
                yield ('delay', rational(stage.get(op+'_nonrunning_ns', 0)))
            for i, kernel in enumerate(profile['kernels']):
                if kernel['barrier_generations']:
                    raise ValueError('internal barriers require an explicit validated replay')
                yield ('cpu', rational(profile['serial_cpu_segments'][i]), [caller])
                yield ('cpu', rational(kernel['launch_prefix_cpu']), [caller])
                work = [rational(w['compute_segments'][0]) for w in kernel['workers']]
                if len(work) != len(mask):
                    raise ValueError('profile logical workers differ from requested threads')
                yield ('fork', work, stage['worker_core_map'])
                # Measured caller tail contains completion waiting. It is NOT
                # intrinsic work and is never added a second time here.
            yield ('cpu', rational(profile['serial_cpu_segments'][-1]), [caller])
            yield ('cpu', rational(stage['get_cpu_ns']), [caller])
            yield ('delay', rational(stage.get('get_nonrunning_ns', 0)))
        elif stage['device'] == 'vta':
            yield ('cpu', rational(stage['set_cpu_ns']), mask)
            yield ('delay', rational(stage.get('set_nonrunning_ns', 0)))
            # CPU work inside device polling is not localized by these input
            # records. Freeze both limiting orders; neither is claimed exact.
            host = ('cpu', rational(stage['run_cpu_ns']), mask)
            device = ('delay', rational(stage['run_nonrunning_ns']))
            yield from ((host, device) if self.host_order == 'before' else (device, host))
            yield ('cpu', rational(stage['get_cpu_ns']), mask)
            yield ('delay', rational(stage.get('get_nonrunning_ns', 0)))
        else:
            raise ValueError('unknown execution resource')

    def start_cpu(self, owner, work, mask, callback, *, waiting=False):
        if work is not None and work < 0:
            raise ValueError('negative CPU work')
        if work == 0:
            callback()
            return None
        key = self.new_id()
        self.active[key] = dict(owner=owner, remaining=work, mask=tuple(mask), callback=callback,
                                waiting=waiting)
        return key

    def advance(self, index):
        action = next(self.state[index]['iterator'], None)
        if action is None:
            self.finish_stage(index)
        elif action[0] == 'cpu':
            self.start_cpu(index, action[1], action[2], lambda: self.advance(index))
        elif action[0] == 'delay':
            if action[1] < 0:
                raise ValueError('negative non-running interval')
            if action[1] == 0:
                self.advance(index)
            else:
                self.delays[self.new_id()] = (self.time + action[1], index)
        elif action[0] == 'fork':
            work, mapping = action[1:]
            fork = self.new_id()
            self.forks[fork] = dict(owner=index, pending=set(range(len(work))), waiter=None,
                                    caller_core=mapping[0], caller_finished=False)
            # Zero work is queued, rather than calling back during construction.
            for slot, (amount, core) in enumerate(zip(work, mapping)):
                key = self.new_id()
                self.active[key] = dict(owner=index, remaining=amount, mask=(core,), waiting=False,
                                        callback=lambda f=fork, s=slot: self.worker_done(f, s))
        else:
            raise ValueError('unknown task action')

    def worker_done(self, fork, slot):
        group = self.forks[fork]
        group['pending'].remove(slot)
        if slot == 0:
            group['caller_finished'] = True
        if not group['pending']:
            if group['waiter'] is not None:
                self.active.pop(group['waiter'], None)
            del self.forks[fork]
            self.advance(group['owner'])
        elif group['caller_finished'] and self.runnable_wait and group['waiter'] is None:
            group['waiter'] = self.start_cpu(group['owner'], None, [group['caller_core']], None,
                                              waiting=True)

    def finish_stage(self, index):
        frame = self.state[index]['frame']
        self.ends[index, frame] = self.time
        if self.device_owner == index:
            self.device_owner = None
        self.state[index] = None
        self.pending_output[index] = frame

    def settle(self):
        """Drain zero-cost control actions while preserving bounded backpressure."""
        changed = True
        while changed:
            changed = False
            while self.source < self.frames and len(self.queues[0]) < self.depth:
                self.queues[0].append(self.source)
                self.source += 1
                changed = True
            for i in reversed(range(len(self.stages))):
                frame = self.pending_output[i]
                if frame is not None:
                    if i == len(self.stages)-1:
                        self.completions[frame] = self.time
                        self.pending_output[i] = None
                        changed = True
                    elif len(self.queues[i+1]) < self.depth:
                        self.queues[i+1].append(frame)
                        self.pending_output[i] = None
                        changed = True
            for i, stage in enumerate(self.stages):
                if self.state[i] is not None or self.pending_output[i] is not None or not self.queues[i]:
                    continue
                if stage['device'] == 'vta' and self.device_owner is not None:
                    continue
                frame = self.queues[i].popleft()
                self.starts[i, frame] = self.time
                self.state[i] = dict(frame=frame, iterator=iter(self.actions(stage)))
                if stage['device'] == 'vta':
                    self.device_owner = i
                self.advance(i)
                changed = True

    def rates(self):
        cores = {c: [] for c in range(4)}
        ordered = sorted(self.active.items(), key=lambda pair: (len(pair[1]['mask']), pair[0]))
        for key, task in ordered:
            core = min(task['mask'], key=lambda c: (len(cores[c]), c))
            cores[core].append(key)
        return {key: Fraction(1, len(keys)) for keys in cores.values() for key in keys}

    def run(self):
        while len(self.completions) < self.frames:
            self.settle()
            if len(self.completions) == self.frames:
                break
            rates = self.rates()
            nexts = [task['remaining']/rates[key] for key, task in self.active.items()
                     if task['remaining'] is not None]
            nexts += [time-self.time for time, _ in self.delays.values()]
            if not nexts:
                raise ValueError('replay deadlocked or has only waiting threads')
            delta = min(nexts)
            if delta < 0:
                raise ValueError('event time went backwards')
            finished = []
            for key, task in self.active.items():
                used = delta*rates[key]
                if task['waiting']:
                    self.wait_work += used
                else:
                    self.cpu_work += used
                    task['remaining'] -= used
                    if task['remaining'] == 0:
                        finished.append(key)
            self.time += delta
            callbacks = [self.active.pop(key)['callback'] for key in finished]
            due = [(key, index) for key, (time, index) in self.delays.items() if time == self.time]
            for key, _ in due:
                del self.delays[key]
            for callback in callbacks:
                callback()
            for _, index in due:
                self.advance(index)
            self.events += 1
            if self.events > 1000000:
                raise ValueError('unbounded replay')
        kept = [self.completions[i] for i in range(self.warmup, self.warmup+self.scored)]
        interval = (kept[-1]-kept[0])/(self.scored-1)
        return dict(interval_ns=ceiling(interval), timestamps_ns=[ceiling(t) for t in kept],
                    stage_service_ns=[ceiling(sum(self.ends[i, f]-self.starts[i, f]
                        for f in range(self.warmup, self.warmup+self.scored))/self.scored)
                        for i in range(len(self.stages))],
                    useful_CPU_ns=ceiling(self.cpu_work), synthetic_wait_CPU_ns=ceiling(self.wait_work),
                    processed_frames=len(self.completions), events=self.events,
                    control_CPU_unmodeled=True, scheduling_is_idealized=True)
