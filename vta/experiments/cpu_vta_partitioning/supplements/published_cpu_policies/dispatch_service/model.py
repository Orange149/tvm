"""Count CPU already consumed in the current dispatch when peers wake.

Reference: Linux v5.4 fair.c check_preempt_tick/htick_start_fair use elapsed
execution since prev_sum_exec_runtime, not a fresh allocation on every load
change. We still use empirical service granules, the old placement rule and
old wake-credit approximation; this is deliberately not a CFS reproduction.
"""
from fractions import Fraction
from finite_service.scheduler import Scheduler
from finite_service_replay import FiniteReplay


class DispatchScheduler(Scheduler):
    def __init__(self, quanta):
        super().__init__(quanta)
        self.dispatch_used = {}
        self.reset_events = 0

    def select(self, tasks):
        old = dict(self.current)
        by_core = {c: [] for c in range(4)}
        for task in tasks.values():
            if task['core'] not in by_core:
                raise ValueError('invalid core')
            by_core[task['core']].append(task['actor'])
        expired = set()
        for core, actors in by_core.items():
            current = old.get(core)
            if current in actors and len(actors) > 1:
                remaining = self.quantum(len(actors))-self.dispatch_used.get(core, Fraction(0))
                self.left[core] = remaining
                if remaining <= 0:
                    expired.add(core)
        selected, _ = super().select(tasks)
        deadlines = []
        for core, key in selected.items():
            current = tasks[key]['actor']
            if current != old.get(core) or old.get(core) not in by_core[core] or core in expired:
                self.dispatch_used[core] = Fraction(0)
                self.reset_events += 1
            used = self.dispatch_used.get(core, Fraction(0))
            self.dispatch_used[core] = used
            if len(by_core[core]) > 1:
                self.left[core] = self.quantum(len(by_core[core]))-used
                if self.left[core] <= 0:
                    raise ValueError('expired dispatch was not renewed')
                deadlines.append(self.left[core])
            else:
                self.left[core] = None
        for core in set(self.dispatch_used)-set(selected):
            del self.dispatch_used[core]
        return selected, min(deadlines) if deadlines else None

    def consume(self, tasks, duration):
        super().consume(tasks, duration)
        for core in self.choices:
            self.dispatch_used[core] += duration


class DispatchReplay(FiniteReplay):
    def __init__(self, stages, *, quanta, **kwargs):
        super().__init__(stages, quanta=quanta, **kwargs)
        self.scheduler = DispatchScheduler(quanta)

    def run(self):
        result = super().run()
        result['replay']['service_model'] = 'finite_service_dispatch_elapsed'
        result['dispatch_elapsed_accounting'] = True
        result['dispatch_reset_events'] = self.scheduler.reset_events
        result['new_runtime_or_affinity_policy'] = False
        return result
