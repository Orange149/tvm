"""Retain consumed service in the existing finite-service approximation.

This is a parameter-free scheduling-rule correction candidate, not exact CFS:
the existing empirical budgets are reused, without real tick or Yield history.
"""
from fractions import Fraction
import hashlib
import inspect
import textwrap
from finite_service.scheduler import Scheduler


class RetainedServiceHistory(Scheduler):
    def __init__(self,original):
        if original.vruntime or original.current:
            raise ValueError('history correction must begin before replay')
        super().__init__(original.quanta)
        self.service_used={}
        source=textwrap.dedent(inspect.getsource(Scheduler.select));before=source
        inherited=getattr(original,'adaptation',None)
        if inherited:
            self.wakeup_granularity_ns=original.wakeup_granularity_ns
            self.sleeper_credit_ns=original.sleeper_credit_ns
            for change in inherited['replacements']:
                if source.count(change['before'])!=1:raise ValueError('kernel substitution anchor changed')
                source=source.replace(change['before'],change['after'])
        changes=[
            ('self.last_count[c]=0;continue', 'self.last_count[c]=0;self.service_used.pop(c,None);continue'),
            ('expired=valid and self.left.get(c) is not None and self.left[c]<=0',
             'expired=valid and q is not None and self.service_used.get(c,Fraction(0))>=q'),
            ('actor=tasks[best][\'actor\'];self.current[c]=actor;self.left[c]=q',
             'actor=tasks[best][\'actor\'];self.current[c]=actor;self.left[c]=q;self.service_used[c]=Fraction(0)'),
            ('elif self.last_count.get(c,0)<2 or self.left.get(c) is None:self.left[c]=q\n            else:self.left[c]=min(self.left[c],q)',
             'else:self.left[c]=q-self.service_used[c]')]
        for old,new in changes:
            if source.count(old)!=1:raise ValueError('service history anchor changed')
            source=source.replace(old,new)
        ns=dict(Scheduler.select.__globals__);exec(compile(source,'<retained model service history>','exec'),ns)
        self.select=ns['select'].__get__(self,type(self))
        self.adaptation=dict(inherited_kernel_substitutions=inherited,
            original_select_sha256=hashlib.sha256(before.encode()).hexdigest(),
            corrected_select_sha256=hashlib.sha256(source.encode()).hexdigest(),
            changes=[dict(before=a,after=b) for a,b in changes],
            new_parameters=0,empirical_budget_values_unchanged=True,
            actual_kernel_tick_or_voluntary_Yield_not_reproduced=True)

    def consume(self,tasks,duration):
        super().consume(tasks,duration)
        for core in self.choices:self.service_used[core]+=duration
