"""Two independently sourced constants, over an unchanged scheduler skeleton."""
from fractions import Fraction
import hashlib
import inspect
import textwrap
from finite_service.scheduler import Scheduler
from device_completion_replay import CompletionFinite


def make_select(*,wake=False,sleeper=False):
    source=textwrap.dedent(inspect.getsource(Scheduler.select));result=source;replacements=[]
    pairs=[]
    if wake:pairs.append(("self.vruntime[old]-self.vruntime[tasks[best]['actor']]>q/2",
                          "self.vruntime[old]-self.vruntime[tasks[best]['actor']]>self.wakeup_granularity_ns"))
    if sleeper:pairs.append(('credit=(q*n/2) if q is not None else Fraction(0)',
                             'credit=self.sleeper_credit_ns'))
    for before,after in pairs:
        if result.count(before)!=1:raise ValueError('audited scheduler anchor differs')
        result=result.replace(before,after);replacements.append(dict(before=before,after=after))
    ns=dict(Scheduler.select.__globals__);exec(compile(result,'<compiled kernel constants only>','exec'),ns)
    return ns['select'],dict(original_sha256=hashlib.sha256(source.encode()).hexdigest(),
        adapted_sha256=hashlib.sha256(result.encode()).hexdigest(),replacements=replacements,
        other_scheduling_rules_unchanged=True)


class CompiledScheduler(Scheduler):
    def __init__(self,quanta,*,wake=False,sleeper=False,wakeup_granularity_ns=None,sleeper_credit_ns=None):
        super().__init__(quanta)
        for on,value in ((wake,wakeup_granularity_ns),(sleeper,sleeper_credit_ns)):
            if on and (type(value) is not int or value<=0):raise ValueError('positive independently sourced integer required')
        self.wakeup_granularity_ns=Fraction(wakeup_granularity_ns or 0)
        self.sleeper_credit_ns=Fraction(sleeper_credit_ns or 0)
        f,self.adaptation=make_select(wake=wake,sleeper=sleeper)
        self.select=f.__get__(self,type(self))


class CompiledFinite(CompletionFinite):
    def __init__(self,*args,quanta,wake=False,sleeper=False,wakeup_granularity_ns=None,sleeper_credit_ns=None,**kw):
        super().__init__(*args,quanta=quanta,**kw)
        self.scheduler=CompiledScheduler(quanta,wake=wake,sleeper=sleeper,
            wakeup_granularity_ns=wakeup_granularity_ns,sleeper_credit_ns=sleeper_credit_ns)

    def run(self):
        r=super().run();r.update(compiled_constant_substitution=self.scheduler.adaptation,
            full_Linux_CFS_reproduced=False,work_profiles_and_busy_service_granules_unchanged=True)
        return r
