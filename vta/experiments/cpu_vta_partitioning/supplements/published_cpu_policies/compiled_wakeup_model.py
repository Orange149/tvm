"""One source-derived wake threshold over the unchanged finite service model."""
from fractions import Fraction
import hashlib
import inspect
import textwrap

from finite_service.scheduler import Scheduler
from device_completion_replay import CompletionFinite


def adapted_select():
    source=textwrap.dedent(inspect.getsource(Scheduler.select))
    before='self.vruntime[old]-self.vruntime[tasks[best][\'actor\']]>q/2'
    after='self.vruntime[old]-self.vruntime[tasks[best][\'actor\']]>self.wakeup_granularity_ns'
    if source.count(before)!=1:raise ValueError('audited scheduler wake comparison changed')
    changed=source.replace(before,after)
    namespace=dict(Scheduler.select.__globals__)
    exec(compile(changed,'<one compiled-kernel wake threshold substitution>','exec'),namespace)
    return namespace['select'],dict(original_sha256=hashlib.sha256(source.encode()).hexdigest(),
        adapted_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        exact_replacement=dict(before=before,after=after),other_scheduler_rules_unchanged=True)


class KernelWakeScheduler(Scheduler):
    select=adapted_select()[0]

    def __init__(self,quanta,wakeup_granularity_ns):
        if type(wakeup_granularity_ns) is not int or wakeup_granularity_ns<=0:
            raise ValueError('positive independently sourced wake granule required')
        super().__init__(quanta);self.wakeup_granularity_ns=Fraction(wakeup_granularity_ns)


class KernelWakeFinite(CompletionFinite):
    def __init__(self,*args,wakeup_granularity_ns,quanta,**kw):
        super().__init__(*args,quanta=quanta,**kw)
        self.scheduler=KernelWakeScheduler(quanta,wakeup_granularity_ns)

    def run(self):
        r=super().run();r.update(wakeup_granularity_ns=int(self.scheduler.wakeup_granularity_ns),
            one_compiled_parameter_substitution=adapted_select()[1],
            full_Linux_CFS_reproduction=False,sleeper_credit_and_service_granules_unchanged=True)
        return r
