"""Controlled aggregation-vs-periodic driver chronology, not hardware simulation.

Both versions consume exactly the same CPU work and nonrunning duration.
Periodic slices use measured isolated poll counts. Host placement stays fixed
inside a device call, so extra slices do not grant extra migration choices.
"""
from fractions import Fraction
from finite_service_replay import FiniteReplay
from static_revision.trace_replay import TraceReplay
from phase_model.simulator import rational,ceiling


def driver_segments(device,periodic):
    CPU=rational(device['CPU']);wall=rational(device['wall']);off=wall-CPU
    if CPU<0 or off<0:raise ValueError('invalid driver CPU/nonrunning work')
    if not periodic:return [('cpu',CPU),('delay',off)]
    checks=ceiling(rational(device['poll_checks']))
    if checks<1:raise ValueError('device polling count must be positive')
    submit=min(CPU,rational(device['submit_mmio_ns']))
    if submit<0:raise ValueError('negative driver submission')
    post=min(off,rational(device['post_start_sleep_ns']))
    if post<0:raise ValueError('negative driver sleep envelope')
    if checks==1:return [('cpu',submit),('delay',off),('cpu',CPU-submit)]
    result=[('cpu',submit),('delay',post)]
    for i in range(checks):
        result.append(('cpu',(CPU-submit)/checks))
        if i+1<checks:result.append(('delay',(off-post)/(checks-1)))
    if sum(v for op,v in result if op=='cpu')!=CPU or sum(v for op,v in result if op=='delay')!=off:
        raise ValueError('periodic driver accounting failed conservation')
    return result


class PollingMixin:
    def __init__(self,*args,driver_timing='aggregate',**kwargs):
        if driver_timing not in ('aggregate','periodic'):raise ValueError('unknown driver timing')
        self.driver_timing=driver_timing
        super().__init__(*args,**kwargs)

    def actions(self,stage):
        if stage['device']=='cpu':
            yield from super().actions(stage);return
        index=next(i for i,s in enumerate(self.stages) if s is stage)
        who=('stage',index,0);mask=stage['mask']
        def CPU(value,op):return ('cpu',rational(value),mask,dict(operation=op,slot=0))
        def delay(value,allow_placement):
            value=rational(value)
            if value>0 and allow_placement:self.potential_wakes.add(who)
            return ('delay',value)
        yield CPU(stage['set_cpu_ns'],'set')
        yield delay(stage.get('set_nonrunning_ns',0),True)
        profile=stage['device_profile']
        for i,device in enumerate(profile['devices']):
            yield CPU(profile['host_serial_CPU'][i],'host_serial')
            periodic=self.driver_timing=='periodic'
            for op,value in driver_segments(device,periodic):
                if op=='cpu':yield CPU(value,'driver_poll' if periodic else 'device_host_CPU')
                else:yield delay(value,not periodic)
            if periodic and rational(device['wall'])>rational(device['CPU']):
                # The baseline already offers placement at the device-call
                # boundary. Preserve that choice, not a new choice per poll.
                self.potential_wakes.add(who)
        yield CPU(profile['host_serial_CPU'][-1],'host_serial')
        yield CPU(stage['get_cpu_ns'],'get')
        yield delay(stage.get('get_nonrunning_ns',0),True)

    def _close_task(self,key):
        trace=self.trace
        if self.active[key]['meta']['operation']=='driver_poll':self.trace=False
        try:return super()._close_task(key)
        finally:self.trace=trace

    def run(self):
        result=super().run()
        result['replay']['driver_timing']=self.driver_timing
        result['within_device_call_extra_migration_choices']=False
        result['poll_slices_are_measured_shape_approximation']=True
        result['device_completion_is_not_independent_hardware_event']=True
        result['per_poll_task_rows_suppressed_but_CPU_totals_preserved']=True
        return result


class PollingFinite(PollingMixin,FiniteReplay):pass
class PollingFluid(PollingMixin,TraceReplay):pass
