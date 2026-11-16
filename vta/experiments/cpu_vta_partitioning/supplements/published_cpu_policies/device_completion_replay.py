"""Independent device progress and endogenous host polling, development only.

Device durations come from isolated MMIO completion brackets. A delayed host
does not stop device progress or have to perform the old isolation poll count.
CPU dispatch and placement remain the existing explicitly approximate models.
"""
from phase_model.simulator import rational
from polling_replay import PollingFinite,PollingFluid,driver_segments


def parameters(d,point):
    CPU=rational(d['CPU']);wall=rational(d['wall']);checks=rational(d['poll_checks'])
    lower=rational(d['completion_lower_ns']);upper=rational(d['completion_upper_ns'])
    if CPU<0 or wall<CPU or checks<1 or not 0<=lower<=upper:
        raise ValueError('invalid independent device profile')
    if point not in ('lower','midpoint','upper'):raise ValueError('unknown completion bracket point')
    duration=lower if point=='lower' else upper if point=='upper' else (lower+upper)/2
    submit=min(CPU,rational(d['submit_mmio_ns']))
    off=wall-CPU;post=min(off,rational(d['post_start_sleep_ns']))
    if submit<0 or post<0:raise ValueError('negative submission or post-start delay')
    sleep=(off-post)/(checks-1) if checks>1 else post
    if sleep<0:raise ValueError('negative inter-poll nonrunning interval')
    return dict(duration=duration,submit_CPU=submit,poll_CPU=(CPU-submit)/checks,
        post=post,sleep=sleep,isolated_checks=checks,
        poll_CPU_is_equal_share_proxy=True,sleep_is_observed_nonrunning_proxy=True)


class CompletionMixin:
    def __init__(self,*args,completion_model='independent',completion_point='midpoint',**kwargs):
        if completion_model not in ('aggregate','independent'):raise ValueError('unknown completion model')
        self.completion_model=completion_model;self.completion_point=completion_point
        self.device_records=[]
        super().__init__(*args,driver_timing='periodic',**kwargs)

    def actions(self,stage):
        if stage['device']=='cpu':yield from super().actions(stage);return
        index=next(i for i,s in enumerate(self.stages) if s is stage)
        who=('stage',index,0);mask=stage['mask']
        def CPU(value,op):return ('cpu',rational(value),mask,dict(operation=op,slot=0))
        def delay(value,placement=False):
            value=rational(value)
            if value>0 and placement:self.potential_wakes.add(who)
            return ('delay',value)
        yield CPU(stage['set_cpu_ns'],'set')
        yield delay(stage.get('set_nonrunning_ns',0),True)
        profile=stage['device_profile']
        for i,d in enumerate(profile['devices']):
            yield CPU(profile['host_serial_CPU'][i],'host_serial')
            if self.completion_model=='aggregate':
                for op,value in driver_segments(d,False):
                    yield CPU(value,'device_host_CPU') if op=='cpu' else delay(value,True)
                continue
            p=parameters(d,self.completion_point)
            yield CPU(p['submit_CPU'],'device_submit')
            start=self.time;deadline=start+p['duration'];checks=0
            yield delay(p['post'])
            while True:
                yield CPU(p['poll_CPU'],'driver_poll')
                checks+=1
                if self.time>=deadline:break
                if p['sleep']==0 and p['poll_CPU']==0:raise ValueError('poll loop cannot make progress')
                if checks>1_000_000:raise ValueError('independent polling unbounded')
                yield delay(p['sleep'])
            frame=self.state[index]['frame']
            self.device_records.append(dict(stage=index,frame=frame,device_index=i,
                start_ns=start,device_done_ns=deadline,host_return_ns=self.time,
                checks=checks,CPU=p['submit_CPU']+checks*p['poll_CPU'],
                isolated_checks=p['isolated_checks']))
            self.potential_wakes.add(who)
        yield CPU(profile['host_serial_CPU'][-1],'host_serial')
        yield CPU(stage['get_cpu_ns'],'get')
        yield delay(stage.get('get_nonrunning_ns',0),True)

    def run(self):
        value=super().run()
        def exact(x):return dict(numerator=x.numerator,denominator=x.denominator)
        value['driver_completion']=[{k:exact(v) if hasattr(v,'denominator') and not isinstance(v,int) else v
                                     for k,v in d.items()} for d in self.device_records]
        value.update(device_completion_model=self.completion_model,completion_bracket_point=self.completion_point,
            device_completion_is_not_independent_hardware_event=self.completion_model!='independent',
            fixed_isolation_poll_count_forced=False if self.completion_model=='independent' else None,
            hardware_duration_is_isolated_bracket_assumption=True,
            hardware_DDR_slowdown_not_predicted=True)
        return value


class CompletionFinite(CompletionMixin,PollingFinite):pass
class CompletionFluid(CompletionMixin,PollingFluid):pass
