"""Same-clock CPU and VTA pieces; no scheduler or FPS model inputs."""
from collections import Counter
import statistics
from phase_model.extract import project_call,signature


def module_key(path,files):
    for prefix in ('programs/','runtime/','runner/'):
        marker='/'+prefix
        if marker in path:
            leaf=prefix+path.split(marker,1)[1]
            if leaf not in files:raise ValueError('unknown callback module')
            return files[leaf]['sha256']
    raise ValueError('unidentified callback module')


def cpu_calls(name,phases,events,files,first,count,mapping):
    launches=[r for r in events if r['kind']==0];callbacks={};barriers={}
    for r in events:
        if r['kind'] in (1,2):(callbacks if r['kind']==1 else barriers).setdefault(r['launch'],[]).append(r)
    projected=[];metrics=[];runs=[]
    for frame in range(first,first+count):
        spans=[p for p in phases if p['stage']==name and p['call']==frame and p['operation']=='run']
        if len(spans)!=1:raise ValueError('missing or duplicate CPU run phase')
        span=spans[0];call=project_call(span,launches,callbacks,barriers)
        chosen=sorted([r for r in launches if r['caller']==span['tid'] and
            span['wall_start_ns']<=r['wall_start_ns']<=r['wall_end_ns']<=span['wall_end_ns']],key=lambda r:r['wall_start_ns'])
        if len(chosen)!=len(call['kernels']):raise ValueError('launch projection differs')
        for index,(launch,kernel) in enumerate(zip(chosen,call['kernels'])):
            kernel['module']=module_key(kernel['module'],files)
            group=sorted(callbacks[launch['launch']],key=lambda r:r['task'])
            workers=[]
            for r,w in zip(group,kernel['workers']):
                cpu=r['cpu_end_ns']-r['cpu_start_ns'];wall=r['wall_end_ns']-r['wall_start_ns']
                if cpu>wall+10000 or wall<0:raise ValueError('callback clocks invalid')
                workers.append(dict(slot=r['task'],core=mapping[r['task']],CPU_ns=cpu,
                    compute_CPU_ns=sum(w['compute_segments_ns']),wall_ns=wall,
                    nonrunning_ns=max(0,wall-cpu),entry_offset_ns=r['wall_start_ns']-launch['wall_start_ns'],
                    callback_start_ns=r['wall_start_ns'],callback_end_ns=r['wall_end_ns']))
            last=max(r['wall_end_ns'] for r in group)
            metrics.append(dict(frame=frame,kernel=index,module_sha256=kernel['module'],
                function_offset=kernel['function_offset'],tasks=kernel['tasks'],
                launch_wall_ns=launch['wall_end_ns']-launch['wall_start_ns'],
                callback_span_ns=last-min(r['wall_start_ns'] for r in group),
                entry_skew_ns=max(r['wall_start_ns'] for r in group)-min(r['wall_start_ns'] for r in group),
                compute_CPU_ns=sum(w['compute_CPU_ns'] for w in workers),
                caller_tail_wall_ns=launch['wall_end_ns']-launch['caller_tail_wall_start_ns'],
                caller_tail_CPU_ns=launch['cpu_end_ns']-launch['caller_tail_cpu_start_ns'],
                last_slots=[r['task'] for r in group if r['wall_end_ns']==last],workers=workers))
        projected.append(call)
        runs.append(dict(frame=frame,run_wall_ns=span['wall_end_ns']-span['wall_start_ns'],
            known_CPU_ns=call['known_CPU_ns'],caller_CPU_ns=call['caller_CPU_ns'],
            serial_CPU_ns=sum(call['serial_cpu_segments_ns']),
            launch_prefix_CPU_ns=sum(k['launch_prefix_cpu_ns'] for k in call['kernels']),
            compute_CPU_ns=sum(sum(w['compute_segments_ns']) for k in call['kernels'] for w in k['workers']),
            caller_wait_and_return_CPU_ns=sum(k['caller_completion_wait_and_return_cpu_ns'] for k in call['kernels'])))
    if any(signature(p)!=signature(projected[0]) for p in projected):raise ValueError('kernel sequence changed')
    grouped=[]
    for i,kernel in enumerate(projected[0]['kernels']):
        values=[v for v in metrics if v['kernel']==i]
        fields=('launch_wall_ns','callback_span_ns','entry_skew_ns','compute_CPU_ns','caller_tail_wall_ns','caller_tail_CPU_ns')
        lasts=Counter(str(slot) for v in values for slot in v['last_slots'])
        grouped.append(dict(kernel=i,module_sha256=kernel['module'],function_offset=kernel['function_offset'],
            means_ms={f.removesuffix('_ns')+'_ms':statistics.mean(v[f] for v in values)/1e6 for f in fields},
            last_slots=dict(lasts),
            worker_compute_CPU_ms=[statistics.mean(v['workers'][slot]['compute_CPU_ns'] for v in values)/1e6 for slot in range(kernel['tasks'])],
            worker_nonrunning_ms=[statistics.mean(v['workers'][slot]['nonrunning_ns'] for v in values)/1e6 for slot in range(kernel['tasks'])]))
    means={k.removesuffix('_ns')+'_ms':statistics.mean(v[k] for v in runs)/1e6 for k in runs[0] if k!='frame'}
    return dict(signature=signature(projected[0]),calls=projected,kernels=grouped,launch_records=metrics,runs=runs,means_ms=means,
        nonrunning_not_all_scheduler_queue=True,callback_entry_not_identical_to_model_first_service=True)


def host_segments(run,events):
    selected=sorted([e for e in events if e['call']==run['call'] and e['tid']==run['tid']],key=lambda e:e['wall_start_ns'])
    if not selected:raise ValueError('no same-TID device work in VTA run')
    wall,cpu=run['wall_start_ns'],run['cpu_start_ns'];parts=[]
    for e in selected:
        if not wall<=e['wall_start_ns']<=e['wall_end_ns']<=run['wall_end_ns']:
            raise ValueError('device outside run wall interval')
        if not cpu<=e['cpu_start_ns']<=e['cpu_end_ns']<=run['cpu_end_ns']:
            raise ValueError('device outside run CPU interval')
        elapsed=e['wall_start_ns']-wall;work=e['cpu_start_ns']-cpu
        if work>elapsed+10000:raise ValueError('host CPU exceeds wall interval')
        parts.append(dict(wall_ns=elapsed,CPU_ns=work,nonrunning_ns=max(0,elapsed-work)))
        wall,cpu=e['wall_end_ns'],e['cpu_end_ns']
    elapsed=run['wall_end_ns']-wall;work=run['cpu_end_ns']-cpu
    if work>elapsed+10000:raise ValueError('host CPU exceeds suffix wall interval')
    parts.append(dict(wall_ns=elapsed,CPU_ns=work,nonrunning_ns=max(0,elapsed-work)))
    driver_wall=sum(e['wall_end_ns']-e['wall_start_ns'] for e in selected)
    driver_CPU=sum(e['cpu_end_ns']-e['cpu_start_ns'] for e in selected)
    if sum(p['wall_ns'] for p in parts)+driver_wall!=run['wall_end_ns']-run['wall_start_ns']:
        raise ValueError('VTA wall work lost or counted twice')
    if sum(p['CPU_ns'] for p in parts)+driver_CPU!=run['cpu_end_ns']-run['cpu_start_ns']:
        raise ValueError('VTA CPU work lost or counted twice')
    return dict(frame=run['call'],segments=parts,driver_wall_ns=driver_wall,driver_CPU_ns=driver_CPU,
        host_wall_ns=sum(p['wall_ns'] for p in parts),host_CPU_ns=sum(p['CPU_ns'] for p in parts),
        host_nonrunning_ns=sum(p['nonrunning_ns'] for p in parts),
        run_wall_ns=run['wall_end_ns']-run['wall_start_ns'],run_CPU_ns=run['cpu_end_ns']-run['cpu_start_ns'])


def host_calls(name,phases,events,first,count):
    calls=[]
    for frame in range(first,first+count):
        runs=[p for p in phases if p['stage']==name and p['call']==frame and p['operation']=='run']
        if len(runs)!=1:raise ValueError('incomplete VTA run phase')
        calls.append(host_segments(runs[0],events))
    if len({len(c['segments']) for c in calls})!=1:raise ValueError('device sequence changed')
    fields=('driver_wall_ns','driver_CPU_ns','host_wall_ns','host_CPU_ns','host_nonrunning_ns','run_wall_ns','run_CPU_ns')
    return dict(calls=calls,means_ms={k.removesuffix('_ns')+'_ms':statistics.mean(c[k] for c in calls)/1e6 for k in fields},
        nonrunning_not_identified_as_all_CPU_queue=True)
