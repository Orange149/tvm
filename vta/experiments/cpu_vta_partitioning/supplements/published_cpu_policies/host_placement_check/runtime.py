from pathlib import Path
from types import FunctionType
from copy import deepcopy
from collections import Counter
from cpu_vta_partitioning.common import digest,read
from cpu_vta_partitioning.transport import SSH
from phase_profile import runtime as old
from .common import protocol


def make_task(bundle,spec):
    if spec['scope']!='pipeline' or spec['case']!='overlap' or spec['network']!='yolov3_tiny416' or spec['probe']:
        raise ValueError('this bounded study has exactly one unmodified pipeline configuration')
    if spec['phase'] not in ('qualify','clean','observed') or (spec['phase']=='observed' and spec['runner']!='phase'):
        raise ValueError('unknown counter task')
    if spec['runner'] not in ('old','phase'):raise ValueError('unknown executor')
    manifest=read(Path(bundle)/'manifest.json')
    if manifest['network']!=spec['network']:raise ValueError('wrong network capsule')
    cfg=read(Path(bundle)/'plans/configurations.json')['pipelines']['overlap']
    qualify=spec['phase']=='qualify'
    task=dict(kind='qualify_pipeline' if qualify else 'performance',configuration=deepcopy(cfg),stages=deepcopy(cfg['stages']),
        spec=deepcopy(spec),environment=dict(cores=4,frequency_khz=1066666,poll_sleep_ns=1000,post_start_sleep_ns=1000,
            spin_count=0,exclude_worker0=1,bind_threads=1),
        **protocol()['qualification' if qualify else 'pipeline'],input_indices=[0],
        executor_sha256=manifest['phase_runner_sha256' if spec['runner']=='phase' else 'old_runner_sha256'],
        adapter_sha256=None,runtime_sha256=manifest['runtime_sha256'],bundle_id=manifest['bundle_id'],
        experiment=protocol()['experiment'],startup_affinity_probe=True)
    task['counter_probe']=spec['phase']=='observed'
    task['counter_scope']='all live TIDs, raw per-core TASK_CLOCK, collector completion-index window'
    task['task_id']=digest({k:v for k,v in task.items() if k!='task_id'})[:24]
    return task


def native_command(doc,bundle,remote_bundle,remote_job):
    function=FunctionType(old.native_command.__code__,{**old.native_command.__globals__,'make_task':make_task})
    argv,inputs=function(doc,bundle,remote_bundle,remote_job)
    if doc['counter_probe']:
        argv+=['--session-profile-json',remote_job+'/profile.json','--session-profile-window',f"{doc['warmup']},{doc['scored']}"]
    if any(x.startswith('VTA_WAIT_') or 'libwait_adapter' in x for x in argv):raise ValueError('unrequested waiting probe')
    return argv,inputs


def validate_profile(profile,affinity,doc):
    if profile.get('status')!='MEASURED' or profile.get('sampling') is not False or profile.get('scaled') is not False or profile.get('cpus')!=[0,1,2,3]:
        raise ValueError('raw counting interface unavailable or unexpected')
    times=[profile[k] for k in ('setup_start_ns','start_reads_end_ns','stop_reads_start_ns','stop_reads_end_ns')]
    if any(type(v) is not int or v<=0 for v in times) or times!=sorted(times) or times[1]>=times[2]:raise ValueError('invalid counter window')
    roles={}
    for s,a in zip(doc['stages'],affinity['stages']):
        tid=int(a['caller_tid']);roles[tid]=dict(role=s['program']+'/caller',kind=s['device'],requested=[s['worker_core_map'][0]] if s['device']=='cpu' else s['mask'])
        for w in a['worker_slots']:
            if w['slot']:
                if w['tid'] in roles:raise ValueError('worker TID used by multiple logical roles')
                roles[w['tid']]=dict(role=s['program']+'/worker'+str(w['slot']),kind='cpu_worker',requested=w['cpus'])
    for c in affinity['control_threads']:
        if c['tid'] in roles:raise ValueError('control and computation TID overlap')
        roles[c['tid']]=dict(role=c['role'],kind='control',requested=c['cpus'])
    tasks=profile['tasks'];tids=[t['tid'] for t in tasks]
    if len(tids)!=len(set(tids)) or not set(roles)<=set(tids):raise ValueError('missing or duplicate live roles')
    result=[]
    for t in tasks:
        a,b=t['before'],t['after'];tid=t['tid']
        if not times[0]<=a['read_start_ns']<=a['read_end_ns']<=times[1]<times[2]<=b['read_start_ns']<=b['read_end_ns']<=times[3]:raise ValueError('per-TID reads escaped window envelope')
        if len(a['counts'])!=5 or len(b['counts'])!=5:raise ValueError('per-core counters incomplete')
        delta=[]
        for first,last in zip(a['counts'],b['counts']):
            if any(type(first[k]) is not int or type(last[k]) is not int or not 0<=first[k]<=last[k] for k in ('value','enabled','running')):raise ValueError('counter moved backwards')
            if last['running']-first['running']>last['enabled']-first['enabled']:raise ValueError('invalid event running clock')
            delta.append(last['value']-first['value'])
        envelope=a['read_end_ns']-a['read_start_ns']+b['read_end_ns']-b['read_start_ns']
        # A task can execute at most one core at a time. Separate FD reads are
        # not simultaneous; retain their envelope instead of rescaling counts.
        if abs(sum(delta[1:])-delta[0])>4*envelope+10000:raise ValueError('per-core CPU conservation failed')
        role=roles.get(tid,dict(role='idle_or_unclassified_'+str(tid),kind='unclassified',requested=t['allowed']))
        if t['allowed']!=role['requested']:raise ValueError('role affinity differs from requested placement')
        if any(c not in (0,1,2,3) for c in t['allowed']):raise ValueError('counter core coverage incomplete')
        if sum(delta[c+1] for c in range(4) if c not in t['allowed'])>4*envelope+10000:raise ValueError('CPU charged outside allowed set')
        if delta[0]>b['read_end_ns']-a['read_start_ns']+envelope:raise ValueError('TID CPU exceeds its wall interval')
        result.append(dict(TID=tid,**role,all_CPU_ns=delta[0],by_core_ns=delta[1:],read_envelope_ns=envelope,
            conservation_error_ns=sum(delta[1:])-delta[0],before_read_ns=[a['read_start_ns'],a['read_end_ns']],after_read_ns=[b['read_start_ns'],b['read_end_ns']]))
    for kind in ('vta','control'):
        if not sum(r['all_CPU_ns'] for r in result if r['kind']==kind):raise ValueError('all target role counts are zero')
    if not 0<profile['completion_start_ns']<=times[0]<profile['completion_end_ns']<=times[2]:raise ValueError('completion anchors use a different clock or order')
    return dict(roles=result,window_start_envelope_ns=times[:2],window_end_envelope_ns=times[2:],
        setup_and_start_read_ns=times[1]-times[0],stop_read_ns=times[3]-times[2],
        completion_start_ns=profile['completion_start_ns'],completion_end_ns=profile['completion_end_ns'],
        start_after_completion_ns=[t-profile['completion_start_ns'] for t in times[:2]],
        stop_after_completion_ns=[t-profile['completion_end_ns'] for t in times[2:]],
        no_rescaling=True,CPU_wait_not_counted_as_running=True,
        window_is_collector_dequeue_not_exact_last_stage_completion=True)


def summary(raw,doc,bundle):
    if doc['counter_probe'] and not (Path(raw)/'profile.json').exists():raise ValueError('requested CPU placement profile absent')
    if not doc['counter_probe'] and (Path(raw)/'profile.json').exists():raise ValueError('unrequested CPU placement profile')
    result=old.summary(raw,doc,bundle)
    result['placement']=validate_profile(read(Path(raw)/'profile.json'),read(Path(raw)/'affinity.json'),doc) if doc['counter_probe'] else None
    return result


def matched_ssh(config,*,execute_board=False):
    measure=FunctionType(SSH.measure.__code__,{**SSH.measure.__globals__,'native_command':native_command})
    return type('HostPlacementSSH',(SSH,),{'measure':measure})(config,execute_board=execute_board)
