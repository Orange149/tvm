"""Reuse tested frame/affinity/output validators without changing old modules."""
from pathlib import Path
from types import FunctionType
import json
from cpu_vta_partitioning.common import digest,read
from cpu_vta_partitioning.transport import SSH
from phase_profile import runtime as old
from .common import protocol


def make_task(bundle,spec):
    function=FunctionType(old.make_task.__code__,{**old.make_task.__globals__,'protocol':protocol})
    task=function(bundle,spec)
    task['queue_counter_scope']='per-TID TASKSTATS runnable CPU delay; query envelopes retained'
    task['original_wait']=True
    task['task_id']=digest({k:v for k,v in task.items() if k!='task_id'})[:24]
    return task


def native_command(doc,bundle,remote_bundle,remote_job):
    function=FunctionType(old.native_command.__code__,{**old.native_command.__globals__,'make_task':make_task})
    argv,blobs=function(doc,bundle,remote_bundle,remote_job)
    if doc['spec']['probe']:
        at=argv.index(remote_bundle+'/runtime/ld-linux-aarch64.so.1')
        argv[at:at]=['VTA_QUEUE_COUNTERS=1','VTA_QUEUE_STATUS='+remote_job+'/queue_status.json']
    return argv,blobs


def validate_queue_rows(rows,*,scope):
    if not rows:raise ValueError('queue trace absent')
    for r in rows:
        if scope=='worker':
            keys=('queue_before_ns','queue_after_ns','queue_read_start0_ns','queue_read_end0_ns','queue_read_start1_ns','queue_read_end1_ns')
        else:
            keys=('queue_before_ns','queue_after_ns','read_start0_ns','read_end0_ns','read_start1_ns','read_end1_ns')
        q0,q1,s0,e0,s1,e1=[r[k] for k in keys]
        if any(type(v) is not int or v<0 for v in (q0,q1,s0,e0,s1,e1)) or not q0<=q1 or not s0<=e0<=s1<=e1:
            raise ValueError('queue counter/time envelope moved backwards')
        if not e0<=r['wall_start_ns']<=r['wall_end_ns']<=s1:
            raise ValueError('counter envelopes do not bracket observed body')
        if q1-q0>e1-s0+100000:
            raise ValueError('per-thread CPU queue exceeds enclosing wall interval')
    return dict(records=len(rows),nonzero_queue_records=sum(r['queue_after_ns']>r['queue_before_ns'] for r in rows),
                scope=scope,queue_waits_not_summed_into_stage_service=True)


def phase_summary(raw,doc,affinity):
    if not doc['spec']['probe']:
        if (Path(raw)/'phase_trace.jsonl').exists():raise ValueError('unrequested phase trace')
        return None
    rows=[json.loads(s) for s in (Path(raw)/'phase_trace.jsonl').read_text().splitlines()]
    total=sum(doc[k] for k in ('warmup','scored','tail'));names={s['program'] for s in doc['stages']}
    expected={(n,i,op) for n in names for i in range(total) for op in ('set','run','get')}
    if doc['spec']['scope']=='pipeline':
        expected|={(n,i,op) for n,op in (('_producer','feed'),('_collector','collect')) for i in range(total)}
    keys=[(r['stage'],r['call'],r['operation']) for r in rows]
    if len(keys)!=len(expected) or set(keys)!=expected:raise ValueError('missing/duplicate stage or control events')
    callers={s['program']:int(a['caller_tid']) for s,a in zip(doc['stages'],affinity['stages'])}
    for r in rows:
        if r['stage'] in callers and r['tid']!=callers[r['stage']]:raise ValueError('stage TID changed')
        if not 0<=r['cpu_start_ns']<=r['cpu_end_ns'] or not 0<=r['wall_start_ns']<=r['wall_end_ns']:
            raise ValueError('phase clocks moved backwards')
        if r['cpu_end_ns']-r['cpu_start_ns']>r['wall_end_ns']-r['wall_start_ns']+10000:
            raise ValueError('phase CPU exceeds wall time')
    for name in names:
        previous=0
        for call in range(total):
            group=[next(r for r in rows if (r['stage'],r['call'],r['operation'])==(name,call,op)) for op in ('set','run','get')]
            if group[0]['wall_start_ns']<previous or any(a['wall_end_ns']>b['wall_start_ns'] for a,b in zip(group,group[1:])):
                raise ValueError('phase sequence overlaps itself')
            previous=group[-1]['wall_end_ns']
    return dict(events=len(rows),stage_calls={n:total for n in names},
                control_events=sum(r['stage'].startswith('_') for r in rows),
                queue=validate_queue_rows(rows,scope='stage'))


def summary(raw,doc,bundle,*,native=True):
    function=FunctionType(old.summary.__code__,{**old.summary.__globals__,'phase_summary':phase_summary})
    if not native and doc['spec']['probe']:
        # QEMU proves ABI and numerics, never native TASKSTATS semantics.
        function=FunctionType(old.summary.__code__,{**old.summary.__globals__,'phase_summary':old.phase_summary})
    result=function(raw,doc,bundle)
    if doc['spec']['probe']:
        status=read(Path(raw)/'queue_status.json')
        if status['errors'] or (native and (not status['enabled'] or not status['reads'])):
            raise ValueError('native queue interception/counters unavailable')
        events=[json.loads(s) for s in (Path(raw)/'wait_trace.jsonl').read_text().splitlines()]
        result['queue']=validate_queue_rows(events,scope='worker') if native else dict(native_counters_not_validated=True)
        result['queue_status']=status
    elif (Path(raw)/'queue_status.json').exists():
        raise ValueError('unrequested queue observer')
    return result


def matched_ssh(config,*,execute_board=False):
    measure=FunctionType(SSH.measure.__code__,{**SSH.measure.__globals__,'native_command':native_command})
    return type('SchedulerGapSSH',(SSH,),{'measure':measure})(config,execute_board=execute_board)
