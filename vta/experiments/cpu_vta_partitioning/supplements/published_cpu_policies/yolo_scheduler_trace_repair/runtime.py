from pathlib import Path
from types import FunctionType
from cpu_vta_partitioning.common import digest,read
from cpu_vta_partitioning.transport import SSH
from cpu_vta_partitioning.measurement import frames
from scheduler_gap_audit import runtime as gap
from yolo_scheduler_trace import runtime as old
from .common import protocol
from .events import validate_profile


def make_task(bundle,spec):
    if spec['runner']!='phase':raise ValueError('repair only uses one frozen executor')
    fn=FunctionType(old.make_task.__code__,{**old.make_task.__globals__,'protocol':protocol})
    doc=fn(bundle,spec);doc.pop('task_id')
    doc['scheduler_recording_policy']=protocol()['drain_policy']
    doc['scheduler_recording_revision']=2
    doc['task_id']=digest(doc)[:24]
    return doc


def native_command(doc,bundle,remote_bundle,remote_job):
    fn=FunctionType(old.native_command.__code__,{**old.native_command.__globals__,'make_task':make_task})
    return fn(doc,bundle,remote_bundle,remote_job)


def summary(raw,doc,bundle):
    result=gap.summary(raw,doc,bundle);path=Path(raw)/'profile.json'
    if doc['scheduler_events']:
        if not path.exists():raise ValueError('scheduler profile absent')
        result['scheduler']=validate_profile(read(path),doc,read(Path(raw)/'affinity.json'))
        rows=frames(Path(raw)/'result.jsonl',sum(doc[k] for k in ('warmup','scored','tail')))
        a,b=doc['warmup'],doc['warmup']+doc['scored']-1
        ns=round((rows[b]['completion_ms']-rows[a]['completion_ms'])*1e6)
        if abs(ns-(result['scheduler']['completion_end_ns']-result['scheduler']['completion_start_ns']))>1000:
            raise ValueError('scheduler completion anchors differ')
    else:
        if path.exists():raise ValueError('unexpected scheduler profile')
        result['scheduler']=None
    return result


def matched_ssh(config,*,execute_board=False):
    fn=FunctionType(SSH.measure.__code__,{**SSH.measure.__globals__,'native_command':native_command})
    return type('DrainedSchedulerSSH',(SSH,),{'measure':fn})(config,execute_board=execute_board)
