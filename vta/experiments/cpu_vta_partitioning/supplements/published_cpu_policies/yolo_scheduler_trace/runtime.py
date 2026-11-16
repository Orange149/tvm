from pathlib import Path
from copy import deepcopy
from types import FunctionType
from cpu_vta_partitioning.common import read,digest
from cpu_vta_partitioning.transport import SSH
from phase_profile import runtime as phase
from scheduler_gap_audit import runtime as gap
from .common import protocol
from .events import validate_profile


def make_task(bundle,spec):
    if (spec['network'],spec['scope'],spec['case'])!=('yolov3_tiny416','pipeline','overlap'):
        raise ValueError('scheduler trace only permits frozen YOLO overlap configuration')
    if spec['phase'] not in ('qualify','clean','observed') or spec['runner'] not in ('old','phase') or spec['probe']!=(spec['phase']=='observed'):
        raise ValueError('scheduler trace task mode differs')
    if spec['probe'] and spec['runner']!='phase':raise ValueError('old executor cannot record new scheduler trace')
    manifest=read(Path(bundle)/'manifest.json');cfg=read(Path(bundle)/'plans/configurations.json')['pipelines']['overlap']
    if manifest['network']!=spec['network']:raise ValueError('wrong network capsule')
    q=spec['phase']=='qualify'
    doc=dict(kind='qualify_pipeline' if q else 'performance',configuration=deepcopy(cfg),stages=deepcopy(cfg['stages']),
        spec=deepcopy(spec),environment=dict(cores=4,frequency_khz=1066666,poll_sleep_ns=1000,post_start_sleep_ns=1000,
            spin_count=0,exclude_worker0=1,bind_threads=1),**protocol()['qualification' if q else 'pipeline'],input_indices=[0],
        executor_sha256=manifest['phase_runner_sha256' if spec['runner']=='phase' else 'old_runner_sha256'],
        adapter_sha256=manifest['adapter_sha256'] if spec['probe'] else None,runtime_sha256=manifest['runtime_sha256'],
        bundle_id=manifest['bundle_id'],experiment=protocol()['experiment'],startup_affinity_probe=True,
        scheduler_events=bool(spec['probe']),original_wait=True,ring_data_pages=32)
    doc['task_id']=digest(doc)[:24];return doc


def native_command(doc,bundle,remote_bundle,remote_job):
    fn=FunctionType(phase.native_command.__code__,{**phase.native_command.__globals__,'make_task':make_task})
    argv,inputs=fn(doc,bundle,remote_bundle,remote_job)
    if doc['spec']['probe']:
        at=argv.index(remote_bundle+'/runtime/ld-linux-aarch64.so.1')
        argv[at:at]=['VTA_QUEUE_COUNTERS=1','VTA_QUEUE_STATUS='+remote_job+'/queue_status.json']
        argv+=['--session-profile-json',remote_job+'/profile.json','--session-profile-window',f"{doc['warmup']},{doc['scored']}"]
    return argv,inputs


def summary(raw,doc,bundle):
    result=gap.summary(raw,doc,bundle)
    path=Path(raw)/'profile.json'
    if doc['scheduler_events']:
        if not path.exists():raise ValueError('requested scheduler profile absent')
        result['scheduler']=validate_profile(read(path),doc,read(Path(raw)/'affinity.json'))
        from cpu_vta_partitioning.measurement import frames
        rows=frames(Path(raw)/'result.jsonl',sum(doc[k] for k in ('warmup','scored','tail')))
        first,last=doc['warmup'],doc['warmup']+doc['scored']-1
        actual=round((rows[last]['completion_ms']-rows[first]['completion_ms'])*1e6)
        if abs(actual-(result['scheduler']['completion_end_ns']-result['scheduler']['completion_start_ns']))>1000:
            raise ValueError('scheduler anchors differ from completion window')
    else:
        if path.exists():raise ValueError('unrequested scheduler profile')
        result['scheduler']=None
    return result


def matched_ssh(config,*,execute_board=False):
    fn=FunctionType(SSH.measure.__code__,{**SSH.measure.__globals__,'native_command':native_command})
    return type('YoloSchedulerTraceSSH',(SSH,),{'measure':fn})(config,execute_board=execute_board)
