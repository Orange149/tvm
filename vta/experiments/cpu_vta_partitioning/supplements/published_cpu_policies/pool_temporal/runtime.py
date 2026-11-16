"""Optional isolated observers, preserving original native scheduling/waiting."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
from types import FunctionType

from cpu_vta_partitioning.common import digest,read
from cpu_vta_partitioning.measurement.tasks import native_command as base_command
from cpu_vta_partitioning.transport import SSH
from device_phase.collect import validate_events
from phase_profile.runtime import summary as old_summary,phase_summary
from phase_profile.worker import validate_workers


def make_task(bundle,spec):
    bundle=Path(bundle);manifest=read(bundle/'manifest.json')
    if spec['network']!=manifest['network'] or spec['mode'] not in ('qualify','observed'):
        raise ValueError('wrong intrinsic profile network or mode')
    cfg=read(bundle/'plans/components.json')[spec['case']]
    cpu=cfg['stages'][0]['device']=='cpu';qualify=spec['mode']=='qualify'
    doc=dict(kind='qualify_component' if qualify else 'calibration',configuration=deepcopy(cfg),
        stages=deepcopy(cfg['stages']),spec=dict(scope='component',probe=False,policy='original'),
        profile_spec=spec,profile_device='cpu' if cpu else 'vta',
        environment=dict(cores=4,frequency_khz=1066666,poll_sleep_ns=1000,post_start_sleep_ns=1000,
                         spin_count=0,exclude_worker0=1,bind_threads=1),
        warmup=0 if qualify else 2,scored=1 if qualify else 8,tail=0,input_indices=[0],
        executor_sha256=manifest['profile_runner_sha256'],runtime_sha256=manifest['runtime_sha256'],
        observer_sha256=manifest['wait_adapter_sha256' if cpu else 'device_adapter_sha256'],
        experiment='cross_network_pool_temporal_01',bundle_id=manifest['bundle_id'],startup_affinity_probe=True)
    doc['task_id']=digest(doc)[:24];return doc


def command(doc,bundle,remote,job):
    if doc!=make_task(bundle,doc['profile_spec']):raise ValueError('intrinsic profile task differs')
    argv,inputs=base_command(doc,bundle,remote,job)
    argv[argv.index(remote+'/runner/vta_stage_pipeline_runner')]=remote+'/runner/vta_stage_pipeline_runner_phase'
    at=argv.index(remote+'/runtime/ld-linux-aarch64.so.1')
    argv[at:at]=['TVM_BIND_THREADS=1','TVM_EXCLUDE_WORKER0=1']
    argv+=['--allow-cpu-affinity-superset','--resource-affinity-probe',
           '--control-cpu-affinity',','.join(map(str,doc['configuration']['control_mask'])),
           '--phase-event-json',job+'/phase_trace.jsonl']
    if doc['profile_device']=='cpu':
        argv.append('--cpu-worker-pinning');library='libwait_adapter.so'
        variables=['VTA_WAIT_POLICY=original','VTA_WAIT_PROBE=1',
                   'VTA_WAIT_TRACE='+job+'/wait_trace.jsonl','VTA_WAIT_STATUS='+job+'/wait_status.json']
    else:
        library='libdevice_phase.so';variables=['VTA_DEVICE_PHASE_TRACE='+job+'/device_trace.jsonl']
    index=next(i for i,s in enumerate(argv) if s.startswith('LD_PRELOAD='))
    argv[index]='LD_PRELOAD='+remote+'/runner/'+library+':'+remote+'/runtime/libvta.so'
    at=argv.index(remote+'/runtime/ld-linux-aarch64.so.1');argv[at:at]=variables
    return argv,inputs


@lru_cache(maxsize=128)
def module_has_parallel_api(path,expected):
    from cpu_vta_partitioning.common import sha256
    if sha256(path)!=expected:raise ValueError('serial-module ELF proof hash differs')
    result=subprocess.run(['readelf','-Ws',path],capture_output=True,text=True,check=True)
    return 'TVMBackendParallelLaunch' in result.stdout


@lru_cache(maxsize=1)
def serial_parser():
    from wait_diagnosis.runtime import trace_summary
    original=inspect.getsource(trace_summary)
    target="    if ('0' not in stage_stats or stage_stats['0']['program']!='cpu_00_02' or\n            not stage_stats['0']['kernel_launches']):raise ValueError('actual target CPU module not intercepted')"
    if original.count(target)!=1:raise ValueError('serial observer adaptation anchor changed')
    patched=original.replace(target,'    # Target ELF has no parallel-launch API; measured caller work is serial.')
    scope=dict(trace_summary.__globals__);exec(compile(patched,'<ELF-proven serial stage observer>','exec'),scope)
    return scope['trace_summary'],hashlib.sha256(patched.encode()).hexdigest()


def cpu_observer(raw,doc,bundle,affinity):
    adapted={**doc,'spec':dict(scope='component',probe=True,policy='original')}
    try:return validate_workers(raw,adapted,bundle,affinity)
    except ValueError as error:
        if str(error)!='actual profiled CPU module not intercepted':raise
        p=read(Path(bundle)/'plans/programs.json')[doc['stages'][0]['program']]
        path=Path(bundle)/'programs'/doc['stages'][0]['program']/'module.so'
        if module_has_parallel_api(str(path),p['files']['module.so']['sha256']):raise
        function,source_sha=serial_parser();result=function(raw,adapted,bundle,affinity)
        result['serial_target_proof']=dict(module_sha256=p['files']['module.so']['sha256'],
            no_parallel_launch_symbol=True,startup_interception_still_required=True,parser_sha256=source_sha)
        return result


def summary(raw,doc,bundle,*,affinity_checker=None):
    def phases(path,document,affinity):
        enabled=deepcopy(document);enabled['spec']['probe']=True
        return phase_summary(path,enabled,affinity)
    scope={**old_summary.__globals__,'phase_summary':phases}
    if affinity_checker is not None:scope['_affinity']=affinity_checker
    check=FunctionType(old_summary.__code__,scope)
    result=check(raw,doc,bundle);raw=Path(raw)
    if doc['profile_device']=='cpu':
        result['worker']=cpu_observer(raw,doc,bundle,read(raw/'affinity.json'))
    else:
        rows=[json.loads(s) for s in (raw/'device_trace.jsonl').read_text().splitlines()]
        spans=[json.loads(s) for s in (raw/'phase_trace.jsonl').read_text().splitlines()]
        result['device']=validate_events(rows,spans,doc)
    return result


def transport(config,*,execute_board=False):
    measure=FunctionType(SSH.measure.__code__,{**SSH.measure.__globals__,'native_command':command})
    return type('PoolIntrinsicSSH',(SSH,),{'measure':measure})(config,execute_board=execute_board)
