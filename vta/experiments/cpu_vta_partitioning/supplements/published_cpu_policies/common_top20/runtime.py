"""One execution identity per configuration, independent of scoring methods."""
from copy import deepcopy
import math
from pathlib import Path
import statistics
from types import FunctionType

from cpu_vta_partitioning.common import digest,read
from cpu_vta_partitioning.measurement import frames,input_sequence,correctness
from cpu_vta_partitioning.measurement.tasks import native_command as base_command
from cpu_vta_partitioning.transport import SSH
from cpu_policies.open_resources.domain import validate_candidate
from cpu_policies.open_resources.runtime import _affinity


def task(bundle,cfg,phase='short'):
    bundle=Path(bundle);manifest=read(bundle/'manifest.json');plan=read(bundle/'plans/protocol.json')
    if phase not in ('qualify','short','long'):raise ValueError('unknown comparison phase')
    validate_candidate(cfg,read(bundle/'plans/programs.json'),unit_count=21 if manifest['network']=='resnet18' else 14)
    window=dict(warmup=0,scored=1,tail=0) if phase=='qualify' else (
        dict(warmup=8,scored=64,tail=32) if phase=='long' else plan['short_window'])
    document=dict(kind='qualify_pipeline' if phase=='qualify' else 'performance',
        phase=phase,configuration=deepcopy(cfg),stages=deepcopy(cfg['stages']),
        environment=dict(cores=4,frequency_khz=1066666,poll_sleep_ns=1000,post_start_sleep_ns=1000,
                         spin_count=0,exclude_worker0=1,bind_threads=1),
        **window,input_indices=[0],network=manifest['network'],executor_sha256=manifest['runner_sha256'],
        runtime_sha256=manifest['runtime_sha256'],bundle_id=manifest['bundle_id'],
        experiment=plan['experiment'],detailed=False,startup_affinity_probe=True)
    document['task_id']=digest(document)[:24]
    return document


def command(doc,bundle,remote,job):
    if doc!=task(bundle,doc['configuration'],doc['phase']):raise ValueError('comparison task identity differs')
    argv,inputs=base_command(doc,bundle,remote,job)
    cfg=doc['configuration'];argv[argv.index('--cpu-stage-gate')+1]=cfg['execution_mode']
    at=argv.index(remote+'/runtime/ld-linux-aarch64.so.1')
    argv[at:at]=['TVM_BIND_THREADS=1','TVM_EXCLUDE_WORKER0=1']
    argv+=['--cpu-worker-pinning','--allow-cpu-affinity-superset','--resource-affinity-probe',
           '--control-cpu-affinity',','.join(map(str,cfg['control_mask']))]
    if doc['phase']!='qualify':argv[argv.index('--output-mode')+1]='raw'
    if '--process-cpu-affinity' in argv or '--queue-timing' in argv:raise ValueError('unrequested execution change')
    return argv,inputs


def transport(config,*,execute_board=False):
    measure=FunctionType(SSH.measure.__code__,{**SSH.measure.__globals__,'native_command':command})
    return type('CommonTop20SSH',(SSH,),{'measure':measure})(config,execute_board=execute_board)


def summary(raw,doc,bundle):
    raw,bundle=Path(raw),Path(bundle)
    if doc!=task(bundle,doc['configuration'],doc['phase']):raise ValueError('comparison raw task differs')
    total=sum(doc[k] for k in ('warmup','scored','tail'))
    rows=frames(raw/'result.jsonl',total);input_sequence(rows,doc)
    report=read(raw/'affinity.json');ledger,idle=_affinity(report,doc['configuration'],True)
    if any(r.get('cpu_stage_gate_mode','off')!=doc['configuration']['execution_mode'] for r in rows):
        raise ValueError('CPU gate execution differs from frozen configuration')
    wanted=len(read(bundle/'plans/programs.json')[doc['stages'][-1]['program']]['output_node_ids'])
    fingerprints=[]
    for row in rows:
        outputs=row.get('raw_outputs',[])
        if len(outputs)!=wanted or sorted(v['index'] for v in outputs)!=list(range(wanted)):
            raise ValueError('not all unique network output/frontier tensors completed')
        fingerprints.append([[v['index'],v['shape'],v['dtype'],v['fnv1a64']] for v in outputs])
    if any(v!=fingerprints[0] for v in fingerprints):raise ValueError('fixed-input outputs changed across frames')
    if doc['phase']=='qualify':
        correctness(rows,doc,bundle,raw);return dict(correct=True,output_fingerprints=fingerprints[0],
                                                   thread_ledger=ledger,idle_helpers=idle)
    selected=rows[doc['warmup']:doc['warmup']+doc['scored']];times=[r['completion_ms'] for r in selected]
    if any(type(t) not in (int,float) or not math.isfinite(t) for t in times) or any(b<=a for a,b in zip(times,times[1:])):
        raise ValueError('nonfinite or nonmonotone completion timestamps')
    interval=(times[-1]-times[0])/(len(times)-1)
    cpu=selected[-1]['completion_process_cpu_ms']-selected[0]['completion_process_cpu_ms']
    if cpu<0:raise ValueError('process CPU clock moved backwards')
    boundaries=[]
    for index in range(len(doc['stages'])):
        key='stage'+str(index)+'_end_ms'
        boundaries.append(dict(stage=doc['stages'][index]['program'],
            counts=[sum(row[key]<=time for row in rows) for time in (times[0],times[-1])]) if key in rows[0]
            else dict(stage=doc['stages'][index]['program'],counts=None,reason='native output lacks this timestamp'))
    return dict(correct=True,II_ms=interval,FPS=1000/interval,kept_timestamps=len(times),gaps=len(times)-1,
        process_CPU_ms_per_completion=cpu/(len(times)-1),completion_times_ms=times,
        stage_service_ms=[statistics.mean(r['stage'+str(i)+'_ms'] for r in selected) for i in range(len(doc['stages']))],
        stage_completion_counts_at_window=boundaries,output_fingerprints=fingerprints[0],
        thread_ledger=ledger,idle_helpers=idle)
