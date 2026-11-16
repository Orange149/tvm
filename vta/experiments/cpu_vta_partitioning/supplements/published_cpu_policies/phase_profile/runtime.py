"""Same-clock evidence; launch observers preserve the original wait policy."""
from copy import deepcopy
import json
import math
from pathlib import Path
import statistics
from types import FunctionType

from cpu_vta_partitioning.common import digest,read
from cpu_vta_partitioning.transport import SSH
from cpu_vta_partitioning.measurement import frames,input_sequence,correctness
from cpu_policies.open_resources.runtime import _affinity
from .prepare import protocol


def make_task(bundle,spec):
    bundle=Path(bundle);manifest=read(bundle/'manifest.json')
    if spec['network']!=manifest['network'] or spec['runner'] not in ('old','phase'):
        raise ValueError('wrong network or executor')
    if spec['scope'] not in ('component','pipeline') or spec['phase'] not in ('qualify','clean','observed'):
        raise ValueError('unknown phase profile task')
    if spec['probe']!=(spec['phase']!='clean') or (spec['probe'] and spec['runner']!='phase'):
        raise ValueError('probe/runner/task identity differs')
    cfg=read(bundle/'plans/configurations.json')['components' if spec['scope']=='component' else 'pipelines'][spec['case']]
    qualify=spec['phase']=='qualify';kind=('qualify_component' if qualify else 'calibration') if spec['scope']=='component' else ('qualify_pipeline' if qualify else 'performance')
    window=protocol()['qualification' if qualify else spec['scope']]
    doc=dict(kind=kind,configuration=deepcopy(cfg),stages=deepcopy(cfg['stages']),spec=deepcopy(spec),
        environment=dict(cores=4,frequency_khz=1066666,poll_sleep_ns=1000,post_start_sleep_ns=1000,
                         spin_count=0,exclude_worker0=1,bind_threads=1),
        **window,input_indices=[0],executor_sha256=manifest['phase_runner_sha256' if spec['runner']=='phase' else 'old_runner_sha256'],
        adapter_sha256=manifest['adapter_sha256'] if spec['probe'] else None,
        runtime_sha256=manifest['runtime_sha256'],bundle_id=manifest['bundle_id'],
        experiment=protocol()['experiment'],startup_affinity_probe=True)
    doc['task_id']=digest(doc)[:24]
    return doc


def native_command(doc,bundle,remote_bundle,remote_job):
    from cpu_vta_partitioning.measurement.tasks import native_command as base
    if doc!=make_task(bundle,doc['spec']):raise ValueError('phase task document changed')
    argv,inputs=base(doc,bundle,remote_bundle,remote_job)
    if doc['spec']['runner']=='phase':
        argv[argv.index(remote_bundle+'/runner/vta_stage_pipeline_runner')]=remote_bundle+'/runner/vta_stage_pipeline_runner_phase'
    at=argv.index(remote_bundle+'/runtime/ld-linux-aarch64.so.1')
    argv[at:at]=['TVM_BIND_THREADS=1','TVM_EXCLUDE_WORKER0=1']
    argv+=['--allow-cpu-affinity-superset','--cpu-worker-pinning','--resource-affinity-probe',
           '--control-cpu-affinity',','.join(map(str,doc['configuration']['control_mask']))]
    if doc['spec']['scope']=='pipeline' and doc['kind']=='performance':
        argv[argv.index('--output-mode')+1]='raw'
    if doc['spec']['probe']:
        i=next(i for i,x in enumerate(argv) if x.startswith('LD_PRELOAD='))
        argv[i]='LD_PRELOAD='+remote_bundle+'/runner/libwait_adapter.so:'+remote_bundle+'/runtime/libvta.so'
        at=argv.index(remote_bundle+'/runtime/ld-linux-aarch64.so.1')
        argv[at:at]=['VTA_WAIT_POLICY=original','VTA_WAIT_PROBE=1',
                    'VTA_WAIT_TRACE='+remote_job+'/wait_trace.jsonl','VTA_WAIT_STATUS='+remote_job+'/wait_status.json']
        argv+=['--phase-event-json',remote_job+'/phase_trace.jsonl']
    if '--process-cpu-affinity' in argv or '--queue-timing' in argv:
        raise ValueError('unrequested affinity override or queue observer')
    return argv,inputs


def matched_ssh(config,*,execute_board=False):
    measure=FunctionType(SSH.measure.__code__,{**SSH.measure.__globals__,'native_command':native_command})
    return type('SameClockPhaseSSH',(SSH,),{'measure':measure})(config,execute_board=execute_board)


def phase_summary(raw,doc,affinity):
    path=Path(raw)/'phase_trace.jsonl'
    if not doc['spec']['probe']:
        if path.exists():raise ValueError('unrequested phase observer')
        return None
    rows=[json.loads(s) for s in path.read_text().splitlines()]
    total=sum(doc[k] for k in ('warmup','scored','tail'));names={s['program'] for s in doc['stages']}
    expected={(name,i,op) for name in names for i in range(total) for op in ('set','run','get')}
    keys=[(r['stage'],r['call'],r['operation']) for r in rows]
    if len(keys)!=len(expected) or set(keys)!=expected:
        raise ValueError('missing/duplicate/extra stage phase events')
    callers={s['program']:int(a['caller_tid']) for s,a in zip(doc['stages'],affinity['stages'])}
    for r in rows:
        if r['tid']!=callers[r['stage']]:raise ValueError('phase caller TID differs from affinity')
        if (not 0<=r['wall_start_ns']<=r['wall_end_ns'] or not 0<=r['cpu_start_ns']<=r['cpu_end_ns']
            or r['cpu_end_ns']-r['cpu_start_ns']>r['wall_end_ns']-r['wall_start_ns']+10000):
            raise ValueError('invalid phase clocks')
    for name in names:
        previous_end=0
        for i in range(total):
            group=[next(r for r in rows if (r['stage'],r['call'],r['operation'])==(name,i,op))
                   for op in ('set','run','get')]
            if group[0]['wall_start_ns']<previous_end or any(a['wall_end_ns']>b['wall_start_ns'] for a,b in zip(group,group[1:])):
                raise ValueError('stage phases are not sequential')
            previous_end=group[-1]['wall_end_ns']
    return dict(events=len(rows),stage_calls={name:total for name in names},
        clock='CLOCK_MONOTONIC/caller CLOCK_THREAD_CPUTIME_ID',
        native_stage_service_not_equal_to_probe_envelope=True)


def summary(raw,doc,bundle):
    raw,bundle=Path(raw),Path(bundle);total=sum(doc[k] for k in ('warmup','scored','tail'))
    rows=frames(raw/'result.jsonl',total);input_sequence(rows,doc)
    affinity=read(raw/'affinity.json');ledger,idle=_affinity(affinity,doc['configuration'],True)
    if any(r.get('cpu_stage_gate_mode','off')!='off' for r in rows):raise ValueError('unexpected gate')
    selected=rows[doc['warmup']:doc['warmup']+doc['scored']]
    fingerprints=[]
    for r in rows:
        outputs=r.get('raw_outputs',[])
        p=read(bundle/'plans/programs.json')[doc['stages'][-1]['program']]
        if len(outputs)!=len(p['output_node_ids']):raise ValueError('network output/frontier incomplete')
        fingerprints.append([[o['index'],o['shape'],o['dtype'],o['fnv1a64']] for o in outputs])
    if any(v!=fingerprints[0] for v in fingerprints):raise ValueError('fixed input output changed between frames')
    if doc['kind'].startswith('qualify'):
        correctness(rows,doc,bundle,raw);result=dict(correct=True)
    elif doc['kind']=='calibration':
        cost={}
        for op in ('set','run','get'):
            for name,suffix in [(op+'_ms','_ms'),(op+'_core_ms','_process_cpu_ms')]:
                values=[r['stage0_'+op+suffix] for r in selected]
                if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in values):
                    raise ValueError('invalid isolated phase clock')
                cost[name]=statistics.mean(values)
        cost.update(service_ms=sum(cost[op+'_ms'] for op in ('set','run','get')),
            run_core_ms=cost['run_core_ms'],boundary_core_ms=cost['set_core_ms']+cost['get_core_ms'])
        result=dict(correct=True,cost=cost)
    else:
        times=[r['completion_ms'] for r in selected]
        if any(b<=a for a,b in zip(times,times[1:])):raise ValueError('invalid completion timestamps')
        ii=(times[-1]-times[0])/(len(times)-1)
        cpu=selected[-1]['completion_process_cpu_ms']-selected[0]['completion_process_cpu_ms']
        result=dict(correct=True,ii_ms=ii,fps=1000/ii,kept_timestamps=len(times),gaps=len(times)-1,
            process_core_ms_per_completion=cpu/(len(times)-1),
            stage_service_ms=[statistics.mean(r['stage'+str(i)+'_ms'] for r in selected) for i in range(len(doc['stages']))])
    phase=phase_summary(raw,doc,affinity)
    if doc['spec']['probe']:
        from .worker import validate_workers
        # Only the observer interface is reused, never its control flow.
        worker=validate_workers(raw,doc,bundle,affinity)
    else:worker=None
    return dict(**result,phase=phase,worker=worker,output_fingerprints=fingerprints[0],
                thread_ledger=ledger,idle_auxiliary_threads=idle)
