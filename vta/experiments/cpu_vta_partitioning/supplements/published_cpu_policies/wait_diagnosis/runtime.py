"""Exact task identities, unchanged native commands plus explicit preload modes."""
from copy import deepcopy
from pathlib import Path
from types import FunctionType
import json
import math
import statistics
from cpu_vta_partitioning.common import digest,read
from cpu_vta_partitioning.transport import SSH
from cpu_policies.open_resources.runtime import _affinity
from yolo_validation.pipeline import work_boundaries
from yolo_transfer.pipeline import check_outputs
from run_yolo_validation_gate_compat import adapted_summary
from .common import protocol


def make_task(bundle,spec,qualify=False):
    bundle=Path(bundle);m=read(bundle/'manifest.json');p=read(bundle/'plans/protocol.json')
    cfg=read(bundle/'plans/configurations.json')['components' if spec['scope']=='component' else 'pipelines'][spec['case']]
    window=p['qualification' if qualify else 'component' if spec['scope']=='component' else 'performance']
    if spec['scope']=='component' and spec['policy']!='original':raise ValueError('no new isolated blocking profiles authorized')
    if spec['policy'] not in ('original','blocking') or type(spec['probe']) is not bool:raise ValueError('invalid wait mode')
    doc=dict(kind=('qualify_component' if qualify else 'calibration') if spec['scope']=='component'
                  else ('qualify_pipeline' if qualify else 'performance'),
        configuration=deepcopy(cfg),stages=deepcopy(cfg['stages']),spec=deepcopy(spec),qualification=qualify,
        input_indices=[0],**window,experiment=p['experiment'],bundle_id=m['bundle_id'],
        executor_sha256=m['runner_sha256'],runtime_sha256=m['runtime_sha256'],adapter_sha256=m['adapter_sha256'],
        environment=dict(cores=4,frequency_khz=1066666,poll_sleep_ns=1000,post_start_sleep_ns=1000,
                         spin_count=0,exclude_worker0=1,bind_threads=1),
        output_mode='all_stage_dumps' if qualify else 'all_final_hashes')
    doc['task_id']=digest(doc)[:24];return doc


def native_command(doc,bundle,remote_bundle,remote_job):
    from cpu_vta_partitioning.measurement.tasks import native_command as base
    if doc!=make_task(bundle,doc['spec'],doc['qualification']):raise ValueError('wait task identity changed')
    argv,inputs=base(doc,bundle,remote_bundle,remote_job);cfg=doc['configuration']
    at=argv.index(remote_bundle+'/runtime/ld-linux-aarch64.so.1')
    argv[at:at]=['TVM_BIND_THREADS=1','TVM_EXCLUDE_WORKER0=1']
    argv+=['--allow-cpu-affinity-superset','--cpu-worker-pinning','--control-cpu-affinity',
           ','.join(map(str,cfg['control_mask']))]
    if doc['qualification']:argv.append('--resource-affinity-probe')
    elif doc['spec']['scope']=='pipeline':argv[argv.index('--output-mode')+1]='raw'
    # Original/unprobed tasks do not load the adapter at all.
    if doc['spec']['probe'] or doc['spec']['policy']=='blocking':
        preload=next(i for i,x in enumerate(argv) if x.startswith('LD_PRELOAD='))
        argv[preload]='LD_PRELOAD='+remote_bundle+'/adapter/libwait_adapter.so:'+remote_bundle+'/runtime/libvta.so'
        at=argv.index(remote_bundle+'/runtime/ld-linux-aarch64.so.1')
        argv[at:at]=['VTA_WAIT_POLICY='+doc['spec']['policy'],'VTA_WAIT_PROBE='+str(int(doc['spec']['probe'])),
                    'VTA_WAIT_TRACE='+remote_job+'/wait_trace.jsonl','VTA_WAIT_STATUS='+remote_job+'/wait_status.json']
    if '--queue-timing' in argv or '--process-cpu-affinity' in argv:raise ValueError('unexpected probe or process binding')
    return argv,inputs


def matched_ssh(config,*,execute_board=False):
    measure=FunctionType(SSH.measure.__code__,{**SSH.measure.__globals__,'native_command':native_command})
    def acknowledge(self,doc,run_dir,boot):
        import shlex
        SSH.acknowledge(self,doc,run_dir,boot)
        self.idle_command('cd '+shlex.quote(self.job_path(doc,run_dir,boot))+' && rm -f wait_trace.jsonl wait_status.json',boot=boot)
    return type('WaitDiagnosisSSH',(SSH,),{'measure':measure,'acknowledge':acknowledge})(config,execute_board=execute_board)


def trace_summary(raw,doc,bundle,affinity):
    raw,bundle=Path(raw),Path(bundle);spec=doc['spec'];status=read(raw/'wait_status.json')
    if (status['policy']!=spec['policy'] or status['probe'] is not spec['probe'] or
        status['errors'] or status['overflow'] or status['launches']<1 or status['callbacks']<1):
        raise ValueError('wait ABI/probe failed, overflowed, or did not intercept')
    if spec['policy']=='blocking' and status['futex_waits']<1:raise ValueError('blocking policy never exercised')
    if not spec['probe']:
        if status['records'] or (raw/'wait_trace.jsonl').exists():raise ValueError('unrequested detailed records')
        return dict(status=status,scope='counters only; no thread timing')
    rows=[json.loads(line) for line in (raw/'wait_trace.jsonl').read_text().splitlines()]
    if len(rows)!=status['records']:raise ValueError('wait trace length differs')
    launches={};tasks={};barriers={};roles={int(s['caller_tid']):(i,s)
        for i,s in enumerate(affinity['stages'])}
    module_ids={}
    for row in rows:
        if row['kind'] not in (0,1,2):raise ValueError('invalid trace event')
        if not (1<=row['tasks']<=4 and row['wall_start_ns']<=row['wall_end_ns']
                and row['cpu_start_ns']<=row['cpu_end_ns']):raise ValueError('invalid wait clocks or task count')
        if row['cpu_end_ns']-row['cpu_start_ns']>row['wall_end_ns']-row['wall_start_ns']+10000:
            raise ValueError('thread CPU exceeds enclosing wall interval')
        if row['kind']==0:
            if row['launch'] in launches:raise ValueError('duplicate launch')
            launches[row['launch']]=row
        elif row['kind']==1:tasks.setdefault(row['launch'],[]).append(row)
        else:barriers.setdefault(row['launch'],[]).append(row)
        if row['result']!=0:raise ValueError('callback or barrier returned an error')
        path=row['module'];relative=None
        for prefix in ('programs/','runtime/','runner/'):
            marker='/'+prefix
            if marker in path:relative=prefix+path.split(marker,1)[1];break
        if relative:
            if relative not in module_ids:module_ids[relative]=__import__('cpu_vta_partitioning.common',fromlist=['sha256']).sha256(bundle/relative)
            row['module_sha256']=module_ids[relative]
    if len(launches)!=status['launches'] or sum(map(len,tasks.values()))!=status['callbacks'] or sum(map(len,barriers.values()))!=status['barriers']:
        raise ValueError('launch/callback/barrier coverage differs')
    stage_stats={};worker_stats={};unassigned=0
    for key,launch in launches.items():
        group=tasks.get(key,[])
        if len(group)!=launch['tasks'] or {r['task'] for r in group}!=set(range(launch['tasks'])):
            raise ValueError('missing or duplicate logical task')
        if len({r['tid'] for r in group})!=launch['tasks']:raise ValueError('logical tasks share a TID')
        if max(r['wall_end_ns'] for r in group)>launch['wall_end_ns']:raise ValueError('callback outlived launch context')
        if launch['caller'] not in roles:unassigned+=1;continue
        index,role=roles[launch['caller']];cfg=doc['stages'][index]
        # VTA's runtime may also invoke CPU helpers; do not call these CPU subgraphs.
        if cfg['device']!='cpu':unassigned+=1;continue
        expected={r['slot']:r['tid'] for r in role.get('worker_slots',[])}
        if len(expected)!=cfg['threads']:raise ValueError('affinity logical worker slots missing')
        for task in group:
            if expected.get(task['task'])!=task['tid']:raise ValueError('trace logical worker/TID differs from affinity probe')
            if task['caller']!=launch['caller']:raise ValueError('trace group caller differs')
            if task['barrier_cpu_ns']>task['cpu_end_ns']-task['cpu_start_ns']+10000:
                raise ValueError('barrier CPU exceeds callback CPU')
            k=(index,task['task'],task['tid']);w=worker_stats.setdefault(k,dict(stage=index,logical_worker=task['task'],
                tid=task['tid'],calls=0,wall_ns=0,cpu_ns=0,barrier_wall_ns=0,barrier_cpu_ns=0,yield_calls=0))
            w['calls']+=1;w['wall_ns']+=task['wall_end_ns']-task['wall_start_ns'];w['cpu_ns']+=task['cpu_end_ns']-task['cpu_start_ns']
            w['barrier_wall_ns']+=task['barrier_wall_ns'];w['barrier_cpu_ns']+=task['barrier_cpu_ns'];w['yield_calls']+=task['yield_calls']
        s=stage_stats.setdefault(str(index),dict(program=cfg['program'],launches=0,launch_wall_ns=0,
            body_cpu_ns=0,barrier_cpu_ns=0,barrier_wall_thread_ns=0,completion_tail_wall_ns=0,
            completion_tail_cpu_ns=0,completion_yield_calls=0,callback_yield_calls=0,last_workers={}))
        s['launches']+=1;s['launch_wall_ns']+=launch['wall_end_ns']-launch['wall_start_ns']
        s['kernel_launches']=s.get('kernel_launches',0)+int('/programs/'+cfg['program']+'/module.so' in launch['module'])
        s['body_cpu_ns']+=sum(r['cpu_end_ns']-r['cpu_start_ns'] for r in group)
        s['barrier_cpu_ns']+=sum(r['barrier_cpu_ns'] for r in group)
        s['barrier_wall_thread_ns']+=sum(r['barrier_wall_ns'] for r in group)
        s['completion_tail_wall_ns']+=launch['wall_end_ns']-launch['caller_tail_wall_start_ns']
        s['completion_tail_cpu_ns']+=launch['cpu_end_ns']-launch['caller_tail_cpu_start_ns']
        s['completion_yield_calls']+=launch['completion_yield_calls'];s['callback_yield_calls']+=sum(r['yield_calls'] for r in group)
        s['barrier_yield_calls']=s.get('barrier_yield_calls',0)+sum(r['yield_calls'] for r in barriers.get(key,[]))
        last=max(group,key=lambda r:r['wall_end_ns'])['task'];s['last_workers'][str(last)]=s['last_workers'].get(str(last),0)+1
    if ('0' not in stage_stats or stage_stats['0']['program']!='cpu_00_02' or
            not stage_stats['0']['kernel_launches']):raise ValueError('actual target CPU module not intercepted')
    for s in stage_stats.values():s['outside_barrier_cpu_ns']=max(0,s['body_cpu_ns']-s['barrier_cpu_ns'])
    for w in worker_stats.values():
        w['nonrunning_ns']=max(0,w['wall_ns']-w['cpu_ns']);w['outside_barrier_cpu_ns']=max(0,w['cpu_ns']-w['barrier_cpu_ns'])
    return dict(status=status,stages=stage_stats,workers=list(worker_stats.values()),module_hashes=module_ids,
        unassigned_startup_or_nonCPU_launches=unassigned,
        scope='whole session; launch statistics include warmup/tail/possible startup, not per-scored-frame costs',
        offCPU_not_separated=True,thread_wait_sums_not_added_to_II=True)


def summary(raw,doc,bundle):
    from cpu_vta_partitioning.measurement import frames,input_sequence,calibration,correctness
    raw,bundle=Path(raw),Path(bundle);count=sum(doc[k] for k in ('warmup','scored','tail'))
    rows=frames(raw/'result.jsonl',count);input_sequence(rows,doc)
    affinity=read(raw/'affinity.json')
    if doc['spec']['scope']=='pipeline':
        base=adapted_summary()[0](raw,doc,bundle)
        if not doc['qualification']:base['window_work']=work_boundaries(rows,doc)
    else:
        cfg=doc['configuration'];ledger,idle=_affinity(affinity,cfg,doc['qualification'])
        p=read(bundle/'plans/programs.json')[cfg['stages'][0]['program']];expected=read(bundle/'plans/expected_outputs.json')
        for row in rows:check_outputs(row.get('raw_outputs',[]),p,expected)
        if doc['qualification']:
            correctness(rows,doc,bundle,raw);base=dict(correct=True)
        else:
            cost=calibration(rows,doc['warmup'],doc['scored'])
            kept=rows[doc['warmup']:doc['warmup']+doc['scored']]
            for op in ('set','run','get'):
                for suffix,field in (('_ms','_ms'),('_core_ms','_process_cpu_ms')):
                    cost[op+suffix]=statistics.mean(r['stage0_'+op+field] for r in kept)
            base=dict(correct=True,cost=cost,stage_service_ms=[cost['service_ms']])
        base.update(thread_ledger=ledger,idle_auxiliary_threads=idle)
    if doc['spec']['probe'] or doc['spec']['policy']=='blocking':base['wait']=trace_summary(raw,doc,bundle,affinity)
    elif (raw/'wait_status.json').exists() or (raw/'wait_trace.jsonl').exists():raise ValueError('unexpected adapter records')
    return base
