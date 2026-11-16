"""Hash-bound tasks and raw validation for independent component actors."""
from copy import deepcopy
from pathlib import Path
from functools import lru_cache
import json
import math
import shlex
import statistics
from .common import api, protocol, DEFAULT_PARENT, load_legacy
from ..core_model.affinity import validate_affinity
from ..core_model.domain import component_key

def task(bundle,kind,name,*,actors=None,targets=None,configuration=None,probe=False,executor='contention'):
    common=api(bundle)
    spec=common.read(Path(bundle)/'plans/executors.json')[executor]
    doc={'kind':kind,'name':name,'actors':deepcopy(actors),'targets':deepcopy(targets),
         'configuration':deepcopy(configuration),'probe':probe,'executor':executor,
         'executor_sha256':spec['sha256'],'bundle_id':common.read(Path(bundle)/'manifest.json')['bundle_id'],
         'protocol':protocol()}
    if kind not in ('isolated','concurrent','diagnostic','pipeline','bridge'): raise ValueError('invalid task kind')
    if probe != (kind=='diagnostic'): raise ValueError('probe is diagnostic-only')
    if kind in ('isolated','concurrent','diagnostic'):
        if not actors or not targets or any(type(i)!=int or i<0 or i>=len(actors) for i in targets): raise ValueError('invalid actors/targets')
        if len(set(targets))!=len(targets): raise ValueError('duplicate target')
        for s in actors:
            if s['device']=='cpu' and (s['mask']!=s['worker_core_map'] or len(s['mask'])!=s['threads']): raise ValueError('worker map differs')
    doc['task_id']=common.digest(doc)[:24]
    return doc

def validate_document(doc,bundle):
    expected=task(bundle,doc['kind'],doc['name'],actors=doc['actors'],targets=doc['targets'],
                  configuration=doc['configuration'],probe=doc['probe'],executor=doc['executor'])
    if expected!=doc: raise ValueError('task identity differs')

def native_spec(doc,bundle,remote_bundle):
    common=api(bundle)
    programs=common.read(Path(bundle)/'plans/programs.json')
    frontiers=common.read(Path(bundle)/'plans/inputs.json')['records'][0]['frontiers']
    actors=[]
    for stage in doc['actors']:
        p=programs[stage['program']]
        paths=lambda ids:[remote_bundle+'/'+frontiers[str(n)]['raw'] for n in ids]
        actors.append({'inputs':paths(p['input_node_ids']),'references':paths(p['output_node_ids'])})
    return {'actors':actors,'targets':doc['targets'],'warmup':2,'scored':8,'probe':int(doc['probe'])}

def pipeline_document(doc,bundle):
    return {'kind':'performance','stages':deepcopy(doc['configuration']['stages']),
            'configuration':deepcopy(doc['configuration']),'environment':protocol()['environment'],
            'warmup':8,'scored':16,'tail':4,'input_indices':[0],'detailed':False}


def native_command(doc,bundle,remote_bundle,remote_job):
    validate_document(doc,bundle)
    c=api(bundle)
    if doc['kind'] in ('pipeline','bridge'):
        from cpu_vta_partitioning.measurement.tasks import native_command as base_command
        base=pipeline_document(doc,bundle)
        argv,inputs=base_command(base,bundle,remote_bundle,remote_job)
        selected=c.read(Path(bundle)/'plans/executors.json')[doc['executor']]['path']
        argv[argv.index(remote_bundle+'/runner/vta_stage_pipeline_runner')]=remote_bundle+'/'+selected
        loader=argv.index(remote_bundle+'/runtime/ld-linux-aarch64.so.1')
        argv[loader:loader]=['TVM_BIND_THREADS=1','TVM_EXCLUDE_WORKER0=1']
        argv+=['--allow-cpu-affinity-superset','--control-cpu-affinity','0,1,2,3','--cpu-worker-pinning',
               '--resource-affinity-probe']
        argv[argv.index('--output-mode')+1]='raw' if doc['kind']=='bridge' else 'raw_all_stages'
        if doc['kind']=='pipeline': argv+=['--output-dump-dir',remote_job+'/outputs']
        return argv,inputs
    programs=c.read(Path(bundle)/'plans/programs.json')
    spec=native_spec(doc,bundle,remote_bundle)
    from cpu_vta_partitioning.measurement.tasks import native_command as base_command
    from .domain import configurations
    base=pipeline_document(task(bundle,'pipeline','env',configuration=configurations()['explanation/overlap4']),bundle)
    original,_=base_command(base,bundle,remote_bundle,remote_job)
    loader=original.index(remote_bundle+'/runtime/ld-linux-aarch64.so.1')
    argv=original[:loader]+['TVM_BIND_THREADS=1','TVM_EXCLUDE_WORKER0=1',remote_bundle+'/runtime/ld-linux-aarch64.so.1','--library-path',remote_bundle+'/runtime',
                           remote_bundle+'/runner/vta_stage_pipeline_runner_contention']
    argv+=['--contention-spec',remote_job+'/contention_spec.json','--input-files',','.join(spec['actors'][0]['inputs']),
           '--runs','8','--runtime-num-threads','1','--queue-depth','2','--cpu-stage-gate','off',
           '--control-cpu-affinity','0,1,2,3','--cpu-worker-pinning','--allow-cpu-affinity-superset',
           '--resource-affinity-probe','--output-mode','raw','--output-jsonl',remote_job+'/result.jsonl',
           '--thread-affinity-report',remote_job+'/affinity.json','--process-timeout-s','90']
    for i,s in enumerate(doc['actors']):
        p=programs[s['program']]; root=remote_bundle+'/programs/'+s['program']
        for flag,value in [('name',s['program']),('graph',root+'/graph.json'),('lib',root+'/module.so'),
                           ('params',root+'/params.bin'),('device',s['device']),('input-names',','.join(p['input_names'])),
                           ('input-sources',','.join('input:'+str(k) for k in range(len(p['input_names'])))),
                           ('runtime-num-threads',str(s['threads'])),('cpu-affinity',','.join(map(str,s['mask'])))]:
            argv += [f'--stage{i}-'+flag,value]
        if s['device']=='cpu': argv += [f'--stage{i}-cpu-worker-map',','.join(map(str,s['worker_core_map']))]
    if doc['probe']: argv+=['--run-thread-counters-json',remote_job+'/run_threads.json','--run-thread-counters-frames','2,6']
    return argv,json.dumps(spec,sort_keys=True,separators=(',',':'))+'\n'

def validate_micro(rows,doc,affinity):
    if not rows: raise ValueError('missing component observations')
    cfg={'stages':doc['actors'],'affinity_mode':'pinned','control_mask':[0,1,2,3]}
    # Replay has one coordinator, not a pipeline producer/collector pair.
    ledger=[]
    for i,s in enumerate(doc['actors']):
        report=deepcopy(affinity); report['stages']=[affinity['stages'][i]]
        part,idle=validate_affinity(report,{'stages':[s],'affinity_mode':'pinned'},True)
        for entry in part: entry['stage_index']=i
        ledger+=part
    all_tids=[t for r in affinity['stages'] for t in r['thread_ids']]
    if len(all_tids)!=len(set(all_tids)): raise ValueError('actors share TID ownership')
    summaries={}
    for i,s in enumerate(doc['actors']):
        group=[r for r in rows if r.get('actor')==i]
        if i in doc['targets'] and len(group)!=8: raise ValueError('missing or duplicate target observations')
        if not group or [r['iteration'] for r in group]!=list(range(len(group))) or len(group)>8: raise ValueError('frame sequence differs')
        for r in group:
            if r.get('schema')!='controlled_contention_v1' or r.get('program')!=s['program'] or r.get('qualification_passed') is not True or r.get('final_output_passed') is not True: raise ValueError('invalid output/schema')
            for k in ('start_ms','end_ms','service_ms','set_ms','run_ms','get_ms','active_start_ms','active_end_ms'):
                if type(r.get(k)) not in (float,int) or not math.isfinite(r[k]) or r[k]<0: raise ValueError('invalid time')
            if abs(r['service_ms']-(r['end_ms']-r['start_ms']))>1e-5 or abs(r['service_ms']-sum(r[k] for k in ('set_ms','run_ms','get_ms')))>1e-5: raise ValueError('wall clock conservation differs')
            if r['calls']<len(group) or r['extra_calls']!=max(0,r['calls']-8): raise ValueError('invalid actual call accounting')
        if any(any(row[k]!=group[0][k] for k in ('calls','extra_calls','active_start_ms','active_end_ms')) for row in group):
            raise ValueError('inconsistent actor window/call totals')
        if any(b['start_ms']<a['end_ms'] for a,b in zip(group,group[1:])): raise ValueError('actor calls overlap')
        fields=['service_ms','set_ms','run_ms','get_ms']
        if doc['kind']=='isolated':
            if any(r['process_cpu_valid'] is not True for r in group): raise ValueError('isolated CPU clock missing')
            fields+=['run_core_ms','set_core_ms','get_core_ms']
            if any(type(row.get(k)) not in (int,float) or not math.isfinite(row[k]) or row[k]<0 for row in group for k in fields): raise ValueError('invalid isolated clock')
        elif any(r['process_cpu_valid'] is not False for r in group): raise ValueError('concurrent process clocks cannot be stage costs')
        summaries[str(i)]={k:statistics.mean(r[k] for r in group) for k in fields}
        summaries[str(i)].update(program=s['program'],calls=group[0]['calls'],extra_calls=group[0]['extra_calls'],
            samples=len(group),active_start_ms=group[0]['active_start_ms'],active_end_ms=group[0]['active_end_ms'])
        if doc['kind']=='isolated': summaries[str(i)]['boundary_core_ms']=summaries[str(i)]['set_core_ms']+summaries[str(i)]['get_core_ms']
    if {r['actor'] for r in rows}!=set(range(len(doc['actors']))): raise ValueError('unexpected actor')
    if doc['kind']!='isolated':
        for target in doc['targets']:
            window=[r for r in rows if r['actor']==target]
            for background in range(len(doc['actors'])):
                if background==target: continue
                b=summaries[str(background)]
                # Actors enter the released loop at slightly different instants.
                # Require actual background execution during every target's measurement window.
                if b['active_start_ms']>window[0]['end_ms'] or b['active_end_ms']<window[-1]['start_ms']:
                    raise ValueError('background did not cover target window')
    return {'actors':summaries,'thread_ledger':ledger,'correct':True}

def validate_final_hashes(rows,bundle,configuration):
    c=api(bundle); import numpy as np
    programs=c.read(Path(bundle)/'plans/programs.json')
    frontiers=c.read(Path(bundle)/'plans/inputs.json')['records'][0]['frontiers']
    expected=[]
    for node in programs[configuration['stages'][-1]['program']]['output_node_ids']:
        raw=np.load(Path(bundle)/frontiers[str(node)]['npy'],allow_pickle=False).tobytes()
        value=1469598103934665603
        for byte in raw: value=((value^byte)*1099511628211)&((1<<64)-1)
        expected.append(f'{value:016x}')
    for row in rows:
        if [item['fnv1a64'] for item in row.get('raw_outputs',[])]!=expected:
            raise ValueError('complete final output hash mismatch')


def validate_trace(trace,doc,affinity):
    expected={(i,f) for i in doc['targets'] for f in (2,6)}
    records=trace.get('records',[])
    if trace.get('schema')!='vta_run_scope_v1' or {(r['stage_index'],r['frame_id']) for r in records}!=expected or len(records)!=len(expected):
        raise ValueError('probe sample coverage differs')
    for record in records:
        index=record['stage_index']; stage=doc['actors'][index]
        if record['program']!=stage['program']: raise ValueError('probe program differs')
        points=[record[k] for k in ('before_start_ms','before_end_ms','run_start_ms','run_end_ms','after_start_ms','after_end_ms')]
        if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in points) or points!=sorted(points): raise ValueError('probe read envelope differs')
        tids=affinity['stages'][index]['thread_ids']; threads=record['threads']
        if len(threads)!=stage['threads'] or {t['tid'] for t in threads}!=set(tids) or sorted(t['logical_worker'] for t in threads)!=list(range(stage['threads'])): raise ValueError('probe TID ownership differs')
        for t in threads:
            if t['core']!=stage['worker_core_map'][t['logical_worker']]: raise ValueError('probe core differs')
            before,after=t['before'],t['after']
            for source,delta in [('runtime_ns','runtime_delta_ns'),('runqueue_wait_ns','runqueue_wait_delta_ns'),('timeslices','timeslices_delta')]:
                if before['available'] and after['available']:
                    a,b=before[source],after[source]
                    if type(a)!=int or type(b)!=int or a<0 or b<a or t[delta]!=b-a: raise ValueError('invalid counter delta')
                elif t[delta] is not None: raise ValueError('missing counter must remain unknown')
    return trace


def validate_summary(rawdir,doc,bundle):
    c=api(bundle); raw=Path(rawdir); validate_document(doc,bundle)
    rows=[json.loads(line) for line in (raw/'result.jsonl').read_text().splitlines() if line]
    affinity=c.read(raw/'affinity.json')
    if doc['kind'] not in ('pipeline','bridge'):
        result=validate_micro(rows,doc,affinity)
        if doc['probe']:
            trace=c.read(raw/'run_threads.json')
            result['probe']=validate_trace(trace,doc,affinity)
        elif (raw/'run_threads.json').exists(): raise ValueError('formal component has probe data')
        return result
    from cpu_vta_partitioning.measurement import frames, input_sequence, correctness
    from cpu_strategy.measurement import whole_pipeline_window
    base=pipeline_document(doc,bundle)
    frames(raw/'result.jsonl',28); input_sequence(rows,base)
    validate_affinity(affinity,doc['configuration'],True)
    if doc['kind']=='pipeline':
        first_frame={**base,'warmup':0,'scored':1,'tail':0}
        correctness([rows[0]],first_frame,bundle,raw)
    validate_final_hashes(rows,bundle,doc['configuration'])
    if any(r.get('queue_timing_enabled') is not False for r in rows): raise ValueError('timing probe enabled')
    window=whole_pipeline_window(rows,8,16,4)
    return {**window,'fps':1000/window['ii_ms'],'correct':True,
            'stage_service_ms':[statistics.mean(r[f'stage{i}_ms'] for r in rows[8:24])
                                for i in range(len(doc['configuration']['stages']))]}

@lru_cache(maxsize=4)
def _ssh_class(bundle):
    load_legacy(bundle)
    from cpu_strategy.policy.transport import SSH as Base
    class SSHReplay(Base):
        def measure(self,doc,bundle,remote_bundle,run_dir,boot):
            import time
            from cpu_vta_partitioning.transport import TransportError
            from cpu_vta_partitioning.transport.scripts import session_script
            c=api(bundle); root=Path(run_dir)/'raw'/boot/doc['task_id']; root.mkdir(parents=True,exist_ok=True)
            remote=self.job_path(doc,run_dir,boot); q=shlex.quote; expected=c.digest(doc)
            c.write(root/'task.json',doc,replace=False) if not (root/'task.json').exists() else None
            if c.read(root/'task.json')!=doc or self.boot_id()!=boot: raise ValueError('task or boot changed')
            argv,inputs=native_command(doc,bundle,remote_bundle,remote)
            (root/'inputs.txt').write_text(inputs)
            script=session_script(self.config['remote_base'],remote,boot,expected,argv)
            (root/'execute.sh').write_text(script)
            self.command('mkdir -p '+q(remote))
            existing=self.command('test -f '+q(remote+'/task_hash')+' && cat '+q(remote+'/task_hash')+' || true').strip()
            if existing and existing!=expected: raise ValueError('remote task differs')
            try:
                if not existing:
                    self.upload(root/'inputs.txt',remote+'/inputs.txt')
                    if doc['kind'] not in ('pipeline','bridge'): self.upload(root/'inputs.txt',remote+'/contention_spec.json')
                    self.upload(root/'execute.sh',remote+'/execute.sh')
                    self.command('cd '+q(remote)+' && nohup sh execute.sh </dev/null > launcher.log 2>&1 &')
                deadline=time.monotonic()+180
                while True:
                    code=self.command('test -f '+q(remote+'/exit_code')+' && cat '+q(remote+'/exit_code')+' || true').strip()
                    if code: break
                    if time.monotonic()>deadline: raise TransportError('replay deadline exceeded')
                    time.sleep(.5)
            except TransportError as error: raise TransportError(str(error)+'; '+str(self.failure_snapshot(remote,root)))
            output=Path(self.collect(remote,root))
            if code!='0' or self.boot_id()!=boot: raise ValueError('native replay failed: '+str(output))
            if (output/'task_hash').read_text().strip()!=expected or (output/'boot_id').read_text().strip()!=boot: raise ValueError('receipt identity differs')
            if (output/'inputs.txt').read_text()!=inputs: raise ValueError('input identity differs')
            return output
    return SSHReplay

def SSH(config,*,execute_board=False,bundle=None):
    return _ssh_class(str(bundle or DEFAULT_PARENT))(config,execute_board=execute_board)
