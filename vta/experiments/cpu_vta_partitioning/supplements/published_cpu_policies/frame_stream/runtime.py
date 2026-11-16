"""Explicit CPU-only tasks and an independent common-frame validator."""
from pathlib import Path
from copy import deepcopy
from functools import lru_cache
import json
import math
import shlex
import statistics
from .common import api, protocol, DEFAULT_PARENT
from .domain import cases
from cpu_policies.core_model.affinity import validate_affinity

def task(bundle,name):
    c=api(bundle); spec=c.read(Path(bundle)/'plans/executors.json')['frame_stream']
    cfg=cases()[name]
    value={'kind':'cpu_frame_stream','case':name,'mode':protocol()['mode'],
           'actors':cfg['actors'],'configuration':cfg['configuration'],'protocol':protocol(),
           'bundle_id':c.read(Path(bundle)/'manifest.json')['bundle_id'],
           'executor_path':spec['path'],'executor_sha256':spec['sha256']}
    value['task_id']=c.digest(value)[:24]
    return value

def validate_document(doc,bundle):
    if doc!=task(bundle,doc['case']):raise ValueError('frame-stream task identity differs')

def native_spec(doc,bundle,remote):
    c=api(bundle); programs=c.read(Path(bundle)/'plans/programs.json')
    frontiers=c.read(Path(bundle)/'plans/inputs.json')['records'][0]['frontiers']
    actors=[]
    for stage in doc['actors']:
        p=programs[stage['program']]
        files=lambda nodes:[remote+'/'+frontiers[str(n)]['raw'] for n in nodes]
        actors.append({'inputs':files(p['input_node_ids']),'references':files(p['output_node_ids'])})
    return {'mode':doc['mode'],'actors':actors,'frames':14,'queue_depth':2}

def native_command(doc,bundle,remote,job):
    validate_document(doc,bundle); c=api(bundle)
    programs=c.read(Path(bundle)/'plans/programs.json'); spec=native_spec(doc,bundle,remote)
    e=protocol()['environment']
    env={'PATH':'/usr/sbin:/usr/bin:/sbin:/bin','LD_LIBRARY_PATH':remote+'/runtime',
         'LD_PRELOAD':remote+'/runtime/libvta.so','TVM_NUM_THREADS':'4','TVM_BIND_THREADS':'1',
         'TVM_EXCLUDE_WORKER0':'1','TVM_THREAD_POOL_SPIN_COUNT':'0','VTA_REPLAY_POLICY':'disabled',
         'VTA_QUEUE_DIAGNOSTICS':'0','C3_COMPLETION_TRACE':'0',
         'AXU5EVB_DRIVER_POLL_SLEEP_NS':str(e['poll_sleep_ns']),
         'AXU5EVB_DRIVER_POST_START_SLEEP_NS':str(e['post_start_sleep_ns'])}
    argv=['env','-i',*[k+'='+v for k,v in env.items()],remote+'/runtime/ld-linux-aarch64.so.1',
          '--library-path',remote+'/runtime',remote+'/'+doc['executor_path'],
          '--cpu-frame-stream-spec',job+'/frame_stream_spec.json',
          '--input-files',','.join(spec['actors'][0]['inputs']),'--runs','14',
          '--runtime-num-threads','1','--queue-depth','2','--cpu-stage-gate','off',
          '--cpu-worker-pinning','--allow-cpu-affinity-superset','--resource-affinity-probe',
          '--control-cpu-affinity','0,1,2,3','--output-mode','raw',
          '--output-jsonl',job+'/result.jsonl','--thread-affinity-report',job+'/affinity.json',
          '--process-timeout-s','90']
    for i,s in enumerate(doc['actors']):
        p=programs[s['program']]; base=remote+'/programs/'+s['program']
        for key,value in [('name',s['program']),('device','cpu'),('graph',base+'/graph.json'),
                          ('lib',base+'/module.so'),('params',base+'/params.bin'),
                          ('input-names',','.join(p['input_names'])),
                          ('input-sources',','.join('input:'+str(k) for k in range(len(p['input_names'])))),
                          ('runtime-num-threads',str(s['threads'])),('cpu-affinity',','.join(map(str,s['mask']))),
                          ('cpu-worker-map',','.join(map(str,s['worker_core_map'])))]:
            argv += [f'--stage{i}-'+key,value]
    return argv,json.dumps(spec,sort_keys=True,separators=(',',':'))+'\n'

def validate_rows(rows,doc,affinity):
    actors=doc['actors']; expected=[(i,f) for i in range(len(actors)) for f in range(14)]
    if [(r.get('actor'),r.get('frame_id')) for r in rows]!=expected:raise ValueError('missing/duplicate/unexpected CPU frame')
    cfg={'stages':actors,'affinity_mode':'pinned','control_mask':[0,1,2,3]}
    ledger,idle=validate_affinity(affinity,cfg,True)
    groups=[]; summaries=[]
    for i,stage in enumerate(actors):
        current=[r for r in rows if r['actor']==i]
        for row in current:
            if any(type(row.get(k))!=int for k in ('actor','frame_id','extra_calls')):raise ValueError('noninteger frame identity')
            if (row.get('schema')!='cpu_frame_stream_v1' or row.get('program')!=stage['program']
                    or row.get('qualification_passed') is not True or row.get('final_output_passed') is not True
                    or type(row.get('input_index'))!=int or row['input_index']!=0
                    or type(row.get('calls'))!=int or row['calls']!=14 or row.get('extra_calls')!=0):
                raise ValueError('invalid frame provenance/output/call count')
            for key in ('wait_start_ns','dequeue_ns','start_ns','end_ns','push_start_ns','push_return_ns'):
                if type(row.get(key))!=int or row[key]<0:raise ValueError('invalid nanosecond timestamp')
            if not (row['wait_start_ns']<=row['dequeue_ns']<=row['start_ns']<row['end_ns']
                    and row['push_start_ns']<=row['push_return_ns'] and row['push_start_ns']<=row['dequeue_ns']):
                raise ValueError('noncausal queue timestamps')
            for key in ('set_ms','run_ms','get_ms','service_ms'):
                if type(row.get(key)) not in (int,float) or not math.isfinite(row[key]) or row[key]<0:raise ValueError('invalid service time')
            if abs(row['service_ms']-sum(row[k] for k in ('set_ms','run_ms','get_ms')))>1e-5:raise ValueError('stage service conservation differs')
        if any(b['wait_start_ns']<a['end_ns'] for a,b in zip(current,current[1:])):raise ValueError('CPU actor calls overlap')
        kept=current[2:10]
        summaries.append({'actor':i,'program':stage['program'],'calls':14,'qualification_calls':1,
            'service_ms':statistics.mean(r['service_ms'] for r in kept),
            'run_ms':statistics.mean(r['run_ms'] for r in kept),
            'queue_wait_ms':statistics.mean((r['dequeue_ns']-r['wait_start_ns'])/1e6 for r in kept),
            'producer_push_wall_ms':statistics.mean((r['push_return_ns']-r['push_start_ns'])/1e6 for r in kept)})
    # Publishing is frame-major and actor-major, but Push return may follow a fast consumer.
    published=sorted(rows,key=lambda r:(r['frame_id'],r['actor']))
    if any(b['push_start_ns']<a['push_return_ns'] for a,b in zip(published,published[1:])):raise ValueError('producer dispatch order differs')
    for f in range(14):groups.append(max(r['end_ns'] for r in rows if r['frame_id']==f))
    if any(b<=a for a,b in zip(groups,groups[1:])):raise ValueError('nonmonotonic group completion')
    total=groups[9]-groups[2]; cpu_ns=(total+6)//7
    return {'cpu_ii_ns':cpu_ns,'cpu_ii_ms':cpu_ns/1e6,'group_completion_ns':groups,
            'scored_completion_count':8,'completion_gaps':7,'actors':summaries,
            'stage_calls':len(actors)*15,'correct':True,'thread_ledger':ledger,
            'idle_auxiliary_threads':idle,'CPU_time_scope':'not_used_as_concurrent_stage_cost'}

def validate_summary(raw,doc,bundle):
    validate_document(doc,bundle);c=api(bundle);raw=Path(raw)
    rows=[json.loads(x) for x in (raw/'result.jsonl').read_text().splitlines() if x]
    return validate_rows(rows,doc,c.read(raw/'affinity.json'))

@lru_cache(maxsize=4)
def _ssh_class(bundle):
    api(bundle)
    from cpu_strategy.policy.transport import SSH as Base
    class FrameSSH(Base):
        def measure(self,doc,bundle,remote,root,boot):
            import time
            from cpu_vta_partitioning.transport import TransportError
            from cpu_vta_partitioning.transport.scripts import session_script
            c=api(bundle); raw=Path(root)/'raw'/boot/doc['task_id'];raw.mkdir(parents=True,exist_ok=True)
            job=self.job_path(doc,root,boot); expected=c.digest(doc); q=shlex.quote
            if (raw/'task.json').exists():
                if c.read(raw/'task.json')!=doc:raise ValueError('task identity collision')
            else:c.write(raw/'task.json',doc)
            if self.boot_id()!=boot:raise ValueError('boot changed before frame stream')
            argv,inputs=native_command(doc,bundle,remote,job)
            (raw/'inputs.txt').write_text(inputs)
            (raw/'execute.sh').write_text(session_script(self.config['remote_base'],job,boot,expected,argv))
            self.command('mkdir -p '+q(job))
            existing=self.command('test -f '+q(job+'/task_hash')+' && cat '+q(job+'/task_hash')+' || true').strip()
            if existing and existing!=expected:raise ValueError('remote task hash differs')
            try:
                if not existing:
                    self.check_space(job,32*1024**2)
                    self.upload(raw/'inputs.txt',job+'/inputs.txt');self.upload(raw/'inputs.txt',job+'/frame_stream_spec.json')
                    self.upload(raw/'execute.sh',job+'/execute.sh')
                    self.command('cd '+q(job)+' && nohup sh execute.sh </dev/null > launcher.log 2>&1 &')
                deadline=time.monotonic()+180
                while True:
                    code=self.command('test -f '+q(job+'/exit_code')+' && cat '+q(job+'/exit_code')+' || true').strip()
                    if code:break
                    if time.monotonic()>deadline:raise TransportError('frame-stream deadline exceeded')
                    time.sleep(.5)
            except TransportError as error:raise TransportError(str(error)+'; '+str(self.failure_snapshot(job,raw)))
            result=Path(self.collect(job,raw))
            if code!='0':raise ValueError('native frame stream failed: '+str(result))
            if self.boot_id()!=boot or (result/'boot_id').read_text().strip()!=boot or (result/'task_hash').read_text().strip()!=expected:raise ValueError('collected task/boot differs')
            if (result/'inputs.txt').read_text()!=inputs:raise ValueError('collected inputs differ')
            return result
    return FrameSSH

def SSH(connection,*,execute_board=False,bundle=None):
    return _ssh_class(str(bundle or DEFAULT_PARENT))(connection,execute_board=execute_board)
