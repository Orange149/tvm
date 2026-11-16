from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import json
import math
import shlex
import statistics
from .common import api, protocol
from cpu_policies.core_model.affinity import validate_affinity

def cpu_task(bundle,group):
    c=api(bundle);plan=c.read(Path(bundle)/'plans/cpu_groups.json')['groups'][group]
    executor=c.read(Path(bundle)/'plans/executors.json')['frame_stream']
    doc={'kind':'cpu_frame_stream','group':group,'actors':deepcopy(plan['actors']),
         'control_mask':plan['control_mask'],'gate':plan['gate'],'core_partition':plan['core_partition'],
         'protocol':protocol(),'executor_path':executor['path'],'executor_sha256':executor['sha256'],
         'bundle_id':c.read(Path(bundle)/'manifest.json')['bundle_id']}
    doc['task_id']=c.digest(doc)[:24]
    return doc

def pipeline_task(bundle,name,qualification=False):
    from cpu_policies.open_resources.runtime import make_task
    c=api(bundle);cfg=c.read(Path(bundle)/'plans/static_pool.json')[name]
    return make_task('qualify_pipeline' if qualification else 'performance',cfg,bundle,
                     warmup=0 if qualification else 8,scored=1 if qualification else 16,
                     tail=0 if qualification else 4,indices=[0],
                     campaign=protocol()['experiment'],candidate_id=name)

def validate_document(doc,bundle):
    if doc['kind']=='cpu_frame_stream':expected=cpu_task(bundle,doc['group'])
    else:expected=pipeline_task(bundle,doc['candidate_id'],doc['kind']=='qualify_pipeline')
    if expected!=doc:raise ValueError('task, executor or candidate identity differs')

def native_command(doc,bundle,remote,job):
    validate_document(doc,bundle)
    if doc['kind']!='cpu_frame_stream':
        from cpu_policies.open_resources.runtime import native_command as base
        return base(doc,bundle,remote,job)
    c=api(bundle);programs=c.read(Path(bundle)/'plans/programs.json')
    frontier=c.read(Path(bundle)/'plans/inputs.json')['records'][0]['frontiers'];actors=[]
    for row in doc['actors']:
        p=programs[row['program']]
        paths=lambda ns:[remote+'/'+frontier[str(n)]['raw'] for n in ns]
        actors.append({'inputs':paths(p['input_node_ids']),'references':paths(p['output_node_ids'])})
    spec={'mode':'bounded_common_frame_stream','actors':actors,'frames':14,'queue_depth':2}
    e=protocol()['environment']
    env={'PATH':'/usr/sbin:/usr/bin:/sbin:/bin','LD_LIBRARY_PATH':remote+'/runtime',
         'LD_PRELOAD':remote+'/runtime/libvta.so','TVM_NUM_THREADS':'4','TVM_BIND_THREADS':'1',
         'TVM_EXCLUDE_WORKER0':'1','TVM_THREAD_POOL_SPIN_COUNT':'0','VTA_REPLAY_POLICY':'disabled',
         'VTA_QUEUE_DIAGNOSTICS':'0','C3_COMPLETION_TRACE':'0',
         'AXU5EVB_DRIVER_POLL_SLEEP_NS':str(e['poll_sleep_ns']),
         'AXU5EVB_DRIVER_POST_START_SLEEP_NS':str(e['post_start_sleep_ns'])}
    argv=['env','-i',*[k+'='+v for k,v in env.items()],remote+'/runtime/ld-linux-aarch64.so.1',
          '--library-path',remote+'/runtime',remote+'/'+doc['executor_path'],
          '--cpu-frame-stream-spec',job+'/frame_stream_spec.json','--input-files',','.join(actors[0]['inputs']),
          '--runs','14','--runtime-num-threads','1','--queue-depth','2','--cpu-stage-gate',doc['gate'],
          '--cpu-worker-pinning','--allow-cpu-affinity-superset','--resource-affinity-probe',
          '--control-cpu-affinity',','.join(map(str,doc['control_mask'])),'--output-mode','raw',
          '--output-jsonl',job+'/result.jsonl','--thread-affinity-report',job+'/affinity.json',
          '--process-timeout-s','90']
    for i,row in enumerate(doc['actors']):
        p=programs[row['program']];base=remote+'/programs/'+row['program']
        fields={'name':row['program'],'device':'cpu','graph':base+'/graph.json','lib':base+'/module.so',
                'params':base+'/params.bin','input-names':','.join(p['input_names']),
                'input-sources':','.join('input:'+str(j) for j in range(len(p['input_names']))),
                'runtime-num-threads':str(row['threads']),'cpu-affinity':','.join(map(str,row['mask'])),
                'cpu-worker-map':','.join(map(str,row['worker_core_map']))}
        for field,value in fields.items():argv += [f'--stage{i}-'+field,value]
    return argv,json.dumps(spec,sort_keys=True,separators=(',',':'))+'\n'

def validate_rows(rows,doc,affinity):
    actors=doc['actors'];expected=[(i,f) for i in range(len(actors)) for f in range(14)]
    if [(r.get('actor'),r.get('frame_id')) for r in rows]!=expected:raise ValueError('missing/duplicate/extra CPU frame')
    ledger,idle=validate_affinity(affinity,{'stages':actors,'affinity_mode':'pinned',
                                         'control_mask':doc['control_mask']},True)
    summaries=[]
    for i,stage in enumerate(actors):
        current=rows[i*14:(i+1)*14]
        for r in current:
            if (any(type(r.get(k))!=int for k in ('actor','frame_id','input_index','calls','extra_calls'))
                    or r['input_index']!=0 or r['calls']!=14 or r['extra_calls']!=0
                    or r.get('schema')!='cpu_frame_stream_top20_v1' or r.get('program')!=stage['program']
                    or r.get('gate_mode')!=doc['gate'] or r.get('qualification_passed') is not True
                    or r.get('final_output_passed') is not True):raise ValueError('invalid CPU frame provenance')
            keys=('wait_start_ns','dequeue_ns','gate_request_ns','gate_acquire_ns','start_ns','end_ns',
                  'gate_release_ns','push_start_ns','push_return_ns')
            if any(type(r.get(k))!=int or r[k]<0 for k in keys):raise ValueError('invalid CPU clock')
            times=[r[k] for k in keys[:7]]
            if (times!=sorted(times) or r['end_ns']<=r['start_ns']
                    or r['push_start_ns']>r['push_return_ns'] or r['push_start_ns']>r['dequeue_ns']):
                raise ValueError('noncausal CPU clock')
            if any(type(r.get(k)) not in (int,float) or not math.isfinite(r[k]) or r[k]<0
                   for k in ('set_ms','run_ms','get_ms','service_ms')):raise ValueError('invalid CPU service')
            if abs(r['service_ms']-sum(r[k] for k in ('set_ms','run_ms','get_ms')))>1e-5:
                raise ValueError('CPU service counted twice or missing')
        if any(b['wait_start_ns']<a['gate_release_ns'] for a,b in zip(current,current[1:])):
            raise ValueError('overlapping calls within CPU actor')
        kept=current[2:10]
        summaries.append({'actor':i,'program':stage['program'],'calls':14,'qualification_calls':1,
            'service_ms':statistics.mean(r['service_ms'] for r in kept),
            'run_ms':statistics.mean(r['run_ms'] for r in kept),
            'queue_wait_ms':statistics.mean((r['dequeue_ns']-r['wait_start_ns'])/1e6 for r in kept),
            'gate_wait_ms':statistics.mean((r['gate_acquire_ns']-r['gate_request_ns'])/1e6 for r in kept),
            'producer_push_wall_ms':statistics.mean((r['push_return_ns']-r['push_start_ns'])/1e6 for r in kept)})
    published=sorted(rows,key=lambda r:(r['frame_id'],r['actor']))
    if any(b['push_start_ns']<a['push_return_ns'] for a,b in zip(published,published[1:])):
        raise ValueError('producer ordering differs')
    if doc['gate']=='resource_group':
        for i,a in enumerate(actors):
            for j,b in enumerate(actors[:i]):
                if a['mask']!=b['mask']:continue
                intervals=sorted([(r['gate_acquire_ns'],r['end_ns']) for r in rows if r['actor'] in (i,j)])
                if any(y[0]<x[1] for x,y in zip(intervals,intervals[1:])):raise ValueError('same-group CPU work overlaps')
    groups=[max(r['end_ns'] for r in rows if r['frame_id']==f) for f in range(14)]
    if any(b<=a for a,b in zip(groups,groups[1:])):raise ValueError('nonmonotonic CPU group completion')
    cpu=(groups[9]-groups[2]+6)//7
    return {'cpu_ii_ns':cpu,'cpu_ii_ms':cpu/1e6,'group_completion_ns':groups,'completion_gaps':7,
            'scored_completion_count':8,'stage_calls':len(actors)*15,'actors':summaries,'correct':True,
            'thread_ledger':ledger,'idle_auxiliary_threads':idle}

def validate_summary(raw,doc,bundle):
    validate_document(doc,bundle);raw=Path(raw);c=api(bundle)
    if doc['kind']=='cpu_frame_stream':
        rows=[json.loads(line) for line in (raw/'result.jsonl').read_text().splitlines() if line]
        return validate_rows(rows,doc,c.read(raw/'affinity.json'))
    from cpu_policies.open_resources.runtime import validate_summary as base
    result=base(raw,doc,bundle)
    if doc['kind']=='performance':
        rows=[json.loads(line) for line in (raw/'result.jsonl').read_text().splitlines() if line]
        if any(r.get('queue_timing_enabled') is not False or r.get('defer_result_write_enabled') is not True for r in rows):
            raise ValueError('detailed timing probe or immediate logging enabled')
        result['fps']=1000/result['ii_ms']
    return result

@lru_cache(maxsize=4)
def _ssh_class(bundle):
    api(bundle)
    from cpu_strategy.policy.transport import SSH as Base
    class SSHTransport(Base):
        def measure(self,doc,bundle,remote,root,boot):
            import time
            from cpu_vta_partitioning.transport import TransportError
            from cpu_vta_partitioning.transport.scripts import session_script
            validate_document(doc,bundle);c=api(bundle);q=shlex.quote
            raw=Path(root)/'raw'/boot/doc['task_id'];raw.mkdir(parents=True,exist_ok=True)
            job=self.job_path(doc,root,boot);expected=c.digest(doc)
            if (raw/'task.json').exists() and c.read(raw/'task.json')!=doc:raise ValueError('task collision')
            c.write(raw/'task.json',doc,replace=True)
            if self.boot_id()!=boot:raise ValueError('boot changed before task')
            argv,inputs=native_command(doc,bundle,remote,job)
            (raw/'inputs.txt').write_text(inputs)
            (raw/'execute.sh').write_text(session_script(self.config['remote_base'],job,boot,expected,argv))
            self.command('mkdir -p '+q(job))
            existing=self.command('test -f '+q(job+'/task_hash')+' && cat '+q(job+'/task_hash')+' || true').strip()
            if existing and existing!=expected:raise ValueError('remote task hash differs')
            try:
                if not existing:
                    self.check_space(job,256*1024**2)
                    self.upload(raw/'inputs.txt',job+'/inputs.txt')
                    if doc['kind']=='cpu_frame_stream':self.upload(raw/'inputs.txt',job+'/frame_stream_spec.json')
                    self.upload(raw/'execute.sh',job+'/execute.sh')
                    self.command('cd '+q(job)+' && nohup sh execute.sh </dev/null > launcher.log 2>&1 &')
                deadline=time.monotonic()+660
                while True:
                    code=self.command('test -f '+q(job+'/exit_code')+' && cat '+q(job+'/exit_code')+' || true').strip()
                    if code:break
                    if time.monotonic()>deadline:raise TransportError('native task deadline exceeded')
                    time.sleep(.5)
                output=Path(self.collect(job,raw))
            except TransportError as error:raise TransportError(str(error)+'; '+str(self.failure_snapshot(job,raw)))
            if self.boot_id()!=boot or (output/'boot_id').read_text().strip()!=boot or (output/'task_hash').read_text().strip()!=expected:
                raise ValueError('collected boot/task differs')
            if code!='0':raise ValueError('native failure retained: '+str(output))
            if (output/'inputs.txt').read_text()!=inputs:raise ValueError('collected inputs differ')
            return output
    return SSHTransport

def SSH(connection,*,execute_board=False,bundle=None):
    obj=_ssh_class(str(bundle))(connection,execute_board=execute_board)
    obj.options += ['-o','ControlMaster=auto','-o','ControlPersist=60','-o','ControlPath=/tmp/cpu-vta-top20-%C']
    return obj
