"""One boot, durable receipts, atomic predictions freeze, no automatic retries."""
from pathlib import Path
from copy import deepcopy
import random
from .common import api, identity, protocol
from .domain import configurations, components, groups, diagnostics
from .runtime import task, validate_document, validate_summary
from ..core_model.campaign import _save, _pin, _publish

def profile_plan(bundle):
    rows=[]
    for name,cfg in components().items():
        rows.append(task(bundle,'isolated','isolated/'+name,actors=cfg['stages'],targets=[0]))
    for name,g in groups().items():
        rows.append(task(bundle,'concurrent','concurrent/'+name,actors=g['actors'],targets=g['targets']))
    assert len(rows)==35
    return rows

def final_plan(bundle):
    rows=[]
    for name,g in diagnostics().items():
        rows.append(task(bundle,'diagnostic','diagnostic/'+name,actors=g['actors'],targets=g['targets'],probe=True))
    cases=configurations(); names=sorted(cases); random.Random(protocol()['seed']).shuffle(names)
    for name in names: rows.append(task(bundle,'pipeline','pipeline/'+name,configuration=cases[name]))
    rows.append(task(bundle,'bridge','bridge/original',configuration=cases['explanation/overlap4'],executor='policy'))
    assert len(rows)==13
    return rows

def validate_state(root):
    root=Path(root); c=api(); state=c.read(root/'state.json')
    if state['controller_identity']!=identity(): raise ValueError('controller identity changed; use frozen controller')
    bundle=Path(state['bundle']); m=c.read(bundle/'manifest.json')
    if m['bundle_id']!=state['bundle_id'] or c.sha256(bundle/'manifest.json')!=state['bundle_manifest_sha256']: raise ValueError('bundle identity differs')
    c.verify_inventory(root,state['frozen_files'])
    if c.read(root/'protocol.json')!=protocol() or c.read(root/'profile_tasks.json')!=profile_plan(bundle): raise ValueError('protocol/tasks differ')
    if state['status']=='COMPLETE' and (state['completed_tasks']!=48 or state['active_boot'] is not None): raise ValueError('incomplete run marked complete')
    return state

def create(bundle,root,connection):
    from .prepare import validate_bundle
    bundle,root=Path(bundle).resolve(),Path(root).resolve(); m=validate_bundle(bundle); c=api(bundle)
    if root.exists(): raise FileExistsError('run exists; use resume')
    root.mkdir(parents=True)
    state={'experiment':protocol()['experiment'],'controller_identity':identity(),'bundle':str(bundle),'bundle_id':m['bundle_id'],
           'bundle_manifest_sha256':c.sha256(bundle/'manifest.json'),'connection':deepcopy(connection),'created':c.now(),
           'status':'CREATED','phase':'CREATED','boot_id':None,'active_boot':None,'completed_boots':[],
           'completed_tasks':0,'expected_tasks':48,'frozen_files':{}}
    c.write(root/'protocol.json',protocol()); c.write(root/'profile_tasks.json',profile_plan(bundle))
    _pin(root,state,'protocol.json','profile_tasks.json')
    return state

class Recorder:
    def __init__(self,root,state,transport=None,remote=None):
        self.root,self.state=Path(root),state
        self.transport,self.remote=transport,remote

    def saved(self,doc):
        c=api(self.state['bundle']); boot=self.state['boot_id']; validate_document(doc,self.state['bundle'])
        receipt=self.root/'accepted'/boot/(doc['task_id']+'.json')
        if receipt.exists():
            item=c.read(receipt)
            if item['task']!=doc or item['boot_id']!=boot: raise ValueError('receipt task or boot differs')
            c.verify_inventory(self.root,item['files'])
            raw=c.relative_file(self.root,item['raw'])
            if not raw.is_relative_to(self.root/'raw'/boot/doc['task_id']): raise ValueError('receipt raw escaped task')
            result=validate_summary(raw,doc,self.state['bundle'])
            if item['summary']!=result: raise ValueError('receipt summary differs from raw')
        else:
            if self.transport is None: raise ValueError('task not yet completed: '+doc['name'])
            print('[受控并发] '+doc['name'],flush=True)
            c.event(self.root,'session_start',task_id=doc['task_id'],name=doc['name'],boot_id=boot)
            raw=Path(self.transport.measure(doc,self.state['bundle'],self.remote,self.root,boot)).resolve()
            if not raw.is_relative_to(self.root/'raw'/boot/doc['task_id']): raise ValueError('transport raw path escaped')
            result=validate_summary(raw,doc,self.state['bundle']); c.sync_tree(raw)
            item={'task':doc,'boot_id':boot,'raw':str(raw.relative_to(self.root)),'summary':result,
                  'files':{str(raw.relative_to(self.root)/n):v for n,v in c.inventory(raw).items()}}
            c.write(receipt,item)
            c.event(self.root,'session_accepted',task_id=doc['task_id'],name=doc['name'],boot_id=boot)
        _pin(self.root,self.state,str(receipt.relative_to(self.root)))
        if self.transport is not None:
            # Acknowledgement is after raw fsync + durable receipt, and can be repeated on resume.
            self.transport.acknowledge(doc,self.root,boot)
        return result

def freeze_predictions(root,state):
    from .model import predictions
    c=api(state['bundle']); recorder=Recorder(root,state); records={}; concurrent={}
    for doc in profile_plan(state['bundle']):
        summary=recorder.saved(doc)
        if doc['kind']=='isolated':
            cost={k:v for k,v in summary['actors']['0'].items() if k in ('service_ms','run_core_ms','boundary_core_ms','set_core_ms','get_core_ms','set_ms','run_ms','get_ms')}
            key=doc['name'].split('/',1)[1]
            records[key]={'cost':cost,'task_id':doc['task_id'],'boot_id':state['boot_id']}
        else: concurrent[doc['name'].split('/',1)[1]]=summary
    _publish(root,state,'costs.json',{'costs':records,'pipeline_feedback_used':False})
    bundle=Path(state['bundle'])
    result=predictions(records,concurrent,c.read(bundle/'plans/baseline_ddr.json'),c.read(bundle/'plans/programs.json'))
    _publish(root,state,'predictions.json',result)
    _publish(root,state,'final_tasks.json',final_plan(bundle))
    if 'prediction_freeze.json' not in state['frozen_files']:
        if 'prediction_freeze_at' not in state:
            state['prediction_freeze_at']=c.now(); _save(root,state)
        _publish(root,state,'prediction_freeze.json',{'frozen_at':state['prediction_freeze_at'],'completed_profile_sessions':35,
                 'predictions_sha256':c.sha256(Path(root)/'predictions.json'),'full_pipeline_sessions_started':0})
    return result

def validated_records(root,state=None):
    state=state or validate_state(root); recorder=Recorder(root,state)
    plan=profile_plan(state['bundle'])+(final_plan(state['bundle']) if (Path(root)/'final_tasks.json').exists() else [])
    return {doc['name']:recorder.saved(doc) for doc in plan}

def run_current(root,transport):
    from .prepare import validate_bundle
    from cpu_vta_partitioning.transport import TransportError
    root=Path(root).resolve(); c=api()
    with c.locked(root):
        state=validate_state(root); validate_bundle(state['bundle'])
        if state['status']=='COMPLETE': validated_records(root,state); return state
        if state['status'].startswith('STOPPED'): raise RuntimeError('invalid execution retained; do not silently retry')
        try:
            boot=transport.boot_id()
            if state['boot_id'] not in (None,boot): raise ValueError('one-boot campaign cannot mix boot IDs')
            state.update(boot_id=boot,active_boot=boot,status='PREFLIGHT',phase='PREFLIGHT'); _save(root,state)
            transport.wait_idle(boot); transport.mount_storage()
            remote=transport.deploy(state['bundle'],root); transport.verify_payload(remote,boot)
            transport.qualify_environment(remote,boot,root,state['bundle'])
            recorder=Recorder(root,state,transport,remote)
            state.update(status='PROFILING',phase='PROFILING'); _save(root,state)
            for i,doc in enumerate(profile_plan(state['bundle']),1):
                recorder.saved(doc); state['completed_tasks']=i; _save(root,state)
            freeze_predictions(root,state)
            state.update(status='VALIDATING',phase='VALIDATING'); _save(root,state)
            for i,doc in enumerate(final_plan(state['bundle']),36):
                recorder.saved(doc); state['completed_tasks']=i; _save(root,state)
            validated_records(root,state)
            if transport.boot_id()!=boot: raise ValueError('boot changed before final acceptance')
            transport.verify_payload(remote,boot); transport.idle_command('sync',boot=boot,timeout=60)
            from .analysis import analyze
            state['analysis']=analyze(root)
            state.update(status='COMPLETE',phase='COMPLETE',active_boot=None,completed_boots=[boot])
            state.pop('last_error',None); _save(root,state); c.sync_tree(root)
            return state
        except TransportError as error:
            state.update(status='INTERRUPTED',last_error=str(error)); _save(root,state); raise
        except Exception as error:
            state.update(status='STOPPED_INVALID_RESULT',last_error=str(error)); _save(root,state); raise

def status(root):
    state=validate_state(root)
    return {k:state.get(k) for k in ('status','phase','boot_id','completed_tasks','expected_tasks','last_error','analysis')}
