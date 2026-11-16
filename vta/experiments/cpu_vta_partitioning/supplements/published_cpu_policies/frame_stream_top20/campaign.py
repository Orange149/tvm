from pathlib import Path
from copy import deepcopy
import random
from .common import api, protocol, identity
from .runtime import cpu_task, pipeline_task, validate_document, validate_summary
from cpu_policies.core_model.campaign import _save, _pin, _publish

def tasks(bundle):
    c=api(bundle);bundle=Path(bundle)
    names=sorted(c.read(bundle/'plans/cpu_groups.json')['groups']);random.Random(protocol()['seed']).shuffle(names)
    cpu=[cpu_task(bundle,name) for name in names]
    names=sorted(c.read(bundle/'plans/static_pool.json'));random.Random(protocol()['seed']+1).shuffle(names)
    pipeline=[doc for name in names for doc in (pipeline_task(bundle,name,True),pipeline_task(bundle,name))]
    return {'cpu':cpu,'pipeline':pipeline}

def create(bundle,root,connection):
    from .prepare import validate_bundle
    bundle,root=Path(bundle).resolve(),Path(root).resolve();m=validate_bundle(bundle);c=api(bundle)
    if root.exists():raise FileExistsError('run exists; use resume')
    plan=tasks(bundle);root.mkdir(parents=True)
    state={'experiment':protocol()['experiment'],'controller_identity':identity(),'bundle':str(bundle),
        'bundle_id':m['bundle_id'],'bundle_manifest_sha256':c.sha256(bundle/'manifest.json'),
        'connection':deepcopy(connection),'created':c.now(),'status':'CREATED','phase':'CREATED',
        'boot_id':None,'active_boot':None,'completed_boots':[],'frozen_files':{},
        'completed_tasks':0,'expected_tasks':sum(map(len,plan.values()))}
    c.write(root/'protocol.json',protocol());c.write(root/'tasks.json',plan)
    _pin(root,state,'protocol.json','tasks.json')
    return state

def validate_state(root):
    root=Path(root);c=api();state=c.read(root/'state.json');bundle=Path(state['bundle'])
    if state['controller_identity']!=identity():raise ValueError('controller changed; use frozen top20_controller')
    if c.sha256(bundle/'manifest.json')!=state['bundle_manifest_sha256']:raise ValueError('bundle manifest changed')
    if c.read(bundle/'manifest.json')['bundle_id']!=state['bundle_id']:raise ValueError('bundle changed')
    c.verify_inventory(root,state['frozen_files'])
    if c.read(root/'tasks.json')!=tasks(bundle) or c.read(root/'protocol.json')!=protocol():raise ValueError('tasks/protocol changed')
    if state['status']=='COMPLETE' and (state['completed_tasks']!=state['expected_tasks'] or state['active_boot'] is not None
            or 'prediction_freeze.json' not in state['frozen_files']):raise ValueError('incomplete run marked complete')
    return state

class Recorder:
    def __init__(self,root,state,transport=None,remote=None):
        self.root,self.state=Path(root).resolve(),state;self.transport,self.remote=transport,remote

    def saved(self,doc):
        c=api(self.state['bundle']);boot=self.state['boot_id'];validate_document(doc,self.state['bundle'])
        receipt=self.root/'accepted'/boot/(doc['task_id']+'.json')
        if receipt.exists():
            value=c.read(receipt)
            if value['task']!=doc or value['boot_id']!=boot:raise ValueError('receipt task/boot differs')
            c.verify_inventory(self.root,value['files']);raw=c.relative_file(self.root,value['raw'])
            if not raw.is_relative_to(self.root/'raw'/boot/doc['task_id']):raise ValueError('raw receipt escaped task')
            summary=validate_summary(raw,doc,self.state['bundle'])
            if value['summary']!=summary:raise ValueError('raw-backed summary changed')
        else:
            if self.transport is None:raise ValueError('missing accepted task '+doc['task_id'])
            print('[DP Top20] '+doc['kind']+' '+doc.get('group',doc.get('candidate_id','')),flush=True)
            c.event(self.root,'session_start',task_kind=doc['kind'],task_id=doc['task_id'],boot_id=boot)
            raw=Path(self.transport.measure(doc,self.state['bundle'],self.remote,self.root,boot)).resolve()
            if not raw.is_relative_to(self.root/'raw'/boot/doc['task_id']):raise ValueError('transport raw path escaped')
            summary=validate_summary(raw,doc,self.state['bundle']);c.sync_tree(raw)
            value={'task':doc,'boot_id':boot,'raw':str(raw.relative_to(self.root)),'summary':summary,
                   'files':{str(raw.relative_to(self.root)/n):v for n,v in c.inventory(raw).items()}}
            c.write(receipt,value);c.event(self.root,'session_accepted',task_kind=doc['kind'],task_id=doc['task_id'],boot_id=boot)
        _pin(self.root,self.state,str(receipt.relative_to(self.root)))
        if self.transport is not None:self.transport.acknowledge(doc,self.root,boot)
        return summary

def cpu_records(root,state=None):
    state=state or validate_state(root);recorder=Recorder(root,state)
    return {d['group']:recorder.saved(d) for d in tasks(state['bundle'])['cpu']}

def static_rank(root):
    state=validate_state(root);c=api(state['bundle'])
    return c.read(Path(state['bundle'])/'plans/static_rankings.json')

def freeze_predictions(root,state=None):
    from .model import rank
    state=state or validate_state(root);c=api(state['bundle']);bundle=Path(state['bundle'])
    result=rank(c.read(bundle/'plans/static_pool.json'),c.read(bundle/'plans/cpu_groups.json'),cpu_records(root,state))
    _publish(root,state,'predictions.json',result)
    if 'prediction_freeze_at' not in state:
        state['prediction_freeze_at']=c.now();_save(root,state)
    _publish(root,state,'prediction_freeze.json',{'frozen_at':state['prediction_freeze_at'],
        'predictions_sha256':c.sha256(Path(root)/'predictions.json'),'pipeline_FPS_used':False,
        'pool_sha256':c.sha256(bundle/'plans/static_pool.json'),'top20_ids':[r['id'] for r in result['top20']]})
    return result

def run_current(root,transport):
    from .prepare import validate_bundle
    from cpu_vta_partitioning.transport import TransportError
    root=Path(root).resolve();c=api()
    with c.locked(root):
        state=validate_state(root);validate_bundle(state['bundle'])
        if state['status']=='COMPLETE':
            recorder=Recorder(root,state)
            for doc in sum(tasks(state['bundle']).values(),[]):recorder.saved(doc)
            return state
        if state['status'].startswith('STOPPED'):raise RuntimeError('invalid evidence retained; use a new batch after correction')
        try:
            boot=transport.boot_id()
            if state['boot_id'] not in (None,boot):raise ValueError('one-boot batch cannot mix boot IDs')
            state.update(boot_id=boot,active_boot=boot,status='PREFLIGHT',phase='PREFLIGHT');_save(root,state)
            transport.wait_idle(boot);transport.mount_storage()
            remote=transport.deploy(state['bundle'],root);transport.verify_payload(remote,boot)
            transport.qualify_environment(remote,boot,root,state['bundle'])
            plan=tasks(state['bundle']);recorder=Recorder(root,state,transport,remote);done=0
            state.update(status='CPU_PROFILING',phase='CPU_PROFILING');_save(root,state)
            for doc in plan['cpu']:
                recorder.saved(doc);done+=1;state['completed_tasks']=done;_save(root,state)
            freeze_predictions(root,state)
            state.update(status='PIPELINE_MEASUREMENT',phase='PIPELINE_MEASUREMENT');_save(root,state)
            for doc in plan['pipeline']:
                recorder.saved(doc);done+=1;state['completed_tasks']=done;_save(root,state)
            state.update(status='ANALYZING',phase='ANALYZING');_save(root,state)
            from .analysis import analyze
            state['analysis']=analyze(root)
            transport.verify_payload(remote,boot);transport.idle_command('sync',boot=boot,timeout=60)
            state.update(status='COMPLETE',phase='COMPLETE',active_boot=None,completed_boots=[boot])
            state.pop('last_error',None);_save(root,state);c.sync_tree(root)
            return state
        except TransportError as error:
            state.update(status='INTERRUPTED',last_error=str(error));_save(root,state);raise
        except Exception as error:
            state.update(status='STOPPED_INVALID_RESULT',last_error=str(error));_save(root,state);raise

def status(root):
    state=validate_state(root)
    return {k:state.get(k) for k in ('status','phase','completed_tasks','expected_tasks','boot_id','last_error','analysis')}
