from copy import deepcopy
from pathlib import Path
import random
import time
from .common import api,identity,protocol
from .runtime import pipeline_task,validate_document,validate_summary
from cpu_policies.core_model.campaign import _save,_pin,_publish

def tasks(bundle):
    c=api(bundle);names=sorted(c.read(Path(bundle)/'plans/static_pool.json'))
    random.Random(protocol()['seed']).shuffle(names)
    return [d for name in names for d in (pipeline_task(bundle,name,True),pipeline_task(bundle,name))]

def freeze_document(bundle):
    c=api(bundle);bundle=Path(bundle);rankings=c.read(bundle/'plans/selection_rankings.json')
    return {'rankings_sha256':c.sha256(bundle/'plans/selection_rankings.json'),
        'pool_sha256':c.sha256(bundle/'plans/static_pool.json'),
        'lists':{key:[r['id'] for r in rankings[key]] for key in ('pure_static20','diverse20','coverage20')},
        'pipeline_FPS_used':False,'frozen_during_preparation':True}

def create(bundle,root,connection):
    from .prepare import validate_bundle
    bundle,root=Path(bundle).resolve(),Path(root).resolve();m=validate_bundle(bundle);c=api(bundle)
    if root.exists():raise FileExistsError('run exists; use resume')
    root.mkdir(parents=True)
    state={'experiment':protocol()['experiment'],'controller_identity':identity(),'bundle':str(bundle),
        'bundle_id':m['bundle_id'],'bundle_manifest_sha256':c.sha256(bundle/'manifest.json'),
        'connection':deepcopy(connection),'created':c.now(),'status':'CREATED','phase':'CREATED',
        'boot_id':None,'active_boot':None,'completed_boots':[],'frozen_files':{},
        'completed_tasks':0,'expected_tasks':len(tasks(bundle))}
    c.write(root/'protocol.json',protocol());c.write(root/'tasks.json',tasks(bundle))
    c.write(root/'selection_freeze.json',freeze_document(bundle))
    _pin(root,state,'protocol.json','tasks.json','selection_freeze.json')
    return state

def validate_state(root):
    root=Path(root);c=api();state=c.read(root/'state.json');bundle=Path(state['bundle'])
    if state['controller_identity']!=identity():raise ValueError('controller changed; use frozen selection_controller')
    if (c.sha256(bundle/'manifest.json')!=state['bundle_manifest_sha256']
            or c.read(bundle/'manifest.json')['bundle_id']!=state['bundle_id']):raise ValueError('bundle changed')
    c.verify_inventory(root,state['frozen_files'])
    if (c.read(root/'protocol.json')!=protocol() or c.read(root/'tasks.json')!=tasks(bundle)
            or c.read(root/'selection_freeze.json')!=freeze_document(bundle)):
        raise ValueError('tasks/protocol/selection freeze changed')
    if state.get('active_boot') not in (None,state['boot_id']):raise ValueError('active boot differs')
    if state['status']=='COMPLETE' and (state['completed_tasks']!=state['expected_tasks']
            or state['active_boot'] is not None or state['completed_boots']!=[state['boot_id']]):
        raise ValueError('incomplete run marked complete')
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
            print('[AutoTVM Top20] '+doc['kind']+' '+doc['candidate_id'],flush=True)
            c.event(self.root,'session_start',task_kind=doc['kind'],task_id=doc['task_id'],boot_id=boot)
            start=time.monotonic()
            raw=Path(self.transport.measure(doc,self.state['bundle'],self.remote,self.root,boot)).resolve()
            elapsed=time.monotonic()-start
            if not raw.is_relative_to(self.root/'raw'/boot/doc['task_id']):raise ValueError('transport raw escaped')
            summary=validate_summary(raw,doc,self.state['bundle']);c.sync_tree(raw)
            value={'task':doc,'boot_id':boot,'raw':str(raw.relative_to(self.root)),'summary':summary,
                'measurement_and_collection_seconds':elapsed,
                'files':{str(raw.relative_to(self.root)/n):v for n,v in c.inventory(raw).items()}}
            c.write(receipt,value);c.event(self.root,'session_accepted',task_kind=doc['kind'],task_id=doc['task_id'],boot_id=boot)
        _pin(self.root,self.state,str(receipt.relative_to(self.root)))
        if self.transport is not None:self.transport.acknowledge(doc,self.root,boot)
        return summary

def rank(root):
    state=validate_state(root);c=api(state['bundle'])
    return c.read(Path(state['bundle'])/'plans/selection_rankings.json')

def run_current(root,transport):
    from .prepare import validate_bundle
    from cpu_vta_partitioning.transport import TransportError
    root=Path(root).resolve();c=api()
    with c.locked(root):
        state=validate_state(root);validate_bundle(state['bundle'])
        if state['status']=='COMPLETE':
            recorder=Recorder(root,state)
            for doc in tasks(state['bundle']):recorder.saved(doc)
            return state
        if state['status'].startswith('STOPPED'):raise RuntimeError('invalid evidence retained; use a new batch')
        try:
            boot=transport.boot_id()
            if state['boot_id'] not in (None,boot):raise ValueError('one-boot batch cannot mix boot IDs')
            state.update(boot_id=boot,active_boot=boot,status='PREFLIGHT',phase='PREFLIGHT');_save(root,state)
            transport.wait_idle(boot);transport.mount_storage()
            remote=transport.deploy(state['bundle'],root);transport.verify_payload(remote,boot)
            transport.qualify_environment(remote,boot,root,state['bundle'])
            if api().read(root/'selection_freeze.json')!=freeze_document(state['bundle']):raise ValueError('selection not frozen')
            state.update(status='PIPELINE_MEASUREMENT',phase='PIPELINE_MEASUREMENT');_save(root,state)
            recorder=Recorder(root,state,transport,remote)
            for done,doc in enumerate(tasks(state['bundle']),1):
                recorder.saved(doc);state['completed_tasks']=done;_save(root,state)
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
