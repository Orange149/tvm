from pathlib import Path
from copy import deepcopy
import random
from .common import api,protocol,identity
from .domain import cases
from .runtime import task,validate_document,validate_summary
from cpu_policies.core_model.campaign import _save,_pin,_publish

def task_plan(bundle):
    names=sorted(cases());random.Random(protocol()['seed']).shuffle(names)
    return [task(bundle,n) for n in names]

def create(bundle,root,connection):
    from .prepare import validate_bundle
    bundle,root=Path(bundle).resolve(),Path(root).resolve();m=validate_bundle(bundle);c=api(bundle)
    if root.exists():raise FileExistsError('run exists; use resume')
    root.mkdir(parents=True)
    state={'experiment':protocol()['experiment'],'controller_identity':identity(),'bundle':str(bundle),
        'bundle_id':m['bundle_id'],'bundle_manifest_sha256':c.sha256(bundle/'manifest.json'),
        'connection':deepcopy(connection),'created':c.now(),'status':'CREATED','phase':'CREATED',
        'boot_id':None,'active_boot':None,'completed_boots':[],'frozen_files':{},
        'completed_tasks':0,'expected_tasks':8,'expected_stage_calls':300}
    c.write(root/'protocol.json',protocol());c.write(root/'tasks.json',task_plan(bundle))
    _pin(root,state,'protocol.json','tasks.json')
    return state

def validate_state(root):
    root=Path(root);c=api();state=c.read(root/'state.json');bundle=Path(state['bundle'])
    if state['controller_identity']!=identity():raise ValueError('controller changed; use frozen frame_stream_controller')
    if c.sha256(bundle/'manifest.json')!=state['bundle_manifest_sha256']:raise ValueError('bundle manifest changed')
    if c.read(bundle/'manifest.json')['bundle_id']!=state['bundle_id']:raise ValueError('bundle ID changed')
    c.verify_inventory(root,state['frozen_files'])
    if c.read(root/'tasks.json')!=task_plan(bundle) or c.read(root/'protocol.json')!=protocol():raise ValueError('task/protocol changed')
    if state['status']=='COMPLETE' and (state['completed_tasks']!=8 or state['active_boot'] is not None
            or any(n not in state['frozen_files'] for n in ('predictions.json','prediction_freeze.json'))):
        raise ValueError('incomplete campaign marked complete')
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
            c.verify_inventory(self.root,value['files'])
            raw=c.relative_file(self.root,value['raw'])
            if not raw.is_relative_to(self.root/'raw'/boot/doc['task_id']):raise ValueError('receipt path escaped task')
            summary=validate_summary(raw,doc,self.state['bundle'])
            if summary!=value['summary']:raise ValueError('summary differs from raw')
        else:
            if self.transport is None:raise ValueError('missing CPU session: '+doc['case'])
            print('[CPU共同帧流] '+doc['case'],flush=True)
            c.event(self.root,'session_start',case=doc['case'],task_id=doc['task_id'],boot_id=boot)
            raw=Path(self.transport.measure(doc,self.state['bundle'],self.remote,self.root,boot)).resolve()
            if not raw.is_relative_to(self.root/'raw'/boot/doc['task_id']):raise ValueError('transport raw path escaped')
            summary=validate_summary(raw,doc,self.state['bundle']);c.sync_tree(raw)
            value={'task':doc,'boot_id':boot,'raw':str(raw.relative_to(self.root)),'summary':summary,
                'files':{str(raw.relative_to(self.root)/n):v for n,v in c.inventory(raw).items()}}
            c.write(receipt,value);c.event(self.root,'session_accepted',task_id=doc['task_id'],boot_id=boot)
        _pin(self.root,self.state,str(receipt.relative_to(self.root)))
        if self.transport is not None:self.transport.acknowledge(doc,self.root,boot)
        return summary

def validated_records(root,state=None):
    state=state or validate_state(root);recorder=Recorder(root,state)
    return {d['case']:recorder.saved(d) for d in task_plan(state['bundle'])}

def freeze_predictions(root,state=None):
    from .model import rank
    state=state or validate_state(root);c=api(state['bundle'])
    measurements=validated_records(root,state)
    model=c.read(Path(state['bundle'])/'plans/model_inputs.json')
    result=rank(model,measurements)
    _publish(root,state,'predictions.json',result)
    if 'prediction_freeze_at' not in state:
        state['prediction_freeze_at']=c.now();_save(root,state)
    _publish(root,state,'prediction_freeze.json',{'frozen_at':state['prediction_freeze_at'],
        'predictions_sha256':c.sha256(Path(root)/'predictions.json'),'new_sessions':8,
        'analysis_reference_read_by_scoring':False,'development_replay':True})
    return result

def run_current(root,transport):
    from .prepare import validate_bundle
    from cpu_vta_partitioning.transport import TransportError
    root=Path(root).resolve();c=api()
    with c.locked(root):
        state=validate_state(root);validate_bundle(state['bundle'])
        if state['status']=='COMPLETE':validated_records(root,state);return state
        if state['status'].startswith('STOPPED'):raise RuntimeError('invalid evidence retained; cannot silently retry')
        try:
            boot=transport.boot_id()
            if state['boot_id'] not in (None,boot):raise ValueError('one-boot run cannot mix boot IDs')
            state.update(boot_id=boot,active_boot=boot,status='PREFLIGHT',phase='PREFLIGHT');_save(root,state)
            transport.wait_idle(boot);transport.mount_storage()
            remote=transport.deploy(state['bundle'],root);transport.verify_payload(remote,boot)
            transport.qualify_environment(remote,boot,root,state['bundle'])
            recorder=Recorder(root,state,transport,remote)
            state.update(status='MEASURING',phase='MEASURING');_save(root,state)
            for i,doc in enumerate(task_plan(state['bundle']),1):
                recorder.saved(doc);state['completed_tasks']=i;_save(root,state)
            freeze_predictions(root,state)
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
    return {k:state.get(k) for k in ('status','phase','completed_tasks','expected_tasks','expected_stage_calls','boot_id','last_error','analysis')}
