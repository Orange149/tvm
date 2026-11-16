"""Bounded no-reboot collection with exact acceptance and resume identities."""
from pathlib import Path
import shlex
import time

from cpu_vta_partitioning.common import digest,inventory,locked,now,read,sha256,sync_tree,verify_inventory,write
from .prepare import validate,protocol,configurations
from .runtime import make_task,matched_ssh,summary


def specs(output,scope):
    output=Path(output)
    if scope=='components':return read(output/'COMPONENT_TASKS.json')
    freeze=output/'PIPELINE_PREDICTIONS_FREEZE.json'
    if not freeze.exists():raise ValueError('component-derived pipeline predictions must be frozen before pipeline tasks')
    document=read(freeze)
    for path,expected in document['files'].items():
        if sha256(output/path)!=expected:raise ValueError('frozen model prediction file changed')
    result=[]
    for network in ('resnet18','yolov3_tiny416'):
        for case in configurations(network)['pipelines']:
            for phase in ('qualify','clean','observed'):
                result.append(dict(network=network,scope='pipeline',case=case,phase=phase,
                    probe=phase!='clean',runner='phase'))
        result.append(dict(network=network,scope='pipeline',case='overlap',phase='clean',probe=False,runner='old'))
    return result


def accept(ssh,task,bundle,remote,run,boot,expected_outputs=None):
    run=Path(run);receipt=run/'accepted'/boot/(task['task_id']+'.json')
    if receipt.exists():
        item=read(receipt)
        if item['task']!=task or item['boot_id']!=boot:raise ValueError('accepted task identity changed')
        verify_inventory(item['raw'],item['files']);calculated=summary(item['raw'],task,bundle)
        if calculated!=item['summary']:raise ValueError('accepted phase evidence does not replay')
    else:
        started=time.monotonic();started_at=now();raw=ssh.measure(task,bundle,remote,run,boot)
        calculated=summary(raw,task,bundle)
        if expected_outputs is not None and calculated['output_fingerprints']!=expected_outputs:
            raise ValueError('profile outputs differ from qualified full outputs')
        sync_tree(raw)
        item=dict(task=task,boot_id=boot,raw=str(raw),files=inventory(raw),summary=calculated,
            measurement_started_at=started_at,accepted_at=now(),measurement_and_collection_seconds=time.monotonic()-started)
        write(receipt,item)
        ssh.acknowledge(task,run,boot)
        ssh.idle_command('cd '+shlex.quote(ssh.job_path(task,run,boot))+' && rm -f phase_trace.jsonl wait_trace.jsonl wait_status.json',boot=boot)
    raw=Path(item['raw'])
    if ((raw/'boot_id').read_text().strip()!=boot or (raw/'task_hash').read_text().strip()!=digest(task)):
        raise ValueError('raw phase task hash or boot differs')
    if expected_outputs is not None and calculated['output_fingerprints']!=expected_outputs:
        raise ValueError('accepted outputs differ from qualification')
    return item


def run(output,run_dir,local_config,*,scope='components',execute_board=False):
    if not execute_board:raise ValueError('--execute-board is required')
    from yolo_validation.components import hardware_check
    output,run_dir=Path(output).resolve(),Path(run_dir).resolve();tasks=specs(output,scope)
    bundles={network:output/network/'capsule' for network in ('resnet18','yolov3_tiny416')}
    verified={network:validate(bundle) for network,bundle in bundles.items()}
    ssh=matched_ssh(read(local_config)['board'],execute_board=True);boot=ssh.boot_id();started=time.monotonic()
    with locked(run_dir):
        marker=run_dir/'run.json';identity=dict(boot_id=boot,output=str(output),protocol=protocol(),
            bundle_ids={network:value['bundle_id'] for network,value in verified.items()})
        if marker.exists() and read(marker)!=identity:raise ValueError('phase campaign boot or identity changed; do not mix executions')
        if not marker.exists():write(marker,identity)
        def state(status,**fields):
            write(run_dir/'state.json',dict(status=status,scope=scope,boot_id=boot,
                updated_at=now(),goal_complete=False,**fields),replace=True)
        try:
            remotes={}
            for network in ('resnet18','yolov3_tiny416'):
                bundle=bundles[network]
                (run_dir/network).mkdir(parents=True,exist_ok=True)
                remotes[network]=ssh.deploy(bundle,run_dir/network)
                ssh.qualify_environment(remotes[network],boot,run_dir/network,bundle)
                write(run_dir/(network+'_hardware.json'),hardware_check(ssh,bundle,boot),replace=True)
            expected={};done=0;stage_calls=0;full=0
            for spec in tasks:
                network=spec['network'];bundle=bundles[network];task=make_task(bundle,spec)
                # A bridge's qualification is the same frozen configuration.
                key=(network,spec['scope'],spec['case'])
                if spec['phase']!='qualify' and key not in expected:
                    qspec={**spec,'phase':'qualify','probe':True,'runner':'phase'}
                    q=make_task(bundle,qspec);qpath=run_dir/'accepted'/boot/(q['task_id']+'.json')
                    if not qpath.exists():raise ValueError('measurement cannot precede qualification')
                    expected[key]=read(qpath)['summary']['output_fingerprints']
                state('RUNNING',accepted_sessions=done,total_sessions=len(tasks),current_task=task['task_id'])
                item=accept(ssh,task,bundle,remotes[network],run_dir,boot,expected.get(key))
                if spec['phase']=='qualify':expected[key]=item['summary']['output_fingerprints']
                done+=1;calls=sum(task[k] for k in ('warmup','scored','tail'))
                if spec['scope']=='component':stage_calls+=calls
                else:full+=calls
                print('同钟阶段观测',done,'/',len(tasks),network,spec['case'],spec['phase'],flush=True)
            for network,remote in remotes.items():
                ssh.idle_command('cd '+shlex.quote(remote)+' && sha256sum -c board.sha256 > phase_post_verify.log && sync',boot=boot,timeout=600)
            state('COMPONENTS_COMPLETE' if scope=='components' else 'PIPELINES_COMPLETE',accepted_sessions=done,
                total_sessions=len(tasks),component_calls=stage_calls,DNN_inferences=full,
                elapsed_seconds=time.monotonic()-started,new_runtime_builds=0,new_module_builds=0)
            sync_tree(run_dir)
        except Exception as error:
            state('STOPPED_ON_ERROR',error_type=type(error).__name__,error=str(error));raise
    return read(run_dir/'state.json')
