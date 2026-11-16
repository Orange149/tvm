"""Exactly eighteen boot-bound sessions, accepted before acknowledgment."""
from pathlib import Path
from types import FunctionType
import shlex
import time
from cpu_vta_partitioning.common import inventory,locked,now,read,sha256,sync_tree,write
from phase_profile.campaign import accept as old_accept
from yolo_validation.components import hardware_check
from .common import protocol,freeze
from .prepare import validate,specs
from .runtime import make_task,matched_ssh,summary

accept=FunctionType(old_accept.__code__,{**old_accept.__globals__,'summary':summary},argdefs=old_accept.__defaults__)


def run(output,run_dir,local_config,*,execute_board=False):
    if not execute_board:raise ValueError('--execute-board is required')
    output,run_dir=Path(output).resolve(),Path(run_dir).resolve();bundles=validate(output)
    if not read(output/'QEMU.json')['passed']:raise ValueError('ARM load/output qualification required')
    ssh=matched_ssh(read(local_config)['board'],execute_board=True);boot=ssh.boot_id()
    capability=output/'capability'/boot/'SELF_TEST.json'
    if not capability.exists() or not read(capability).get('passed'):raise ValueError('current boot native TASKSTATS self-test required')
    if not (output/'PREDICTIONS_FREEZE.json').exists():raise ValueError('freeze unchanged predictions before performance')
    identity=dict(boot_id=boot,bundles=bundles,protocol=protocol(),
                  prediction_sha256=sha256(output/'PREDICTIONS_FREEZE.json'),capability_sha256=sha256(capability))
    started=time.monotonic()
    with locked(run_dir):
        freeze(run_dir/'run.json',identity)
        def state(status,**fields):
            write(run_dir/'state.json',dict(status=status,boot_id=boot,updated_at=now(),**fields),replace=True)
        try:
            remotes={}
            for network in bundles:
                bundle=output/network/'capsule';deployment=run_dir/network;deployment.mkdir(parents=True,exist_ok=True)
                remotes[network]=ssh.deploy(bundle,deployment)
                ssh.qualify_environment(remotes[network],boot,deployment,bundle)
                write(run_dir/(network+'_hardware.json'),hardware_check(ssh,bundle,boot),replace=True)
            expected={};full=0;isolated=0
            for done,spec in enumerate(specs(),1):
                network=spec['network'];bundle=output/network/'capsule';task=make_task(bundle,spec)
                key=(network,spec['scope'],spec['case'])
                if spec['phase']!='qualify' and key not in expected:raise ValueError('qualification must precede timings')
                state('RUNNING',accepted_sessions=done-1,total_sessions=18,current_spec=spec)
                item=accept(ssh,task,bundle,remotes[network],run_dir,boot,expected.get(key))
                if spec['phase']=='qualify':expected[key]=item['summary']['output_fingerprints']
                count=sum(task[k] for k in ('warmup','scored','tail'))
                if spec['scope']=='component':isolated+=count
                else:full+=count
                print('排队诊断',done,'/18',network,spec['scope'],spec['case'],spec['phase'],flush=True)
            for network,remote in remotes.items():
                ssh.idle_command('cd '+shlex.quote(remote)+' && sha256sum -c board.sha256 > scheduler_post_verify.log && sync',boot=boot,timeout=600)
            if full!=627 or isolated!=63:raise ValueError('native task budget changed')
            state('COMPLETE',accepted_sessions=18,total_sessions=18,DNN_inferences=full,
                  component_calls=isolated,seconds=time.monotonic()-started,no_reboot=True)
            sync_tree(run_dir)
        except Exception as error:
            state('STOPPED_ON_ERROR',error_type=type(error).__name__,error=str(error));raise
    return read(run_dir/'state.json')
