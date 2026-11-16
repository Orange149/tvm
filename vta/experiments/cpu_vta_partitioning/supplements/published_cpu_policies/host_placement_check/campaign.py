from pathlib import Path
from types import FunctionType
import shlex,time
from cpu_vta_partitioning.common import locked,now,read,sha256,sync_tree,write
from phase_profile.campaign import accept as original_accept
from yolo_validation.components import hardware_check
from .common import CAPABILITY,freeze,protocol,specs
from .prepare import validate
from .runtime import make_task,matched_ssh,summary

accept=FunctionType(original_accept.__code__,{**original_accept.__globals__,'summary':summary},argdefs=original_accept.__defaults__)


def full_specs():
    return [dict(network='yolov3_tiny416',scope='pipeline',case='overlap',**s) for s in specs()]


def run(output,run_dir,local_config,*,execute_board=False):
    if not execute_board:raise ValueError('--execute-board is required')
    output,run_dir=Path(output).resolve(),Path(run_dir).resolve();valid=validate(output)
    if not read(output/'QEMU.json')['passed']:raise ValueError('ARM loading/output qualification required')
    capability=read(CAPABILITY/'BOARD_SELF_TEST.json')
    if not capability['validation']['passed']:raise ValueError('native per-core counting capability unverified')
    ssh=matched_ssh(read(local_config)['board'],execute_board=True);boot=ssh.boot_id()
    if boot!=capability['boot_id']:raise ValueError('boot differs from native counter test; no automatic restart')
    bundle=output/'capsule';tick=time.monotonic()
    with locked(run_dir):
        freeze(run_dir/'run.json',dict(boot_id=boot,bundle_id=valid['bundle_id'],protocol=protocol(),counter_capability_sha256=sha256(CAPABILITY/'BOARD_SELF_TEST.json')))
        def state(status,**fields):write(run_dir/'state.json',dict(status=status,boot_id=boot,updated_at=now(),**fields),replace=True)
        try:
            remote=ssh.deploy(bundle,run_dir)
            ssh.qualify_environment(remote,boot,run_dir,bundle)
            write(run_dir/'hardware.json',hardware_check(ssh,bundle,boot),replace=True)
            qualified={};reference=None;calls=0
            for index,spec in enumerate(full_specs(),1):
                task=make_task(bundle,spec);runner=spec['runner']
                if spec['phase']!='qualify' and runner not in qualified:raise ValueError('timing cannot precede executor qualification')
                state('RUNNING',accepted_sessions=index-1,total_sessions=5,current_spec=spec)
                item=accept(ssh,task,bundle,remote,run_dir,boot,qualified.get(runner,reference))
                if spec['phase']=='qualify':
                    qualified[runner]=item['summary']['output_fingerprints'];reference=qualified[runner]
                calls+=sum(task[k] for k in ('warmup','scored','tail'))
                print('主机逐核诊断',index,'/5',spec['phase'],runner,flush=True)
            if calls!=314:raise ValueError('bounded inference count changed')
            ssh.idle_command('cd '+shlex.quote(remote)+' && sha256sum -c board.sha256 > placement_post_verify.log && sync',boot=boot,timeout=300)
            if ssh.boot_id()!=boot:raise ValueError('boot changed after measurement')
            state('COMPLETE',accepted_sessions=5,total_sessions=5,DNN_inferences=calls,seconds=time.monotonic()-tick,
                  no_reboot=True,no_extra_measurement=True)
            sync_tree(run_dir)
        except Exception as e:
            state('STOPPED_ON_ERROR',error=str(e),error_type=type(e).__name__);raise
    return read(run_dir/'state.json')
