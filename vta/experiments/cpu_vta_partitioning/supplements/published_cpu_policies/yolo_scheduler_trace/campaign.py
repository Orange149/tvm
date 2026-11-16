from pathlib import Path
from types import FunctionType
import shlex,time
from cpu_vta_partitioning.common import locked,now,read,sha256,sync_tree,write
from phase_profile.campaign import accept as original_accept
from yolo_validation.components import hardware_check
from .common import CAPABILITY,PARENT,freeze,protocol,specs
from .runtime import make_task,matched_ssh,summary
from .prepare import validate

accept=FunctionType(original_accept.__code__,{**original_accept.__globals__,'summary':summary},argdefs=original_accept.__defaults__)


def run(output,run_dir,local_config,*,execute_board=False):
    if not execute_board:raise ValueError('--execute-board is required')
    output,run_dir=Path(output).resolve(),Path(run_dir).resolve();valid=validate(output)
    if not read(output/'QEMU.json')['passed']:raise ValueError('AArch64 loading and CPU output check required')
    capability=read(CAPABILITY/'BOARD_SELF_TEST.json')
    if not capability['validation']['passed']:raise ValueError('native scheduler event capability missing')
    ssh=matched_ssh(read(local_config)['board'],execute_board=True);boot=ssh.boot_id()
    if boot!=capability['boot_id']:raise ValueError('current boot differs from event capability test')
    delay_capability=PARENT/'capability'/boot/'SELF_TEST.json'
    if not delay_capability.exists() or not read(delay_capability).get('passed'):raise ValueError('current boot runnable-delay capability missing')
    bundle=output/'capsule';tick=time.monotonic()
    with locked(run_dir):
        freeze(run_dir/'run.json',dict(boot_id=boot,bundle_id=valid['bundle_id'],protocol=protocol(),
            switch_capability_sha256=sha256(CAPABILITY/'BOARD_SELF_TEST.json'),delay_capability_sha256=sha256(delay_capability)))
        def state(status,**kw):write(run_dir/'state.json',dict(status=status,boot_id=boot,updated_at=now(),**kw),replace=True)
        try:
            remote=ssh.deploy(bundle,run_dir);ssh.qualify_environment(remote,boot,run_dir,bundle)
            write(run_dir/'hardware.json',hardware_check(ssh,bundle,boot),replace=True)
            outputs={};common=None;count=0
            for index,spec in enumerate(specs(),1):
                doc=make_task(bundle,spec)
                if spec['phase']!='qualify' and spec['runner'] not in outputs:raise ValueError('measurement cannot precede qualification')
                state('RUNNING',accepted_sessions=index-1,total_sessions=5,current_spec=spec)
                item=accept(ssh,doc,bundle,remote,run_dir,boot,outputs.get(spec['runner'],common))
                if spec['phase']=='qualify':outputs[spec['runner']]=common=item['summary']['output_fingerprints']
                count+=sum(doc[k] for k in ('warmup','scored','tail'))
                print('YOLO调度时间线',index,'/5',spec['phase'],spec['runner'],flush=True)
            if count!=146:raise ValueError('scheduler diagnostic inference bound changed')
            ssh.idle_command('cd '+shlex.quote(remote)+' && sha256sum -c board.sha256 > scheduler_trace_post_verify.log && sync',boot=boot,timeout=300)
            if ssh.boot_id()!=boot:raise ValueError('boot changed after scheduler diagnostic')
            state('COMPLETE',accepted_sessions=5,total_sessions=5,DNN_inferences=count,seconds=time.monotonic()-tick,no_reboot=True,no_extra_measurement=True)
            sync_tree(run_dir)
        except Exception as e:
            state('STOPPED_ON_ERROR',error_type=type(e).__name__,error=str(e));raise
    return read(run_dir/'state.json')
