from pathlib import Path
from types import FunctionType
import shlex,time
from cpu_vta_partitioning.common import locked,now,read,sha256,sync_tree,write
from phase_profile.campaign import accept as previous_accept
from yolo_validation.components import hardware_check
from .common import CAPABILITY,PARENT,freeze,protocol,specs
from .runtime import make_task,matched_ssh,summary
from .prepare import validate

accept=FunctionType(previous_accept.__code__,{**previous_accept.__globals__,'summary':summary},argdefs=previous_accept.__defaults__)


def run(output,run_dir,local_config,*,execute_board=False):
    if not execute_board:raise ValueError('--execute-board is required')
    output,run_dir=Path(output).resolve(),Path(run_dir).resolve();valid=validate(output)
    if not read(output/'QEMU.json')['passed']:raise ValueError('QEMU check required')
    capability=read(CAPABILITY/'BOARD_SELF_TEST.json')
    if not capability['validation']['passed']:raise ValueError('native switch capability absent')
    ssh=matched_ssh(read(local_config)['board'],execute_board=True);boot=ssh.boot_id()
    if boot!=capability['boot_id']:raise ValueError('boot differs from switch capability')
    delay=PARENT/'capability'/boot/'SELF_TEST.json'
    if not delay.exists() or not read(delay).get('passed'):raise ValueError('runnable-delay capability absent')
    bundle=output/'capsule';tick=time.monotonic()
    with locked(run_dir):
        freeze(run_dir/'run.json',dict(boot_id=boot,bundle_id=valid['bundle_id'],protocol=protocol(),
            switch_capability_sha256=sha256(CAPABILITY/'BOARD_SELF_TEST.json'),delay_capability_sha256=sha256(delay)))
        def state(status,**kw):write(run_dir/'state.json',dict(status=status,boot_id=boot,updated_at=now(),**kw),replace=True)
        try:
            remote=ssh.deploy(bundle,run_dir);ssh.qualify_environment(remote,boot,run_dir,bundle)
            write(run_dir/'hardware.json',hardware_check(ssh,bundle,boot),replace=True)
            expected=None;count=0
            for index,spec in enumerate(specs(),1):
                doc=make_task(bundle,spec)
                if index>1 and expected is None:raise ValueError('qualification must precede timing')
                state('RUNNING',accepted_sessions=index-1,total_sessions=3,current_spec=spec)
                item=accept(ssh,doc,bundle,remote,run_dir,boot,expected)
                if index==1:expected=item['summary']['output_fingerprints']
                count+=sum(doc[k] for k in ('warmup','scored','tail'))
                print('调度事件及时回收',index,'/3',spec['phase'],flush=True)
            if count!=97:raise ValueError('repair inference limit changed')
            ssh.idle_command('cd '+shlex.quote(remote)+' && sha256sum -c board.sha256 > drained_trace_post_verify.log && sync',boot=boot,timeout=300)
            if ssh.boot_id()!=boot:raise ValueError('boot changed')
            state('COMPLETE',accepted_sessions=3,total_sessions=3,DNN_inferences=count,
                  seconds=time.monotonic()-tick,no_reboot=True,no_extra_measurement=True)
            sync_tree(run_dir)
        except Exception as e:
            state('STOPPED_ON_ERROR',error_type=type(e).__name__,error=str(e));raise
    return read(run_dir/'state.json')
