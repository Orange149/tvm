"""One performance boot; original native tasks, receipts and recovery semantics."""
from pathlib import Path
import shlex
import time

from cpu_vta_partitioning.common import locked, now, read, sha256, sync_tree, write
from resnet50_validation import runtime, components
from resnet50_validation.campaign import accept
from resnet50_validation.deployment_reuse import deploy
from .common import SOURCE, protocol, frozen, assert_controller
from .workflow import validate_bundle


def ensure_storage(ssh,boot):
    """Same storage check as the old board cohort, now inside this sealed package."""
    source='/media/sd-mmcblk1p3';target=ssh.config['storage_mount'];q=shlex.quote
    script='\n'.join(['set -eu',
        'test "$(cat /proc/sys/kernel/random/boot_id)" = '+q(boot),
        'awk -v p='+q(source)+" '$2 == p && $4 ~ /(^|,)rw(,|$)/ {ok=1} END {exit !ok}' /proc/mounts",
        'if ! awk -v p='+q(target)+" '$2 == p {ok=1} END {exit !ok}' /proc/mounts; then",
        '  test ! -L '+q(target),'  mkdir -p '+q(target),
        '  mount --bind '+q(source)+' '+q(target),'fi',
        'test "$(stat -c %d '+q(source)+')" = "$(stat -c %d '+q(target)+')"',
        'df -Pk '+q(target)])
    return ssh.command(script,timeout=30)


def health(ssh, boot, bundle):
    path=shlex.quote(ssh.config['storage_mount'])
    script='\n'.join(['set -eu',
        'test "$(cat /proc/sys/kernel/random/boot_id)" = '+shlex.quote(boot),
        "awk '/MemAvailable:/ {print \"MEM_AVAILABLE_BYTES\",$2*1024}' /proc/meminfo",
        'df -Pk '+path+" | awk 'NR==2 {printf \"SD_AVAILABLE_BYTES %.0f\\n\",$4*1024}'"])
    raw=ssh.idle_command(script,boot=boot,timeout=30)
    values={k:int(float(v)) for k,v in (line.split() for line in raw.splitlines())}
    if values.get('MEM_AVAILABLE_BYTES',0)<256*1024**2 or values.get('SD_AVAILABLE_BYTES',0)<256*1024**2:
        raise RuntimeError('idle memory/storage margin below 256 MiB; no cleanup or resize performed')
    return dict(boot_id=boot,values=values,scope='between processes, not measured peak',time=now())


def run(output,run_dir,local_config,*,execute_board=False):
    if not execute_board:
        raise ValueError('--execute-board is required')
    output,run_dir=Path(output).resolve(),Path(run_dir).resolve()
    assert_controller(output);bundle=output/'pipeline_bundle';manifest=validate_bundle(bundle)
    ssh=runtime.transport(read(local_config)['board'],execute_board=True)
    with locked(run_dir):
        boot=ssh.boot_id()
        frozen(run_dir/'run.json',dict(boot_id=boot,bundle_id=manifest['bundle_id'],
            controller_sha256=sha256(output/'CONTROLLERS.json'),
            ranking_freeze_sha256=sha256(output/'ranking/FREEZE.json')))
        def state(status,**extra):
            write(run_dir/'state.json',dict(status=status,boot_id=boot,updated_at=now(),**extra),replace=True)
        try:
            state('STORAGE')
            write(run_dir/'storage.json',dict(boot_id=boot,raw=ensure_storage(ssh,boot)),replace=True)
            write(run_dir/'health_initial.json',health(ssh,boot,bundle),replace=True)
            state('DEPLOYING');began=time.monotonic()
            remote=deploy(ssh,bundle,run_dir/'pipeline',source_bundle=SOURCE/'profile_capsule')
            write(run_dir/'deployment_cost.json',dict(seconds=time.monotonic()-began,
                source_bundle=str(SOURCE/'profile_capsule'),profile_repeated=False),replace=True)
            state('ENVIRONMENT')
            ssh.qualify_environment(remote,boot,run_dir/'pipeline',bundle)
            write(run_dir/'hardware.json',components.check_hardware(ssh,bundle,boot),replace=True)
            state('PIPELINE');order=read(bundle/'plans/execution_order.json');pool=read(bundle/'plans/pool.json')
            if len(order)>103 or len(order)!=len(set(order)) or set(order)!=set(pool):
                raise ValueError('measurement union differs from the approved frozen pool')
            pipeline=run_dir/'pipeline'
            for index,name in enumerate(order):
                if (run_dir/'STOP_REQUESTED').exists():
                    raise InterruptedError('stop requested; the last accepted candidate is preserved')
                if index%5==0:
                    write(run_dir/'health'/f'{index:03}.json',health(ssh,boot,bundle),replace=True)
                q=accept(ssh,runtime.task(bundle,pool[name],'qualify'),bundle,remote,pipeline,boot)
                accept(ssh,runtime.task(bundle,pool[name],'short'),bundle,remote,pipeline,boot,
                       expected=q['summary']['output_fingerprints'])
                progress=dict(status='MEASURING',boot_id=boot,candidate_count=len(order),
                    done_candidates=index+1,accepted_sessions=2*(index+1),updated_at=now())
                write(pipeline/'state.json',progress,replace=True)
                print('ResNet50 ablation',index+1,'/',len(order),name,flush=True)
            ssh.idle_command('cd '+shlex.quote(remote)+' && sha256sum -c board.sha256 > post_run_verification.log && sync',
                             boot=boot,timeout=900)
            finished=dict(status='COMPLETE',boot_id=boot,candidate_count=len(order),accepted_sessions=2*len(order),
                complete_DNN_inferences=49*len(order),post_deployment_hashes_passed=True,finished_at=now())
            write(pipeline/'state.json',finished,replace=True)
            write(run_dir/'health_final.json',health(ssh,boot,bundle),replace=True)
            state('ANALYZING')
            from .analysis import analyze
            report=analyze(output,run_dir)
            state('COMPLETE',report=report);sync_tree(run_dir)
            return report
        except Exception as error:
            state('STOPPED_ON_ERROR',error_type=type(error).__name__,error=str(error))
            raise
