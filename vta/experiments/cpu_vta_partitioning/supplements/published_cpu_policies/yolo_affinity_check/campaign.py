"""Exactly twenty tasks; byte-bound receipts survive transport interruption."""
from pathlib import Path
from types import FunctionType
import shlex
import time

from cpu_vta_partitioning.common import locked, now, read, sha256, sync_tree, write
from yolo_validation.pipeline import accept as old_accept
from yolo_validation.components import hardware_check
from .common import assert_prepared, frozen
from .prepare import validate_bundle
from .runtime import make_task, matched_ssh, summary

accept = FunctionType(old_accept.__code__, {**old_accept.__globals__, 'summary': summary})


def run(output, run_dir, local_config, *, execute_board=False):
    if not execute_board:
        raise ValueError('--execute-board is required')
    output, run_dir = Path(output).resolve(), Path(run_dir).resolve()
    assert_prepared(output)
    bundle = output / 'capsule'
    manifest = validate_bundle(bundle)
    ssh = matched_ssh(read(local_config)['board'], execute_board=True)
    boot, started = ssh.boot_id(), time.monotonic()
    with locked(run_dir):
        frozen(run_dir / 'run.json', dict(boot_id=boot, bundle_id=manifest['bundle_id'], bundle=str(bundle)))
        frozen(run_dir / 'protocol_freeze.json', dict(preparation_sha256=sha256(output/'PREPARATION.json'),
            order_sha256=sha256(bundle/'plans/task_order.json'),
            configuration_sha256=sha256(bundle/'plans/configurations.json')))
        def state(status, **values):
            write(run_dir / 'state.json', dict(status=status, boot_id=boot, bundle=str(bundle),
                updated_at=now(), **values), replace=True)
        try:
            remote = ssh.deploy(bundle, run_dir)
            ssh.qualify_environment(remote, boot, run_dir, bundle)
            write(run_dir/'hardware.json', hardware_check(ssh, bundle, boot), replace=True)
            frequencies = ssh.command('for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; do cat "$f"; done').split()
            if frequencies != ['1066666']*4:
                raise ValueError('CPU frequency changed; no implicit retuning')
            done = full = components = 0
            for spec in read(bundle/'plans/task_order.json'):
                doc = make_task(bundle, spec)
                item = accept(ssh, doc, bundle, remote, run_dir, boot)
                done += 1
                count = sum(doc[k] for k in ('warmup', 'scored', 'tail'))
                if spec['scope'] == 'pipeline':
                    full += count
                else:
                    components += count
                state('MEASURING', accepted_sessions=done, DNN_inferences=full,
                      isolated_CPU_stage_calls=components, last_spec=spec)
                print('绑定验证', done, '/20', spec,
                      item['summary'].get('fps', item['summary'].get('stage_service_ms', 'qualified')), flush=True)
            if (done, full, components) != (20, 836, 44):
                raise ValueError('experiment exceeded or failed to meet the frozen task budget')
            ssh.idle_command('cd '+shlex.quote(remote)+' && sha256sum -c board.sha256 > post_run_verification.log && sync',
                             boot=boot, timeout=600)
            state('COMPLETE', accepted_sessions=done, DNN_inferences=full,
                isolated_CPU_stage_calls=components, post_deployment_hashes_passed=True,
                execution_elapsed_seconds=time.monotonic()-started, same_boot=True)
            sync_tree(run_dir)
        except Exception as error:
            state('STOPPED_ON_ERROR', error_type=type(error).__name__, error=str(error))
            raise
    return read(run_dir/'state.json')


def status(output, run_dir):
    return dict(prepared=(Path(output)/'capsule/manifest.json').exists(),
        state=read(Path(run_dir)/'state.json') if (Path(run_dir)/'state.json').exists() else dict(status='NOT_STARTED'))
