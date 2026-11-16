"""Check old completion-tail evidence without per-call TASKSTATS queries.

One existing layout, one existing binary, three bounded sessions. No runtime,
module, waiting, affinity, score, or ranking changes.
"""
from pathlib import Path
from types import FunctionType
import argparse
import json
import shlex
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from cpu_vta_partitioning.common import read, write, sha256, digest, locked, now, sync_tree
from cpu_vta_partitioning.measurement import frames
from phase_profile import runtime as phase
from phase_profile.campaign import accept as original_accept
from yolo_scheduler_trace_repair.prepare import validate as validate_parent
from yolo_scheduler_trace_repair.runtime import make_task as parent_task
from yolo_scheduler_trace_repair.events import validate_profile
from run_resnet_middle_switch import phase_summary, disabled_queue_rows
from board_transport_without_reprogram import transport

ROOT = Path(__file__).resolve().parent
PARENT = ROOT / 'artifacts/yolo_scheduler_trace_repair_01'
BUNDLE = PARENT / 'capsule'
OUT = ROOT / 'artifacts/yolo_switch_without_queries_01'
RUN = ROOT / 'runs/yolo_switch_without_queries_01'
CONFIG = ROOT / 'artifacts/cross_network_model_01/board.local.json'


def protocol():
    return dict(experiment='yolo_switch_without_queries_01', input_index=0,
        qualification=dict(warmup=0, scored=1, tail=0),
        pipeline=dict(warmup=8, scored=16, tail=4), sessions=3, DNN_inferences=57,
        queue_counter_reads=False, same_frozen_recorder_binary=True,
        new_modules=0, new_runtime=0, new_affinity_policy=False,
        no_reboot=True, no_FPGA_reprogram=True, no_system_retuning=True,
        no_anchor=True, no_drift_rejection=True, no_automatic_extra_measurements=True,
        purpose='Verify which critical-worker competition survives removal of per-call counter queries.')


def sources():
    files = [Path(__file__), ROOT / 'tests/test_yolo_switch_without_queries.py',
        ROOT / 'board_transport_without_reprogram.py', ROOT / 'phase_profile/runtime.py',
        ROOT / 'phase_profile/campaign.py', ROOT / 'run_resnet_middle_switch.py',
        ROOT / 'yolo_scheduler_trace_repair/runtime.py', ROOT / 'yolo_scheduler_trace_repair/events.py',
        ROOT / 'yolo_scheduler_trace/events.py', ROOT / 'scheduler_gap_audit/runtime.py',
        ROOT / 'run_three_thread_worker_gap_compat.py']
    return {str(path): sha256(path) for path in files}


def freeze(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError('frozen counter-free observation identity differs: ' + str(path))
    else:
        write(path, value)
    return value


def specs():
    return [dict(network='yolov3_tiny416', scope='pipeline', case='overlap',
                 phase=p, runner='phase', probe=p == 'observed') for p in ('qualify', 'clean', 'observed')]


def make_task(bundle, spec):
    if spec not in specs():
        raise ValueError('unplanned counter-free trace session')
    doc = parent_task(bundle, spec)
    doc.pop('task_id')
    doc.update(protocol()['qualification' if spec['phase'] == 'qualify' else 'pipeline'])
    doc.update(experiment=protocol()['experiment'], queue_counter_reads=False,
               diagnostic_protocol_sha256=digest(protocol()), controller_sources_sha256=digest(sources()))
    doc['task_id'] = digest(doc)[:24]
    return doc


def native_command(doc, bundle, remote_bundle, remote_job):
    command = FunctionType(phase.native_command.__code__, {**phase.native_command.__globals__,
                                                          'make_task': make_task})
    argv, inputs = command(doc, bundle, remote_bundle, remote_job)
    if doc['spec']['probe']:
        at = argv.index(remote_bundle + '/runtime/ld-linux-aarch64.so.1')
        argv[at:at] = ['VTA_QUEUE_COUNTERS=0', 'VTA_QUEUE_STATUS=' + remote_job + '/queue_status.json']
        argv += ['--session-profile-json', remote_job + '/profile.json',
                 '--session-profile-window', f"{doc['warmup']},{doc['scored']}"]
    return argv, inputs


def summary(raw, doc, bundle):
    fn = FunctionType(phase.summary.__code__, {**phase.summary.__globals__, 'phase_summary': phase_summary})
    result = fn(raw, doc, bundle)
    raw = Path(raw)
    if doc['scheduler_events']:
        status = read(raw / 'queue_status.json')
        if status['enabled'] or status['errors'] or status['reads']:
            raise ValueError('counter-free observation made a TASKSTATS query')
        trace = [json.loads(line) for line in (raw / 'wait_trace.jsonl').read_text().splitlines()]
        disabled_queue_rows(trace, scope='worker')
        extra = ('queue_read_cpu0_ns', 'queue_read_cpu1_ns', 'caller_tail_queue_start_ns')
        if any(record[key] != 0 for record in trace for key in extra):
            raise ValueError('counter-free trace has unexpected counter values')
        if any(value != 0 for record in trace for item in record['worker_queue_baselines'] for value in item.values()):
            raise ValueError('counter-free trace contains worker counter queries')
        result['scheduler'] = validate_profile(read(raw / 'profile.json'), doc, read(raw / 'affinity.json'))
        rows = frames(raw / 'result.jsonl', sum(doc[k] for k in ('warmup', 'scored', 'tail')))
        a, b = doc['warmup'], doc['warmup'] + doc['scored'] - 1
        duration = round((rows[b]['completion_ms'] - rows[a]['completion_ms']) * 1e6)
        scheduler = result['scheduler']
        if abs(duration - (scheduler['completion_end_ns'] - scheduler['completion_start_ns'])) > 1000:
            raise ValueError('scheduler and completion anchors differ')
        result['queue_status'] = status
        result['zero_queue_fields_are_not_wait_measurements'] = True
    else:
        if (raw / 'profile.json').exists() or (raw / 'queue_status.json').exists():
            raise ValueError('unexpected observer in non-observed session')
        result['scheduler'] = None
    return result


def accept(*args, **kwargs):
    fn = FunctionType(original_accept.__code__, {**original_accept.__globals__, 'summary': summary})
    return fn(*args, **kwargs)


def prepare():
    parent = validate_parent(PARENT)
    qemu = read(PARENT / 'QEMU.json')
    if qemu.get('passed') is not True:
        raise ValueError('reused recorder has no accepted QEMU loading/output check')
    commands = []
    for spec in specs():
        task = make_task(BUNDLE, spec)
        argv, _ = native_command(task, BUNDLE, '/bundle', '/job')
        commands.append(dict(task=task, argv=argv))
    if len({command['task']['executor_sha256'] for command in commands}) != 1:
        raise ValueError('sessions use different executors')
    plan = dict(protocol=protocol(), sources=sources(), bundle_id=parent['bundle_id'],
        parent_manifest_sha256=sha256(BUNDLE / 'manifest.json'), commands=commands,
        reused_binary_QEMU_sha256=sha256(PARENT / 'QEMU.json'),
        hypothesis='TASKSTATS-free switch evidence will distinguish real critical-core contention from query-induced ordering changes.',
        decision='If ordering changes materially, do not reuse query-on per-kernel attribution quantitatively; preserve total-stage and native-performance limits.',
        no_static_inputs_or_formula_changed=True)
    freeze(OUT / 'PLAN.json', plan)
    return plan


def validate():
    plan = read(OUT / 'PLAN.json')
    if plan['protocol'] != protocol() or plan['sources'] != sources():
        raise ValueError('counter-free observation controller or protocol changed')
    parent = validate_parent(PARENT)
    if parent['bundle_id'] != plan['bundle_id'] or sha256(BUNDLE / 'manifest.json') != plan['parent_manifest_sha256']:
        raise ValueError('reused recorder capsule changed')
    return plan


def board(config=CONFIG, run_dir=RUN, *, execute_board=False):
    if not execute_board:
        raise ValueError('--execute-board is required')
    plan = validate()
    ssh = transport(read(config)['board'], native_command, execute_board=True)
    boot = ssh.boot_id()
    run_dir = Path(run_dir)
    started = time.monotonic()
    with locked(run_dir):
        freeze(run_dir / 'run.json', dict(boot_id=boot, plan_sha256=sha256(OUT / 'PLAN.json'),
                                         bundle_id=plan['bundle_id']))
        def state(status, **fields):
            write(run_dir / 'state.json', dict(status=status, boot_id=boot, updated_at=now(), **fields), replace=True)
        try:
            remote = ssh.deploy(BUNDLE, run_dir)
            ssh.qualify_environment(remote, boot, run_dir, BUNDLE)
            expected = None
            count = 0
            for index, spec in enumerate(specs(), 1):
                task = make_task(BUNDLE, spec)
                state('RUNNING', current_phase=spec['phase'], accepted_sessions=index - 1, total_sessions=3)
                item = accept(ssh, task, BUNDLE, remote, run_dir, boot, expected)
                if index == 1:
                    expected = item['summary']['output_fingerprints']
                count += sum(task[k] for k in ('warmup', 'scored', 'tail'))
                print('无逐次查询的线程观测', index, '/3', spec['phase'], flush=True)
            if count != 57:
                raise ValueError('bounded inference count differs')
            ssh.idle_command('cd ' + shlex.quote(remote) + ' && sha256sum -c board.sha256 > switch_no_queries_post_verify.log && sync', boot=boot, timeout=300)
            if ssh.boot_id() != boot:
                raise ValueError('boot changed')
            state('COMPLETE', accepted_sessions=3, total_sessions=3, DNN_inferences=count,
                  seconds=time.monotonic() - started, no_reboot=True, no_new_components=True)
            sync_tree(run_dir)
        except Exception as error:
            state('STOPPED_ON_ERROR', error_type=type(error).__name__, error=str(error))
            raise
    return read(run_dir / 'state.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'validate', 'board', 'status'))
    parser.add_argument('--execute-board', action='store_true')
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--run-dir', type=Path, default=RUN)
    args = parser.parse_args()
    if args.command == 'prepare':
        plan = prepare()
        result = dict(prepared=True, sessions=3, DNN_inferences=57, new_compilation=False, bundle_id=plan['bundle_id'])
    elif args.command == 'validate':
        plan = validate()
        result = dict(validated=True, bundle_id=plan['bundle_id'])
    elif args.command == 'status':
        result = read(args.run_dir / 'state.json') if (args.run_dir / 'state.json').exists() else dict(status='NOT_STARTED')
    else:
        result = board(args.config, args.run_dir, execute_board=args.execute_board)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
