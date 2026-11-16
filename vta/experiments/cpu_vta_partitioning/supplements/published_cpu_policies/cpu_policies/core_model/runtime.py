"""Explicit diagnostic tasks and validated, hash-bound optional counters."""
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import shlex
import time

from .common import api, load_legacy, protocol, DEFAULT_PARENT
from ..open_resources.runtime import _FIELDS as BASE_FIELDS
from .affinity import validate_affinity as _affinity
from .domain import validate_candidate


_FIELDS = BASE_FIELDS | {'run_probe', 'probe_frames'}


def make_task(kind, config, bundle, *, warmup=0, scored=1, tail=0, indices=None,
              executor='core_model', run_probe=False, probe_frames=None, **labels):
    common = api(bundle)
    if _FIELDS.intersection(labels):
        raise ValueError('labels override execution identity')
    if kind not in ('calibration', 'qualify_component', 'qualify_pipeline', 'performance'):
        raise ValueError('unsupported diagnostic task kind')
    if any(type(n) is not int or n < 0 for n in (warmup, scored, tail)) or scored < 1:
        raise ValueError('invalid measurement window')
    indices = list(indices if indices is not None else [0])
    if indices != [0] or type(indices[0]) is not int:
        raise ValueError('only fixed input zero is authorized')
    component = kind in ('calibration', 'qualify_component')
    validate_candidate(config, common.read(Path(bundle) / 'plans/programs.json'), component=component)
    if labels.get('detailed', False) is not False or labels.get('diagnosis_detail', False) is not False:
        raise ValueError('session and queue probes are disabled in this campaign')
    if executor not in ('policy', 'core_model') or type(run_probe) is not bool:
        raise ValueError('invalid executor/probe selection')
    if run_probe and executor != 'core_model':
        raise ValueError('old executor does not support run counters')
    if run_probe and (kind not in ('calibration', 'qualify_component')
                      or len(config['stages']) != 1 or config['stages'][0]['device'] != 'cpu'
                      or config['stages'][0]['threads'] <= 1):
        raise ValueError('only multithread component costs use the run probe')
    if executor == 'policy' and kind != 'performance':
        raise ValueError('old executor is confined to the performance bridge')
    if executor == 'policy' and any(s['device'] == 'cpu' and s['worker_core_map'] != s['mask']
                                   for s in config['stages']):
        raise ValueError('old executor cannot execute a reordered worker map')
    if probe_frames is None:
        probe_frames = (list(range(warmup, warmup + scored, 4 if kind == 'performance' else 1))
                        if run_probe else [])
    if (not isinstance(probe_frames, list)
            or any(type(n) is not int or n < 0 or n >= warmup + scored + tail for n in probe_frames)
            or probe_frames != sorted(set(probe_frames))
            or bool(probe_frames) is not run_probe):
        raise ValueError('invalid run-counter frame selection')
    if component and (len(config['stages']) != 1 or tail):
        raise ValueError('component must be isolated without a tail')
    if kind == 'performance' and scored < 2:
        raise ValueError('completion intervals require at least two timestamps')
    spec = common.read(Path(bundle) / 'plans/executors.json')[executor]
    doc = {'kind': kind, 'configuration': deepcopy(config), 'stages': deepcopy(config['stages']),
           'environment': deepcopy(protocol()['environment']), 'experiment': protocol()['experiment'],
           'warmup': warmup, 'scored': scored, 'tail': tail, 'input_indices': indices,
           'executor': executor, 'executor_path': spec['path'], 'executor_sha256': spec['sha256'],
           'native_warmup_outside_results': False, 'run_probe': run_probe,
           'probe_frames': deepcopy(probe_frames), **labels}
    doc['bundle_id'] = common.read(Path(bundle) / 'manifest.json')['bundle_id']
    doc['task_id'] = common.digest(doc)[:24]
    return doc


def validate_document(doc, bundle):
    labels = {k: v for k, v in doc.items() if k not in _FIELDS}
    expected = make_task(doc['kind'], doc['configuration'], bundle, warmup=doc['warmup'],
                         scored=doc['scored'], tail=doc['tail'], indices=doc['input_indices'],
                         executor=doc['executor'], run_probe=doc['run_probe'],
                         probe_frames=doc['probe_frames'], **labels)
    if expected != doc:
        raise ValueError('diagnostic task identity differs')
    return expected


def native_command(document, bundle, remote_bundle, remote_job):
    validate_document(document, bundle)
    from cpu_vta_partitioning.measurement.tasks import native_command as base_command
    argv, inputs = base_command(document, bundle, remote_bundle, remote_job)
    cfg = document['configuration']
    argv[argv.index(remote_bundle + '/runner/vta_stage_pipeline_runner')] = remote_bundle + '/' + document['executor_path']
    argv[argv.index('--cpu-stage-gate') + 1] = cfg['execution_mode']
    loader = argv.index(remote_bundle + '/runtime/ld-linux-aarch64.so.1')
    argv[loader:loader] = ['TVM_BIND_THREADS=1', 'TVM_EXCLUDE_WORKER0=1']
    argv += ['--allow-cpu-affinity-superset', '--control-cpu-affinity', ','.join(map(str, cfg['control_mask']))]
    if cfg['affinity_mode'] == 'pinned':
        argv.append('--cpu-worker-pinning')
    if document['executor'] == 'core_model':
        for index, stage in enumerate(cfg['stages']):
            if stage['device'] == 'cpu':
                argv += [f'--stage{index}-cpu-worker-map', ','.join(map(str, stage['worker_core_map']))]
    if document['kind'].startswith('qualify'):
        argv.append('--resource-affinity-probe')
    if document['run_probe']:
        argv += ['--run-thread-counters-json', remote_job + '/run_threads.json',
                 '--run-thread-counters-frames', ','.join(map(str, document['probe_frames']))]
    if '--process-cpu-affinity' in argv:
        raise ValueError('process affinity would overwrite pinned compute threads')
    return argv, inputs


def validate_summary(rawdir, doc, bundle, *, sched_schedstats=None):
    validate_document(doc, bundle)
    common = api(bundle)
    from cpu_vta_partitioning.measurement import frames, input_sequence, correctness
    from cpu_strategy.measurement import isolated_components, whole_pipeline_window, thread_activity
    rawdir = Path(rawdir)
    rows = frames(rawdir / 'result.jsonl', sum(doc[k] for k in ('warmup', 'scored', 'tail')))
    if any(type(row.get(k)) is not int for row in rows for k in ('frame_id', 'completion_index', 'input_index')):
        raise ValueError('noninteger frame/input identity')
    input_sequence(rows, doc)
    if any(row.get('queue_timing_enabled') is not False
           or row.get('defer_result_write_enabled') is not True for row in rows):
        raise ValueError('actual queue probe or deferred output switch differs')
    cfg = doc['configuration']
    report = common.read(rawdir / 'affinity.json')
    ledger, idle = _affinity(report, cfg, doc['kind'].startswith('qualify'))
    if any(row.get('cpu_stage_gate_mode', 'off') != cfg['execution_mode'] for row in rows):
        raise ValueError('native CPU gate differs')
    if doc['kind'].startswith('qualify'):
        correctness(rows, doc, bundle, rawdir)
        result = {'correct': True}
    elif doc['kind'] == 'calibration':
        result = isolated_components(rows, doc['warmup'], doc['scored'])
    else:
        result = whole_pipeline_window(rows, doc['warmup'], doc['scored'], doc['tail'])
    if doc['run_probe']:
        if not (rawdir / 'run_threads.json').is_file():
            raise ValueError('required run_threads.json is missing')
        from ..run_scope.analysis import validate_probe
        result['run_counter_quality'] = validate_probe(common.read(rawdir / 'run_threads.json'), doc, report, rows,
                                                       sched_schedstats=sched_schedstats)
    elif (rawdir / 'run_threads.json').exists():
        raise ValueError('probe-off session unexpectedly produced run counters')
    return {**result, 'thread_activity': thread_activity(report, cfg['stages']),
            'thread_ledger': ledger, 'idle_auxiliary_threads': idle}


class Evidence:
    def __init__(self, root, bundle):
        self.root, self.bundle = Path(root).resolve(), Path(bundle).resolve()
        self.common = api(bundle)

    def summary(self, value, doc, boot):
        common = self.common
        validate_document(doc, self.bundle)
        name = f'accepted/{boot}/{doc["task_id"]}.json'
        path = common.relative_file(self.root, name)
        ref = value.get('_receipt', {})
        if ref.get('path') != name or not path.is_file() or common.sha256(path) != ref.get('sha256'):
            raise ValueError('receipt missing or modified')
        receipt = common.read(path)
        if receipt['task'] != doc or receipt['boot_id'] != boot:
            raise ValueError('receipt task or boot differs')
        files = receipt['files']
        raw = self.root / 'raw' / boot / doc['task_id']
        if not files or any(not common.relative_file(self.root, p).is_relative_to(raw) for p in files):
            raise ValueError('receipt file escaped task scope')
        common.verify_inventory(self.root, files)
        logs = [common.relative_file(self.root, p) for p in files if Path(p).name == 'result.jsonl']
        if len(logs) != 1:
            raise ValueError('missing/duplicate frame evidence')
        rawdir = logs[0].parent
        bound = lambda p: str(p.relative_to(self.root)) in files
        for leaf in ['affinity.json'] + (['run_threads.json'] if doc['run_probe'] else []):
            if not bound(rawdir / leaf):
                raise ValueError('unbound diagnostic artifact: ' + leaf)
        if doc['kind'].startswith('qualify'):
            from cpu_vta_partitioning.measurement import frames
            for row in frames(logs[0], sum(doc[k] for k in ('warmup', 'scored', 'tail'))):
                for dump in row['stage_raw_outputs']:
                    for output in dump['output_files']:
                        if not bound(rawdir / 'outputs' / ('run_' + str(row['completion_index'])) / Path(output['path']).name):
                            raise ValueError('unbound correctness output')
        calculated = validate_summary(rawdir, doc, self.bundle, sched_schedstats=receipt.get('sched_schedstats'))
        if receipt['summary'] != calculated or calculated != {k: v for k, v in value.items() if k != '_receipt'}:
            raise ValueError('summary differs from raw evidence')
        return receipt


def _measure(self, doc, bundle, remote_bundle, run_dir, boot):
    validate_document(doc, bundle)
    common = api(bundle)
    from cpu_vta_partitioning.prepare.payload import job_storage_bytes
    from cpu_vta_partitioning.transport import TransportError
    from cpu_vta_partitioning.transport.scripts import session_script
    root = Path(run_dir) / 'raw' / boot / doc['task_id']
    root.mkdir(parents=True, exist_ok=True)
    remote, q = self.job_path(doc, run_dir, boot), shlex.quote
    expected = common.digest(doc)
    taskfile = root / 'task.json'
    if taskfile.exists() and common.read(taskfile) != doc:
        raise ValueError('raw task identity collision')
    if not taskfile.exists():
        common.write(taskfile, doc)
    if self.boot_id() != boot:
        raise ValueError('board boot changed before task')
    self.command('mkdir -p ' + q(remote))
    argv, inputs = native_command(doc, bundle, remote_bundle, remote)
    (root / 'inputs.txt').write_text(inputs)
    (root / 'execute.sh').write_text(session_script(self.config['remote_base'], remote, boot, expected, argv))
    existing = self.command('test -f ' + q(remote + '/task_hash') + ' && cat ' + q(remote + '/task_hash') + ' || true').strip()
    if existing and existing != expected:
        raise ValueError('remote task hash differs')
    try:
        if not existing:
            margin = common.read(Path(bundle) / 'plans/board_execution.json')['storage_margin_bytes']
            self.check_space(remote, job_storage_bytes(doc, common.read(Path(bundle) / 'plans/programs.json')) + margin)
            self.upload(root / 'inputs.txt', remote + '/inputs.txt')
            self.upload(root / 'execute.sh', remote + '/execute.sh')
            self.command('cd ' + q(remote) + ' && nohup sh execute.sh </dev/null > launcher.log 2>&1 &')
        deadline = time.monotonic() + 660
        while True:
            result = self.command('test -f ' + q(remote + '/exit_code') + ' && cat ' + q(remote + '/exit_code') + ' || true').strip()
            if result:
                break
            if time.monotonic() >= deadline:
                raise TransportError('native task deadline exceeded')
            time.sleep(.5)
    except TransportError as error:
        raise TransportError(str(error) + '; retained ' + self.failure_snapshot(remote, root)) from error
    output = self.collect(remote, root)
    if ((output / 'boot_id').read_text().strip() != boot
            or (output / 'task_hash').read_text().strip() != expected or self.boot_id() != boot):
        raise ValueError('collected task or boot differs')
    if result != '0':
        raise ValueError('native execution failed; evidence retained: ' + str(output))
    if (output / 'inputs.txt').read_text() != inputs:
        raise ValueError('native input sequence differs')
    return output


@lru_cache(maxsize=4)
def _ssh_class(bundle):
    load_legacy(bundle)
    from cpu_strategy.policy.transport import SSH as FrozenSSH
    class DiagnosticSSH(FrozenSSH):
        measure = _measure
    return DiagnosticSSH


def SSH(config, *, execute_board=False, bundle=None):
    result = _ssh_class(str(Path(bundle or DEFAULT_PARENT).resolve()))(config, execute_board=execute_board)
    result.options += ['-o', 'ControlMaster=auto', '-o', 'ControlPersist=60',
                       '-o', 'ControlPath=/tmp/cpu-vta-core-model-%C']
    return result
