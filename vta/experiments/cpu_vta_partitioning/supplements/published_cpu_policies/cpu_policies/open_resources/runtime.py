"""Nine-context task construction and raw, byte-bound native evidence.

Only transport lifecycle and clock parsers are reused from the frozen runner.
The old validator is intentionally not used: it conflates host-core binding
with a reserved compute core.
"""
from copy import deepcopy
from functools import lru_cache
import math
from pathlib import Path
import shlex
import time

from .common import api, load_legacy, protocol, DEFAULT_PARENT

_FIELDS = {'kind', 'configuration', 'stages', 'environment', 'experiment', 'warmup',
           'scored', 'tail', 'input_indices', 'executor', 'executor_path',
           'executor_sha256', 'bundle_id', 'task_id', 'native_warmup_outside_results'}


def make_task(kind, config, bundle, *, warmup=0, scored=1, tail=0, indices=None, **labels):
    from .domain import validate_candidate
    common = api(bundle)
    if _FIELDS.intersection(labels):
        raise ValueError('labels override task execution identity')
    if kind not in ('calibration', 'qualify_component', 'qualify_pipeline', 'performance', 'traffic'):
        raise ValueError('unsupported measurement kind')
    if any(type(n) is not int or n < 0 for n in (warmup, scored, tail)) or scored < 1:
        raise ValueError('invalid measurement window')
    indices = list(indices if indices is not None else [0])
    if indices != [0] or type(indices[0]) is not int:
        raise ValueError('this frozen experiment uses input zero only')
    component = kind in ('calibration', 'qualify_component', 'traffic')
    validate_candidate(config, common.read(Path(bundle) / 'plans/programs.json'), component=component)
    if component and (len(config['stages']) != 1 or tail):
        raise ValueError('component task must have one isolated stage and no tail')
    if kind == 'traffic' and (config['stages'][0]['device'] != 'vta' or (warmup, scored, tail) != (1, 1, 0)):
        raise ValueError('traffic requires one VTA stage, one warmup and one measured run')
    if not component and scored < (1 if kind.startswith('qualify') else 2):
        raise ValueError('a pipeline interval requires at least two timestamps')
    spec = common.read(Path(bundle) / 'plans/executors.json')['policy']
    doc = {'kind': kind, 'configuration': deepcopy(config), 'stages': deepcopy(config['stages']),
           'environment': deepcopy(protocol()['environment']), 'experiment': protocol()['experiment'],
           'warmup': warmup, 'scored': scored, 'tail': tail, 'input_indices': indices,
           'executor': 'policy', 'executor_path': spec['path'], 'executor_sha256': spec['sha256'],
           'native_warmup_outside_results': kind == 'traffic', **labels}
    doc['bundle_id'] = common.read(Path(bundle) / 'manifest.json')['bundle_id']
    doc['task_id'] = common.digest(doc)[:24]
    return doc


def validate_document(document, bundle):
    labels = {k: v for k, v in document.items() if k not in _FIELDS}
    expected = make_task(document['kind'], document['configuration'], bundle,
                         warmup=document['warmup'], scored=document['scored'],
                         tail=document['tail'], indices=document['input_indices'], **labels)
    if document != expected:
        raise ValueError('task, executor, context or window identity differs')
    return document


def native_command(document, bundle, remote_bundle, remote_job):
    validate_document(document, bundle)
    from cpu_vta_partitioning.measurement.tasks import native_command as base_command
    config = document['configuration']
    argv, inputs = base_command(document, bundle, remote_bundle, remote_job)
    argv[argv.index(remote_bundle + '/runner/vta_stage_pipeline_runner')] = remote_bundle + '/' + document['executor_path']
    argv[argv.index('--cpu-stage-gate') + 1] = config['execution_mode']
    loader = argv.index(remote_bundle + '/runtime/ld-linux-aarch64.so.1')
    argv[loader:loader] = ['TVM_BIND_THREADS=1', 'TVM_EXCLUDE_WORKER0=1']
    argv += ['--allow-cpu-affinity-superset', '--control-cpu-affinity', ','.join(map(str, config['control_mask']))]
    if config['affinity_mode'] == 'pinned':
        argv.append('--cpu-worker-pinning')
    if document['kind'].startswith('qualify'):
        argv.append('--resource-affinity-probe')
    if document['kind'] == 'traffic':
        # Native warmup clears profiler counters before the one returned frame.
        # These rows are deliberately distinct from timed calibration windows.
        argv[argv.index('--runs') + 1] = '1'
        argv[argv.index('--warmup-runs') + 1] = '1'
        argv += ['--serial', '--vta-runtime-profile-dir', remote_job + '/traffic_profile']
        argv[argv.index('--output-mode') + 1] = 'raw'
    if '--process-cpu-affinity' in argv:
        raise ValueError('process-wide affinity would overwrite fixed workers')
    return argv, inputs


def _finite_nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError('invalid ' + name)
    return value


def _affinity(report, config, qualification):
    if report.get('cpu_worker_pinning') is not (config['affinity_mode'] == 'pinned'):
        raise ValueError('native fixed-worker mode differs')
    if len(report.get('stages', [])) != len(config['stages']):
        raise ValueError('affinity stage coverage differs')
    ledger, idle = [], []
    stage_tids = set()
    for s, row in zip(config['stages'], report['stages']):
        if row['requested_cpus'] != s['mask'] or row['active_runtime_threads'] != s['threads']:
            raise ValueError('requested CPU group or thread count differs')
        tids, entries = row['thread_ids'], row['effective_affinities']
        if (any(type(t) is not int or t <= 0 for t in tids)
                or type(row['caller_tid']) is not int or len(entries) != len(tids)
                or len(tids) != len(set(tids)) or {e['tid'] for e in entries} != set(tids)
                or row['caller_tid'] not in tids):
            raise ValueError('invalid thread ownership')
        if stage_tids.intersection(tids):
            raise ValueError('different stage pools share a thread identity')
        stage_tids.update(tids)
        for e in entries:
            if (type(e['tid']) is not int or not e['cpus']
                    or any(type(c) is not int for c in e['cpus'])
                    or e['cpus'] != sorted(set(e['cpus']))
                    or not set(e['cpus']) <= set(s['mask'])):
                raise ValueError('thread escaped its allowed stage group')
            delta = _finite_nonnegative(e.get('cpu_time_delta_ms'), 'whole-session thread CPU increment')
            item = {'stage': s['program'], 'role': 'caller' if e['tid'] == row['caller_tid'] else 'worker',
                    'tid': e['tid'], 'requested_mask': s['mask'], 'actual_mask': e['cpus'],
                    'cpu_time_delta_ms': delta, 'scope': 'whole_session_including_warmup_and_tail'}
            if s['device'] == 'cpu':
                item['participation'] = 'configured_compute_slot'
                item['logical_worker'] = s['mask'].index(e['cpus'][0]) if len(e['cpus']) == 1 else None
                if delta == 0:
                    item['activity_note'] = 'no_observed_increment_not_proof_of_no_work'
            elif delta == 0 and e['tid'] != row['caller_tid']:
                item['participation'] = 'VTA_noncompute_helper_without_observed_increment'
                idle.append(item)
            ledger.append(item)
        if s['device'] == 'cpu':
            if len(tids) != s['threads'] or any(len(e['cpus']) != 1 for e in entries):
                raise ValueError('fixed CPU pool is not one thread per selected core')
            if sorted(e['cpus'][0] for e in entries) != s['worker_core_map']:
                raise ValueError('CPU worker/core mapping is not bijective')
            caller = next(e for e in entries if e['tid'] == row['caller_tid'])
            if caller['cpus'] != [s['worker_core_map'][0]]:
                raise ValueError('CPU caller is not logical worker zero')
            if qualification:
                slots = row.get('worker_slots', [])
                if len(slots) != s['threads'] or sorted(v['slot'] for v in slots) != list(range(s['threads'])):
                    raise ValueError('missing logical CPU worker probe')
                if len({v['tid'] for v in slots}) != len(slots):
                    raise ValueError('duplicate logical worker thread')
                for v in slots:
                    if v['cpus'] != [s['worker_core_map'][v['slot']]] or v['tid'] not in tids:
                        raise ValueError('logical worker mapping differs')
                if next(v for v in slots if v['slot'] == 0)['tid'] != row['caller_tid']:
                    raise ValueError('logical worker zero is not the caller')
    if len(config['stages']) > 1:
        controls = report.get('control_threads', [])
        if (len(controls) != 2 or {c['role'] for c in controls} != {'collector', 'producer'}
                or len({c['tid'] for c in controls}) != 2
                or any(type(c['tid']) is not int or c['tid'] <= 0
                       or any(type(core) is not int for core in c['cpus'])
                       or c['cpus'] != config['control_mask'] or c['tid'] in stage_tids for c in controls)):
            raise ValueError('producer/main-collector affinity or identity differs')
        for c in controls:
            ledger.append({'role': c['role'], 'tid': c['tid'], 'requested_mask': config['control_mask'],
                           'actual_mask': c['cpus'], 'cpu_time_delta_ms': None,
                           'scope': 'not_collected_by_frozen_executor'})
    return ledger, idle


def traffic_summary(rawdir):
    common = api()
    path = Path(rawdir) / 'traffic_profile/single_run_status.json'
    if not path.is_file():
        raise ValueError('VTA profiler hook unavailable: missing single-run counter file')
    counters = common.read(path)
    keys = ('synchronize_load_bytes', 'synchronize_store_bytes', 'synchronize_calls')
    if any(type(counters.get(k)) is not int or counters[k] <= 0 for k in keys):
        raise ValueError('missing/empty VTA DMA request counters; never substitute zero')
    total = common.read(Path(rawdir) / 'traffic_profile/benchmark_totals_status.json')
    if any(total.get(k) != counters[k] for k in keys):
        raise ValueError('traffic counters do not represent exactly one post-warmup run')
    return {'load_bytes': counters[keys[0]], 'store_bytes': counters[keys[1]],
            'synchronize_calls': counters[keys[2]], 'hooks_present': True,
            'count': 1, 'traffic_scope': 'VTA_LOAD_STORE_requested_DMA_payload_not_AXI_bus_total'}


def validate_summary(rawdir, document, bundle):
    validate_document(document, bundle)
    common = api(bundle)
    from cpu_vta_partitioning.measurement import frames, input_sequence, correctness
    from cpu_strategy.measurement import thread_activity, isolated_components, whole_pipeline_window
    total = document['scored'] if document['kind'] == 'traffic' else sum(document[k] for k in ('warmup', 'scored', 'tail'))
    rows = frames(Path(rawdir) / 'result.jsonl', total)
    if any(type(row.get(k)) is not int for row in rows for k in ('frame_id', 'completion_index', 'input_index')):
        raise ValueError('noninteger frame/input identities')
    input_sequence(rows, {**document, 'warmup': 0} if document['kind'] == 'traffic' else document)
    config = document['configuration']
    report = common.read(Path(rawdir) / 'affinity.json')
    ledger, idle = _affinity(report, config, document['kind'].startswith('qualify'))
    if any(r.get('cpu_stage_gate_mode', 'off') != config['execution_mode'] for r in rows):
        raise ValueError('native CPU gate differs from frozen configuration')
    activity = thread_activity(report, config['stages'])
    if document['kind'].startswith('qualify'):
        correctness(rows, document, bundle, rawdir)
        result = {'correct': True}
    elif document['kind'] == 'calibration':
        result = isolated_components(rows, document['warmup'], document['scored'])
    elif document['kind'] == 'traffic':
        result = traffic_summary(rawdir)
    else:
        result = whole_pipeline_window(rows, document['warmup'], document['scored'], document['tail'])
    return {**result, 'thread_activity': activity, 'thread_ledger': ledger, 'idle_auxiliary_threads': idle,
            'thread_ledger_scope': 'reported_stage_pools_and_main_producer_not_global_TID_census',
            'unreported_main_pool_auxiliary_threads': 'not_collected_by_frozen_executor'}


class Evidence:
    def __init__(self, root, bundle):
        self.root, self.bundle = Path(root).resolve(), Path(bundle).resolve()
        self.common = api(bundle)
        self.bundle_id = self.common.read(self.bundle / 'manifest.json')['bundle_id']

    def summary(self, value, document, boot):
        common = self.common
        expected = validate_document(document, self.bundle)
        if expected['bundle_id'] != self.bundle_id:
            raise ValueError('wrong reference bundle')
        name = f"accepted/{boot}/{expected['task_id']}.json"
        path = common.relative_file(self.root, name)
        ref = value.get('_receipt', {})
        if ref.get('path') != name or not path.is_file() or common.sha256(path) != ref.get('sha256'):
            raise ValueError('receipt missing or modified')
        receipt = common.read(path)
        if receipt['task'] != expected or receipt['boot_id'] != boot:
            raise ValueError('receipt belongs to another task or boot')
        files = receipt['files']
        raw = common.relative_file(self.root, f"raw/{boot}/{expected['task_id']}")
        if not files or any(not common.relative_file(self.root, p).is_relative_to(raw) for p in files):
            raise ValueError('raw receipt escaped task scope')
        common.verify_inventory(self.root, files)
        logs = [common.relative_file(self.root, p) for p in files if Path(p).name == 'result.jsonl']
        if len(logs) != 1:
            raise ValueError('missing/duplicate frame evidence')
        rawdir = logs[0].parent
        bound = lambda p: str(p.relative_to(self.root)) in files
        if not bound(rawdir / 'affinity.json'):
            raise ValueError('unbound affinity evidence')
        if expected['kind'] == 'traffic':
            for leaf in ('single_run_status.json', 'benchmark_totals_status.json'):
                if not bound(rawdir / 'traffic_profile' / leaf):
                    raise ValueError('unbound profiler evidence')
        if expected['kind'].startswith('qualify'):
            from cpu_vta_partitioning.measurement import frames
            for row in frames(logs[0], sum(expected[k] for k in ('warmup', 'scored', 'tail'))):
                for dump in row['stage_raw_outputs']:
                    for output in dump['output_files']:
                        p = rawdir / 'outputs' / ('run_' + str(row['completion_index'])) / Path(output['path']).name
                        if not bound(p):
                            raise ValueError('unbound correctness output')
        calculated = validate_summary(rawdir, expected, self.bundle)
        if receipt['summary'] != calculated or calculated != {k: v for k, v in value.items() if k != '_receipt'}:
            raise ValueError('summary differs from raw evidence')
        return receipt


def _measure(self, document, bundle, remote_bundle, run_dir, boot):
    validate_document(document, bundle)
    common = api(bundle)
    from cpu_vta_partitioning.prepare.payload import job_storage_bytes
    from cpu_vta_partitioning.transport import TransportError
    from cpu_vta_partitioning.transport.scripts import session_script
    root = Path(run_dir) / 'raw' / boot / document['task_id']
    root.mkdir(parents=True, exist_ok=True)
    remote, q = self.job_path(document, run_dir, boot), shlex.quote
    expected = common.digest(document)
    taskfile = root / 'task.json'
    if taskfile.exists() and common.read(taskfile) != document:
        raise ValueError('raw task identity collision')
    if not taskfile.exists():
        common.write(taskfile, document)
    if self.boot_id() != boot:
        raise ValueError('board boot changed before task')
    self.command('mkdir -p ' + q(remote))
    argv, inputs = native_command(document, bundle, remote_bundle, remote)
    (root / 'inputs.txt').write_text(inputs)
    (root / 'execute.sh').write_text(session_script(self.config['remote_base'], remote, boot, expected, argv))
    existing = self.command('test -f ' + q(remote + '/task_hash') + ' && cat ' + q(remote + '/task_hash') + ' || true').strip()
    if existing and existing != expected:
        raise ValueError('remote task hash differs')
    try:
        if not existing:
            margin = common.read(Path(bundle) / 'plans/board_execution.json')['storage_margin_bytes']
            self.check_space(remote, job_storage_bytes(document, common.read(Path(bundle) / 'plans/programs.json')) + margin)
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
        raise ValueError('collected task or boot identity differs')
    if result != '0':
        raise ValueError('native execution failed; evidence retained: ' + str(output))
    if (output / 'inputs.txt').read_text() != inputs:
        raise ValueError('native input sequence differs')
    return output


@lru_cache(maxsize=4)
def _ssh_class(bundle):
    load_legacy(bundle)
    from cpu_strategy.policy.transport import SSH as FrozenSSH
    class OpenSSH(FrozenSSH):
        measure = _measure
    return OpenSSH


def SSH(config, *, execute_board=False, bundle=None):
    instance = _ssh_class(str(Path(bundle or DEFAULT_PARENT).resolve()))(config, execute_board=execute_board)
    instance.options += ['-o', 'ControlMaster=auto', '-o', 'ControlPersist=60',
                         '-o', 'ControlPath=/tmp/cpu-vta-open-%C']
    return instance
