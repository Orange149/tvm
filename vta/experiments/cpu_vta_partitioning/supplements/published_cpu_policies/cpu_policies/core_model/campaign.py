"""Common isolated profiles, frozen micro-domain ranking, then one-boot validation."""
from copy import deepcopy
from pathlib import Path
import random
import statistics

from .common import api, identity, protocol
from .runtime import Evidence, make_task, validate_summary

STOPPED = {'STOPPED_BOOT_CHANGED', 'STOPPED_INVALID_RESULT', 'STOPPED_IDENTITY'}


def component_plan(bundle):
    common = api(bundle)
    cases = common.read(Path(bundle) / 'plans/diagnosis_configurations.json')['components']
    rows = []
    for name, cfg in sorted(cases.items()):
        stage = cfg['stages'][0]
        probe = stage['device'] == 'cpu' and stage['threads'] > 1
        for kind, block, window in (('qualify_component', 'qualify', (0, 1, 0)),
                                     ('calibration', 'component', (2, 8, 0))):
            doc = make_task(kind, cfg, bundle, warmup=window[0], scored=window[1], tail=window[2],
                            run_probe=probe, diagnosis_case=name, diagnosis_block=block, detailed=False)
            rows.append({'name': 'components/' + name + '/' + block, 'document': doc})
    if len(rows) != 22:
        raise ValueError('eleven component qualifications and profiles required')
    return rows


def bridge_task(bundle):
    cfg = api(bundle).read(Path(bundle) / 'plans/diagnosis_configurations.json')['pipelines']['static_first']
    return {'name': 'bridge/old', 'document': make_task('performance', cfg, bundle,
            warmup=8, scored=16, tail=4, executor='policy', diagnosis_case='static_first',
            diagnosis_block='old_bridge', detailed=False)}


def pipeline_plan(bundle, pool, order):
    if len(pool) > 12 or len(order) != len(set(order)) or set(order) != set(pool):
        raise ValueError('invalid frozen performance pool')
    result = []
    for key in order:
        for kind, block, window in (('qualify_pipeline', 'qualify', (0, 1, 0)),
                                     ('performance', 'performance', (8, 16, 4))):
            result.append({'name': 'pipelines/' + key + '/' + block,
                'document': make_task(kind, pool[key], bundle, warmup=window[0],
                                     scored=window[1], tail=window[2],
                                     diagnosis_case=key, diagnosis_block=block,
                                     candidate_id=key, detailed=False)})
    return result


def task_plan(bundle, root=None):
    rows = component_plan(bundle)
    if root is not None and (Path(root) / 'search_pool.json').is_file():
        c = api(bundle)
        rows += [bridge_task(bundle)] + pipeline_plan(bundle, c.read(Path(root) / 'search_pool.json'),
                                                     c.read(Path(root) / 'screening_order.json'))
    return rows


def _save(root, state):
    api(state['bundle']).write(Path(root) / 'state.json', state, replace=True)


def _pin(root, state, *names):
    common = api(state['bundle'])
    for name in names:
        path = common.relative_file(root, name)
        info = {'sha256': common.sha256(path), 'bytes': path.stat().st_size}
        if name in state['frozen_files'] and state['frozen_files'][name] != info:
            raise ValueError('frozen diagnostic artifact changed: ' + name)
        state['frozen_files'][name] = info
    _save(root, state)


def _publish(root, state, name, value):
    common = api(state['bundle'])
    intended = common.digest(value)
    pending = state.setdefault('pending_publications', {})
    if name in pending and pending[name] != intended:
        raise ValueError('pending publication changed: ' + name)
    pending[name] = intended
    _save(root, state)
    path = common.relative_file(root, name)
    if path.exists():
        if common.read(path) != value:
            raise ValueError('frozen model artifact differs: ' + name)
    else:
        common.write(path, value)
    _pin(root, state, name)
    pending.pop(name, None)
    _save(root, state)
    return value


def component_cost(doc, summary):
    row = doc['stages'][0]
    fields = ('service_ms', 'run_core_ms', 'boundary_core_ms', 'set_ms', 'run_ms', 'get_ms',
              'set_core_ms', 'get_core_ms')
    cost = {key: summary[key] for key in fields}
    if doc['run_probe']:
        quality = summary['run_counter_quality']
        if not quality['all_runtime_samples_available']:
            raise ValueError('required isolated worker runtime unavailable; no ranking can be generated')
        samples = quality['samples']
        if len(samples) != doc['scored']:
            raise ValueError('component run-scope sample coverage differs')
        workers = quality['threads']
        vector = [sum(t['runtime_delta_ns'] for t in workers if t['logical_worker'] == slot)
                  / doc['scored'] / 1_000_000 for slot in range(row['threads'])]
        if any(sum(t['logical_worker'] == slot for t in workers) != doc['scored']
               for slot in range(row['threads'])):
            raise ValueError('isolated logical worker sample coverage differs')
        total = sum(vector)
        if total <= 0:
            raise ValueError('isolated runtime vector is empty')
        cost.update(run_core_ms=total, worker_run_core_ms=vector,
                    cpu_core_weights=[v / total for v in vector],
                    run_ms=statistics.mean(t['diagnostic_run_ms'] for t in samples))
        cost['service_ms'] = cost['set_ms'] + cost['run_ms'] + cost['get_ms']
        cost['measurement_scope'] = 'run_call_worker_schedstat_with_read_envelopes_probe_wall_excluded'
    elif row['device'] == 'cpu':
        if row['threads'] != 1:
            raise ValueError('multithread CPU profiles require worker measurements')
        cost.update(worker_run_core_ms=[cost['run_core_ms']], cpu_core_weights=[1.],
                    measurement_scope='exclusive_process_CPU_time_single_compute_thread')
    return cost


def load_costs(root, state=None):
    state = state or validate_state(root)
    common = api(state['bundle'])
    from .domain import component_key
    recorder = Recorder(root, state)
    records = {}
    rows = component_plan(state['bundle'])
    for qitem, item in zip(rows[::2], rows[1::2]):
        qsummary = recorder.saved(qitem['name'], qitem['document'])
        summary = recorder.saved(item['name'], item['document'])
        doc = item['document']
        key = component_key(doc['configuration'])
        if key in records:
            raise ValueError('duplicate isolated execution signature')
        records[key] = {'configuration': doc['configuration'], 'cost': component_cost(doc, summary),
                        'measurement': {'document': doc, 'summary': summary, 'boot_id': state['boot_id']},
                        'qualification': {'document': qitem['document'], 'summary': qsummary,
                                          'boot_id': state['boot_id']}}
    if len(records) != 11:
        raise ValueError('incomplete common profile table')
    return records


def rank(root, state=None):
    root = Path(root).resolve()
    state = state or validate_state(root)
    common = api(state['bundle'])
    records = load_costs(root, state)
    frozen = common.read(root / 'costs.json')
    if frozen['costs'] != records or frozen.get('pipeline_feedback_used') is not False:
        raise ValueError('cost table differs from the isolated raw evidence')
    from .search import rank as static_rank
    bundle = Path(state['bundle'])
    result = static_rank({'costs': records}, common.read(bundle / 'plans/baseline_ddr.json'),
                         common.read(bundle / 'plans/programs.json'))
    _publish(root, state, 'rankings.json', result)
    _publish(root, state, 'search_pool.json', result['pool'])
    order = sorted(result['pool'])
    random.Random(protocol().get('seed', 20261107)).shuffle(order)
    _publish(root, state, 'screening_order.json', order)
    return result


def create(bundle, root, connection):
    from .prepare import validate_bundle
    bundle, root = Path(bundle).resolve(), Path(root).resolve()
    manifest = validate_bundle(bundle)
    if root.exists():
        raise FileExistsError('run already exists; use board resume')
    common = api(bundle)
    plan = component_plan(bundle)
    root.mkdir(parents=True)
    state = {'experiment': protocol()['experiment'], 'controller_identity': identity(),
             'bundle': str(bundle), 'bundle_id': manifest['bundle_id'],
             'bundle_manifest_sha256': common.sha256(bundle / 'manifest.json'),
             'connection': deepcopy(connection), 'created': common.now(), 'status': 'CREATED',
             'phase': 'CREATED', 'boot_id': None, 'active_boot': None, 'completed_boots': [],
             'frozen_files': {}, 'completed_tasks': 0, 'expected_tasks': 47,
             'expected_forwards': 497}
    common.write(root / 'protocol.json', protocol())
    common.write(root / 'tasks.json', plan)
    _pin(root, state, 'protocol.json', 'tasks.json')
    return state


def validate_state(root, *, validate_payload=False):
    root = Path(root).resolve()
    common = api()
    state = common.read(root / 'state.json')
    if state.get('controller_identity') != identity() or state.get('experiment') != protocol()['experiment']:
        raise ValueError('diagnostic code identity changed; use frozen controller snapshot')
    bundle = Path(state['bundle'])
    manifest = common.read(bundle / 'manifest.json')
    if (manifest['bundle_id'] != state['bundle_id']
            or manifest['controller_identity'] != identity()
            or common.sha256(bundle / 'manifest.json') != state['bundle_manifest_sha256']):
        raise ValueError('diagnostic bundle identity changed')
    common.verify_inventory(bundle, {k: v for k, v in manifest['files'].items() if k.startswith('plans/')})
    common.verify_inventory(root, state['frozen_files'])
    for name, expected in state.get('pending_publications', {}).items():
        path = common.relative_file(root, name)
        if path.exists() and common.digest(common.read(path)) != expected:
            raise ValueError('pending artifact differs: ' + name)
    if common.read(root / 'protocol.json') != protocol() or common.read(root / 'tasks.json') != component_plan(bundle):
        raise ValueError('diagnostic protocol or task plan changed')
    if state.get('active_boot') not in (None, state.get('boot_id')):
        raise ValueError('active boot identity differs')
    if state['status'] == 'COMPLETE' and (state['completed_tasks'] != state['expected_tasks']
            or state['active_boot'] is not None or state['completed_boots'] != [state['boot_id']]):
        raise ValueError('incomplete diagnostic campaign marked complete')
    if state['status'] == 'COMPLETE':
        required = ('costs.json', 'rankings.json', 'search_pool.json', 'screening_order.json',
                    'pipeline_tasks.json', 'screening.json')
        if any(name not in state['frozen_files'] for name in required):
            raise ValueError('complete campaign lacks frozen artifacts')
        pool, order = common.read(root / 'search_pool.json'), common.read(root / 'screening_order.json')
        plan = [bridge_task(bundle)] + pipeline_plan(bundle, pool, order)
        if (state['expected_tasks'] != 23 + 2 * len(pool)
                or state['expected_forwards'] != 149 + 29 * len(pool)
                or common.read(root / 'pipeline_tasks.json') != plan):
            raise ValueError('complete campaign coverage differs')
        screen = common.read(root / 'screening.json')
        if (screen['boot_id'] != state['boot_id'] or screen['rounds'] != 1
                or screen['order'] != order or set(screen['measurements']) != set(pool)):
            raise ValueError('complete campaign screening identity differs')
    if validate_payload:
        from .prepare import validate_bundle
        validate_bundle(bundle)
    return state


def _linked(root, path, receipt):
    common = api()
    return {**receipt['summary'], '_receipt': {'path': str(path.relative_to(root)), 'sha256': common.sha256(path)}}


class Recorder:
    def __init__(self, root, state, transport=None, remote=None):
        self.root, self.state = Path(root).resolve(), state
        self.transport, self.remote, self.boot = transport, remote, state['boot_id']
        self.evidence = Evidence(self.root, state['bundle'])

    def acknowledge(self, doc):
        if self.transport is None:
            return
        common = api(self.state['bundle'])
        pending = self.root / 'cleanup_pending' / self.boot / (doc['task_id'] + '.json')
        try:
            self.transport.acknowledge(doc, self.root, self.boot)
        except (OSError, RuntimeError) as error:
            common.write(pending, {'document': doc, 'boot_id': self.boot}, replace=True)
            common.event(self.root, 'cleanup_pending', task_id=doc['task_id'], error=str(error))
        else:
            pending.unlink(missing_ok=True)

    def saved(self, name, doc):
        common = api(self.state['bundle'])
        path = common.relative_file(self.root, f'records/{self.boot}/{name}.json')
        receipt = self.root / 'accepted' / self.boot / (doc['task_id'] + '.json')
        if path.exists():
            record = common.read(path)
            if record['document'] != doc or record['boot_id'] != self.boot:
                raise ValueError('record belongs to another task/boot')
            value = record['summary']
            self.evidence.summary(value, doc, self.boot)
        else:
            if receipt.exists():
                value = _linked(self.root, receipt, common.read(receipt))
                self.evidence.summary(value, doc, self.boot)
            else:
                if self.transport is None:
                    raise ValueError('diagnostic task is missing: ' + name)
                if self.transport.boot_id() != self.boot:
                    raise ValueError('board boot changed before measurement')
                print('[逐核模型验证] ' + name, flush=True)
                common.event(self.root, 'session_start', task_id=doc['task_id'], boot_id=self.boot)
                raw = Path(self.transport.measure(doc, self.state['bundle'], self.remote, self.root, self.boot)).resolve()
                if self.transport.boot_id() != self.boot or not raw.is_relative_to(self.root / 'raw' / self.boot / doc['task_id']):
                    raise ValueError('result belongs to another boot or raw directory')
                kernel = self.state.get('sched_schedstats')
                summary = validate_summary(raw, doc, self.state['bundle'], sched_schedstats=kernel)
                common.sync_tree(raw)
                files = {str(raw.relative_to(self.root) / n): info for n, info in common.inventory(raw).items()}
                common.write(receipt, {'task': doc, 'boot_id': self.boot, 'summary': summary,
                                       'files': files, 'sched_schedstats': kernel})
                value = _linked(self.root, receipt, common.read(receipt))
                self.evidence.summary(value, doc, self.boot)
                common.event(self.root, 'session_accepted', task_id=doc['task_id'], boot_id=self.boot)
            common.write(path, {'document': doc, 'boot_id': self.boot, 'summary': value})
        _pin(self.root, self.state, str(path.relative_to(self.root)), str(receipt.relative_to(self.root)))
        self.acknowledge(doc)
        return value


def validated_records(root, state=None):
    state = state or validate_state(root)
    recorder = Recorder(root, state)
    return {item['name']: recorder.saved(item['name'], item['document']) for item in task_plan(state['bundle'], root)}


def run_current(root, transport):
    root = Path(root).resolve()
    common = api()
    from cpu_vta_partitioning.transport import TransportError
    with common.locked(root):
        state = validate_state(root, validate_payload=True)
        if state['status'] == 'COMPLETE':
            validated_records(root, state)
            return state
        if state['status'] in STOPPED:
            raise RuntimeError('campaign is stopped; do not silently resume ' + state['status'])
        try:
            boot = transport.boot_id()
            if state['boot_id'] not in (None, boot):
                state.update(status='STOPPED_BOOT_CHANGED', last_error='one-boot diagnosis cannot mix another boot')
                _save(root, state)
                raise RuntimeError(state['last_error'])
            state.update(boot_id=boot, active_boot=boot, phase='PREFLIGHT', status='PREFLIGHT')
            state.pop('last_error', None)
            _save(root, state)
            transport.wait_idle(boot)
            transport.mount_storage()
            remote = transport.deploy(state['bundle'], root)
            transport.verify_payload(remote, boot)
            transport.qualify_environment(remote, boot, root, state['bundle'])
            environment = 'environment-' + boot + '.json'
            if (root / environment).exists():
                _pin(root, state, environment)
            capabilities = root / 'scheduler_capabilities.json'
            if capabilities.exists():
                kernel = common.read(capabilities)['sched_schedstats']
            else:
                raw = transport.command('if test -r /proc/sys/kernel/sched_schedstats; then cat /proc/sys/kernel/sched_schedstats; else echo unavailable; fi').strip()
                kernel = int(raw) if raw in ('0', '1') else None
                common.write(capabilities, {'boot_id': boot, 'sched_schedstats': kernel,
                                           'raw': raw, 'kernel_settings_changed': False})
            state['sched_schedstats'] = kernel
            _pin(root, state, 'scheduler_capabilities.json')
            recorder = Recorder(root, state, transport, remote)
            for path in sorted((root / 'cleanup_pending' / boot).glob('*.json')):
                doc = common.read(path)['document']
                receipt = root / 'accepted' / boot / (doc['task_id'] + '.json')
                recorder.evidence.summary(_linked(root, receipt, common.read(receipt)), doc, boot)
                recorder.acknowledge(doc)
            state.update(status='MEASURING', phase='MEASURING')
            _save(root, state)
            for n, item in enumerate(component_plan(state['bundle']), 1):
                recorder.saved(item['name'], item['document'])
                state['completed_tasks'] = n
                _save(root, state)
            costs = load_costs(root, state)
            _publish(root, state, 'costs.json', {'costs': costs,
                     'source': 'fresh_common_isolated_profiles', 'boot_id': boot,
                     'pipeline_feedback_used': False})
            state.update(status='RANKING', phase='RANKING')
            _save(root, state)
            rankings = rank(root, state)
            pool = common.read(root / 'search_pool.json')
            order = common.read(root / 'screening_order.json')
            plan = [bridge_task(state['bundle'])] + pipeline_plan(state['bundle'], pool, order)
            _publish(root, state, 'pipeline_tasks.json', plan)
            state.update(expected_tasks=23 + 2 * len(pool), expected_forwards=149 + 29 * len(pool),
                         completed_pipeline_configurations=0, status='PIPELINE', phase='PIPELINE')
            _save(root, state)
            measurements = {}
            for n, item in enumerate(plan, 23):
                value = recorder.saved(item['name'], item['document'])
                doc = item['document']
                if doc['diagnosis_block'] == 'performance':
                    measurements[doc['candidate_id']] = {'document': doc, 'summary': value, 'boot_id': boot}
                    state['completed_pipeline_configurations'] = len(measurements)
                state['completed_tasks'] = n
                _save(root, state)
            _publish(root, state, 'screening.json', {'boot_id': boot, 'rounds': 1,
                     'order': order, 'measurements': measurements})
            validated_records(root, state)
            if transport.boot_id() != boot:
                raise ValueError('board boot changed before final acceptance')
            transport.verify_payload(remote, boot)
            transport.idle_command('sync', boot=boot, timeout=60)
            state.update(status='ANALYZING', phase='ANALYZING')
            _save(root, state)
            from .analysis import analyze
            state['analysis'] = analyze(root)
            state.update(status='COMPLETE', phase='COMPLETE', active_boot=None, completed_boots=[boot])
            state.pop('last_error', None)
            _save(root, state)
            common.sync_tree(root)
            return state
        except TransportError as error:
            state.update(status='INTERRUPTED', last_error=str(error))
            _save(root, state)
            common.event(root, 'transport_interrupted', error=str(error))
            raise
        except Exception as error:
            if state['status'] not in STOPPED:
                state.update(status='STOPPED_INVALID_RESULT', last_error=str(error))
                _save(root, state)
            common.event(root, 'diagnosis_stopped', error=str(error))
            raise


def status(root):
    state = validate_state(root)
    return {k: state.get(k) for k in ('status', 'phase', 'boot_id', 'completed_tasks',
                                     'expected_tasks', 'expected_forwards', 'last_error', 'analysis')}
