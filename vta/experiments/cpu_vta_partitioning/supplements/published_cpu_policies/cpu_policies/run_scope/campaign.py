"""Twenty-two tasks on one boot, independent of calibration/search campaigns."""
from copy import deepcopy
from pathlib import Path

from .common import api, identity, protocol
from .runtime import Evidence, make_task, validate_summary

STOPPED = {'STOPPED_BOOT_CHANGED', 'STOPPED_INVALID_RESULT', 'STOPPED_IDENTITY'}


def task_plan(bundle):
    common = api(bundle)
    configs = common.read(Path(bundle) / 'plans/diagnosis_configurations.json')
    result = []
    for case in ('cpu00_t3', 'cpu00_t4', 'cpu05_t1', 'cpu05_t4'):
        cfg = configs['components'][case]
        doc = make_task('qualify_component', cfg, bundle, run_probe=True,
                        diagnosis_case=case, diagnosis_block='qualify', detailed=False)
        result.append({'name': case + '/qualify', 'document': doc})
        for block, probe in (('component_off', False), ('component_probe', True)):
            doc = make_task('calibration', cfg, bundle, warmup=2, scored=8, run_probe=probe,
                            diagnosis_case=case, diagnosis_block=block, detailed=False)
            result.append({'name': case + '/' + block, 'document': doc})
    for case in ('static_first', 'history_host2'):
        cfg = configs['pipelines'][case]
        doc = make_task('qualify_pipeline', cfg, bundle, run_probe=True,
                        diagnosis_case=case, diagnosis_block='qualify', detailed=False)
        result.append({'name': case + '/qualify', 'document': doc})
        for block in ('old_plain', 'new_plain_before', 'run_probe', 'new_plain_after'):
            doc = make_task('performance', cfg, bundle, warmup=8, scored=16, tail=4,
                            executor='policy' if block == 'old_plain' else 'run_scope',
                            run_probe=block == 'run_probe', diagnosis_case=case,
                            diagnosis_block=block, detailed=False)
            result.append({'name': case + '/' + block, 'document': doc})
    if len(result) != 22 or len({v['document']['task_id'] for v in result}) != 22:
        raise ValueError('run-scope plan must contain twenty-two unique tasks')
    if sum(sum(v['document'][k] for k in ('warmup', 'scored', 'tail')) for v in result) != 310:
        raise ValueError('run-scope frame coverage differs')
    return result


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


def create(bundle, root, connection):
    from .prepare import validate_bundle
    bundle, root = Path(bundle).resolve(), Path(root).resolve()
    manifest = validate_bundle(bundle)
    if root.exists():
        raise FileExistsError('run already exists; use board resume')
    common = api(bundle)
    plan = task_plan(bundle)
    root.mkdir(parents=True)
    state = {'experiment': protocol()['experiment'], 'controller_identity': identity(),
             'bundle': str(bundle), 'bundle_id': manifest['bundle_id'],
             'bundle_manifest_sha256': common.sha256(bundle / 'manifest.json'),
             'connection': deepcopy(connection), 'created': common.now(), 'status': 'CREATED',
             'phase': 'CREATED', 'boot_id': None, 'active_boot': None, 'completed_boots': [],
             'frozen_files': {}, 'completed_tasks': 0, 'expected_tasks': 22,
             'expected_forwards': 310}
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
    if common.read(root / 'protocol.json') != protocol() or common.read(root / 'tasks.json') != task_plan(bundle):
        raise ValueError('diagnostic protocol or task plan changed')
    if state.get('active_boot') not in (None, state.get('boot_id')):
        raise ValueError('active boot identity differs')
    if state['status'] == 'COMPLETE' and (state['completed_tasks'] != 22
            or state['active_boot'] is not None or state['completed_boots'] != [state['boot_id']]):
        raise ValueError('incomplete diagnostic campaign marked complete')
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
                print('[计算区间诊断] ' + name, flush=True)
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
    return {item['name']: recorder.saved(item['name'], item['document']) for item in task_plan(state['bundle'])}


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
            for n, item in enumerate(task_plan(state['bundle']), 1):
                recorder.saved(item['name'], item['document'])
                state['completed_tasks'] = n
                _save(root, state)
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
