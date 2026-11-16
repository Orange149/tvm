"""Read-only receipt replay. Diagnostics are opened only after replay freeze."""
from pathlib import Path
from types import FunctionType

from cpu_vta_partitioning.common import digest, read, sha256, verify_inventory
from .common import WAIT_RUN, WAIT_BUNDLE, AFFINITY_RUN, protocol


def wait_receipt(case, scope, probe, qualification=False):
    from wait_diagnosis.runtime import make_task, summary as frozen_summary, trace_summary
    from run_yolo_wait_diagnosis_compat import passive_slots
    spec = dict(stage=1, policy='original', scope=scope, case=case, probe=probe)
    task = make_task(WAIT_BUNDLE, spec, qualification)
    identity = read(WAIT_RUN / 'run.json')
    if identity['bundle_id'] != read(WAIT_BUNDLE / 'manifest.json')['bundle_id']:
        raise ValueError('waiting source capsule changed')
    path = WAIT_RUN / 'accepted' / identity['boot_id'] / (task['task_id'] + '.json')
    value = checked_receipt(path, WAIT_RUN, identity, task)
    def compatible_trace(raw, document, bundle, affinity):
        return trace_summary(raw, document, bundle, passive_slots(document, affinity))
    replay = FunctionType(frozen_summary.__code__,
        {**frozen_summary.__globals__, 'trace_summary': compatible_trace})
    calculated = replay(value['raw'], task, WAIT_BUNDLE)
    if calculated != value['summary']:
        raise ValueError('waiting summary differs from verified raw data')
    return value, dict(path=str(path), sha256=sha256(path), task_id=task['task_id'],
        boot_id=value['boot_id'], scope=scope, probe=probe, qualification=qualification,
        raw=value['raw'], raw_files=value['files'])


def checked_receipt(path, run, identity, task):
    value = read(path)
    if value['task'] != task or value['boot_id'] != identity['boot_id']:
        raise ValueError('source task/boot identity differs')
    raw = Path(value['raw']).resolve()
    expected = Path(run).resolve() / 'raw' / identity['boot_id'] / task['task_id']
    if not raw.is_relative_to(expected):
        raise ValueError('source raw data escaped the task directory')
    parts = raw.relative_to(expected).parts
    if len(parts) != 3 or parts[0] != 'transfers' or not parts[1].isdigit() or parts[2] != 'collected':
        raise ValueError('unknown immutable collection layout')
    verify_inventory(raw, value['files'])
    if ((raw/'boot_id').read_text().strip() != value['boot_id']
            or (raw/'task_hash').read_text().strip() != digest(task)):
        raise ValueError('raw source boot/task hash differs')
    return value


def audit_isolation():
    from wait_diagnosis.prepare import validate_bundle
    manifest = validate_bundle(WAIT_BUNDLE)
    refs, cases = [], {}
    for case in ('overlap4', 'split3'):
        for probe in (False, True):
            for qualification in (True, False):
                item, ref = wait_receipt(case, 'component', probe, qualification)
                refs.append(ref)
                if not qualification:
                    cases[case + ('_observed' if probe else '_clean')] = dict(
                        task_id=item['task']['task_id'], raw=item['raw'],
                        configuration=item['task']['configuration'], cost=item['summary']['cost'])
    return dict(protocol=protocol(), capsule_id=manifest['bundle_id'],
        capsule_manifest_sha256=sha256(WAIT_BUNDLE/'manifest.json'),
        module_sha256=sha256(WAIT_BUNDLE/'programs/cpu_00_02/module.so'),
        runtime_sha256=manifest['runtime_sha256'], runner_sha256=manifest['runner_sha256'],
        adapter_sha256=manifest['adapter_sha256'],
        source_run_identity=read(WAIT_RUN/'run.json'), cases=cases, receipts=refs,
        diagnostic_performance_read=False, no_new_measurement=True)


def diagnostic_sources():
    refs, cases = [], {}
    for case in ('overlap4', 'split3'):
        for probe in (False, True):
            for qualification in (True, False):
                item, ref = wait_receipt(case, 'pipeline', probe, qualification)
                refs.append(ref)
                if not qualification:
                    cases[case + ('_observed' if probe else '_clean')] = dict(
                        raw=item['raw'], configuration=item['task']['configuration'],
                        summary=item['summary'], task=item['task'])
    # The new affinity batch supplies background observations, not kernel work.
    from yolo_affinity_check.runtime import summary as affinity_summary, make_task
    from yolo_affinity_check.prepare import validate_bundle
    bundle = Path(read(AFFINITY_RUN/'run.json')['bundle'])
    validate_bundle(bundle)
    identity = read(AFFINITY_RUN/'run.json')
    affinity = []
    for spec in read(bundle/'plans/task_order.json'):
        if spec['scope'] != 'pipeline' or spec['phase'] != 'measure':
            continue
        task = make_task(bundle, spec)
        path = AFFINITY_RUN/'accepted'/identity['boot_id']/(task['task_id']+'.json')
        item = checked_receipt(path, AFFINITY_RUN, identity, task)
        if affinity_summary(item['raw'], task, bundle) != item['summary']:
            raise ValueError('affinity summary no longer replays')
        refs.append(dict(path=str(path), sha256=sha256(path), task_id=task['task_id'],
                         raw=item['raw'], raw_files=item['files'], boot_id=item['boot_id']))
        affinity.append(dict(case=spec['case'], round=spec['round'], **item['summary']))
    return dict(cases=cases, affinity_observations=affinity, receipts=refs)
