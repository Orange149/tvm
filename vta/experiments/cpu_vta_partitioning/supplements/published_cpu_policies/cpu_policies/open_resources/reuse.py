"""Reuse valid measurements after a search-controller correction.

Original task, boot, bundle and raw receipts are immutable. A frozen source
runtime replays them against byte-identical execution assets. Search failures
do not turn successfully qualified isolated components into invalid samples.
"""
from copy import deepcopy
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile

from .common import api
from .domain import component_key
from .costs import FIELDS

_SOURCE = 'provenance/calibration_source'
_MANIFEST = 'provenance/calibration_manifest.json'
_PLANS = 'provenance/calibration_plans'
_CONTROLLER = 'sources/calibration_controller'

_REPLAY = r'''
import json,sys
from pathlib import Path
snapshot,root,bundle,expected=sys.argv[1:]
sys.path.insert(0,snapshot)
from cpu_policies.open_resources.common import identity
from cpu_policies.open_resources.runtime import Evidence
if identity()!=expected:
    raise ValueError('archived measurement controller identity differs')
root=Path(root);evidence=Evidence(root,Path(bundle));count=0
for path in sorted((root/'components').glob('*.json')):
    record=json.loads(path.read_text())
    for role in ('qualification','measurement'):
        p=record[role];evidence.summary(p['summary'],p['document'],p['boot_id']);count+=1
for path in sorted((root/'traffic').glob('*.json')):
    p=json.loads(path.read_text())['measurement']
    evidence.summary(p['summary'],p['document'],p['boot_id']);count+=1
print(json.dumps({'passed':True,'replayed_receipts':count}))
'''


def _copy_file(common, source, target, expected=None):
    source, target = Path(source), Path(target)
    if source.is_symlink() or not source.is_file():
        raise ValueError('missing or linked measurement source: ' + str(source))
    if expected is not None and (common.sha256(source) != expected['sha256']
                                 or source.stat().st_size != expected['bytes']):
        raise ValueError('source measurement hash differs: ' + str(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _asset_check(bundle, manifest):
    common = api(bundle)
    assets = {n: v for n, v in manifest['files'].items()
              if n.startswith(('programs/', 'runtime/', 'runner/', 'firmware/', 'inputs/', 'references/'))}
    if not assets:
        raise ValueError('calibration source has no execution assets')
    common.verify_inventory(bundle, assets)
    return assets


def _replay(bundle, state):
    bundle = Path(bundle).resolve()
    with tempfile.TemporaryDirectory(prefix='cpu-vta-calibration-replay-') as temp:
        view = Path(temp)
        shutil.copy2(bundle / _MANIFEST, view / 'manifest.json')
        (view / 'plans').symlink_to(bundle / _PLANS, target_is_directory=True)
        (view / 'sources').symlink_to(bundle / 'sources', target_is_directory=True)
        for name in ('programs', 'runtime', 'runner', 'firmware', 'inputs', 'references'):
            (view / name).symlink_to(bundle / name, target_is_directory=True)
        result = subprocess.run([sys.executable, '-c', _REPLAY, str(bundle / _CONTROLLER),
            str(bundle / _SOURCE), str(view), state['controller_identity']], capture_output=True,
            text=True, timeout=600, env={**os.environ, 'PYTHONNOUSERSITE': '1'})
        if result.returncode:
            raise ValueError('frozen calibration replay failed:\n' + result.stdout + result.stderr)
        return json.loads(result.stdout.strip().splitlines()[-1])


def _summary(bundle):
    common = api(bundle)
    bundle = Path(bundle)
    root = bundle / _SOURCE
    state = common.read(root / 'state.json')
    manifest = common.read(bundle / _MANIFEST)
    if (common.digest({k: v for k, v in manifest.items() if k != 'bundle_id'})[:24] != manifest['bundle_id']
            or manifest['bundle_id'] != state['bundle_id']
            or common.sha256(bundle / _MANIFEST) != state['bundle_manifest_sha256']
            or manifest['controller_identity'] != state['controller_identity']):
        raise ValueError('source calibration batch/manifest identity differs')
    if state.get('completed_components') != 775 or state.get('new_components') != 261:
        raise ValueError('source did not complete all 261 missing component measurements')
    _asset_check(bundle, manifest)
    controller_inventory = common.read(bundle / 'plans/calibration_controller_inventory.json')
    common.verify_inventory(bundle, controller_inventory)
    evidence_inventory = common.read(bundle / 'plans/calibration_evidence_inventory.json')
    common.verify_inventory(root, evidence_inventory)
    frozen = state['frozen_files']
    all_costs = common.read(root / 'costs.json')['costs']
    if len(all_costs) != 775:
        raise ValueError('source cost table lacks complete 775 coverage')
    costs, traffic, source_index = {}, {}, {}
    for path in sorted((root / 'components').glob('*.json')):
        relative = str(path.relative_to(root))
        if relative not in frozen:
            raise ValueError('source component was not durably frozen')
        common.verify_inventory(root, {relative: frozen[relative]})
        record = common.read(path)
        cfg = record['configuration']
        key = component_key(cfg)
        if path.stem != key or record.get('kind') != 'measured' or all_costs.get(key) != record:
            raise ValueError('source component differs from its cost table')
        m, q = record['measurement'], record['qualification']
        doc = m['document']
        if (doc['kind'] != 'calibration' or doc['input_indices'] != [0]
                or (doc['warmup'], doc['scored'], doc['tail']) != (2, 8, 0)
                or m['boot_id'] != state['boot_id'] or q['boot_id'] != state['boot_id']
                or cfg['stages'][0]['device'] != 'vta' or cfg['host_mask'] not in ([0], [1], [2])
                or any(record['cost'][f] != m['summary'][f] for f in FIELDS)):
            raise ValueError('source calibration execution scope differs')
        provenance = {'source_bundle_id': state['bundle_id'], 'source_boot_id': m['boot_id'],
                      'source_task_id': doc['task_id'], 'source_record': relative,
                      'source_record_sha256': common.sha256(path),
                      'source_kind': 'qualified_component_before_search_controller_correction',
                      'execution_equivalence': 'identical_programs_runtime_executor_inputs_and_measurement_protocol'}
        copied = deepcopy(record)
        copied['kind'] = 'imported'
        copied['provenance'] = provenance
        costs[key] = copied
        source_index[key] = provenance
    if len(costs) != 261:
        raise ValueError('source must supply exactly 261 additional VTA components')
    for path in sorted((root / 'traffic').glob('*.json')):
        relative = str(path.relative_to(root))
        if relative not in frozen:
            raise ValueError('source traffic was not frozen')
        common.verify_inventory(root, {relative: frozen[relative]})
        record = common.read(path)
        m = record['measurement']
        doc, summary = m['document'], m['summary']
        if (record['program'] != path.stem or doc['kind'] != 'traffic'
                or (doc['warmup'], doc['scored'], doc['tail']) != (1, 1, 0)
                or doc['input_indices'] != [0] or m['boot_id'] != state['boot_id']
                or summary.get('hooks_present') is not True or summary.get('count') != 1
                or any(record[k] != summary[k] or summary[k] <= 0 for k in ('load_bytes', 'store_bytes'))):
            raise ValueError('source traffic execution scope differs')
        copied = deepcopy(record)
        copied['kind'] = 'imported'
        copied['provenance'] = {'source_bundle_id': state['bundle_id'], 'source_boot_id': m['boot_id'],
                                'source_task_id': doc['task_id'], 'source_record': relative,
                                'source_record_sha256': common.sha256(path)}
        traffic[path.stem] = copied
    if len(traffic) > 87:
        raise ValueError('unexpected source traffic coverage')
    replay = _replay(bundle, state)
    if replay != {'passed': True, 'replayed_receipts': 522 + len(traffic)}:
        raise ValueError('source evidence replay coverage differs')
    return {'costs': costs, 'traffic': traffic, 'source_index': source_index,
            'source_bundle_id': state['bundle_id'], 'source_boot_ids': [state['boot_id']],
            'source_measurement_controller_identity': state['controller_identity'],
            'source_costs_sha256': common.sha256(root / 'costs.json'),
            'replay': replay, 'execution_assets_changed': False,
            'original_tasks_relabelled_as_new_measurements': False}


def freeze_calibration_source(source_run, target_bundle):
    """Copy only valid completed records; failed traffic attempts remain in source."""
    common = api(target_bundle)
    source, target = Path(source_run).resolve(), Path(target_bundle).resolve()
    state = common.read(source / 'state.json')
    if state.get('status') not in ('STOPPED_INVALID_RESULT', 'INTERRUPTED', 'COMPLETE'):
        raise ValueError('calibration source must be stopped or complete before reuse')
    parent = Path(state['bundle']).resolve()
    manifest = common.read(parent / 'manifest.json')
    _asset_check(target, manifest)
    dest = target / _SOURCE
    if dest.exists():
        shutil.rmtree(dest)
    files = {}
    for name in ('state.json', 'protocol.json', 'costs.json'):
        path = common.relative_file(source, name)
        files[name] = {'sha256': common.sha256(path), 'bytes': path.stat().st_size}
    for name in ('protocol.json', 'costs.json'):
        expected = state['frozen_files'].get(name)
        if expected is None or files[name] != expected:
            raise ValueError('source frozen metadata differs: ' + name)
    for folder in ('components', 'traffic'):
        for path in sorted((source / folder).glob('*.json')):
            relative = str(path.relative_to(source))
            expected = state['frozen_files'].get(relative)
            if expected is None:
                raise ValueError('unfrozen source result: ' + relative)
            files[relative] = expected
            record = common.read(path)
            roles = ('measurement', 'qualification') if folder == 'components' else ('measurement',)
            for role in roles:
                proof = record[role]
                ref = proof['summary']['_receipt']
                receipt_path = common.relative_file(source, ref['path'])
                if common.sha256(receipt_path) != ref['sha256']:
                    raise ValueError('source receipt changed')
                receipt = common.read(receipt_path)
                if receipt['task'] != proof['document'] or receipt['boot_id'] != proof['boot_id']:
                    raise ValueError('source proof identity differs')
                common.verify_inventory(source, receipt['files'])
                files[ref['path']] = {'sha256': ref['sha256'], 'bytes': receipt_path.stat().st_size}
                for name, value in receipt['files'].items():
                    if name in files and files[name] != value:
                        raise ValueError('source raw inventory collision')
                    files[name] = value
    for name, expected in files.items():
        _copy_file(common, common.relative_file(source, name), dest / name, expected)
    _copy_file(common, parent / 'manifest.json', target / _MANIFEST)
    controller = parent / 'sources/open_resources_controller'
    destination = target / _CONTROLLER
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(controller, destination, ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
    inventory = {}
    prefix = 'sources/open_resources_controller/'
    for name, expected in manifest['files'].items():
        if name.startswith(prefix):
            local = _CONTROLLER + '/' + name[len(prefix):]
            inventory[local] = expected
    if not inventory:
        raise ValueError('source frozen measurement controller missing')
    for leaf in ('programs.json', 'inputs.json', 'executors.json'):
        _copy_file(common, parent / 'plans' / leaf, target / _PLANS / leaf,
                   manifest['files']['plans/' + leaf])
    common.write(target / 'plans/calibration_controller_inventory.json', inventory, replace=True)
    common.write(target / 'plans/calibration_evidence_inventory.json', files, replace=True)
    if common.sha256(source / 'state.json') != files['state.json']['sha256']:
        raise ValueError('source batch changed during reuse preparation')
    return _summary(target)


def validate_reused_calibration(bundle):
    return _summary(Path(bundle).resolve())
