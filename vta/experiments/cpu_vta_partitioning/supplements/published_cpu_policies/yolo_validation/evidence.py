"""Replay individual receipts without rewriting their original boot identities."""
from collections import Counter
from pathlib import Path
import time
from cpu_vta_partitioning.common import (inventory, now, read, sha256, verify_inventory, write)
from yolo_transfer import profile
from .common import ASSETS, PROFILE, OLD_RUN, METHODS, frozen, protocol, source_identity, assert_sources

MATCH_FIELDS = ('configuration', 'stages', 'environment', 'warmup', 'scored', 'tail',
                'input_indices', 'executor_sha256', 'affinity_probe', 'expected_outputs')

def receipt_record(bundle, key, path, *, expected_record=None):
    """A receipt's own capsule and byte identities, never an aggregate boot, bind it."""
    bundle, path = Path(bundle), Path(path)
    item = read(path)
    doc = item['task']
    current = profile.make_task(bundle, key)
    if doc.get('kind') != 'calibration' or doc.get('component_key') != key:
        raise ValueError('not an isolated component receipt')
    if any(doc.get(k) != current[k] for k in MATCH_FIELDS):
        raise ValueError('component placement/window/environment differs')
    if doc.get('task_id') != __import__('cpu_vta_partitioning.common', fromlist=['digest']).digest(
            {k:v for k,v in doc.items() if k != 'task_id'})[:24]:
        raise ValueError('original task seal differs')
    # Resolve the original capsule from its original run marker. Imports can
    # legitimately originate in profile_01 rather than the stopped profile_02.
    original_run = path.parents[2]
    original_id = read(original_run / 'run.json')
    if item['boot_id'] != original_id['boot_id'] or path.parent.name != item['boot_id']:
        raise ValueError('receipt boot differs from its original cohort')
    old_bundle = (ASSETS / ('profile_bundle' if doc['bundle_id'] == 'ff5dbfe325ae02a7c578960b'
                           else 'profile_bundle_bridge02'))
    old_manifest = read(old_bundle / 'manifest.json')
    if doc['bundle_id'] != old_manifest['bundle_id'] or original_id['bundle_id'] != doc['bundle_id']:
        raise ValueError('unknown original component capsule')
    current_manifest = read(bundle / 'manifest.json')
    for prefix in ('runtime/', 'runner/', 'firmware/'):
        old = {n:v for n,v in old_manifest['files'].items() if n.startswith(prefix)}
        new = {n:v for n,v in current_manifest['files'].items() if n.startswith(prefix)}
        if old != new:
            raise ValueError('component executable/runtime/firmware changed')
    name = doc['stages'][0]['program']
    for leaf in ('graph.json', 'module.so', 'params.bin'):
        if sha256(old_bundle / 'programs' / name / leaf) != sha256(bundle / 'programs' / name / leaf):
            raise ValueError('superseded module is not reusable: ' + name)
    for leaf in ('inputs.json', 'expected_outputs.json'):
        if sha256(old_bundle / 'plans' / leaf) != sha256(bundle / 'plans' / leaf):
            raise ValueError('component frontier/reference changed')
    verify_inventory(item['raw'], item['files'])
    raw = Path(item['raw'])
    if (raw / 'boot_id').read_text().strip() != item['boot_id']:
        raise ValueError('raw boot identity differs')
    result = profile.summary(raw, doc, bundle)
    if result != item['summary']:
        raise ValueError('component receipt summary does not replay')
    if expected_record:
        if ('receipt_sha256' in expected_record['source']
                and sha256(path) != expected_record['source']['receipt_sha256']):
            raise ValueError('component receipt hash changed')
        if expected_record['source']['boot_id'] != item['boot_id']:
            raise ValueError('source index boot differs')
        if any(sha256(bundle/'programs'/name/f)!=value
               for f,value in expected_record['source'].get('program_files',{}).items()):
            raise ValueError('source index program hash changed')
        if any(expected_record.get(k) != v for k,v in result.items()):
            raise ValueError('stored isolated costs do not replay')
    return {**result, 'configuration': current['configuration'], 'source': {
        'receipt': str(path.resolve()), 'receipt_sha256': sha256(path),
        'boot_id': item['boot_id'], 'bundle_id': doc['bundle_id'], 'task_id': doc['task_id'],
        'program_files': {f:sha256(bundle / 'programs' / name / f) for f in ('graph.json','module.so','params.bin')},
        'reuse': 'exact graph/module/params/runtime/firmware/frontier/window/environment; original boot and task preserved'}}

def audit(output, bundle=PROFILE, old_run=OLD_RUN):
    output, bundle, old_run = map(lambda p:Path(p).resolve(), (output,bundle,old_run))
    started = time.monotonic()
    manifest = profile.validate(bundle)
    programs = read(bundle / 'plans/programs.json')
    configs = read(bundle / 'plans/component_configurations.json')
    if (len(programs) != 159 or len(configs) != 714
            or Counter(p['device'] for p in programs.values()) != {'cpu':81,'vta':78}):
        raise ValueError('YOLO compiled domain differs')
    units = read(ASSETS / 'units/UNIT_STATUS.json')
    if (units['units'] != 14 or not units['all_105_interval_recompositions_exact']
            or units['terminal_output_count'] != 8
            or units['source_sha256'] != sha256(ASSETS / 'graph/canonical_qir.json')):
        raise ValueError('unit/graph structural evidence differs')
    snapshot = read(old_run / 'partial_cost_snapshot.json')
    records, rejected = {}, {}
    for key, record in snapshot['costs'].items():
        source = record['source']
        path = Path(source.get('receipt', old_run / 'accepted' / source['boot_id'] / (source['task_id'] + '.json')))
        # Identity mismatches are not silently substituted with same-name files.
        # Only that signature is rescheduled; damaged hashes fail the audit.
        records[key] = receipt_record(bundle, key, path, expected_record=record)
    missing = sorted(set(configs) - set(records))
    report = dict(status='AUDITED', profile_bundle=str(bundle), profile_bundle_id=manifest['bundle_id'],
        profile_manifest_sha256=sha256(bundle/'manifest.json'), old_snapshot_sha256=sha256(old_run/'partial_cost_snapshot.json'),
        reusable=len(records), required=len(configs), missing=len(missing),
        missing_by_device=dict(Counter(configs[k]['stages'][0]['device'] for k in missing)),
        source_boot_ids=sorted({r['source']['boot_id'] for r in records.values()}),
        original_receipts_not_modified=True, old_pipeline_FPS_read=False,
        graph_status=units, elapsed_seconds=time.monotonic()-started)
    output.mkdir(parents=True,exist_ok=True)
    frozen(output/'imported_costs.json',dict(status='PARTIAL',costs=records,missing=missing,
        accepted=len(records),required=len(configs),cross_boot_sources_allowed=True))
    # Elapsed time is an audit log, not part of the immutable execution contract.
    write(output/'AUDIT.json',report,replace=True)
    return report

def prepare(output, bundle=PROFILE, old_run=OLD_RUN):
    output = Path(output).resolve()
    if not (output/'AUDIT.json').exists():
        audit(output,bundle,old_run)
    info = read(output/'AUDIT.json')
    value = dict(protocol=protocol(),sources=source_identity(),profile_bundle=info['profile_bundle'],
        profile_manifest_sha256=info['profile_manifest_sha256'],
        imports_sha256=sha256(output/'imported_costs.json'),
        old_results_read_only=True, status='PREPARED_FOR_COMPONENT_COMPLETION')
    frozen(output/'PREPARATION.json',value)
    return value

def replay_imports(output):
    output=Path(output); prep=assert_sources(output);bundle=Path(prep['profile_bundle'])
    profile.validate(bundle)
    if sha256(output/'imported_costs.json') != prep['imports_sha256']:
        raise ValueError('cross-boot source index changed')
    records=read(output/'imported_costs.json')['costs']
    for key,record in records.items():
        if receipt_record(bundle,key,record['source']['receipt'],expected_record=record) != record:
            raise ValueError('import source changed after preparation')
    return records
