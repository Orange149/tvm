"""Read-only recovery of a stopped compiler batch and exact CPU profile reuse.

An archived controller is evidence, never permission to execute that controller.
Only byte-identical isolated CPU work may be carried into a repaired batch.
"""
from pathlib import Path

from cpu_vta_partitioning.common import digest, read, sha256, verify_inventory
from yolo_transfer import profile

from .common import ROOT, assert_controllers, frozen

OLD_OUTPUT = ROOT / 'artifacts/resnet50_v2_validation_01'
OLD_RUN = ROOT / 'runs/resnet50_v2_validation_01'
MODULE_FILES = ('module.so', 'graph.json', 'params.bin')
MATCH_FIELDS = ('configuration', 'stages', 'environment', 'warmup', 'scored', 'tail',
                'input_indices', 'executor_sha256', 'affinity_probe', 'expected_outputs')


def _manifest(bundle):
    bundle = Path(bundle); doc = read(bundle / 'manifest.json')
    if doc['bundle_id'] != digest({k: v for k, v in doc.items() if k != 'bundle_id'})[:24]:
        raise ValueError('archived capsule seal differs')
    verify_inventory(bundle, doc['files'])
    if doc.get('network') != 'resnet50_v2' or doc.get('unit_count') != 38:
        raise ValueError('archived graph/model identity differs')
    return doc


def _snapshot(output):
    output = Path(output); root = output / 'controller_snapshot'
    doc = read(root / 'manifest.json'); controllers = read(output / 'CONTROLLERS.json')
    if doc['controller_sha256'] != sha256(output / 'CONTROLLERS.json'):
        raise ValueError('archived controller identity differs')
    if set(doc['files']) != set(controllers['sources']):
        raise ValueError('archived source coverage differs')
    for original, row in doc['files'].items():
        relative = Path(row['path'])
        if (relative.is_absolute() or '..' in relative.parts
                or row['sha256'] != controllers['sources'][original]
                or sha256(root / relative) != row['sha256']):
            raise ValueError('archived source hash differs: ' + str(relative))
    # Only the generic decoder is used below. New R50 compiler/preparation
    # sources may differ; they are not compared with their archived versions.
    for tail in ('yolo_transfer/profile.py', 'cpu_vta_partitioning/common.py',
                 'cpu_vta_partitioning/measurement/__init__.py'):
        matches = [(p, row) for p, row in doc['files'].items() if p.endswith('/' + tail)]
        if len(matches) != 1 or sha256(matches[0][0]) != matches[0][1]['sha256']:
            raise ValueError('generic archived evidence decoder changed: ' + tail)
    return doc


def audit(output=OLD_OUTPUT, run_dir=OLD_RUN):
    """Audit the stopped cohort without SSH, writes, current R50 prepare, or FPS."""
    output, run_dir = Path(output).resolve(), Path(run_dir).resolve()
    if (output / 'DEPLOYMENT_PREFLIGHT_STOP.json').is_file():
        if (output / 'COMPILER_REGRESSION_STOP.json').exists():
            raise ValueError('ambiguous compiler and transport archive types')
        return _transport_preflight(output, run_dir)
    stop = read(output / 'COMPILER_REGRESSION_STOP.json')
    if stop.get('status') != 'STOPPED_COMPILER_REGRESSION':
        raise ValueError('not the sealed compiler-regression archive')
    snapshot = _snapshot(output)
    bundle = output / 'profile_capsule'; manifest = _manifest(bundle)
    run = read(run_dir / 'components/run.json'); boot = run['boot_id']
    if run['bundle_id'] != manifest['bundle_id'] or boot != stop['boot_id']:
        raise ValueError('archived run or stop boot identity differs')
    rows, calls, cpu = [], 0, 0
    for path in sorted((run_dir / 'components/accepted' / boot).glob('*.json')):
        item = read(path); doc = item['task']
        if (doc.get('kind') != 'calibration' or doc != profile.make_task(bundle, doc['component_key'])
                or item['boot_id'] != boot):
            raise ValueError('archived isolated task identity differs')
        verify_inventory(item['raw'], item['files'])
        raw = Path(item['raw'])
        if ((raw / 'boot_id').read_text().strip() != boot
                or (raw / 'task_hash').read_text().strip() != digest(doc)):
            raise ValueError('archived raw task or boot identity differs')
        if profile.summary(raw, doc, bundle) != item['summary']:
            raise ValueError('archived generic component summary no longer replays')
        count = sum(doc[k] for k in ('warmup', 'scored', 'tail')) * len(doc['stages'])
        calls += count; cpu += doc['stages'][0]['device'] == 'cpu'
        rows.append(dict(receipt=str(path), receipt_sha256=sha256(path), task_id=doc['task_id'],
            component_key=doc['component_key'], boot_id=boot, device=doc['stages'][0]['device'],
            program=doc['stages'][0]['program'], stage_calls=count,
            measurement_and_collection_seconds=item.get('measurement_and_collection_seconds')))
    if len(rows) != stop['accepted'] or calls != stop['scheduled_component_stage_calls']:
        raise ValueError('archived stop counters do not match accepted evidence')
    if not set(stop.get('recovered_without_reexecution', [])) <= {r['task_id'] for r in rows}:
        raise ValueError('recovered task is missing from archived accepted evidence')
    other = list((run_dir / 'traffic/accepted').glob('*/*.json')) + list((run_dir / 'pipeline/accepted').glob('*/*.json'))
    if other or stop.get('new_DMA_sessions') != 0 or stop.get('full_DNN_inferences') != 0:
        raise ValueError('stopped cohort scope changed; do not hide DMA/network work')
    builds = sorted((output / 'build/build_reports').glob('*.json'))
    compiled = sorted((output / 'build/programs').glob('*/record.json'))
    durations = [r['measurement_and_collection_seconds'] for r in rows]
    return dict(status='ARCHIVED_EVIDENCE_VERIFIED', archived=True, executable=False,
        source_snapshot_sha256=sha256(output / 'controller_snapshot/manifest.json'),
        archived_controller_sha256=snapshot['controller_sha256'], source_files=len(snapshot['files']),
        archived_output=str(output), archived_run=str(run_dir), bundle_id=manifest['bundle_id'], boot_id=boot,
        stop_sha256=sha256(output / 'COMPILER_REGRESSION_STOP.json'),
        accepted_components=len(rows), CPU_components=cpu, VTA_components=len(rows) - cpu,
        scheduled_component_stage_calls=calls, new_DMA_sessions=0, full_DNN_inferences=0,
        recovered_without_reexecution=stop.get('recovered_without_reexecution', []), records=rows,
        preparation_cost=dict(successful_program_records=len(compiled),
            successful_compile_worker_seconds=sum(read(p)['compile_seconds'] for p in compiled) if compiled else None,
            recorded_compile_invocation_wall_seconds=sum(read(p)['wall_seconds'] for p in builds) if builds else None,
            component_measurement_collection_seconds=sum(durations) if all(v is not None for v in durations) else None,
            failed_compile_worker_seconds=None, initialization_probe_calls=None),
        new_measurements=0, board_accessed=False, pipeline_FPS_used=False)


def _transport_preflight(output, run_dir):
    """A failed payload verification is not a native measurement or compiler failure."""
    output, run_dir = Path(output), Path(run_dir)
    stop_path = output / 'DEPLOYMENT_PREFLIGHT_STOP.json'
    stop = read(stop_path)
    if stop.get('status') != 'STOPPED_DEPLOYMENT_VERIFICATION_TIMEOUT':
        raise ValueError('not the sealed deployment-preflight archive')
    snapshot = _snapshot(output)
    if len(snapshot['files']) != 43:
        raise ValueError('transport archive source snapshot must retain all forty-three files')
    expected_snapshot = output / 'controller_snapshot/manifest.json'
    if (Path(stop['source_snapshot']).resolve() != expected_snapshot.resolve()
            or stop['controller_sha256'] != snapshot['controller_sha256']):
        raise ValueError('transport stop controller or snapshot identity differs')
    state_path = run_dir / 'state.json'
    if sha256(state_path) != stop['state_sha256']:
        raise ValueError('transport archive state identity differs')
    state, run = read(state_path), read(run_dir / 'run.json')
    bundle = output / 'profile_capsule'; manifest = _manifest(bundle)
    component_run = read(run_dir / 'components/run.json')
    boot = stop['boot_id']
    if (run['boot_id'] != boot or component_run['boot_id'] != boot or state['boot_id'] != boot
            or run['controller_sha256'] != stop['controller_sha256']
            or run['profile_capsule_id'] != manifest['bundle_id']
            or component_run['bundle_id'] != manifest['bundle_id']
            or Path(run['output']).resolve() != output.resolve()
            or state['status'] != 'STOPPED_ON_ERROR' or state['error_type'] != 'TransportError'):
        raise ValueError('transport source boot/controller/state binding differs')
    accepted, tasks = [], []
    for phase in ('components', 'traffic', 'pipeline'):
        accepted += list((run_dir / phase / 'accepted').glob('*/*.json'))
        tasks += list((run_dir / phase / 'raw').rglob('task.json'))
    if accepted or tasks or any(stop.get(k) != 0 for k in
            ('new_native_component_sessions', 'new_DMA_sessions', 'new_complete_DNN_inferences')):
        raise ValueError('transport-only archive contains own native task or accepted measurement')
    imported = imported_records(bundle, run_dir / 'components')
    index_path = run_dir / 'components/imported_components.json'; index = read(index_path)
    if (len(imported) != stop['imported_CPU_costs']
            or sorted({r['source']['boot_id'] for r in imported.values()}) != index['source_boot_ids']):
        raise ValueError('transport archive imported CPU source count or boot differs')
    origin = index['archive_audit']; origin_output = Path(index['archived_output'])
    if (Path(origin['archived_output']).resolve() != origin_output.resolve()
            or sha256(origin_output / 'CONTROLLERS.json') != origin['archived_controller_sha256']
            or sha256(origin_output / 'controller_snapshot/manifest.json') != origin['source_snapshot_sha256']
            or index['source_boot_ids'] != [origin['boot_id']]):
        raise ValueError('transport imported source controller/snapshot/boot binding differs')
    deployments = sorted((run_dir / 'components/deployments').glob('*.json'))
    deployment_rows = []
    for path in deployments:
        item = read(path)
        if item['manifest_sha256'] != sha256(bundle / 'manifest.json'):
            raise ValueError('transport deployment used a different capsule identity')
        deployment_rows.append(dict(path=str(path), sha256=sha256(path), **item))
    if not deployment_rows or deployment_rows[-1]['status'] != 'FAILED':
        raise ValueError('transport archive lacks its failed deployment receipt')
    return dict(status='ARCHIVED_TRANSPORT_PREFLIGHT_VERIFIED', archive_type='deployment_preflight',
        transport_preflight_validated=True, archived=True, executable=False,
        source_snapshot_sha256=sha256(expected_snapshot), source_files=len(snapshot['files']),
        archived_controller_sha256=snapshot['controller_sha256'], stop_sha256=sha256(stop_path),
        state_sha256=sha256(state_path), archived_output=str(output.resolve()), archived_run=str(run_dir.resolve()),
        bundle_id=manifest['bundle_id'], boot_id=boot, imported_CPU_costs=len(imported),
        imported_source_boot_ids=index['source_boot_ids'], imported_index_sha256=sha256(index_path),
        imported_source_controller_sha256=origin['archived_controller_sha256'],
        original_source_identities_preserved=True,
        new_native_component_sessions=0, new_component_stage_calls=0, new_DMA_sessions=0,
        full_DNN_inferences=0, new_measurements=0, deployments=deployment_rows,
        deployment_and_verification_seconds=sum(row['elapsed_seconds'] for row in deployment_rows),
        timing_scope='recorded deployment attempts including transfer and local/remote verification, not hash-only time',
        board_accessed=False, pipeline_FPS_used=False)


def imported_record(bundle, key, item):
    """Revalidate old bytes and outputs against the repaired bundle, preserving source."""
    bundle = Path(bundle); new_doc = profile.make_task(bundle, key)
    doc, source = item['task'], item['source']
    if doc['stages'][0]['device'] != 'cpu' or any(doc[k] != new_doc[k] for k in MATCH_FIELDS):
        raise ValueError('imported CPU execution semantics differ')
    origin = Path(source['receipt'])
    if sha256(origin) != source['receipt_sha256']:
        raise ValueError('original imported receipt changed')
    original = read(origin)
    if any(original[k] != item[k] for k in ('task', 'raw', 'files', 'summary')):
        raise ValueError('imported evidence differs from original receipt')
    if (original['boot_id'] != source['boot_id'] or doc['task_id'] != source['task_id']
            or doc['bundle_id'] != source['bundle_id']):
        raise ValueError('imported source task, bundle or boot identity differs')
    verify_inventory(item['raw'], item['files'])
    raw = Path(item['raw'])
    if ((raw / 'boot_id').read_text().strip() != source['boot_id']
            or (raw / 'task_hash').read_text().strip() != digest(doc)):
        raise ValueError('imported raw task or boot differs')
    name = doc['stages'][0]['program']
    if set(source['program_files']) != set(MODULE_FILES) or any(
            sha256(bundle / 'programs' / name / leaf) != source['program_files'][leaf] for leaf in MODULE_FILES):
        raise ValueError('imported CPU executable bytes differ')
    for leaf in ('plans/inputs.json', 'plans/expected_outputs.json'):
        if sha256(bundle / leaf) != source['input_reference_files'][leaf]:
            raise ValueError('imported input or repaired expected output differs')
    manifest = read(bundle / 'manifest.json')
    runtime = {k: v for k, v in manifest['files'].items() if k.startswith(('runtime/', 'runner/'))}
    if runtime != source['runtime_files']:
        raise ValueError('imported runtime or runner differs')
    # Deliberately use NEW expected_outputs here, rather than the old capsule.
    result = profile.summary(raw, doc, bundle)
    if result != item['summary']:
        raise ValueError('imported output does not revalidate against the repaired reference')
    return {**result, 'configuration': new_doc['configuration'], 'source': dict(source)}


def imported_records(bundle, run_dir):
    path = Path(run_dir) / 'imported_components.json'
    if not path.exists():
        return {}
    doc = read(path)
    if (doc.get('artifact_sha256') != digest({k: v for k, v in doc.items() if k != 'artifact_sha256'})
            or doc.get('bundle_id') != read(Path(bundle) / 'manifest.json')['bundle_id']):
        raise ValueError('imported component index identity differs')
    if doc.get('CPU_only') is not True or any(
            item['source']['archive_audit_sha256'] != digest(doc['archive_audit'])
            or item['source']['archived_output'] != doc['archived_output']
            for item in doc['records'].values()):
        raise ValueError('imported archive provenance differs')
    return {key: imported_record(bundle, key, item) for key, item in doc['records'].items()}


def import_components(output, run_dir, *, archived_output=OLD_OUTPUT, archived_run=OLD_RUN):
    """Reuse the proven matcher; only exact CPU matches enter repaired costs."""
    output, run_dir = Path(output).resolve(), Path(run_dir).resolve()
    archived_output, archived_run = Path(archived_output).resolve(), Path(archived_run).resolve()
    if output == archived_output or (output / 'COMPILER_REGRESSION_STOP.json').exists():
        raise ValueError('an archived compiler batch must never resume execution')
    assert_controllers(output)
    from .prepare import validate_profile
    bundle = output / 'profile_capsule'; current = validate_profile(bundle)
    proof = audit(archived_output, archived_run)
    # A staging index preserves the original helper's single-source boot marker.
    # The repaired measurement run will get its own actual boot marker later.
    staging = run_dir / 'components/import_matching_origin'
    matched = profile.import_matching(archived_output / 'profile_capsule', archived_run / 'components', bundle, staging)
    original = read(staging / 'imported_components.json')
    records = {}
    for key, item in original['records'].items():
        if item['task']['stages'][0]['device'] != 'cpu':
            continue
        source = item['source']
        source['input_reference_files'] = {leaf: sha256(bundle / leaf) for leaf in ('plans/inputs.json', 'plans/expected_outputs.json')}
        source['runtime_files'] = {k: v for k, v in current['files'].items() if k.startswith(('runtime/', 'runner/'))}
        source['archived_output'] = str(archived_output)
        source['archive_audit_sha256'] = digest(proof)
        imported_record(bundle, key, item)
        records[key] = item
    doc = dict(bundle_id=current['bundle_id'], records=records,
        source_boot_ids=sorted({item['source']['boot_id'] for item in records.values()}),
        archived_output=str(archived_output), archived_run=str(archived_run),
        archive_audit=proof, original_matcher_sha256=sha256(Path(profile.__file__)),
        CPU_only=True, original_source_identities_preserved=True, new_measurements=0)
    doc['artifact_sha256'] = digest(doc)
    frozen(run_dir / 'components/imported_components.json', doc)
    return dict(imported=len(records), matcher_exact_matches=matched['imported'],
        excluded_matching_VTA_components=matched['imported'] - len(records), new_measurements=0,
        original_boot_ids=doc['source_boot_ids'], original_task_ids_preserved=True)
