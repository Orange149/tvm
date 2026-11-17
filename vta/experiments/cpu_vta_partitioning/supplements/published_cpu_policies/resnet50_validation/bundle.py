"""Self-contained selected-stage pipeline capsule; reuses frozen native binaries."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil
import subprocess
import time

from cpu_vta_partitioning.common import digest, inventory, read, sha256, sync_tree, write
from cpu_vta_partitioning.prepare.build import copy_checked

from .common import ROOT, PROJECT, assert_controllers, protocol
from .validation import (BUFFER_BYTES, MODULE_FILES, REQUIRED_FLAGS, dependency_report, memory_report,
                         validate_bundle, validate_lists)


def _copy_sources(output, destination):
    controllers = assert_controllers(output)
    mapping = {}
    for original, expected in sorted(controllers['sources'].items()):
        path = Path(original)
        try:
            relative = Path('sources/policies') / path.relative_to(ROOT)
        except ValueError:
            relative = Path('sources/base_controller') / path.relative_to(PROJECT / 'src')
        copy_checked(path, destination / relative, expected)
        mapping[original] = dict(path=str(relative), sha256=expected)
    write(destination / 'provenance/controller_sources.json', mapping)
    copy_checked(Path(output) / 'CONTROLLERS.json', destination / 'provenance/CONTROLLERS.json')


def _loader_check(destination, manifest):
    qemu = shutil.which('qemu-aarch64-static')
    if not qemu:
        raise RuntimeError('qemu-aarch64-static is required for portable loader validation')
    result = subprocess.run([qemu, str(destination / 'runtime/ld-linux-aarch64.so.1'),
        '--library-path', str(destination / 'runtime'),
        str(destination / 'runner/vta_stage_pipeline_runner'), '--help'],
        capture_output=True, text=True, timeout=60)
    text = result.stdout + result.stderr
    if result.returncode or any(flag not in text for flag in REQUIRED_FLAGS):
        raise ValueError('frozen AArch64 runner cannot load or lacks requested parameters: ' + text[-2000:])
    return dict(passed=True, runner_sha256=manifest['runner_sha256'],
                runtime_sha256=sha256(destination / 'runtime/libtvm_runtime.so'),
                supported_flags=[flag for flag in REQUIRED_FLAGS if flag in text],
                returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
                network_inference_calls=0, board_accessed=False)


def prepare(output):
    from .prepare import validate_profile
    output = Path(output).resolve()
    assert_controllers(output)
    profile = output / 'profile_capsule'
    parent = validate_profile(profile)
    ranking = output / 'ranking'
    freeze = read(ranking / 'FREEZE.json')
    if freeze.get('pipeline_FPS_used') is not False:
        raise ValueError('pipeline feedback cannot select a transfer-verification pool')
    for leaf, expected in freeze['files'].items():
        if Path(leaf).is_absolute() or '..' in Path(leaf).parts or sha256(ranking / leaf) != expected:
            raise ValueError('frozen rankings differ: ' + leaf)
    pool = read(ranking / 'pool.json')
    lists = read(ranking / 'rankings.json')['lists']
    full = read(profile / 'plans/programs.json')
    validate_lists(lists, pool, full, unit_count=parent['unit_count'])
    if freeze['candidate_count'] != len(pool):
        raise ValueError('ranked union size differs from freeze')
    names = {stage['program'] for cfg in pool.values() for stage in cfg['stages']}
    programs = {name: deepcopy(full[name]) for name in sorted(names)}
    destination = output / 'pipeline_bundle'
    if (destination / 'manifest.json').is_file():
        previous = validate_bundle(destination)
        if (previous['ranking_freeze_sha256'] != sha256(ranking / 'FREEZE.json')
                or previous['source_profile_bundle_id'] != parent['bundle_id']
                or previous['canonical_sha256'] != parent['canonical_sha256']):
            raise ValueError('existing pipeline capsule belongs to a different profile or frozen selection')
        return previous
    if destination.exists():
        raise FileExistsError('partial pipeline capsule retained; use a separate preparation directory')
    destination.mkdir(parents=True)
    started = time.monotonic()
    for name, info in parent['files'].items():
        if name.startswith(('runtime/', 'runner/', 'firmware/')):
            copy_checked(profile / name, destination / name, info['sha256'])
    for name, record in programs.items():
        if set(record['files']) != set(MODULE_FILES):
            raise ValueError('profile module runtime inventory must be normalized')
        for leaf in MODULE_FILES:
            copy_checked(profile / 'programs' / name / leaf,
                         destination / 'programs' / name / leaf, record['files'][leaf]['sha256'])
        write(destination / 'programs' / name / 'record.json', record)
    inputs = read(profile / 'plans/inputs.json')
    if len(inputs['records']) != 1 or inputs['records'][0]['index'] != 0:
        raise ValueError('profile reference must contain exactly input0')
    record = deepcopy(inputs['records'][0])
    nodes = {str(node) for program in programs.values()
             for node in program['input_node_ids'] + program['output_node_ids']}
    if not nodes <= set(record['frontiers']):
        raise ValueError('selected modules lack complete frontier reference coverage')
    record['frontiers'] = {key: value for key, value in record['frontiers'].items() if key in nodes}
    required = {record['input']}
    for value in record['frontiers'].values():
        required.update((value['raw'], value['npy']))
    for name in sorted(required):
        copy_checked(profile / name, destination / name, parent['files'][name]['sha256'])
    expected = read(profile / 'plans/expected_outputs.json')
    if not nodes <= set(expected):
        raise ValueError('selected frontier output fingerprints missing')
    write(destination / 'plans/programs.json', programs)
    write(destination / 'plans/inputs.json', dict(records=[record], available_indices=[0]))
    write(destination / 'plans/expected_outputs.json', {node: expected[node] for node in sorted(nodes)})
    for name in freeze['files']:
        copy_checked(ranking / name, destination / 'plans' / name, freeze['files'][name])
    copy_checked(ranking / 'FREEZE.json', destination / 'plans/ranking_freeze.json')
    for path in sorted((ranking / 'inputs').iterdir()):
        if path.is_file():
            copy_checked(path, destination / 'provenance/static_inputs' / path.name)
    copy_checked(profile / 'manifest.json', destination / 'provenance/profile_manifest.json')
    _copy_sources(output, destination)
    write(destination / 'plans/qemu_loader.json', _loader_check(destination, parent))
    write(destination / 'plans/ELF_DEPENDENCIES.json', dependency_report(destination, programs))
    memory = memory_report(destination, pool, programs, queue_depth=protocol()['queue_depth'])
    write(destination / 'plans/memory_estimate.json', memory)
    write(destination / 'plans/board_execution.json', dict(
        boot_policy='one_boot_no_automatic_reboot', udmabuf_bytes=BUFFER_BYTES,
        storage_margin_bytes=268435456, queue_depth=protocol()['queue_depth'],
        largest_visible_VTA_storage_estimate_bytes=memory['maximum_visible_VTA_storage_bytes'],
        visible_estimate_exceeds_existing_buffer=memory['visible_estimate_exceeds_existing_buffer'],
        actual_memory_sufficiency_proved=False))
    payload = {name: item for name, item in inventory(destination).items()
               if name.startswith(('programs/', 'runtime/', 'runner/', 'firmware/', 'inputs/'))
               and name.rsplit('/', 1)[-1] != 'record.json'}
    for name in ('plans/board_execution.json',):
        payload[name] = dict(bytes=(destination / name).stat().st_size,
                             sha256=sha256(destination / name))
    write(destination / 'plans/board_payload.json', dict(files=payload,
        unpacked_bytes=sum(value['bytes'] for value in payload.values())))
    write(destination / 'plans/preparation_cost.json', dict(seconds=time.monotonic() - started,
        candidates=len(pool), programs=len(programs), new_compilations=0,
        payload_bytes=sum(value['bytes'] for value in payload.values()),
        runtime_or_executor_recompiled=False, board_accessed=False))
    manifest = dict(network='resnet50_v2', model='MXNet Gluon pretrained ResNet50-v2',
        unit_count=38, canonical_sha256=parent['canonical_sha256'],
        experiment=protocol()['experiment'], candidate_count=len(pool),
        runner_sha256=parent['runner_sha256'],
        runtime_sha256=sha256(destination / 'runtime/libtvm_runtime.so'),
        source_profile_bundle_id=parent['bundle_id'], ranking_freeze_sha256=sha256(ranking / 'FREEZE.json'),
        files=inventory(destination))
    manifest['bundle_id'] = digest(manifest)[:24]
    write(destination / 'manifest.json', manifest)
    sync_tree(destination)
    return validate_bundle(destination)


prepare_pipeline = prepare
