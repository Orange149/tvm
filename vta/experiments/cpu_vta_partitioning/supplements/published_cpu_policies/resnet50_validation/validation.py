"""Portable offline capsule checks, exact list validation and visible memory accounting."""
from __future__ import annotations

from pathlib import Path
import re
import struct
import subprocess

from cpu_vta_partitioning.common import digest, read, sha256, verify_inventory
from cpu_policies.open_resources.domain import canonical_tuple, configuration, validate_candidate

from .assets import EXPECTED
from .common import METHODS, protocol

BUFFER_BYTES = 201326592
MODULE_FILES = ('graph.json', 'module.so', 'params.bin')
REQUIRED_FLAGS = ('--cpu-worker-pinning', '--allow-cpu-affinity-superset',
                  '--resource-affinity-probe', '--control-cpu-affinity', '--cpu-stage-gate')


def validate_lists(lists, pool, programs=None, *, unit_count=38):
    if set(lists) != set(METHODS):
        raise ValueError('exactly the two frozen methods are required')
    union = set()
    for method in METHODS:
        rows = lists[method]
        if len(rows) != 20 or len({r['id'] for r in rows}) != 20:
            raise ValueError('each method requires twenty distinct configurations')
        ordered = []
        for rank, row in enumerate(rows, 1):
            name = row['id']
            if name not in pool or configuration(row) != configuration(pool[name]):
                raise ValueError('method list and execution pool differ')
            if name != 'cfg_' + digest(configuration(row))[:24]:
                raise ValueError('configuration identity differs')
            if type(row['score_ns']) is not int or row['score_ns'] < 0:
                raise ValueError('scores must be nonnegative integer nanoseconds')
            if row.get('static_rank', rank) != rank:
                raise ValueError('static ranks are not consecutive')
            if programs is not None:
                validate_candidate(row, programs, unit_count=unit_count, max_vta_islands=3)
            if method == 'tarnawski' and (row['execution_mode'] != 'resource_group'
                    or any(s['threads'] != 1 for s in row['stages'] if s['device'] == 'cpu')):
                raise ValueError('Tarnawski physical single-core execution differs')
            ordered.append((row['score_ns'], canonical_tuple(row)))
            union.add(name)
        if ordered != sorted(ordered):
            raise ValueError('method lists violate score/canonical ordering')
    if union != set(pool) or len(pool) > 40:
        raise ValueError('execution union or forty-configuration budget differs')
    return True


def tensor_bytes(shape, dtype):
    if dtype == 'bool':
        bits, lanes = 1, 1
    else:
        match = re.fullmatch(r'(?:u?int|float|bfloat)(\d+)(?:x(\d+))?', str(dtype))
        if match is None:
            raise ValueError('unsupported graph dtype: ' + str(dtype))
        bits, lanes = int(match[1]), int(match[2] or 1)
    if bits <= 0 or lanes <= 0:
        raise ValueError('invalid graph dtype width')
    count = 1
    for dimension in shape:
        if type(dimension) is not int or dimension < 0:
            raise ValueError('graph memory accounting requires static nonnegative shapes')
        count *= dimension
    return count * ((bits * lanes + 7) // 8)


def graph_memory(graph, *, device):
    """Count visible pooled tensor storage by physical device, once per slot."""
    attrs = graph['attrs']
    shapes, types, slots = (attrs[k][1] for k in ('shape', 'dltype', 'storage_id'))
    default = 12 if device == 'vta' else 1
    devices = attrs.get('device_index', [None, [default] * len(shapes)])[1]
    if not len(shapes) == len(types) == len(slots) == len(devices):
        raise ValueError('graph storage metadata lengths differ')
    storage, slot_device, sizes = {}, {}, []
    for shape, dtype, slot, target in zip(shapes, types, slots, devices):
        size = tensor_bytes(shape, dtype)
        slot, target = int(slot), int(target)
        sizes.append(size)
        if slot < 0:
            continue
        if slot in slot_device and slot_device[slot] != target:
            raise ValueError('one graph storage slot belongs to different devices')
        slot_device[slot] = target
        key = target, slot
        storage[key] = max(storage.get(key, 0), size)
    rows = graph['node_row_ptr']
    if len(rows) != len(graph['nodes']) + 1 or rows[-1] != len(sizes):
        raise ValueError('graph node/entry metadata differs')
    incoming = [rows[n] for n in graph['arg_nodes']
                if graph['nodes'][n]['name'] == 'data'
                or graph['nodes'][n]['name'].startswith('edge')]
    outgoing = [rows[n] + port for n, port, _ in graph['heads']]
    if any(i < 0 or i >= len(sizes) for i in incoming + outgoing):
        raise ValueError('graph input/output entry is outside storage metadata')
    return dict(visible_VTA_storage_bytes=sum(v for (d, _), v in storage.items() if d == 12),
                visible_CPU_storage_bytes=sum(v for (d, _), v in storage.items() if d == 1),
                other_device_storage_bytes=sum(v for (d, _), v in storage.items() if d not in (1, 12)),
                input_tensor_bytes=sum(sizes[i] for i in incoming),
                output_tensor_bytes=sum(sizes[i] for i in outgoing),
                storage_slots=len(storage), is_measured_peak=False)


def memory_report(bundle, pool, programs, *, queue_depth=2, buffer_bytes=BUFFER_BYTES):
    bundle = Path(bundle)
    stages = {}
    for name, record in programs.items():
        stages[name] = {**graph_memory(read(bundle / 'programs' / name / 'graph.json'),
                                     device=record['device']),
                        'parameter_blob_host_bytes': (bundle / 'programs' / name / 'params.bin').stat().st_size}
        if record['device'] == 'cpu' and stages[name]['visible_VTA_storage_bytes']:
            raise ValueError('CPU stage graph unexpectedly requires VTA storage')
    configurations = {}
    for identity, cfg in pool.items():
        rows = [stages[stage['program']] for stage in cfg['stages']]
        # All graph executors are live together, including all VTA islands.
        vta = sum(row['visible_VTA_storage_bytes'] for row in rows)
        # Ordinary-copy outputs are CPU snapshots retained in their frame.
        # This is a visible-array proxy, not a complete process-memory bound.
        frames = (len(rows) + 1) * queue_depth + len(rows) + 2
        frame_bytes = rows[0]['input_tensor_bytes'] + sum(row['output_tensor_bytes'] for row in rows)
        configurations[identity] = dict(visible_VTA_storage_bytes=vta,
            visible_CPU_graph_storage_bytes=sum(row['visible_CPU_storage_bytes'] for row in rows),
            parameter_blob_host_bytes=sum(row['parameter_blob_host_bytes'] for row in rows),
            in_flight_frame_count_proxy_upper=frames,
            visible_frame_arrays_host_proxy_bytes=frames * frame_bytes,
            visible_estimate_exceeds_existing_buffer=vta > buffer_bytes)
    maximum = max((r['visible_VTA_storage_bytes'] for r in configurations.values()), default=0)
    return dict(buffer_bytes=buffer_bytes, queue_depth=queue_depth, programs=stages,
        configurations=configurations, maximum_visible_VTA_storage_bytes=maximum,
        visible_estimate_exceeds_existing_buffer=maximum > buffer_bytes,
        actual_memory_sufficiency_proved=False, is_measured_peak=False,
        interpretation='all graph executors coexist; graph storage_id is counted once per program; ordinary-copy frame outputs are host arrays',
        unknown=['TIR workspace and scratch', 'allocator overhead and alignment',
                 'driver runtime allocations', 'actual frame/storage aliasing', 'other process and system memory'])


def check_aarch64(path):
    with Path(path).open('rb') as stream:
        header = stream.read(64)
    if len(header) < 64 or header[:4] != b'\x7fELF' or header[4:6] != bytes((2, 1)):
        raise ValueError('required ELF64 little-endian object missing: ' + str(path))
    kind, machine = struct.unpack_from('<HH', header, 16)
    if machine != 183 or kind not in (2, 3):
        raise ValueError('required AArch64 executable/shared object differs: ' + str(path))
    return True


def dependency_report(bundle, programs):
    """Inspect actual ELF DT_NEEDED entries; loading still needs separate checks."""
    bundle = Path(bundle)
    paths = ['runner/vta_stage_pipeline_runner']
    for path in sorted((bundle / 'runtime').iterdir()):
        if path.is_file():
            with path.open('rb') as stream:
                if stream.read(4) == b'\x7fELF':
                    paths.append(str(path.relative_to(bundle)))
    paths += ['programs/' + name + '/module.so' for name in sorted(programs)]
    result = {}
    for relative in paths:
        path = bundle / relative
        check_aarch64(path)
        command = subprocess.run(['readelf', '-d', str(path)], capture_output=True,
                                 text=True, timeout=30)
        if command.returncode:
            raise ValueError('cannot inspect shared-library dependencies: ' + relative)
        needed = re.findall(r'\(NEEDED\).*?\[([^]]+)\]', command.stdout)
        for name in needed:
            if Path(name).name != name or not (bundle / 'runtime' / name).is_file():
                raise ValueError('nonportable or missing shared library: ' + relative + ' -> ' + name)
        result[relative] = dict(sha256=sha256(path), needed=needed)
    return dict(files=result, all_dependencies_bundled=True, board_accessed=False,
                stage_module_outputs_executed=False)


def validate_bundle(bundle):
    bundle = Path(bundle)
    manifest = read(bundle / 'manifest.json')
    if manifest['bundle_id'] != digest({k: v for k, v in manifest.items() if k != 'bundle_id'})[:24]:
        raise ValueError('pipeline capsule seal differs')
    if manifest['network'] != 'resnet50_v2' or manifest['unit_count'] != 38:
        raise ValueError('pipeline model/unit identity differs')
    for relative in manifest['files']:
        if Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ValueError('capsule inventory contains a nonportable path')
    verify_inventory(bundle, manifest['files'])
    for name, expected in EXPECTED.items():
        if not (bundle / name).is_file() or sha256(bundle / name) != expected:
            raise ValueError('frozen runtime, firmware or input identity differs: ' + name)
    if manifest['runner_sha256'] != EXPECTED['runner/vta_stage_pipeline_runner']:
        raise ValueError('formal executor identity changed')
    if manifest['runtime_sha256'] != EXPECTED['runtime/libtvm_runtime.so']:
        raise ValueError('formal runtime identity changed')
    seal = read(bundle / 'plans/ranking_freeze.json')
    if sha256(bundle / 'plans/ranking_freeze.json') != manifest['ranking_freeze_sha256']:
        raise ValueError('ranking freeze provenance identity differs')
    for leaf, expected in seal['files'].items():
        if Path(leaf).is_absolute() or '..' in Path(leaf).parts or sha256(bundle / 'plans' / leaf) != expected:
            raise ValueError('frozen method selection differs: ' + leaf)
    pool = read(bundle / 'plans/pool.json')
    programs = read(bundle / 'plans/programs.json')
    validate_lists(read(bundle / 'plans/rankings.json')['lists'], pool, programs)
    selected = {stage['program'] for cfg in pool.values() for stage in cfg['stages']}
    if selected != set(programs) or manifest['candidate_count'] != len(pool):
        raise ValueError('selected module/candidate coverage differs')
    order = read(bundle / 'plans/execution_order.json')
    if len(order) != len(pool) or set(order) != set(pool):
        raise ValueError('execution order is not a permutation of the pool')
    if read(bundle / 'plans/protocol.json') != protocol():
        raise ValueError('frozen performance protocol changed')
    for name, record in programs.items():
        if record['canonical_sha256'] != manifest['canonical_sha256']:
            raise ValueError('selected stage belongs to a different common graph')
        if set(record['files']) != set(MODULE_FILES):
            raise ValueError('pipeline stage runtime closure differs')
        if read(bundle / 'programs' / name / 'record.json') != record:
            raise ValueError('runtime stage record and program catalog differ')
        verify_inventory(bundle / 'programs' / name, record['files'])
        check_aarch64(bundle / 'programs' / name / 'module.so')
    check_aarch64(bundle / 'runner/vta_stage_pipeline_runner')
    for name in ('libtvm_runtime.so', 'libvta.so', 'ld-linux-aarch64.so.1'):
        check_aarch64(bundle / 'runtime' / name)
    dependencies = read(bundle / 'plans/ELF_DEPENDENCIES.json')
    expected_elf = {'runner/vta_stage_pipeline_runner'} | {
        'programs/' + name + '/module.so' for name in programs}
    if not expected_elf <= set(dependencies['files']) or dependencies.get('all_dependencies_bundled') is not True:
        raise ValueError('selected ELF dependency coverage missing')
    for relative, entry in dependencies['files'].items():
        if relative not in manifest['files'] or sha256(bundle / relative) != entry['sha256']:
            raise ValueError('ELF dependency proof identity differs')
        if any(Path(name).name != name or 'runtime/' + name not in manifest['files']
               for name in entry['needed']):
            raise ValueError('a shared library is absent from the portable capsule')
    controls = read(bundle / 'provenance/CONTROLLERS.json')
    source_map = read(bundle / 'provenance/controller_sources.json')
    if controls['protocol'] != protocol() or set(source_map) != set(controls['sources']):
        raise ValueError('controller snapshot coverage differs')
    for original, entry in source_map.items():
        relative = entry['path']
        if (Path(relative).is_absolute() or '..' in Path(relative).parts
                or relative not in manifest['files']
                or entry['sha256'] != controls['sources'][original]
                or sha256(bundle / relative) != entry['sha256']):
            raise ValueError('controller source snapshot identity differs')
    loading = read(bundle / 'plans/qemu_loader.json')
    if (loading.get('passed') is not True or loading['runner_sha256'] != manifest['runner_sha256']
            or loading['runtime_sha256'] != manifest['runtime_sha256']
            or any(flag not in loading['supported_flags'] for flag in REQUIRED_FLAGS)):
        raise ValueError('AArch64 loader/argument-support proof differs')
    inputs = read(bundle / 'plans/inputs.json')
    if inputs.get('available_indices') != [0] or len(inputs['records']) != 1:
        raise ValueError('exactly fixed input0 is required')
    if inputs['records'][0]['index'] != 0 or inputs['records'][0]['input'] != 'inputs/000.bin':
        raise ValueError('input identity contract differs')
    wanted = {str(node) for p in programs.values() for node in p['input_node_ids'] + p['output_node_ids']}
    if set(inputs['records'][0]['frontiers']) != wanted:
        raise ValueError('frontier coverage differs')
    for row in inputs['records'][0]['frontiers'].values():
        for key in ('raw', 'npy'):
            relative = row[key]
            if (Path(relative).is_absolute() or '..' in Path(relative).parts
                    or relative not in manifest['files'] or not (bundle / relative).is_file()):
                raise ValueError('frontier reference/runtime input missing or nonportable')
    proof = read(bundle / 'plans/proof.json')
    if proof.get('pipeline_FPS_used') is not False or proof.get('exact_DP') is not True:
        raise ValueError('ranking is not a frozen isolated-cost exact-DP result')
    if proof.get('independent_global_enumeration') is not False:
        raise ValueError('do not overstate ResNet50 independent full enumeration')
    if proof.get('small_domain_and_ResNet18_acceptance', {}).get('passed') is not True:
        raise ValueError('offline DP acceptance missing')
    policy = read(bundle / 'plans/board_execution.json')
    if policy['udmabuf_bytes'] != BUFFER_BYTES or policy['boot_policy'] != 'one_boot_no_automatic_reboot':
        raise ValueError('hardware resource or boot policy changed')
    memory = memory_report(bundle, pool, programs, queue_depth=protocol()['queue_depth'])
    if read(bundle / 'plans/memory_estimate.json') != memory:
        raise ValueError('pipeline memory estimate does not replay')
    if memory['visible_estimate_exceeds_existing_buffer']:
        raise MemoryError('visible aggregate VTA storage exceeds the unchanged 192 MiB allocator; do not deploy or resize automatically')
    if any(row['other_device_storage_bytes'] for row in memory['programs'].values()):
        raise ValueError('selected graph requires a device outside the frozen CPU/VTA runtime')
    return manifest


def validate(output, run_dir=None):
    """Validate completed local stages; never load a graph or contact hardware."""
    output = Path(output)
    result = dict(status='PARTIALLY_PREPARED', board_accessed=False, new_inferences=0, checks={})
    if (output / 'CONTROLLERS.json').is_file():
        from .common import assert_controllers
        assert_controllers(output)
        result['checks']['controllers'] = True
    if (output / 'graph/graph_manifest.json').is_file():
        from .graph import validate_graph
        validate_graph(output / 'graph')
        result['checks']['common_graph'] = True
    if (output / 'units/UNIT_STATUS.json').is_file():
        units = read(output / 'units/UNIT_STATUS.json')
        if units['source_sha256'] != sha256(output / 'graph/canonical_qir.json') or not units['all_interval_recompositions_exact'] or units['interval_checks'] != 741:
            raise ValueError('typed unit/frontier proof differs')
        result['checks']['units_and_741_intervals'] = True
    if (output / 'build/plans/compiler_identity.json').is_file():
        from .prepare import validate_build
        state = validate_build(output / 'build', require_all=False)
        result['checks']['build'] = dict(complete=len(state['complete']), missing=len(state['missing']))
    if (output / 'references/REFERENCE_PROOF.json').is_file():
        reference = read(output / 'references/REFERENCE_PROOF.json')
        if (reference['status'] != 'ARM_COMPLETE_GRAPH_PROBE_BYTE_EXACT' or reference['outputs'] != 1
                or reference['canonical_sha256'] != sha256(output / 'graph/canonical_qir.json')):
            raise ValueError('complete ARM reference proof belongs to a different graph')
        verify_inventory(output / 'references/references', reference['files'])
        result['checks']['ARM_reference'] = True
    if (output / 'profile_capsule/manifest.json').is_file():
        from .prepare import validate_profile
        result['checks']['profile_capsule'] = validate_profile(output / 'profile_capsule')['bundle_id']
    if (output / 'pipeline_bundle/manifest.json').is_file():
        value = validate_bundle(output / 'pipeline_bundle')
        result.update(status='PREPARED_OFFLINE', bundle_id=value['bundle_id'], candidates=value['candidate_count'])
        result['checks']['pipeline_capsule'] = True
    if run_dir is not None and (Path(run_dir) / 'state.json').is_file():
        result['existing_run_status'] = read(Path(run_dir) / 'state.json')['status']
    return result
