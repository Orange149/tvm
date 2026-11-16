"""True isolated YOLO profiles with frozen references and explicit thread placement.

One short calibration session also checks every returned tensor hash and the
logical worker map. No separate qualification inference is added. Thread shares
remain whole-session measurements, including warmup and the startup probe.
"""
from copy import deepcopy
import math
from pathlib import Path
import statistics
from types import FunctionType

from cpu_vta_partitioning.common import (digest, inventory, locked, read, sha256,
                                        sync_tree, verify_inventory, write)
from cpu_vta_partitioning.prepare.build import copy_checked


def fnv1a64(data):
    value = 1469598103934665603
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return format(value, '016x')


def prepare(build, references, parent, output, *, repair_build=None):
    import cpu_vta_partitioning
    from cpu_policies.open_resources.domain import requirements
    build, references, parent, output = [Path(p).resolve() for p in (build, references, parent, output)]
    if output.exists():
        raise FileExistsError('profile capsules are immutable')
    proof = read(references / 'REFERENCE_PROOF.json')
    if proof['status'] != 'ARM_COMPLETE_GRAPH_PROBE_BYTE_EXACT' or proof['outputs'] != 8:
        raise ValueError('independent ARM two-head references are required')
    verify_inventory(references / 'references', proof['files'])
    specs = read(build / 'plans/programs.json')
    builds = [build] + ([Path(repair_build).resolve()] if repair_build else [])
    identities = {p: read(p / 'plans/compiler_identity.json') for p in builds}
    for p, identity in identities.items():
        verify_inventory(p, identity['files'])
        if read(p / 'plans/programs.json') != specs:
            raise ValueError('repair compiler changed the stage domain')
    parent_manifest = read(parent / 'manifest.json')
    with locked(output):
        programs = {}
        for spec in specs:
            name = spec['name']
            matches = [p for p in builds if (p / 'programs' / name / 'record.json').is_file()]
            if not matches or (len(matches) > 1 and (not repair_build or spec['device'] != 'vta')):
                raise ValueError('ambiguous or missing stage build source: ' + name)
            # Repair all VTA bridges, including cases that type-inferred but
            # concatenated blocked tensors with ordinary-layout semantics.
            origin = matches[-1]
            identity = identities[origin]
            source = origin / 'programs' / name
            record = read(source / 'record.json')
            if (record['spec'] != spec or record['build_identity'] != digest(identity)
                    or record['canonical_sha256'] != proof['canonical_sha256']):
                raise ValueError('compiled module does not match the complete frontier reference')
            verify_inventory(source, record['files'])
            for filename in ('graph.json', 'module.so', 'params.bin', 'record.json'):
                copy_checked(source / filename, output / 'programs' / name / filename)
            programs[name] = {**record, **spec, 'build_source_identity': digest(identity),
                'superseded_build_identities': [digest(identities[p]) for p in matches[:-1]]}
        input_record = {'index': 0, 'input': 'inputs/000.bin', 'frontiers': {}}
        expected = {}
        for node, typ in proof['frontiers'].items():
            raw = 'inputs/frontiers/000/' + node + '.bin'
            npy = 'references/000/' + node + '.npy'
            copy_checked(references / typ['raw'], output / raw)
            copy_checked(references / typ['npy'], output / npy)
            input_record['frontiers'][node] = {**typ, 'raw': raw, 'npy': npy}
            data = (output / raw).read_bytes()
            expected[node] = {'shape': typ['shape'], 'dtype': typ['dtype'], 'nbytes': len(data),
                              'fnv1a64': fnv1a64(data), 'sha256': sha256(output / raw)}
        copy_checked(build / 'inputs/000.bin', output / 'inputs/000.bin', proof['input']['sha256'])
        for filename, entry in parent_manifest['files'].items():
            if filename.startswith(('runtime/', 'firmware/')):
                copy_checked(parent / filename, output / filename, entry['sha256'])
        runner = 'runner/vta_stage_pipeline_runner_pinned'
        runner_hash = copy_checked(parent / runner, output / 'runner/vta_stage_pipeline_runner',
                                   parent_manifest['files'][runner]['sha256'])
        package = Path(__file__).resolve().parent
        for source in package.rglob('*.py'):
            copy_checked(source, output / 'sources/yolo_transfer' / source.relative_to(package))
        core = Path(cpu_vta_partitioning.__file__).resolve().parent
        for source in core.rglob('*.py'):
            copy_checked(source, output / 'sources/base_controller/cpu_vta_partitioning' / source.relative_to(core))
        policies = package.parent / 'cpu_policies/open_resources'
        for filename in ('domain.py', '__init__.py'):
            copy_checked(policies / filename, output / 'sources/open_resources' / filename)
        for origin, identity in identities.items():
            for filename in ('compiler_identity.json', 'programs.json'):
                copy_checked(origin / 'plans' / filename, output / 'sources/build' / digest(identity) / filename)
        copy_checked(references / 'REFERENCE_PROOF.json', output / 'sources/REFERENCE_PROOF.json')
        write(output / 'plans/programs.json', programs)
        write(output / 'plans/inputs.json', {'records': [input_record]})
        write(output / 'plans/expected_outputs.json', expected)
        configs = requirements(programs)
        write(output / 'plans/component_configurations.json', configs)
        # Easy-to-audit compatibility checks first; every item belongs to the same table.
        ordered = sorted(configs, key=lambda key: (
            configs[key]['stages'][0]['program'] not in ('cpu_00_00', 'cpu_13_13', 'vta_01_12'), key))
        write(output / 'plans/profile_order.json', ordered)
        write(output / 'plans/board_execution.json', {'boot_policy': 'one_boot_no_automatic_reboot',
            'udmabuf_bytes': 201326592, 'storage_margin_bytes': 268435456,
            'calibration': {'warmup': 2, 'scored': 8, 'tail': 0, 'input_index': 0},
            'qualification': 'all-frame exact frontier hashes plus one startup affinity probe',
            'thread_share_scope': 'whole_native_session_including_warmup_and_startup_probe',
            'new_pipeline_measurements': 0})
        files = {name: item for name, item in inventory(output).items()
                 if name.startswith(('programs/', 'runtime/', 'runner/', 'firmware/', 'inputs/'))
                 and name.rsplit('/', 1)[-1] != 'record.json'}
        files['plans/board_execution.json'] = inventory(output / 'plans')['board_execution.json']
        write(output / 'plans/board_payload.json', {'files': files, 'unpacked_bytes': sum(v['bytes'] for v in files.values())})
        manifest = {'model': 'YOLOv3-tiny416', 'scope': 'true_isolated_profiles',
            'canonical_sha256': proof['canonical_sha256'], 'runner_sha256': runner_hash,
            'component_count': len(configs), 'files': inventory(output, ['.lock'])}
        manifest['bundle_id'] = digest(manifest)[:24]
        write(output / 'manifest.json', manifest)
        sync_tree(output)
    return {'bundle': str(output), 'bundle_id': manifest['bundle_id'], 'components': len(configs),
            'board_payload_bytes': sum(v['bytes'] for v in files.values())}


def validate(bundle):
    import cpu_vta_partitioning
    bundle = Path(bundle).resolve()
    manifest = read(bundle / 'manifest.json')
    if manifest['bundle_id'] != digest({k: v for k, v in manifest.items() if k != 'bundle_id'})[:24]:
        raise ValueError('profile capsule identity differs')
    verify_inventory(bundle, manifest['files'])
    package = Path(__file__).resolve().parent
    for source in (bundle / 'sources/yolo_transfer').rglob('*.py'):
        if sha256(source) != sha256(package / source.relative_to(bundle / 'sources/yolo_transfer')):
            raise ValueError('loaded YOLO source differs from frozen profile capsule')
    core = Path(cpu_vta_partitioning.__file__).resolve().parent
    snapshot = bundle / 'sources/base_controller/cpu_vta_partitioning'
    for source in snapshot.rglob('*.py'):
        if sha256(source) != sha256(core / source.relative_to(snapshot)):
            raise ValueError('loaded SSH lifecycle differs from frozen capsule')
    from cpu_policies.open_resources import domain
    if sha256(Path(domain.__file__)) != sha256(bundle / 'sources/open_resources/domain.py'):
        raise ValueError('loaded component identity logic differs')
    return manifest


def make_task(bundle, key):
    bundle = Path(bundle)
    configuration = read(bundle / 'plans/component_configurations.json')[key]
    doc = {'kind': 'calibration', 'configuration': configuration,
        'stages': deepcopy(configuration['stages']), 'environment': {'poll_sleep_ns': 1000, 'post_start_sleep_ns': 1000},
        'warmup': 2, 'scored': 8, 'tail': 0, 'input_indices': [0],
        'component_key': key, 'bundle_id': read(bundle / 'manifest.json')['bundle_id'],
        'executor_sha256': read(bundle / 'manifest.json')['runner_sha256'],
        'affinity_probe': 'startup_only_outside_stage_timers', 'expected_outputs': 'exact_ARM_frontier_hashes'}
    doc['task_id'] = digest(doc)[:24]
    return doc


def native_command(doc, bundle, remote_bundle, remote_job):
    from cpu_vta_partitioning.measurement.tasks import native_command as base
    if doc != make_task(bundle, doc['component_key']):
        raise ValueError('profile task identity differs')
    argv, inputs = base(doc, bundle, remote_bundle, remote_job)
    loader = argv.index(remote_bundle + '/runtime/ld-linux-aarch64.so.1')
    argv[loader:loader] = ['TVM_BIND_THREADS=1', 'TVM_EXCLUDE_WORKER0=1']
    argv += ['--allow-cpu-affinity-superset', '--cpu-worker-pinning', '--resource-affinity-probe',
             '--control-cpu-affinity', ','.join(map(str, doc['configuration']['control_mask']))]
    if '--process-cpu-affinity' in argv:
        raise ValueError('process affinity must not overwrite per-worker placement')
    return argv, inputs


def matched_ssh(config, *, execute_board=False):
    from cpu_vta_partitioning.transport import SSH
    # A private globals dictionary preserves the proven lifecycle exactly without
    # mutating the old transport module or other campaigns' command builders.
    measure = FunctionType(SSH.measure.__code__, {**SSH.measure.__globals__, 'native_command': native_command},
                           SSH.measure.__name__, SSH.measure.__defaults__, SSH.measure.__closure__)
    return type('YOLOProfileSSH', (SSH,), {'measure': measure})(config, execute_board=execute_board)


def summary(raw, doc, bundle):
    from cpu_vta_partitioning.measurement import affinity, calibration, frames, input_sequence
    raw, bundle = Path(raw), Path(bundle)
    rows = frames(raw / 'result.jsonl', 10)
    input_sequence(rows, doc)
    report = read(raw / 'affinity.json')
    affinity(report, doc['stages'])
    row = doc['stages'][0]
    program = read(bundle / 'plans/programs.json')[row['program']]
    expected = read(bundle / 'plans/expected_outputs.json')
    for frame in rows:
        actual = frame.get('raw_outputs', [])
        if len(actual) != len(program['output_node_ids']):
            raise ValueError('component output frontier is incomplete')
        for index, (node, item) in enumerate(zip(program['output_node_ids'], actual)):
            if item.get('index') != index or any(item.get(k) != expected[str(node)][k]
                    for k in ('shape', 'dtype', 'nbytes', 'fnv1a64')):
                raise ValueError('component output differs from independent ARM frontier: ' + str(node))
    cost = calibration(rows, 2, 8)
    for operation in ('set', 'run', 'get'):
        for suffix, native in (('_ms', '_ms'), ('_core_ms', '_process_cpu_ms')):
            values = [r['stage0_' + operation + native] for r in rows[2:]]
            if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
                raise ValueError('invalid isolated component clock')
            cost[operation + suffix] = statistics.mean(values)
    shares = None
    stage = report['stages'][0]
    if row['device'] == 'cpu':
        if len(stage['thread_ids']) != row['threads'] or len(stage['effective_affinities']) != row['threads']:
            raise ValueError('reported CPU pool exceeds or misses the configured slots')
        slots = sorted(stage.get('worker_slots', []), key=lambda v: v['slot'])
        if len(slots) != row['threads'] or [v['slot'] for v in slots] != list(range(row['threads'])):
            raise ValueError('logical CPU worker slots missing')
        if len({v['tid'] for v in slots}) != row['threads'] or slots[0]['tid'] != stage['caller_tid']:
            raise ValueError('caller or logical worker identity differs')
        ledger = {entry['tid']: entry for entry in stage['effective_affinities']}
        increments = []
        for slot, core in zip(slots, row['worker_core_map']):
            entry = ledger.get(slot['tid'])
            if entry is None or slot['cpus'] != [core] or entry['cpus'] != [core]:
                raise ValueError('actual logical worker placement differs')
            value = entry.get('cpu_time_delta_ms')
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError('invalid whole-session thread clock')
            increments.append(value)
        if sum(increments) > 0:
            shares = [value / sum(increments) for value in increments]
    result = {'cost': cost, 'correct': True, 'actual_affinity_checked': True,
              'frame_count': 10, 'session_thread_weight_scope': 'warmup_scored_and_startup_probe_not_scored_run_only'}
    if shares is not None:
        result['session_thread_weights'] = shares
    else:
        result['thread_share_note'] = 'uniform fallback required when a CPU clock has no observable increment' if row['device'] == 'cpu' else 'VTA host thread shares not used'
    return result


def import_matching(old_bundle, old_run, bundle, run_dir):
    """Reuse exact executable/input/environment matches with original task IDs."""
    old_bundle, old_run, bundle, run_dir = [Path(p).resolve() for p in (old_bundle, old_run, bundle, run_dir)]
    current = validate(bundle)
    prior = read(old_bundle / 'manifest.json')
    if prior['bundle_id'] != digest({k: v for k, v in prior.items() if k != 'bundle_id'})[:24]:
        raise ValueError('old profile capsule seal differs')
    verify_inventory(old_bundle, prior['files'])
    if current['canonical_sha256'] != prior['canonical_sha256']:
        raise ValueError('profile reuse requires the same quantized graph')
    for name in ('plans/inputs.json', 'plans/expected_outputs.json'):
        if sha256(old_bundle / name) != sha256(bundle / name):
            raise ValueError('profile reuse changed frontier inputs or references')
    old_runtime = {k: v for k, v in prior['files'].items() if k.startswith(('runtime/', 'runner/'))}
    new_runtime = {k: v for k, v in current['files'].items() if k.startswith(('runtime/', 'runner/'))}
    if old_runtime != new_runtime:
        raise ValueError('profile reuse changed executor or runtime')
    old_programs, new_programs = (read(p / 'plans/programs.json') for p in (old_bundle, bundle))
    boot = read(old_run / 'run.json')['boot_id']
    records = {}
    for receipt in (old_run / 'accepted' / boot).glob('*.json'):
        item = read(receipt); doc = item['task']; key = doc['component_key']
        if doc != make_task(old_bundle, key) or item['boot_id'] != boot:
            raise ValueError('old profile task identity differs')
        new_doc = make_task(bundle, key)
        fields = ('configuration', 'stages', 'environment', 'warmup', 'scored', 'tail', 'input_indices',
                  'executor_sha256', 'affinity_probe', 'expected_outputs')
        if any(doc[k] != new_doc[k] for k in fields):
            continue
        name = doc['stages'][0]['program']
        if any(sha256(old_bundle / 'programs' / name / f) != sha256(bundle / 'programs' / name / f)
               for f in ('graph.json', 'params.bin', 'module.so')):
            continue
        if any(old_programs[name][k] != new_programs[name][k] for k in
               ('input_names', 'input_node_ids', 'output_node_ids', 'input_types', 'output_types')):
            raise ValueError('identical module bytes have conflicting frontier metadata')
        verify_inventory(item['raw'], item['files'])
        result = summary(item['raw'], doc, bundle)
        if result != item['summary']:
            raise ValueError('old matched profile no longer replays exactly')
        records[key] = {'task': doc, 'raw': item['raw'], 'files': item['files'], 'summary': result,
            'source': {'boot_id': boot, 'task_id': doc['task_id'], 'bundle_id': prior['bundle_id'],
                       'receipt_sha256': sha256(receipt), 'receipt': str(receipt),
                       'program_files': {f: sha256(bundle / 'programs' / name / f)
                                         for f in ('graph.json', 'params.bin', 'module.so')},
                       'reuse': 'byte-identical graph/module/params/runtime/inputs; same placement/window/environment'}}
    with locked(run_dir):
        identity = {'boot_id': boot, 'bundle_id': current['bundle_id']}
        marker = run_dir / 'run.json'
        if marker.exists() and read(marker) != identity:
            raise ValueError('import destination belongs to another boot or capsule')
        if not marker.exists():
            write(marker, identity)
        imports = {**identity, 'records': records}
        imports['artifact_sha256'] = digest(imports)
        write(run_dir / 'imported_components.json', imports, replace=True)
    return {'imported': len(records), 'original_boot_id': boot,
            'original_task_ids_preserved': True, 'new_measurements': 0}


def run(bundle, run_dir, local_config, *, execute_board=False, limit=None):
    bundle, run_dir = Path(bundle).resolve(), Path(run_dir).resolve()
    manifest = validate(bundle)
    ssh = matched_ssh(read(local_config)['board'], execute_board=execute_board)
    with locked(run_dir):
        boot = ssh.boot_id()
        identity = {'boot_id': boot, 'bundle_id': manifest['bundle_id']}
        marker = run_dir / 'run.json'
        if marker.exists() and read(marker) != identity:
            raise ValueError('profile batch boot or program identity changed')
        if not marker.exists():
            write(marker, identity)
        remote = ssh.deploy(bundle, run_dir)
        ssh.wait_idle(boot)
        frequencies = ssh.command('for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; do cat "$f"; done').split()
        if frequencies != ['1066666'] * 4:
            raise ValueError('CPU frequency changed; no automatic retuning')
        order = read(bundle / 'plans/profile_order.json')
        import_path = run_dir / 'imported_components.json'
        imported = read(import_path) if import_path.exists() else {**identity, 'records': {}}
        if any(imported[k] != identity[k] for k in identity):
            raise ValueError('imported profile boot or capsule differs')
        if import_path.exists() and imported.get('artifact_sha256') != digest(
                {k: v for k, v in imported.items() if k != 'artifact_sha256'}):
            raise ValueError('imported profile seal differs')
        records = {}
        newly_measured = 0
        for key in order:
            doc = make_task(bundle, key)
            receipt = run_dir / 'accepted' / boot / (doc['task_id'] + '.json')
            if key in imported['records']:
                item = imported['records'][key]
                verify_inventory(item['raw'], item['files'])
                if sha256(item['source']['receipt']) != item['source']['receipt_sha256']:
                    raise ValueError('original imported receipt changed')
                original_receipt = read(item['source']['receipt'])
                if any(original_receipt[k] != item[k] for k in ('task', 'raw', 'files', 'summary')):
                    raise ValueError('imported profile fields differ from original receipt')
                if item['task']['configuration'] != doc['configuration']:
                    raise ValueError('imported resource configuration differs')
                name = doc['stages'][0]['program']
                if any(sha256(bundle / 'programs' / name / f) != h
                       for f, h in item['source']['program_files'].items()):
                    raise ValueError('imported program bytes differ')
                result = summary(item['raw'], item['task'], bundle)
                if result != item['summary'] or item['source']['boot_id'] != boot:
                    raise ValueError('imported isolated profile does not replay')
                source = item['source']
            elif receipt.exists():
                item = read(receipt)
                if item['task'] != doc or item['boot_id'] != boot:
                    raise ValueError('profile receipt identity differs')
                verify_inventory(item['raw'], item['files'])
                result = summary(item['raw'], doc, bundle)
                if result != item['summary']:
                    raise ValueError('replayed profile summary differs')
                source = {'boot_id': boot, 'task_id': doc['task_id'], 'receipt_sha256': sha256(receipt)}
            else:
                if limit is not None and newly_measured >= limit:
                    continue
                raw = ssh.measure(doc, bundle, remote, run_dir, boot)
                result = summary(raw, doc, bundle)
                write(receipt, {'task': doc, 'boot_id': boot, 'raw': str(raw),
                    'files': inventory(raw), 'summary': result})
                # collect() already fsynced the raw tree; write() fsynced the receipt.
                ssh.acknowledge(doc, run_dir, boot)
                newly_measured += 1
                source = {'boot_id': boot, 'task_id': doc['task_id'], 'receipt_sha256': sha256(receipt)}
            records[key] = {**result, 'configuration': doc['configuration'],
                'source': source}
            print('accepted', len(records), '/', len(order), doc['stages'][0]['signature'], flush=True)
        result = {'status': 'COMPLETE' if len(records) == len(order) else 'PARTIAL',
            'accepted': len(records), 'required': len(order), 'boot_id': boot,
            'imported_components': len(imported['records']),
            'pipeline_FPS_used': False, 'costs': records}
        write(run_dir / 'isolated_costs.json', result, replace=True)
        sync_tree(run_dir)
        return {k: v for k, v in result.items() if k != 'costs'}
