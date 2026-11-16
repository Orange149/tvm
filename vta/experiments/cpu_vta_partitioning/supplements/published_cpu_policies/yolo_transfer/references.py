"""Independent ARM full-output/frontier references, reusing checked SSH lifecycle."""
from pathlib import Path

from cpu_vta_partitioning.common import (digest, inventory, locked, read, sha256,
                                        sync_tree, verify_inventory, write)
from cpu_vta_partitioning.prepare.build import copy_checked


def prepare(build, output, parent):
    build, output, parent = map(lambda p: Path(p).resolve(), (build, output, parent))
    if output.exists():
        raise FileExistsError('reference capsules are immutable; select a new output directory')
    manifest = read(parent / 'manifest.json')
    programs = {}
    with locked(output):
        for name in ('reference_full', 'reference_frontiers'):
            source = build / 'programs' / name
            record = read(source / 'record.json')
            verify_inventory(source, record['files'])
            for filename in ('graph.json', 'params.bin', 'module.so', 'record.json'):
                copy_checked(source / filename, output / 'programs' / name / filename)
            programs[name] = {**record, 'name': name, 'begin': 0, 'end': 14,
                              'device': 'cpu'}
        if programs['reference_full']['canonical_sha256'] != programs['reference_frontiers']['canonical_sha256']:
            raise ValueError('full graph and probe do not describe the same common DAG')
        for filename, entry in manifest['files'].items():
            if filename.startswith(('runtime/', 'firmware/')):
                copy_checked(parent / filename, output / filename, entry['sha256'])
        runner = 'runner/vta_stage_pipeline_runner_pinned'
        runner_hash = copy_checked(parent / runner, output / 'runner/vta_stage_pipeline_runner',
                                   manifest['files'][runner]['sha256'])
        copy_checked(build / 'inputs/000.bin', output / 'inputs/000.bin')
        for filename in ('compiler_identity.json', 'programs.json'):
            copy_checked(build / 'plans' / filename, output / 'sources/build' / filename)
        source_package = Path(__file__).resolve().parent
        for filename in ('references.py', '__init__.py'):
            copy_checked(source_package / filename, output / 'sources/yolo_transfer' / filename)
        import cpu_vta_partitioning
        core_source = Path(cpu_vta_partitioning.__file__).resolve().parent
        for source in core_source.rglob('*.py'):
            copy_checked(source, output / 'sources/base_controller/cpu_vta_partitioning'
                         / source.relative_to(core_source))
        write(output / 'plans/programs.json', programs)
        write(output / 'plans/inputs.json', {'records': [{'index': 0, 'input': 'inputs/000.bin', 'frontiers': {}}]})
        write(output / 'plans/board_execution.json', {
            'boot_policy': 'one_boot_no_automatic_reboot', 'udmabuf_bytes': 201326592,
            'storage_margin_bytes': 268435456, 'scope': 'two_single_frame_CPU_reference_sessions'})
        files = {name: entry for name, entry in inventory(output).items()
                 if name.startswith(('programs/', 'runtime/', 'runner/', 'firmware/', 'inputs/'))
                 and name.rsplit('/', 1)[-1] != 'record.json'}
        files['plans/board_execution.json'] = inventory(output / 'plans')['board_execution.json']
        write(output / 'plans/board_payload.json', {'files': files,
            'unpacked_bytes': sum(r['bytes'] for r in files.values())})
        identity = {'model': 'YOLOv3-tiny416', 'scope': 'independent_ARM_reference',
                    'canonical_sha256': programs['reference_full']['canonical_sha256'],
                    'runner_sha256': runner_hash, 'files': inventory(output, ['.lock'])}
        write(output / 'manifest.json', {'bundle_id': digest(identity)[:24], **identity})
        sync_tree(output)
    return {'bundle': str(output), 'bundle_id': read(output / 'manifest.json')['bundle_id'],
            'payload_bytes': sum(r['bytes'] for r in files.values())}


def validate(bundle):
    bundle = Path(bundle).resolve()
    manifest = read(bundle / 'manifest.json')
    if manifest['bundle_id'] != digest({k: v for k, v in manifest.items() if k != 'bundle_id'})[:24]:
        raise ValueError('reference capsule identity differs')
    verify_inventory(bundle, manifest['files'])
    verify_inventory(bundle, read(bundle / 'plans/board_payload.json')['files'])
    import cpu_vta_partitioning
    core = Path(cpu_vta_partitioning.__file__).resolve().parent
    snapshot = bundle / 'sources/base_controller/cpu_vta_partitioning'
    for source in snapshot.rglob('*.py'):
        if sha256(source) != sha256(core / source.relative_to(snapshot)):
            raise ValueError('loaded reference transport differs from the frozen source')
    if sha256(Path(__file__)) != sha256(bundle / 'sources/yolo_transfer/references.py'):
        raise ValueError('loaded reference controller differs from the frozen source')
    return manifest


def task(bundle, name):
    manifest = validate(bundle)
    programs = read(Path(bundle) / 'plans/programs.json')
    if name not in programs:
        raise ValueError('unknown reference program')
    doc = {'kind': 'qualify_component', 'stages': [{'program': name, 'device': 'cpu',
            'threads': 1, 'mask': [0]}], 'environment': {'poll_sleep_ns': 1000,
            'post_start_sleep_ns': 1000}, 'warmup': 0, 'scored': 1, 'tail': 0,
           'input_indices': [0], 'bundle_id': manifest['bundle_id'],
           'purpose': 'independent_complete_ARM_reference_not_performance'}
    doc['task_id'] = digest(doc)[:24]
    return doc


def output_arrays(raw, document, bundle):
    import numpy as np
    from cpu_vta_partitioning.measurement import affinity, frames, input_sequence
    raw, bundle = Path(raw), Path(bundle)
    rows = frames(raw / 'result.jsonl', 1)
    input_sequence(rows, document)
    affinity(read(raw / 'affinity.json'), document['stages'])
    row = rows[0]
    if len(row['stage_raw_outputs']) != 1:
        raise ValueError('reference stage outputs missing')
    dumps = row['stage_raw_outputs'][0]
    program = read(bundle / 'plans/programs.json')[document['stages'][0]['program']]
    if len(dumps['output_files']) != len(program['output_types']) or len(dumps['outputs']) != len(program['output_types']):
        raise ValueError('reference tensor outputs incomplete')
    arrays = []
    for index, (file, summary, typ) in enumerate(zip(dumps['output_files'], dumps['outputs'], program['output_types'])):
        if (file['index'] != index or summary['index'] != index or file['shape'] != typ['shape']
                or file['dtype'] != typ['dtype'] or summary['shape'] != typ['shape']
                or summary['dtype'] != typ['dtype']):
            raise ValueError('reference tensor type/order differs')
        path = raw / 'outputs/run_0' / Path(file['path']).name
        data = path.read_bytes()
        expected_bytes = int(np.prod(typ['shape'])) * np.dtype(typ['dtype']).itemsize
        value = 1469598103934665603
        for byte in data:
            value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
        if len(data) != expected_bytes or file['nbytes'] != len(data) or summary['nbytes'] != len(data):
            raise ValueError('reference tensor byte count differs')
        if summary['fnv1a64'].lower() != format(value, '016x'):
            raise ValueError('reference tensor payload hash differs')
        array = np.frombuffer(data, dtype=typ['dtype']).reshape(typ['shape'])
        if not np.isfinite(array).all():
            raise ValueError('nonfinite reference values')
        arrays.append(array)
    return program, arrays


def finish(bundle, run_dir, accepted):
    import numpy as np
    bundle, run_dir = Path(bundle), Path(run_dir)
    full, a = output_arrays(accepted['reference_full'], task(bundle, 'reference_full'), bundle)
    probe, b = output_arrays(accepted['reference_frontiers'], task(bundle, 'reference_frontiers'), bundle)
    if len(a) != 8 or any(i is None for i in probe['terminal_output_indices']):
        raise ValueError('both detection heads and metadata must be present')
    for expected, index in zip(a, probe['terminal_output_indices']):
        actual = b[index]
        if expected.dtype != actual.dtype or expected.shape != actual.shape or expected.tobytes() != actual.tobytes():
            raise ValueError('independent full graph and frontier probe are not ARM byte-exact')
    frontiers = run_dir / 'references'
    frontiers.mkdir(exist_ok=True)
    records = {}
    for node, array in zip(probe['output_node_ids'], b):
        np.save(frontiers / (str(node) + '.npy'), array)
        array.tofile(frontiers / (str(node) + '.bin'))
        records[str(node)] = {'dtype': str(array.dtype), 'shape': list(array.shape),
                             'npy': 'references/' + str(node) + '.npy',
                             'raw': 'references/' + str(node) + '.bin'}
    proof = {'status': 'ARM_COMPLETE_GRAPH_PROBE_BYTE_EXACT', 'outputs': 8,
        'frontier_tensor_count': len(records), 'canonical_sha256': full['canonical_sha256'],
        'boot_id': read(run_dir / 'run.json')['boot_id'], 'reference_execution_target': 'AArch64 LLVM CPU',
        'reference_capsule_id': read(bundle / 'manifest.json')['bundle_id'],
        'new_performance_windows': 0, 'input': {'path': str(bundle / 'inputs/000.bin'),
            'sha256': sha256(bundle / 'inputs/000.bin')}, 'frontiers': records,
        'files': inventory(frontiers), 'raw_sources': {k: str(v) for k, v in accepted.items()}}
    write(run_dir / 'REFERENCE_PROOF.json', proof, replace=True)
    sync_tree(run_dir)
    return proof


def run(bundle, run_dir, local_config, *, execute_board=False):
    from cpu_vta_partitioning.transport import SSH
    bundle, run_dir = Path(bundle).resolve(), Path(run_dir).resolve()
    manifest = validate(bundle)
    ssh = SSH(read(local_config)['board'], execute_board=execute_board)
    with locked(run_dir):
        boot = ssh.boot_id()
        path = run_dir / 'run.json'
        identity = {'boot_id': boot, 'bundle_id': manifest['bundle_id']}
        if path.exists():
            if read(path) != identity:
                raise ValueError('reference run boot or capsule identity changed')
        else:
            write(path, identity)
        remote = ssh.deploy(bundle, run_dir)
        accepted = {}
        for name in ('reference_full', 'reference_frontiers'):
            document = task(bundle, name)
            receipt = run_dir / 'accepted' / boot / (document['task_id'] + '.json')
            if receipt.exists():
                item = read(receipt)
                if item['task'] != document or item['boot_id'] != boot:
                    raise ValueError('reference receipt identity differs')
                raw = Path(item['raw'])
                verify_inventory(raw, item['files'])
                output_arrays(raw, document, bundle)
            else:
                raw = ssh.measure(document, bundle, remote, run_dir, boot)
                output_arrays(raw, document, bundle)
                write(receipt, {'task': document, 'boot_id': boot, 'raw': str(raw), 'files': inventory(raw)})
                sync_tree(run_dir)
                ssh.acknowledge(document, run_dir, boot)
            accepted[name] = raw
            print(name, 'ARM reference accepted', flush=True)
        proof = finish(bundle, run_dir, accepted)
        ssh.idle_command('sync', boot=boot)
        return {'status': proof['status'], 'outputs': proof['outputs'], 'frontiers': proof['frontier_tensor_count']}
