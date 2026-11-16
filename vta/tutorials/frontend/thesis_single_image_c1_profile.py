#!/usr/bin/env python3
"""Profile actual canonical components, verify bytes, then freeze C1 orders."""
import argparse
import json
from pathlib import Path
import shlex
import statistics
import tarfile
import numpy as np
from run_thesis_single_image_smoke import SSH, checked, ROOT
from thesis_completion_context_reset import reset
from thesis_supplement_run import RunStore, sha256, write_new_json
from freeze_thesis_c1_fair_support import freeze, SCHEMA
from thesis_single_image_board_check import numeric
from thesis_single_image_native import runner_name


def prepare(store):
    r = store.directory
    components = r / 'raw/c1_components_int8'
    manifest = json.loads((components / 'component_manifest.json').read_text())
    with np.load(r / 'raw/host_reference/reference_boundaries.npz', allow_pickle=False) as data:
        values = {key: data[key] for key in data.files}
    profile = r / 'raw/c1_component_inputs'
    profile.mkdir()
    for record in manifest['programs']:
        directory = profile / record['name']
        directory.mkdir()
        for i, node in enumerate(record['input_node_ids']):
            (directory / f'input{i}.bin').write_bytes(values[f'node_{node}'].tobytes())
    archive = r / 'raw/c1_components_bundle.tgz'
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(components, arcname=components.name)
        tar.add(profile, arcname=profile.name)
        tar.add(r / 'raw' / runner_name(r), arcname=runner_name(r))
    board = json.loads((r / 'analysis/board_directory.json').read_text())['path']
    checked(store, SSH + ['cat > ' + board + '/c1_components_bundle.tgz'], 'c1_components_transfer', 'one_time_preparation', 120, archive)
    checked(store, SSH + ['cd ' + board + ' && tar -xzf c1_components_bundle.tgz && chmod 755 ' + runner_name(r)],
            'c1_components_unpack', 'one_time_preparation', 60)
    # Verify every compiled component in RAM before any profiling.
    script = r / 'raw/c1_component_hash_check.sh'
    checks = ['set -eu', 'cd ' + board]
    for record in manifest['programs']:
        for name, value in record['files'].items():
            path = components.name + '/' + record['name'] + '/' + name
            checks.append('test "$(sha256sum ' + path + ' | cut -d \' \' -f 1)" = ' + value['sha256'])
    checks.append('test "$(sha256sum ' + runner_name(r) + ' | cut -d \' \' -f 1)" = ' + sha256(r / 'raw' / runner_name(r)))
    script.write_text('\n'.join(checks) + '\n')
    checked(store, SSH + ['sh -s'], 'c1_component_hash_verification', 'one_time_preparation', 60, script)


def profile(store):
    r = store.directory
    board = json.loads((r / 'analysis/board_directory.json').read_text())['path']
    boot = json.loads((r / 'analysis/board_boot.json').read_text())['boot_id']
    manifest = json.loads((r / 'raw/c1_components_int8/component_manifest.json').read_text())
    with np.load(r / 'raw/host_reference/reference_boundaries.npz', allow_pickle=False) as data:
        references = {key: data[key] for key in data.files}
    observations = []
    for component in manifest['programs']:
        for threads in ((1, 4) if component['device'] == 'cpu' else (1,)):
            identity = component['name'] + '_t' + str(threads)
            reset(store, 1, boot, 'component_' + identity, board)
            destination = r / 'raw/c1_profiles' / identity
            destination.mkdir(parents=True, exist_ok=False)
            path = 'c1_components_int8/' + component['name']
            remote_output = board + '/c1_profiles/' + identity
            command = [board + '/' + runner_name(r),
                '--stage0-name', component['name'], '--stage0-device', component['device'],
                '--stage0-graph', board + '/' + path + '/graph.json', '--stage0-lib', board + '/' + path + '/module.so',
                '--stage0-params', board + '/' + path + '/params.bin', '--stage0-input-names', ','.join(component['input_names']),
                '--stage0-runtime-num-threads', str(threads), '--runtime-num-threads', str(threads),
                '--input-files', ','.join(board + '/c1_component_inputs/' + component['name'] + f'/input{i}.bin' for i in range(len(component['input_names']))),
                '--runs', '20', '--serial', '--output-mode', 'raw', '--output-jsonl', remote_output + '/result.jsonl',
                '--output-dump-dir', remote_output + '/outputs']
            if component['device'] == 'cpu':
                command += ['--stage0-cpu-affinity', ','.join(str(i) for i in range(threads))]
            script = destination / 'native_profile.sh'
            script.write_text('set -eu\nmkdir -p ' + remote_output + '\nexport LD_LIBRARY_PATH=' + board + '\nexport LD_PRELOAD=' + board + '/libvta.so\nexport TVM_NUM_THREADS=' + str(threads) + '\nexport VTA_REPLAY_POLICY=disabled VTA_QUEUE_DIAGNOSTICS=1 C3_COMPLETION_TRACE=1\nexport TVM_THREAD_POOL_SPIN_COUNT=0\nunset VTA_INSN_BUFFER_BYTES VTA_UOP_BUFFER_BYTES VTA_INSN_SUBMIT_THRESHOLD_BYTES\n' + shlex.join(command) + ' > ' + remote_output + '/native.log 2>&1\ntar -cf - -C ' + remote_output + ' result.jsonl outputs native.log\n')
            result = checked(store, SSH + ['sh -s'], 'native_component_profile_' + identity, 'one_time_preparation', 60, script)
            with tarfile.open(r / result['stdout']) as tar:
                for item in tar.getmembers():
                    if item.name.startswith('/') or '..' in Path(item.name).parts or item.issym() or item.islnk():
                        raise ValueError('Unsafe archive member')
                tar.extractall(destination)
            rows = [json.loads(line) for line in (destination / 'result.jsonl').read_text().splitlines()]
            if len(rows) != 20:
                raise ValueError('Component profile missing calls')
            for row in rows:
                if len(row['raw_output_files']) != len(component['output_node_ids']):
                    raise ValueError('Missing complete frontier output')
                for item, node in zip(row['raw_output_files'], component['output_node_ids']):
                    filename = Path(item['path']).relative_to(remote_output)
                    actual = np.frombuffer((destination / filename).read_bytes(), dtype=item['dtype']).reshape(item['shape'])
                    numeric(actual, references[f'node_{node}'], store.contract['numeric'])
            observations.append({'component': component['name'], 'threads': threads,
                'service_ms': statistics.median(row['stage0_ms'] for row in rows[5:]),
                'raw_rows': str((destination / 'result.jsonl').relative_to(r)),
                'raw_rows_sha256': sha256(destination / 'result.jsonl'),
                'numeric_calls': 20, 'all_frontier_outputs_passed': True})
            print(identity + ': 20 complete frontier checks PASS, component profile saved', flush=True)
    write_new_json(r / 'analysis/c1_component_profile.json', {'scope': 'component preparation costs, not candidate II labels',
        'boot_id': boot, 'discard_initial_calls': 5, 'scored_calls_per_component': 15, 'observations': observations})
    costs = {(row['component'], row['threads']): row['service_ms'] for row in observations}
    lookup = {(p['begin'], p['end_exclusive'], p['device']): p for p in manifest['programs']}
    candidates = []
    for topology in manifest['topologies']:
        for threads in ([4, 1, 4], [1, 1, 1]):
            scores = [costs[(lookup[tuple(spec)]['name'], count)] for spec, count in zip(topology['ranges'], threads)]
            bytes_materialized = 0
            for spec in topology['ranges']:
                record = lookup[tuple(spec)]
                bytes_materialized += sum(references[f'node_{n}'].nbytes for n in record['input_node_ids'])
                bytes_materialized += sum(references[f'node_{n}'].nbytes for n in record['output_node_ids'])
            identity = {'canonical_qir_sha256': store.contract['canonical_qir_sha256'],
                'input_sha256': sha256(r / 'raw/inputs/input.npy'),
                'runner_sha256': sha256(r / 'raw' / runner_name(r)),
                'baseline_history_sha256': store.contract['baseline_history_sha256'],
                'runtime_bindings': json.loads((r / 'analysis/board_artifact_bindings.json').read_text()),
                'stages': [lookup[tuple(spec)]['files'] for spec in topology['ranges']],
                'threads': threads, 'affinity': ['prefix', None, 'prefix'], 'policy': 'ordinary_copy'}
            import hashlib
            digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            candidates.append({'candidate_id': topology['topology_id'] + '_t' + '_'.join(map(str, threads)) + '_' + digest[:12],
                'full_identity_sha256': digest, 'full_identity': identity,
                'topology_id': topology['topology_id'], 'm1_score_ms': max(scores), 'component_service_ms': scores,
                'vta_islands': 1, 'stage_devices': ['cpu', 'vta', 'cpu'], 'thread_vector': threads,
                'static_legal': True, 'realized_policy': 'ordinary_copy', 'reasonable_a': topology['reasonable_a'],
                'framework_materialization_bytes': bytes_materialized, 'vta_dma_calls': None,
                'vta_internal_dma_bytes': None, 'additional_slot_bytes': 0, 'ranges': topology['ranges']})
    payload = {'schema': SCHEMA, 'new_performance_labels_read': False, 'candidates': candidates,
        'm1_definition': 'max of current canonical native component median service costs at matching prefix-mask threads; copy adapters included',
        'knownness': 'internal DMA fields unknown; semantic band must fall back to M1, never fill zero',
        'candidate_scope': '8 one-island integer-entry cut pairs; multi-island queue handoff unqualified; not exhaustive topology search'}
    write_new_json(r / 'raw/c1_support_input.json', payload)
    frozen = freeze(payload)
    directory = r / 'raw/c1_frozen'
    directory.mkdir()
    write_new_json(directory / 'fair_support_contract.json', frozen)
    write_new_json(directory / 'artifact_hashes.json', {'fair_support_contract.json': sha256(directory / 'fair_support_contract.json')})
    store.event('c1_support_frozen_before_candidate_windows', support_sha256=sha256(directory / 'fair_support_contract.json'))
    print('C1 support frozen: 8 topologies /16 full thread configurations, matched four orders', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation', choices=('prepare', 'profile'))
    p.add_argument('run_dir', type=Path)
    args = p.parse_args()
    s = RunStore(args.run_dir)
    if (s.directory / 'invalid_session.json').exists():
        raise ValueError('Invalid run')
    try:
        prepare(s) if args.operation == 'prepare' else profile(s)
    except BaseException as error:
        write_new_json(s.directory / 'invalid_session.json', {'operation': 'c1_' + args.operation, 'reason': str(error), 'all_paid_costs_retained': True})
        raise
