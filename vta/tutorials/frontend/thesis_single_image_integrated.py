#!/usr/bin/env python3
"""Qualify current-program K2/anchor/Q1 and collect the approved smoke blocks."""
import argparse
import json
import os
from pathlib import Path
import tarfile
import uuid
from types import SimpleNamespace
import numpy as np
from run_thesis_single_image_smoke import SSH, checked
from thesis_supplement_run import RunStore, sha256, write_new_json
from thesis_single_image_native import native, runner_name
from thesis_single_image_c1_windows import stages_for
from audit_shared_buffer_storage import audit_package, sha
from make_shared_buffer_plan import make_plan
from aggregate_thesis_vta_queue_profile import aggregate, load_records
from build_thesis_command_capacity_certificate import create_profile, create_certificate


def boot_identity(r):
    path = r / 'analysis/c1_boot3_board.json'
    if not path.exists():
        path = r / 'analysis/board_boot.json'
    return json.loads(path.read_text())['boot_id'], 3 if 'boot3' in path.name else 1


def make_package(store, name, stages, candidate_id):
    r = store.directory
    directory = r / 'raw/condition_packages' / name
    directory.mkdir(parents=True, exist_ok=False)
    rows = []
    with np.load(r / 'raw/host_reference/reference_boundaries.npz', allow_pickle=False) as data:
        refs = {key: data[key] for key in data.files}
    for index, stage in enumerate(stages):
        output = directory / 'stages' / str(index)
        output.mkdir(parents=True)
        source = r / 'raw' / stage['directory']
        for filename in ('graph.json', 'module.so', 'params.bin'):
            if sha256(source / filename) != stage['files'][filename]['sha256']:
                raise ValueError('Frozen stage changed before package assembly')
            os.link(source / filename, output / filename)
        rows.append({'index': index, 'name': stage['name'], 'device': stage['device'],
            'graph': f'stages/{index}/graph.json', 'lib': f'stages/{index}/module.so',
            'params': f'stages/{index}/params.bin', 'input_names': stage['input_names'],
            'artifact_sha256': {'graph_sha256': sha256(output / 'graph.json'),
                'lib_sha256': sha256(output / 'module.so'), 'params_sha256': sha256(output / 'params.bin')},
            'output_schema': {'slots': [{'node_id': node, 'shape': list(refs[f'node_{node}'].shape),
                'dtype': str(refs[f'node_{node}'].dtype)} for node in stage['output_node_ids']]}})
    binding = json.loads((r / 'analysis/board_artifact_bindings.json').read_text())
    os.link(binding['libtvm_runtime.so']['host_path'], directory / 'libtvm_runtime.so')
    write_new_json(directory / 'manifest.json', {'schema': 'thesis_single_image_package_v1',
        'candidate_id': candidate_id, 'canonical_qir_sha256': store.contract['canonical_qir_sha256'],
        'extra_quantize_calls': 0, 'stages': rows})
    checks = directory / 'expected_checks.sha256'
    checks.write_text(''.join(sha256(p) + '  ' + str(p.relative_to(directory)) + '\n'
        for p in sorted(directory.rglob('*')) if p.is_file() and p != checks))
    return directory


def deploy_controls(store, name):
    r = store.directory
    directory = r / 'raw/condition_controls' / name
    archive = r / 'raw' / (name + '_controls_' + uuid.uuid4().hex[:8] + '.tgz')
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(directory, arcname='condition_controls/' + name)
    board = json.loads((r / 'analysis/board_directory.json').read_text())['path']
    checked(store, SSH + ['cat > ' + board + '/' + archive.name], name + '_control_transfer',
            'one_time_preparation', 60, archive)
    script = r / 'raw' / (archive.stem + '_unpack.sh')
    lines = ['set -eu', 'cd ' + board, 'tar -xzf ' + archive.name]
    for path in sorted(directory.rglob('*')):
        if path.is_file():
            relative = 'condition_controls/' + name + '/' + str(path.relative_to(directory))
            lines.append('test "$(sha256sum ' + relative + ' | cut -d \' \' -f 1)" = ' + sha256(path))
    script.write_text('\n'.join(lines) + '\n')
    checked(store, SSH + ['sh -s'], name + '_control_verify', 'one_time_preparation', 60, script)


def make_anchor(store, name, package):
    r = store.directory
    audit = audit_package(package, package / 'expected_checks.sha256')
    audit['audit_source_sha256'] = sha(Path(__file__).with_name('audit_shared_buffer_storage.py'))
    destination = r / 'analysis' / (name + '_input_pool_audit.json')
    write_new_json(destination, audit)
    binding = json.loads((r / 'analysis/board_artifact_bindings.json').read_text())
    plan = make_plan(audit, {'artifacts': {'libvta.so': binding['libvta.so']['sha256']}})
    plan['audit_sha256'] = sha256(destination)
    plan['planner_sha256'] = sha(Path(__file__).with_name('make_shared_buffer_plan.py'))
    output = r / 'raw/condition_controls' / name
    output.mkdir(parents=True, exist_ok=False)
    write_new_json(output / 'shared_buffer_plan.json', plan)
    return 'condition_controls/' + name + '/shared_buffer_plan.json'


def make_q1(store, name, package, stages, threads, anchor):
    r = store.directory
    boot, index = boot_identity(r)
    identity = 'b' + str(index) + '_' + name + '_default_worker_profile'
    native(store, identity, stages, threads, index, boot, 'profile', 20,
           managed=True, anchor_plan=anchor, profile=True)
    source = r / 'raw/native_sessions' / identity / 'native.stderr'
    environment = {'VTA_REPLAY_POLICY': 'disabled', 'VTA_INSN_SUBMIT_THRESHOLD_BYTES': None,
        'VTA_INSN_BUFFER_BYTES': 1 << 25, 'VTA_UOP_BUFFER_BYTES': 1 << 25}
    status = aggregate(load_records(source), environment)
    status_path = r / 'analysis' / (name + '_all_worker_queue_status.json')
    write_new_json(status_path, {**status, 'raw_stderr_sha256': sha256(source)})
    binding = json.loads((r / 'analysis/board_artifact_bindings.json').read_text())
    args = SimpleNamespace(package_dir=str(package), runner=str(r / 'raw' / runner_name(r)),
        runtime=binding['libtvm_runtime.so']['host_path'], driver=binding['libvta.so']['host_path'],
        bitstream=binding['vta_hpc.bit']['host_path'], shared_buffer_plan=str(r / 'raw' / anchor) if anchor else None,
        vta_max_xfer_bytes=1 << 25, queue_status=str(status_path), evidence_scope='board_profiled_program',
        boot_id=boot, output_dir=str(r / 'raw/condition_controls' / name / 'profile'))
    profile_path = create_profile(args)
    args.peak_profile = str(profile_path)
    args.mode = 'q1'
    args.output_dir = str(r / 'raw/condition_controls' / name / 'q1')
    create_certificate(args)
    deploy_controls(store, name)
    return 'condition_controls/' + name + '/q1/command_capacity_certificate.json'


def qualify(store, name, condition):
    r = store.directory
    boot, index = boot_identity(r)
    identity = f'b{index}_{name}_integrated_correctness'
    metric = native(store, identity, condition['stages'], condition['threads'], index, boot,
        'correctness', 10, managed=condition.get('managed', False),
        anchor_plan=condition.get('anchor'), capacity=condition.get('capacity'))
    if condition.get('anchor'):
        plan = json.loads((r / 'raw' / condition['anchor']).read_text())
        text = (r / 'raw/native_sessions' / identity / 'native.stdout').read_text()
        for edge in plan['edges']:
            if edge['source'] == 'pool-anchor' and f'edge={edge["producer_stage"]}' not in text:
                raise ValueError('Planned anchor has no runtime evidence')
            if edge['source'] == 'pool-anchor' and f'edge={edge["producer_stage"]} tensor=0 source=pool-anchor' not in text:
                # Actual marker puts tensor and source in a fixed order.
                import re
                if not re.search(r'edge=' + str(edge['producer_stage']) + r'[^\n]*source=pool-anchor', text):
                    raise ValueError('Runtime rejected a planned anchor; do not label it Ca')
    pipeline_metric = native(store, identity + '_pipeline', condition['stages'], condition['threads'], index, boot,
        'pipeline_correctness', 10, managed=condition.get('managed', False),
        anchor_plan=condition.get('anchor'), capacity=condition.get('capacity'))
    metric['pipeline_correctness_identity'] = pipeline_metric['identity']
    return metric


def exact_pair(store, names):
    r = store.directory
    boot, index = boot_identity(r)
    board = json.loads((r / 'analysis/board_directory.json').read_text())['path']
    hashes = []
    for suffix in ('', '_pipeline'):
        for name in names:
            directory = r / 'raw/native_sessions' / (f'b{index}_{name}_integrated_correctness' + suffix)
            rows = [json.loads(line) for line in (directory / 'result.jsonl').read_text().splitlines()]
            remote_output = Path(board) / 'native_sessions' / directory.name
            hashes.append([sha256(directory / Path(row['raw_output_files'][0]['path']).relative_to(remote_output)) for row in rows])
    if any(values != hashes[0] for values in hashes[1:]):
        raise ValueError('Same-program copy/shared final output is not byte-exact')
    write_new_json(r / 'analysis' / ('_'.join(names) + '_byte_exact.json'),
        {'conditions': names, 'modes': ['serial', 'pipeline'], 'calls_per_condition_per_mode': 10,
            'output_sha256': hashes, 'passed': True})


def integrate(store):
    r = store.directory
    selected = json.loads((r / 'analysis/B_selection.json').read_text())
    frozen = json.loads((r / 'raw/c1_frozen/fair_support_contract.json').read_text())
    a_candidates = [candidate for candidate in frozen['support']
                    if candidate['reasonable_a'] and candidate['thread_vector'] == store.contract['A_thread_vector']]
    if len(a_candidates) != 1:
        raise ValueError('Predetermined reasonable A must resolve once in the frozen support')
    a_candidate = a_candidates[0]
    a_stages = stages_for(r, a_candidate)
    renderer = json.loads((r / 'analysis/native_runner_selection.json').read_text())
    write_new_json(r / 'analysis/selection_deployment_lineage.json', {
        'source_A_candidate_id': a_candidate['candidate_id'],
        'source_B_candidate_id': selected['candidate_id'],
        'source_selection_runner': renderer.get('source_C1_runner_selection', renderer),
        'final_deployment_runner': renderer,
        'source_selection_contract_sha256': sha256(r / 'raw/c1_measurement_contract.json'),
        'boundary': 'C1 candidate IDs identify the original selection deployment. Original windows remain original-runner measurements; final six conditions use the same repaired runner and are freshly qualified and measured.'})
    write_new_json(r / 'analysis/A_fair_program_binding.json', {
        'candidate_id': a_candidate['candidate_id'], 'stages': a_stages,
        'support_sha256': sha256(r / 'raw/c1_frozen/fair_support_contract.json'),
        'selection_rule': 'reasonable_a designation frozen before labels; root fixed thread vector; no performance-based baseline choice',
        'fairness': 'Same compilation and parameter serialization strategy as B; original A diagnostic observations retain original artifacts and are not final A timings'})
    b = {'stages': selected['stages'], 'threads': selected['candidate']['thread_vector'],
         'managed': False, 'c1_candidate_id': selected['candidate_id'],
         'c1_candidate_id_scope': 'original selection deployment', 'final_runner_sha256': renderer['sha256']}
    package = make_package(store, 'B', b['stages'], selected['candidate_id'])
    c = {**b, 'managed': True}
    anchor = make_anchor(store, 'Ca', package)
    deploy_controls(store, 'Ca')
    ca = {**c, 'anchor': anchor}
    write_new_json(r / 'raw/integrated_qualification_contract.json', {
        'serial_correctness': {'calls_per_condition': 10, 'actual_storage_delayed_read_ms': 10, 'all_frontiers': True},
        'managed_slot_policy': {'serial_C': 1, 'serial_anchor_variants': 2, 'pipeline_all_managed': 2,
            'limitation': 'Existing serial CLI constraints require this difference; C-to-Ca serial latency cannot isolate anchor alone. Pipeline allocator comparisons hold K=2.'},
        'pipeline_correctness': {'calls_per_condition': 10, 'all_complete_final_outputs': True,
            'copy_route_all_frontiers': True, 'extra_delay_in_pipeline': False},
        'performance_contract_sha256': sha256(r / 'contract.json'), 'extra_quantize_calls': 0,
        'purpose': 'verify actual pipeline handoff in addition to serial completion stability; no task accuracy claim'})
    initial = {'A': {'stages': a_stages, 'threads': store.contract['A_thread_vector'], 'managed': False},
        'B': b, 'C': c, 'Ca': ca}
    initial['A']['c1_candidate_id'] = a_candidate['candidate_id']
    initial['A']['c1_candidate_id_scope'] = 'original selection deployment'
    initial['A']['final_runner_sha256'] = renderer['sha256']
    write_new_json(r / 'raw/ABCCa_conditions.json', initial)
    for name, condition in initial.items():
        qualify(store, name, condition)
        print(name + ': 10 complete outputs/all frontiers/10ms stability PASS', flush=True)
    exact_pair(store, ['B', 'C', 'Ca'])
    # D carries the same anchor plan bytes, not an altered graph or new quantization.
    d_directory = r / 'raw/condition_controls/D'
    d_directory.mkdir(parents=True)
    d_anchor = 'condition_controls/D/shared_buffer_plan.json'
    os.link(r / 'raw' / anchor, r / 'raw' / d_anchor)
    deploy_controls(store, 'D')
    capacity = make_q1(store, 'D', package, b['stages'], b['threads'], d_anchor)
    d = {**c, 'anchor': d_anchor, 'capacity': capacity}
    qualify(store, 'D', d)
    exact_pair(store, ['B', 'C', 'Ca', 'D'])
    write_new_json(r / 'raw/ABCCaD_conditions.json', {**initial, 'D': d})
    print('D: actual-worker FINISH peak Q1 certificate +10 complete outputs/10ms stability PASS', flush=True)


def performance(store):
    r = store.directory
    conditions = json.loads((r / 'raw/frozen_six_conditions.json').read_text())
    if set(conditions) != {'A', 'B', 'C', 'Ca', 'D', 'E'}:
        raise ValueError('Six qualified conditions required')
    boot, index = boot_identity(r)
    results = []
    for block, order in enumerate(store.contract['performance']['orders']):
        for name in order:
            condition = conditions[name]
            for mode, runs in [('serial', 20), ('window', 307)]:
                identity = f'b{index}_block{block}_{name}_{mode}'
                metric = native(store, identity, condition['stages'], condition['threads'], index, boot,
                    mode, runs, managed=condition.get('managed', False),
                    anchor_plan=condition.get('anchor'), capacity=condition.get('capacity'))
                results.append({'condition': name, 'block': block, **metric})
                print(f'block{block} {name} {mode}: completed {runs}, saved', flush=True)
    write_new_json(r / 'analysis/smoke_performance_blocks.json', {'scope': 'one-boot paired forward/reverse smoke only',
        'boot_id': boot, 'inferential_confidence_interval': None, 'results': results})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('integrate', 'performance'))
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    store = RunStore(args.run_dir)
    if (store.directory / 'invalid_session.json').exists():
        raise ValueError('Invalid run cannot continue')
    try:
        integrate(store) if args.operation == 'integrate' else performance(store)
    except BaseException as error:
        write_new_json(store.directory / 'invalid_session.json', {'operation': args.operation,
            'reason': str(error), 'all_paid_costs_retained': True, 'stop_later_conditions': True})
        store.event('single_image_run_invalid', operation=args.operation, reason=str(error))
        raise
