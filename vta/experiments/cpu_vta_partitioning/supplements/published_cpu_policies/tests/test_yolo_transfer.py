"""Branch/frontier and raw ARM reference integrity, independent of board access."""
from pathlib import Path
import json
import numpy as np
import pytest


def test_full_interval_domain_has_all_alternating_paths_and_atomic_curves():
    from yolo_transfer.domain import programs, summary, topologies
    rows = programs()
    info = summary()
    assert info['topology_count'] == 2509
    assert info['search_CPU_stages'] == 79
    assert info['CPU_atomic_only_extras'] == 2
    assert info['VTA_stages'] == 78
    assert info['total_basic_signature_count'] == 714
    assert len({r['name'] for r in rows}) == len(rows) == 159
    # Independent traversal of the qualified interval DAG, no combinations recurrence.
    edges = {(r['begin'], r['end'], r['device']) for r in rows if r['eligible_search']}
    def walk(position, device, islands, path):
        if position == 14:
            return {path} if device == 'vta' and 1 <= islands <= 3 else set()
        found = set()
        for begin, end, target in edges:
            count = islands + int(target == 'vta')
            if begin == position and target == device and count <= 3:
                found |= walk(end, 'vta' if device == 'cpu' else 'cpu', count,
                              path + ((target, begin, end),))
        return found
    assert walk(0, 'cpu', 0, ()) == set(topologies())
    assert all(any(r['device'] == 'cpu' and r['begin'] == i and r['end'] == i+1 for r in rows)
               for i in range(14))


def branch_graph():
    import tvm
    from tvm import relay
    from yolo_transfer.prepare.units import UnitGraph
    x = relay.var('data', shape=(1, 1, 4, 4))
    w = relay.const(np.ones((1, 1, 1, 1), dtype='float32'))
    a = relay.nn.relu(relay.nn.conv2d(x, w, channels=1, kernel_size=(1, 1)))
    b = relay.nn.conv2d(a, w, channels=1, kernel_size=(1, 1))
    c = relay.nn.conv2d(a, w, channels=1, kernel_size=(1, 1))
    # Distinct weights keep the two heads distinct even under structural comparison.
    c = relay.add(c, relay.const(np.float32(2)))
    result = relay.Tuple([relay.sigmoid(b), relay.sigmoid(c), relay.const(np.array([1, 2], 'int32'))])
    mod = tvm.IRModule.from_expr(relay.Function([x], result))
    return UnitGraph(mod, conv_order=(0, 1, 2))


def test_two_head_frontier_keeps_live_branch_and_every_output():
    from yolo_transfer.prepare.compiler import program_ir
    g = branch_graph()
    assert len(g.units) == 4
    assert len(g.frontier(2)) == 2
    assert len(g.frontier(3)) == 2
    assert len(g.frontier(4)) == 3
    for begin in range(4):
        for end in range(begin + 1, 5):
            assert g.recompose([(i, i + 1) for i in range(begin)] + [(begin, end)]
                               + [(i, i + 1) for i in range(end, 4)])
    full, incoming, outgoing = program_ir(g, {'kind': 'complete_graph'})
    probe, _, frontiers = program_ir(g, {'kind': 'complete_frontier_probe'})
    assert len(incoming) == 1 and len(outgoing) == 3
    assert all(node in frontiers for node in outgoing)
    assert len(probe['main'].body.fields) > len(full['main'].body.fields)


def test_bad_convolution_order_and_incomplete_dependency_are_rejected():
    from yolo_transfer.prepare.units import UnitGraph
    g = branch_graph()
    with pytest.raises(ValueError):
        UnitGraph(g.mod, conv_order=(1, 0, 2))
    original = g.frontier
    g.frontier = lambda boundary: original(boundary)[1:] if boundary == 2 else original(boundary)
    with pytest.raises(ValueError):
        g.stage(2, 3)


@pytest.mark.parametrize('both_packed', [False, True])
def test_routing_bridge_normalizes_mixed_layout_tuple_without_changing_values(both_packed):
    import tvm
    from tvm import relay
    from yolo_transfer.prepare.packing import pack_frontiers
    a = relay.var('a', shape=(1, 16, 2, 2))
    b = relay.var('b', shape=(1, 16, 2, 2))
    packed_a = relay.Call(tvm.ir.Op.get('annotation.bitpack_start'), [a])
    second = relay.Call(tvm.ir.Op.get('annotation.bitpack_start'), [b]) if both_packed else b
    concat = relay.concatenate([packed_a, second], axis=1)
    original = relay.Function([a, b], concat)
    transformed = pack_frontiers(original, 1, 16, 8)
    mod = relay.transform.InferType()(tvm.IRModule.from_expr(transformed))
    assert list(map(int, mod['main'].body.checked_type.shape)) == [1, 32, 2, 2]
    x = np.arange(64, dtype='float32').reshape(1, 16, 2, 2)
    y = -x
    result = relay.create_executor('vm', mod=mod, device=tvm.cpu(), target='llvm').evaluate()(x, y).numpy()
    np.testing.assert_array_equal(result, np.concatenate([x, y], axis=1))


def test_detection_logits_passthrough_does_not_require_256_channels():
    import tvm
    from tvm import relay
    from yolo_transfer.prepare.packing import pack_frontiers
    x = relay.var('x', shape=(1, 255, 2, 2), dtype='int8')
    start = relay.Call(tvm.ir.Op.get('annotation.bitpack_start'), [x])
    stop = relay.Call(tvm.ir.Op.get('annotation.bitpack_end'), [start])
    result = pack_frontiers(relay.Function([x], stop), 1, 16, 8)
    mod = relay.transform.InferType()(tvm.IRModule.from_expr(result))
    data = np.arange(1020, dtype='int64').astype('int8').reshape(1, 255, 2, 2)
    actual = relay.create_executor('vm', mod=mod, device=tvm.cpu(), target='llvm').evaluate()(data).numpy()
    np.testing.assert_array_equal(actual, data)


def reference_raw(tmp_path):
    def write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    raw = tmp_path / 'raw'
    bundle = tmp_path / 'bundle'
    data = np.array([1., 2.], 'float32').tobytes()
    path = raw / 'outputs/run_0/stage0_output_0.bin'
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    value = 1469598103934665603
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    file = {'index': 0, 'dtype': 'float32', 'shape': [2], 'nbytes': 8,
            'path': '/remote/outputs/run_0/stage0_output_0.bin'}
    summary = {**file, 'fnv1a64': format(value, '016x')}
    row = {'frame_id': 0, 'completion_index': 0, 'completion_ms': 1., 'input_index': 0,
           'stage_raw_outputs': [{'outputs': [summary], 'output_files': [file]}]}
    write(raw / 'result.jsonl', row)
    write(raw / 'affinity.json', {'stages': [{'requested_cpus': [0], 'thread_ids': [123],
        'effective_affinities': [{'cpus': [0], 'tid': 123}], 'active_runtime_threads': 1}]})
    write(bundle / 'plans/programs.json', {'test': {'output_types': [{'shape': [2], 'dtype': 'float32'}]}})
    document = {'kind': 'qualify_component', 'stages': [{'program': 'test', 'threads': 1, 'mask': [0]}],
                'warmup': 0, 'scored': 1, 'tail': 0, 'input_indices': [0]}
    return raw, bundle, document, row


def test_raw_reference_payload_is_checked_against_native_summary(tmp_path):
    from yolo_transfer.references import output_arrays
    raw, bundle, document, _ = reference_raw(tmp_path)
    _, arrays = output_arrays(raw, document, bundle)
    np.testing.assert_array_equal(arrays[0], np.array([1, 2], 'float32'))
    (raw / 'outputs/run_0/stage0_output_0.bin').write_bytes(np.array([3, 4], 'float32').tobytes())
    with pytest.raises(ValueError, match='hash'):
        output_arrays(raw, document, bundle)


@pytest.mark.parametrize('fault', ['duplicate_frame', 'missing_output', 'wrong_shape', 'wrong_affinity'])
def test_invalid_reference_evidence_is_rejected(tmp_path, fault):
    from yolo_transfer.references import output_arrays
    raw, bundle, document, row = reference_raw(tmp_path)
    if fault == 'duplicate_frame':
        (raw / 'result.jsonl').write_text(json.dumps(row) + '\n' + json.dumps(row))
    elif fault == 'missing_output':
        row['stage_raw_outputs'][0]['output_files'] = []
    elif fault == 'wrong_shape':
        row['stage_raw_outputs'][0]['outputs'][0]['shape'] = [1, 2]
    else:
        d = json.loads((raw / 'affinity.json').read_text())
        d['stages'][0]['effective_affinities'][0]['cpus'] = [1]
        (raw / 'affinity.json').write_text(json.dumps(d))
    if fault != 'duplicate_frame':
        (raw / 'result.jsonl').write_text(json.dumps(row))
    with pytest.raises(ValueError):
        output_arrays(raw, document, bundle)


def profile_raw(tmp_path):
    from yolo_transfer.profile import fnv1a64
    raw, bundle = tmp_path / 'raw', tmp_path / 'bundle'
    raw.mkdir(); (bundle / 'plans').mkdir(parents=True)
    expected = {'shape': [2], 'dtype': 'float32', 'nbytes': 8,
                'fnv1a64': fnv1a64(np.array([1, 2], 'float32').tobytes())}
    rows = []
    for index in range(10):
        row = {'frame_id': index, 'completion_index': index, 'completion_ms': index + 1.,
               'input_index': 0, 'stage_cpu_time_scope': 'exclusive_process_cpu_time',
               'raw_outputs': [{'index': 0, **expected}]}
        for op, wall, cpu in [('set', .1, .1), ('run', 1., 1.5), ('get', .2, .2)]:
            row['stage0_' + op + '_ms'] = wall
            row['stage0_' + op + '_process_cpu_ms'] = cpu
        rows.append(row)
    (raw / 'result.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
    stage = {'requested_cpus': [0, 1], 'thread_ids': [111, 222], 'active_runtime_threads': 2,
             'caller_tid': 111, 'effective_affinities': [
                 {'tid': 111, 'cpus': [0], 'cpu_time_delta_ms': 6.},
                 {'tid': 222, 'cpus': [1], 'cpu_time_delta_ms': 4.}],
             'worker_slots': [{'slot': 0, 'tid': 111, 'cpus': [0]}, {'slot': 1, 'tid': 222, 'cpus': [1]}]}
    (raw / 'affinity.json').write_text(json.dumps({'stages': [stage]}))
    (bundle / 'plans/programs.json').write_text(json.dumps({'cpu_00_00': {'output_node_ids': [123]}}))
    (bundle / 'plans/expected_outputs.json').write_text(json.dumps({'123': expected}))
    doc = {'kind': 'calibration', 'warmup': 2, 'scored': 8, 'tail': 0, 'input_indices': [0],
           'stages': [{'device': 'cpu', 'program': 'cpu_00_00', 'threads': 2, 'mask': [0, 1], 'worker_core_map': [0, 1]}]}
    return raw, bundle, doc


def test_isolated_service_and_whole_session_shares_remain_distinct(tmp_path):
    from yolo_transfer.profile import summary
    raw, bundle, doc = profile_raw(tmp_path)
    result = summary(raw, doc, bundle)
    assert result['cost']['service_ms'] == pytest.approx(1.3)
    assert result['cost']['run_core_ms'] == 1.5
    assert result['cost']['boundary_core_ms'] == pytest.approx(.3)
    assert result['session_thread_weights'] == [.6, .4]
    assert 'not_scored_run_only' in result['session_thread_weight_scope']


@pytest.mark.parametrize('fault', ['output_hash', 'worker_mapping', 'unexpected_thread', 'bad_clock'])
def test_profile_output_and_pool_errors_are_not_performance_samples(tmp_path, fault):
    from yolo_transfer.profile import summary
    raw, bundle, doc = profile_raw(tmp_path)
    if fault in ('worker_mapping', 'unexpected_thread'):
        d = json.loads((raw / 'affinity.json').read_text())
        if fault == 'worker_mapping':
            d['stages'][0]['worker_slots'][1]['cpus'] = [0]
        else:
            d['stages'][0]['thread_ids'].append(333)
        (raw / 'affinity.json').write_text(json.dumps(d))
    else:
        rows = [json.loads(line) for line in (raw / 'result.jsonl').read_text().splitlines()]
        if fault == 'output_hash':
            rows[0]['raw_outputs'][0]['fnv1a64'] = '0000000000000000'
        else:
            rows[2]['stage0_run_process_cpu_ms'] = -1.
        (raw / 'result.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError):
        summary(raw, doc, bundle)


def test_new_command_builder_does_not_mutate_old_transport_globals():
    from cpu_vta_partitioning.transport import SSH
    from yolo_transfer.profile import matched_ssh, native_command
    original = SSH.measure.__globals__['native_command']
    with pytest.raises(ValueError, match='execute-board'):
        matched_ssh({'host': 'example', 'user': 'root', 'remote_base': '/tmp/experiment'})
    instance = matched_ssh({'host': 'example', 'user': 'root', 'remote_base': '/tmp/experiment'}, execute_board=True)
    assert SSH.measure.__globals__['native_command'] is original
    assert instance.measure.__func__.__globals__['native_command'] is native_command


def test_wrong_boot_is_rejected_before_deployment_or_measurement(tmp_path, monkeypatch):
    from yolo_transfer import profile
    bundle, run_dir = tmp_path / 'bundle', tmp_path / 'run'
    run_dir.mkdir()
    (run_dir / 'run.json').write_text(json.dumps({'boot_id': 'old-boot', 'bundle_id': 'capsule'}))
    local = tmp_path / 'local.json'; local.write_text(json.dumps({'board': {}}))
    class FakeSSH:
        def boot_id(self): return 'new-boot'
        def deploy(self, *args): raise AssertionError('must not deploy on wrong boot')
        def measure(self, *args): raise AssertionError('must not run on wrong boot')
    monkeypatch.setattr(profile, 'validate', lambda bundle: {'bundle_id': 'capsule'})
    monkeypatch.setattr(profile, 'matched_ssh', lambda *args, **kwargs: FakeSSH())
    with pytest.raises(ValueError, match='boot'):
        profile.run(bundle, run_dir, local, execute_board=True)


def traffic_raw(tmp_path):
    raw, bundle, _ = profile_raw(tmp_path)
    row = json.loads((raw / 'result.jsonl').read_text().splitlines()[0])
    (raw / 'result.jsonl').write_text(json.dumps(row))
    (raw / 'affinity.json').write_text(json.dumps({'stages': [{'requested_cpus': [0, 1, 2, 3],
        'thread_ids': [111], 'active_runtime_threads': 1,
        'effective_affinities': [{'tid': 111, 'cpus': [0, 1, 2, 3]}]}]}))
    (bundle / 'plans/programs.json').write_text(json.dumps({'vta_01_01': {'output_node_ids': [123]}}))
    counters = {'synchronize_load_bytes': 64, 'synchronize_store_bytes': 32, 'synchronize_calls': 2}
    (raw / 'traffic_profile').mkdir()
    for name in ('single_run_status.json', 'benchmark_totals_status.json'):
        (raw / 'traffic_profile' / name).write_text(json.dumps(counters))
    doc = {'kind': 'traffic', 'warmup': 1, 'scored': 1, 'tail': 0, 'input_indices': [0],
           'stages': [{'device': 'vta', 'program': 'vta_01_01', 'threads': 1, 'mask': [0, 1, 2, 3]}]}
    return raw, bundle, doc


def test_traffic_counts_are_exactly_one_post_warmup_run(tmp_path):
    from yolo_transfer.traffic import summary
    raw, bundle, doc = traffic_raw(tmp_path)
    result = summary(raw, doc, bundle)
    assert (result['load_bytes'], result['store_bytes'], result['count']) == (64, 32, 1)
    total = raw / 'traffic_profile/benchmark_totals_status.json'
    d = json.loads(total.read_text()); d['synchronize_load_bytes'] *= 2
    total.write_text(json.dumps(d))
    with pytest.raises(ValueError, match='exactly one'):
        summary(raw, doc, bundle)


@pytest.mark.parametrize('changed_program', [False, True])
def test_import_preserves_original_task_and_never_reuses_changed_program(tmp_path, monkeypatch, changed_program):
    import shutil
    from cpu_vta_partitioning.common import digest, inventory, read, write
    from cpu_policies.open_resources.domain import component, component_key, stage
    from yolo_transfer import profile
    raw, old, _ = profile_raw(tmp_path)
    metadata = {'cpu_00_00': {'input_names': ['data'], 'input_node_ids': [0], 'output_node_ids': [123],
        'input_types': [{'shape': [2], 'dtype': 'float32'}], 'output_types': [{'shape': [2], 'dtype': 'float32'}]}}
    write(old / 'plans/programs.json', metadata, replace=True)
    cfg = component(stage('cpu', 0, 1, 2, [0, 1])); key = component_key(cfg)
    write(old / 'plans/component_configurations.json', {key: cfg})
    write(old / 'plans/profile_order.json', [key])
    write(old / 'plans/inputs.json', {'records': [{'index': 0, 'input': 'inputs/000.bin'}]})
    for name in ('graph.json', 'params.bin', 'module.so'):
        path = old / 'programs/cpu_00_00' / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'original-' + name.encode())
    def seal(path):
        manifest = {'canonical_sha256': 'common', 'runner_sha256': 'runner', 'files': inventory(path, ['manifest.json'])}
        manifest['bundle_id'] = digest(manifest)[:24]
        write(path / 'manifest.json', manifest, replace=True)
        return manifest
    prior = seal(old)
    new = tmp_path / 'new'; shutil.copytree(old, new)
    if changed_program:
        (new / 'programs/cpu_00_00/module.so').write_bytes(b'changed')
    # Ensure a distinct capsule identity even for byte-identical executables.
    write(new / 'new_preparation_identity.json', {'revision': 2})
    current = seal(new)
    old_run = tmp_path / 'old_run'
    write(old_run / 'run.json', {'boot_id': 'same-boot', 'bundle_id': prior['bundle_id']})
    document = profile.make_task(old, key)
    result = profile.summary(raw, document, old)
    receipt = old_run / 'accepted/same-boot' / (document['task_id'] + '.json')
    write(receipt, {'task': document, 'boot_id': 'same-boot', 'raw': str(raw), 'files': inventory(raw), 'summary': result})
    monkeypatch.setattr(profile, 'validate', lambda bundle: current)
    run_dir = tmp_path / 'new_run'
    imported = profile.import_matching(old, old_run, new, run_dir)
    assert imported['imported'] == (0 if changed_program else 1)
    if changed_program:
        return
    saved = read(run_dir / 'imported_components.json')['records'][key]
    assert saved['source']['task_id'] == document['task_id']
    assert saved['source']['task_id'] != profile.make_task(new, key)['task_id']
    class FakeSSH:
        def boot_id(self): return 'same-boot'
        def deploy(self, *args): return '/remote/bundle'
        def wait_idle(self, *args): pass
        def command(self, *args): return '1066666\n' * 4
        def measure(self, *args): raise AssertionError('imported task must not run again')
    monkeypatch.setattr(profile, 'matched_ssh', lambda *args, **kwargs: FakeSSH())
    local = tmp_path / 'local.json'; write(local, {'board': {}})
    state = profile.run(new, run_dir, local, execute_board=True)
    assert state['status'] == 'COMPLETE' and state['imported_components'] == 1
    assert read(run_dir / 'isolated_costs.json')['costs'][key]['source']['task_id'] == document['task_id']
    assert not list((run_dir / 'accepted').glob('*/*.json'))


def test_transferred_search_matches_independent_oracle_on_non21_unit_graph(tmp_path):
    from cpu_vta_partitioning.common import read
    from cpu_policies.open_resources.costs import Costs
    from cpu_policies.open_resources.domain import context, requirements, validate_candidate
    from resource_envelope.model import score
    from resource_envelope.search_adapter import generated
    from resource_envelope.weighted_search import original_engine
    from yolo_transfer.domain import programs as interval_programs
    from yolo_transfer.ranking import compact_records
    programs = {r['name']: r for r in interval_programs(5, 1, include_atomic_cpu=False)}
    records = {}
    for key, cfg in requirements(programs).items():
        row = cfg['stages'][0]; n = row['threads']; duration = (row['end'] - row['begin']) * 10.
        records[key] = {'configuration': cfg, 'cost': {'service_ms': duration / min(n, 2) + .2,
            'run_core_ms': duration * .7, 'boundary_core_ms': .2},
            'session_thread_weights': [2. / (n + 1)] + ([1. / (n + 1)] * (n - 1))}
    ddr = {'stage_service_ms': {name: {str(n): .01 for n in range(1, 5)} for name in programs}}
    balanced = Costs(records, programs=programs, ddr=ddr)
    weighted = Costs(records, programs=programs, ddr=ddr, distribution='session_weights')
    original = original_engine(tmp_path / 'original')
    enhanced = generated(tmp_path / 'enhanced')
    old = original.rank_all(programs, balanced, ddr=ddr, unit_count=5, max_vta_islands=1,
                            contexts=[context('shared4')], output_dir=tmp_path / 'original/search', verify=True)
    new = enhanced.rank_all(programs, weighted, ddr=ddr, unit_count=5, max_vta_islands=1,
                           contexts=[context('shared4')], output_dir=tmp_path / 'enhanced/search', verify=True)
    assert old['counts'] == new['counts'] and new['counts']['topologies'] == 6
    assert old['verification']['passed'] and new['verification']['passed']
    for result, costs, variant in [(old, balanced, 'original'), (new, weighted, 'combined')]:
        for cfg in result['pure_static20']:
            validate_candidate(cfg, programs, unit_count=5, max_vta_islands=1)
            assert score(cfg, costs, ddr)[variant] == cfg['score_ns']
    compact, expand = compact_records(programs, [read(tmp_path / 'enhanced/search/shared4.json')], 5)
    assert compact
    for record in compact:
        cfg = expand(record)
        validate_candidate(cfg, programs, unit_count=5, max_vta_islands=1)
        assert score(cfg, weighted, ddr)['combined'] == record['score_ns']
