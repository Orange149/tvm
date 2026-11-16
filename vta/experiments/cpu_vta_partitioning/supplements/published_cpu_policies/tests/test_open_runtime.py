"""Execution semantics and evidence checks for the nine-context runner."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from cpu_policies.open_resources import runtime
from cpu_policies.open_resources import common
from cpu_policies.open_resources.domain import candidate, component, stage


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    api = common.api()
    root = tmp_path / 'bundle'
    (root / 'plans').mkdir(parents=True)
    for name in ('programs.json', 'inputs.json', 'executors.json'):
        (root / 'plans' / name).write_bytes((common.DEFAULT_PARENT / 'plans' / name).read_bytes())
    (root / 'manifest.json').write_text(json.dumps({'bundle_id': 'open-test'}))
    monkeypatch.setattr(runtime, 'api', lambda unused=None: api)
    return root


def config(ctx='shared4', gate='off'):
    from cpu_policies.open_resources.domain import context
    c = context(ctx)
    cores = c['compute_pool']
    return candidate([stage('cpu', 0, 3, 2, cores[:2]),
                      stage('vta', 3, 18, 1, c['host_mask']),
                      stage('cpu', 18, 21, 1, cores[-1:])], c, gate)


def affinity(cfg, probe=False):
    rows = []
    for i, s in enumerate(cfg['stages']):
        tids = [100 + 10 * i + j for j in range(s['threads'])]
        masks = [[c] for c in s['mask']] if s['device'] == 'cpu' else [s['mask']]
        entries = [{'tid': t, 'cpus': m, 'cpu_time_delta_ms': 1.} for t, m in zip(tids, masks)]
        rows.append({'requested_cpus': s['mask'], 'active_runtime_threads': s['threads'],
                     'thread_ids': tids, 'caller_tid': tids[0], 'effective_affinities': entries,
                     'worker_slots': [{'slot': j, 'tid': t, 'cpus': m} for j, (t, m) in enumerate(zip(tids, masks))] if probe else []})
    return {'stages': rows, 'cpu_worker_pinning': cfg['affinity_mode'] == 'pinned',
            'control_threads': [{'role': role, 'tid': tid, 'cpus': cfg['control_mask']}
                                for role, tid in [('producer', 10), ('collector', 11)]] if len(rows) > 1 else []}


def raw_performance(path, doc):
    path.mkdir(parents=True)
    completion, rows = 0, []
    for i in range(28):
        completion += 10 if i % 2 else 30
        rows.append({'frame_id': i, 'completion_index': i, 'input_index': 0,
                     'completion_ms': completion, 'total_latency_ms': 80,
                     'completion_process_cpu_ms': completion / 2,
                     'cpu_stage_gate_mode': doc['configuration']['execution_mode']})
    (path / 'result.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (path / 'affinity.json').write_text(json.dumps(affinity(doc['configuration'])))


@pytest.mark.parametrize('ctx', ['shared4', 'host_shared_3', 'host_reserved_2'])
def test_arguments_keep_stage_host_and_control_independent(bundle, ctx):
    doc = runtime.make_task('performance', config(ctx), bundle, warmup=8, scored=16, tail=4)
    argv, _ = runtime.native_command(doc, bundle, '/board/b', '/board/j')
    c = doc['configuration']
    assert argv[argv.index('--stage1-cpu-affinity') + 1] == ','.join(map(str, c['host_mask']))
    assert argv[argv.index('--control-cpu-affinity') + 1] == ','.join(map(str, c['control_mask']))
    assert '--cpu-worker-pinning' in argv
    assert '--process-cpu-affinity' not in argv
    assert '--serial' not in argv
    assert argv[argv.index('--runs') + 1] == '28'


def test_shared_host_three_does_not_reserve_computation_core(bundle):
    c = config('host_shared_3')
    assert c['stages'][-1]['mask'] == [3]
    runtime.make_task('performance', c, bundle, scored=16)
    c['reserved_cores'] = [3]
    with pytest.raises(ValueError):
        runtime.make_task('performance', c, bundle, scored=16)


def test_traffic_has_native_warmup_outside_profiler_count(bundle, tmp_path):
    c = component(stage('vta', 3, 18, 1, [0, 1, 2, 3]))
    doc = runtime.make_task('traffic', c, bundle, warmup=1, scored=1)
    argv, _ = runtime.native_command(doc, bundle, '/b', '/j')
    assert argv[argv.index('--warmup-runs') + 1] == '1'
    assert argv[argv.index('--runs') + 1] == '1'
    assert argv[argv.index('--vta-runtime-profile-dir') + 1] == '/j/traffic_profile'
    assert '--serial' in argv
    raw = tmp_path / 'traffic'
    (raw / 'traffic_profile').mkdir(parents=True)
    (raw / 'result.jsonl').write_text(json.dumps({'frame_id': 0, 'completion_index': 0,
        'input_index': 0, 'completion_ms': 1, 'cpu_stage_gate_mode': 'off'}) + '\n')
    (raw / 'affinity.json').write_text(json.dumps(affinity(c)))
    counters = {'synchronize_load_bytes': 100, 'synchronize_store_bytes': 20, 'synchronize_calls': 2}
    for leaf in ('single_run_status.json', 'benchmark_totals_status.json'):
        (raw / 'traffic_profile' / leaf).write_text(json.dumps(counters))
    result = runtime.validate_summary(raw, doc, bundle)
    assert result['count'] == 1 and result['load_bytes'] == 100
    (raw / 'traffic_profile/benchmark_totals_status.json').write_text(json.dumps({**counters, 'synchronize_load_bytes': 200}))
    with pytest.raises(ValueError, match='exactly one'):
        runtime.validate_summary(raw, doc, bundle)


def test_no_missing_hook_or_empty_counter_is_priced_zero(tmp_path):
    with pytest.raises(ValueError, match='hook'):
        runtime.traffic_summary(tmp_path)
    p = tmp_path / 'traffic_profile'
    p.mkdir()
    (p / 'single_run_status.json').write_text('{}')
    with pytest.raises(ValueError, match='counters'):
        runtime.traffic_summary(tmp_path)


def test_timing_variation_accepted_fifteen_intervals_and_scope(bundle, tmp_path):
    doc = runtime.make_task('performance', config('host_reserved_2'), bundle, warmup=8, scored=16, tail=4)
    raw = tmp_path / 'raw'
    raw_performance(raw, doc)
    s = runtime.validate_summary(raw, doc, bundle)
    assert s['gaps'] == 15 and s['kept_timestamps'] == 16 and s['p95_gap_ms'] == 30
    assert s['cpu_time_scope'] == 'whole_process_between_scored_completions'
    controls = [r for r in s['thread_ledger'] if r['role'] in ('producer', 'collector')]
    assert len(controls) == 2 and all(r['cpu_time_delta_ms'] is None for r in controls)


def test_controls_and_pool_identity_and_frames_are_checked(bundle, tmp_path):
    doc = runtime.make_task('performance', config(), bundle, warmup=8, scored=16, tail=4)
    raw = tmp_path / 'raw'
    raw_performance(raw, doc)
    report = affinity(doc['configuration'])
    report['control_threads'][0]['cpus'] = [2]
    (raw / 'affinity.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='producer'):
        runtime.validate_summary(raw, doc, bundle)
    (raw / 'affinity.json').write_text(json.dumps(affinity(doc['configuration'])))
    rows = [json.loads(s) for s in (raw / 'result.jsonl').read_text().splitlines()]
    rows[1]['frame_id'] = 0
    (raw / 'result.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    with pytest.raises(ValueError, match='duplicate/missing'):
        runtime.validate_summary(raw, doc, bundle)


def test_evidence_rejects_boot_hash_and_raw_changes(bundle, tmp_path):
    api = common.api()
    doc = runtime.make_task('performance', config(), bundle, warmup=8, scored=16, tail=4)
    root, boot = tmp_path / 'run', 'boot-a'
    raw = root / 'raw' / boot / doc['task_id']
    raw_performance(raw, doc)
    s = runtime.validate_summary(raw, doc, bundle)
    receipt = root / 'accepted' / boot / (doc['task_id'] + '.json')
    files = {str(raw.relative_to(root) / k): v for k, v in api.inventory(raw).items()}
    api.write(receipt, {'task': doc, 'boot_id': boot, 'summary': s, 'files': files})
    v = {**s, '_receipt': {'path': str(receipt.relative_to(root)), 'sha256': api.sha256(receipt)}}
    assert runtime.Evidence(root, bundle).summary(v, doc, boot)['summary'] == s
    with pytest.raises(ValueError):
        runtime.Evidence(root, bundle).summary(v, doc, 'boot-b')
    (raw / 'result.jsonl').write_text('changed')
    with pytest.raises(ValueError):
        runtime.Evidence(root, bundle).summary(v, doc, boot)


@pytest.mark.parametrize('bad', [{'indices': [False]}, {'indices': [16]}, {'scored': True}, {'scored': 1}])
def test_invalid_protocol_and_identity(bundle, bad):
    with pytest.raises(ValueError):
        runtime.make_task('performance', config(), bundle, **bad)


def test_changed_executor_rejected(bundle):
    doc = runtime.make_task('performance', config(), bundle, scored=16)
    doc['executor_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='identity'):
        runtime.validate_document(doc, bundle)


def test_zero_cpu_ticks_do_not_turn_a_configured_worker_into_an_idle_helper():
    c = config()
    report = affinity(c)
    report['stages'][0]['effective_affinities'][1]['cpu_time_delta_ms'] = 0
    ledger, idle = runtime._affinity(report, c, False)
    worker = next(r for r in ledger if r['tid'] == 101)
    assert worker['participation'] == 'configured_compute_slot'
    assert worker not in idle
    report['stages'][1]['effective_affinities'].append(deepcopy(report['stages'][1]['effective_affinities'][0]))
    with pytest.raises(ValueError, match='ownership'):
        runtime._affinity(report, c, False)


def test_full_output_qualification_rejects_a_single_wrong_element(bundle, tmp_path):
    import numpy as np
    c = component(stage('cpu', 0, 3, 2, [0, 1]))
    doc = runtime.make_task('qualify_component', c, bundle)
    programs = json.loads((bundle / 'plans/programs.json').read_text())
    inputs = json.loads((bundle / 'plans/inputs.json').read_text())
    raw = tmp_path / 'qualification'
    (raw / 'outputs/run_0').mkdir(parents=True)
    outputs = []
    for node in programs[c['stages'][0]['program']]['output_node_ids']:
        rel = f'references/node_{node}.npy'
        (bundle / 'references').mkdir(exist_ok=True)
        values = np.asarray([1, 2], dtype='int8')
        np.save(bundle / rel, values)
        inputs['records'][0]['frontiers'][str(node)]['npy'] = rel
        name = f'node_{node}.bin'
        values.tofile(raw / 'outputs/run_0' / name)
        outputs.append({'path': '/board/' + name, 'shape': [2], 'dtype': 'int8'})
    (bundle / 'plans/inputs.json').write_text(json.dumps(inputs))
    row = {'frame_id': 0, 'completion_index': 0, 'input_index': 0, 'completion_ms': 1.,
           'cpu_stage_gate_mode': 'off', 'stage_raw_outputs': [{'output_files': outputs}]}
    (raw / 'result.jsonl').write_text(json.dumps(row) + '\n')
    (raw / 'affinity.json').write_text(json.dumps(affinity(c, probe=True)))
    assert runtime.validate_summary(raw, doc, bundle)['correct'] is True
    np.asarray([1, 3], dtype='int8').tofile(raw / 'outputs/run_0' / Path(outputs[0]['path']).name)
    with pytest.raises(ValueError, match='incorrect full output'):
        runtime.validate_summary(raw, doc, bundle)


def test_generated_options_are_accepted_by_real_packaged_executor():
    """Exercise the real parser, rather than echoing the builder's flag names."""
    import shutil
    import subprocess
    emulator = shutil.which('qemu-aarch64-static')
    assert emulator, 'packaged native option acceptance requires QEMU'
    b = common.DEFAULT_PARENT
    from cpu_policies.open_resources.domain import contexts
    configurations = [('qualify_pipeline', config(c['id'])) for c in contexts()]
    configurations.append(('traffic', component(stage('vta', 3, 18, 1, [0, 1, 2, 3]))))
    for kind, c in configurations:
        doc = runtime.make_task(kind, c, b, warmup=1 if kind == 'traffic' else 0, scored=1)
        argv, _ = runtime.native_command(doc, b, '/board/b', '/board/j')
        exe = '/board/b/' + doc['executor_path']
        args = argv[argv.index(exe) + 1:]
        command = [emulator, str(b / 'runtime/ld-linux-aarch64.so.1'), '--library-path',
                   str(b / 'runtime'), str(b / doc['executor_path']), *args, '--help']
        result = subprocess.run(command, capture_output=True, text=True, timeout=20)
        assert result.returncode == 0, result.stdout + result.stderr
        assert 'Unknown option' not in result.stderr
        rejected = subprocess.run([*command[:-1], '--invalid-native-parser-test', '--help'],
                                 capture_output=True, text=True, timeout=20)
        assert rejected.returncode != 0 and 'Unknown option' in rejected.stderr
