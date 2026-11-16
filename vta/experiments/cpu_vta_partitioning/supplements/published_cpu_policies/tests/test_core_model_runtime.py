"""Ordered-map execution, shared-cost derivation, and durable evidence tests."""
from copy import deepcopy
import json

import pytest

from cpu_policies.core_model import campaign, common, runtime
from cpu_policies.core_model.affinity import validate_affinity
from cpu_policies.core_model.domain import configurations, baseline_candidate, stage, candidate


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    api = common.api()
    root = tmp_path / 'bundle'
    (root / 'plans').mkdir(parents=True)
    for name in ('programs.json', 'inputs.json', 'executors.json'):
        (root / 'plans' / name).write_bytes((common.DEFAULT_PARENT / 'plans' / name).read_bytes())
    specs = api.read(root / 'plans/executors.json')
    specs['core_model'] = {**specs['policy'], 'path': 'runner/vta_stage_pipeline_runner_core_model'}
    api.write(root / 'plans/executors.json', specs, replace=True)
    api.write(root / 'manifest.json', {'bundle_id': 'core-model-test'})
    api.write(root / 'plans/diagnosis_configurations.json', configurations())
    monkeypatch.setattr(runtime, 'api', lambda unused=None: api)
    monkeypatch.setattr(campaign, 'api', lambda unused=None: api)
    return root


def report(cfg, qualified=False):
    rows = []
    for index, s in enumerate(cfg['stages']):
        tids = [100 + index * 10 + k for k in range(s['threads'])]
        masks = [[c] for c in s['worker_core_map']] if s['device'] == 'cpu' else [s['mask']]
        rows.append({'requested_cpus': s['mask'], 'active_runtime_threads': s['threads'],
                     'thread_ids': tids, 'caller_tid': tids[0],
                     'effective_affinities': [{'tid': tid, 'cpus': mask, 'cpu_time_delta_ms': 1.}
                                              for tid, mask in zip(tids, masks)],
                     'worker_slots': [{'slot': k, 'tid': tid, 'cpus': mask}
                                      for k, (tid, mask) in enumerate(zip(tids, masks))] if qualified else []})
    return {'cpu_worker_pinning': cfg['affinity_mode'] == 'pinned', 'stages': rows,
            'control_threads': [{'role': role, 'tid': tid, 'cpus': cfg['control_mask']}
                                for role, tid in [('producer', 10), ('collector', 11)]] if len(rows) > 1 else []}


def mapped():
    cfg = baseline_candidate()
    cfg['stages'][0] = stage('cpu', 0, 3, 4, [0, 1, 2, 3], [3, 0, 1, 2])
    return candidate(cfg['stages'])


def raw_performance(path, doc):
    path.mkdir(parents=True, exist_ok=True)
    rows = [{'frame_id': i, 'completion_index': i, 'input_index': 0,
             'completion_ms': i * 10., 'completion_process_cpu_ms': i * 20.,
             'queue_timing_enabled': False, 'defer_result_write_enabled': True,
             'cpu_stage_gate_mode': 'off', 'total_latency_ms': 80.} for i in range(28)]
    (path / 'result.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in rows))
    (path / 'affinity.json').write_text(json.dumps(report(doc['configuration'])))


def test_11_profiles_and_maximum_campaign_budget(bundle):
    base = campaign.component_plan(bundle)
    assert len(base) == 22
    assert sum(sum(x['document'][k] for k in ('warmup', 'scored', 'tail')) for x in base) == 121
    assert sum(x['document']['run_probe'] and x['document']['kind'] == 'calibration' for x in base) == 6
    from cpu_policies.core_model.domain import allocations
    configs = list(allocations(True))[:12]
    pool = {c['id']: c for c in configs}
    full = base + [campaign.bridge_task(bundle)] + campaign.pipeline_plan(bundle, pool, sorted(pool))
    assert len(full) == len({x['document']['task_id'] for x in full}) == 47
    assert sum(sum(x['document'][k] for k in ('warmup', 'scored', 'tail')) for x in full) == 497
    assert all(not x['document']['run_probe'] for x in full if x['document']['kind'] == 'performance')


def test_native_worker_order_is_not_sorted_away(bundle):
    cfg = mapped()
    doc = runtime.make_task('performance', cfg, bundle, warmup=8, scored=16, tail=4)
    argv, _ = runtime.native_command(doc, bundle, '/b', '/j')
    assert argv[argv.index('--stage0-cpu-affinity') + 1] == '0,1,2,3'
    assert argv[argv.index('--stage0-cpu-worker-map') + 1] == '3,0,1,2'
    assert '--queue-timing' not in argv and '--session-profile-json' not in argv
    with pytest.raises(ValueError, match='reordered'):
        runtime.make_task('performance', cfg, bundle, scored=16, executor='policy')
    old = campaign.bridge_task(bundle)['document']
    argv, _ = runtime.native_command(old, bundle, '/b', '/j')
    assert not any(x.endswith('-cpu-worker-map') for x in argv)


def test_actual_logical_worker_mapping_is_checked():
    cfg = mapped()
    value = report(cfg, True)
    ledger, _ = validate_affinity(value, cfg, True)
    assert next(x for x in ledger if x['tid'] == 100)['actual_mask'] == [3]
    value['stages'][0]['worker_slots'][0]['cpus'] = [0]
    with pytest.raises(ValueError, match='mapping'):
        validate_affinity(value, cfg, True)


def test_common_costs_exclude_probe_wall_and_conserve_worker_time(bundle):
    cfg = configurations()['components']['cpu00_t2']
    doc = runtime.make_task('calibration', cfg, bundle, warmup=2, scored=8, run_probe=True)
    summary = dict(service_ms=13., run_core_ms=7.5, boundary_core_ms=.5, set_ms=1.,
                   run_ms=10., get_ms=2., set_core_ms=.3, get_core_ms=.2)
    samples = [{'diagnostic_run_ms': 9.} for _ in range(8)]
    workers = [{'logical_worker': k, 'runtime_delta_ns': value}
               for _ in range(8) for k, value in enumerate([2_000_000, 4_000_000])]
    summary['run_counter_quality'] = {'samples': samples, 'threads': workers,
                                     'all_runtime_samples_available': True}
    cost = campaign.component_cost(doc, summary)
    assert cost['service_ms'] == 12.
    assert cost['run_core_ms'] == sum(cost['worker_run_core_ms']) == 6.
    assert cost['cpu_core_weights'] == [1/3, 2/3]
    summary['run_counter_quality']['all_runtime_samples_available'] = False
    with pytest.raises(ValueError, match='unavailable'):
        campaign.component_cost(doc, summary)


def test_probe_is_confined_to_multithread_isolated_costs(bundle):
    with pytest.raises(ValueError):
        runtime.make_task('performance', baseline_candidate(), bundle, scored=16, run_probe=True)
    single = configurations()['components']['cpu00_t1']
    with pytest.raises(ValueError):
        runtime.make_task('calibration', single, bundle, scored=8, run_probe=True)


def test_receipt_recovery_raw_hash_and_wrong_boot(bundle, tmp_path):
    api = common.api()
    doc = runtime.make_task('performance', mapped(), bundle, warmup=8, scored=16, tail=4)
    root, boot = tmp_path / 'run', 'boot-a'
    raw = root / 'raw' / boot / doc['task_id']; raw_performance(raw, doc)
    summary = runtime.validate_summary(raw, doc, bundle)
    receipt = root / 'accepted' / boot / (doc['task_id'] + '.json')
    files = {str(raw.relative_to(root) / k): v for k, v in api.inventory(raw).items()}
    api.write(receipt, {'task': doc, 'boot_id': boot, 'summary': summary, 'files': files})
    state = {'bundle': str(bundle), 'boot_id': boot, 'frozen_files': {}}
    recorder = campaign.Recorder(root, state)
    value = recorder.saved('pipeline/case/performance', doc)
    assert recorder.saved('pipeline/case/performance', doc) == value
    with pytest.raises(ValueError): runtime.Evidence(root, bundle).summary(value, doc, 'wrong-boot')
    (raw / 'affinity.json').write_text('{}')
    with pytest.raises(ValueError): recorder.saved('pipeline/case/performance', doc)


def test_frozen_publication_recovers_after_write_before_pin(bundle, tmp_path):
    api = common.api(); root = tmp_path / 'run'; root.mkdir()
    state = {'bundle': str(bundle), 'frozen_files': {}, 'pending_publications': {'a.json': api.digest({'v': 1})}}
    api.write(root / 'a.json', {'v': 1})
    assert campaign._publish(root, state, 'a.json', {'v': 1}) == {'v': 1}
    assert state['pending_publications'] == {}
    with pytest.raises(ValueError): campaign._publish(root, state, 'a.json', {'v': 2})


def test_changed_boot_stops_before_any_deployment(bundle, tmp_path, monkeypatch):
    state = {'bundle': str(bundle), 'boot_id': 'first', 'status': 'INTERRUPTED'}
    monkeypatch.setattr(campaign, 'validate_state', lambda *a, **k: state)
    class OtherBoot:
        def boot_id(self): return 'second'
        def deploy(self, *args): raise AssertionError('must stop')
    with pytest.raises(RuntimeError, match='another boot'):
        campaign.run_current(tmp_path / 'run', OtherBoot())
    assert state['status'] == 'STOPPED_BOOT_CHANGED'
