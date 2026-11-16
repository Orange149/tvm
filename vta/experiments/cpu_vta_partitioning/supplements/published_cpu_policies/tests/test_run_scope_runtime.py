"""Sampling, binary identity, and exactly-once campaign regression checks."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from cpu_policies.run_scope import campaign, common, runtime
from cpu_policies.run_scope.domain import configurations


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    frozen = common.api()
    root = tmp_path / 'bundle'
    (root / 'plans').mkdir(parents=True)
    for name in ('programs.json', 'inputs.json', 'executors.json'):
        (root / 'plans' / name).write_bytes((common.DEFAULT_PARENT / 'plans' / name).read_bytes())
    executors = frozen.read(root / 'plans/executors.json')
    executors['run_scope'] = {**executors['policy'], 'path': 'runner/vta_stage_pipeline_runner_run_scope'}
    frozen.write(root / 'plans/executors.json', executors, replace=True)
    frozen.write(root / 'manifest.json', {'bundle_id': 'run-scope-test'})
    frozen.write(root / 'plans/diagnosis_configurations.json', configurations())
    monkeypatch.setattr(runtime, 'api', lambda unused=None: frozen)
    monkeypatch.setattr(campaign, 'api', lambda unused=None: frozen)
    return root


def affinity(cfg):
    stages = []
    for i, stage in enumerate(cfg['stages']):
        tids = [100 + 10 * i + j for j in range(stage['threads'])]
        masks = [[c] for c in stage['mask']] if stage['device'] == 'cpu' else [stage['mask']]
        stages.append({'requested_cpus': stage['mask'], 'active_runtime_threads': stage['threads'],
                       'thread_ids': tids, 'caller_tid': tids[0], 'effective_affinities': [
                           {'tid': tid, 'cpus': mask, 'cpu_time_delta_ms': 1.}
                           for tid, mask in zip(tids, masks)]})
    return {'stages': stages, 'cpu_worker_pinning': True, 'control_threads': [
        {'role': role, 'tid': tid, 'cpus': cfg['control_mask']}
        for role, tid in [('producer', 10), ('collector', 11)]] if len(stages) > 1 else []}


def performance(root, doc):
    root.mkdir(parents=True, exist_ok=True)
    rows = [{'frame_id': i, 'completion_index': i, 'input_index': 0,
             'completion_ms': 100 * i, 'completion_process_cpu_ms': 200 * i,
             'queue_timing_enabled': False, 'defer_result_write_enabled': True,
             'total_latency_ms': 500, 'cpu_stage_gate_mode': doc['configuration']['execution_mode']}
            for i in range(28)]
    (root / 'result.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    (root / 'affinity.json').write_text(json.dumps(affinity(doc['configuration'])))
    return rows


def test_exact_22_tasks_310_forwards_and_sparse_sampling(bundle):
    plan = campaign.task_plan(bundle)
    docs = [row['document'] for row in plan]
    assert len(docs) == len({row['task_id'] for row in docs}) == 22
    assert sum(sum(row[k] for k in ('warmup', 'scored', 'tail')) for row in docs) == 310
    assert sum(row['kind'].startswith('qualify') for row in docs) == 6
    assert sum(row['kind'] == 'calibration' for row in docs) == 8
    assert sum(row['kind'] == 'performance' for row in docs) == 8
    assert sum(row['executor'] == 'policy' for row in docs) == 2
    for row in docs:
        if row['run_probe']:
            assert row['probe_frames'] == ([8, 12, 16, 20] if row['kind'] == 'performance'
                                           else list(range(2, 10)) if row['kind'] == 'calibration' else [0])
        else:
            assert row['probe_frames'] == []
        assert row.get('detailed') is False


def test_probe_identity_and_before_after_sessions_are_distinct(bundle):
    docs = {row['name']: row['document'] for row in campaign.task_plan(bundle)}
    before, after = [docs['static_first/' + name] for name in ('new_plain_before', 'new_plain_after')]
    assert before['configuration'] == after['configuration']
    assert before['task_id'] != after['task_id']
    old = docs['static_first/old_plain']
    assert old['executor_sha256'] == before['executor_sha256']  # fake test payload shares bytes
    assert old['executor_path'] != before['executor_path']
    assert old['task_id'] != before['task_id']
    bad = deepcopy(before)
    bad['run_probe'] = True
    with pytest.raises(ValueError):
        runtime.validate_document(bad, bundle)


def test_commands_never_enable_expensive_probes(bundle):
    for row in campaign.task_plan(bundle):
        doc = row['document']
        args, _ = runtime.native_command(doc, bundle, '/bundle', '/job')
        assert '--queue-timing' not in args
        assert '--session-profile-json' not in args
        assert '--profile-pmu' not in args
        assert '--process-cpu-affinity' not in args
        assert ('--run-thread-counters-json' in args) is doc['run_probe']
        if doc['run_probe']:
            assert args[args.index('--run-thread-counters-frames') + 1] == ','.join(map(str, doc['probe_frames']))
        assert ('--serial' in args) is (doc['kind'] in ('calibration', 'qualify_component'))


@pytest.mark.parametrize('changes', [dict(executor='policy', run_probe=True),
                                      dict(detailed=True), dict(run_probe=True, probe_frames=[]),
                                      dict(run_probe=True, probe_frames=[8, 8]),
                                      dict(run_probe=True, probe_frames=[True])])
def test_invalid_probe_or_executor_combinations(bundle, changes):
    cfg = configurations()['pipelines']['static_first']
    with pytest.raises(ValueError):
        runtime.make_task('performance', cfg, bundle, warmup=8, scored=16, tail=4, **changes)


def test_missing_required_trace_and_probe_off_unexpected_trace(bundle, tmp_path):
    docs = {row['name']: row['document'] for row in campaign.task_plan(bundle)}
    doc = docs['static_first/run_probe']
    raw = tmp_path / 'raw'
    performance(raw, doc)
    with pytest.raises(ValueError, match='run_threads.json'):
        runtime.validate_summary(raw, doc, bundle)
    off = docs['static_first/new_plain_before']
    performance(raw, off)
    (raw / 'run_threads.json').write_text('{}')
    with pytest.raises(ValueError, match='unexpectedly'):
        runtime.validate_summary(raw, off, bundle)


def test_duplicate_frames_and_affinity_errors_rejected(bundle, tmp_path):
    doc = next(row['document'] for row in campaign.task_plan(bundle)
               if row['document']['diagnosis_block'] == 'new_plain_before')
    raw = tmp_path / 'raw'
    rows = performance(raw, doc)
    rows[1]['frame_id'] = 0
    (raw / 'result.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError):
        runtime.validate_summary(raw, doc, bundle)
    performance(raw, doc)
    report = affinity(doc['configuration']); report['stages'][0]['effective_affinities'][0]['cpus'] = [3]
    (raw / 'affinity.json').write_text(json.dumps(report))
    with pytest.raises(ValueError):
        runtime.validate_summary(raw, doc, bundle)


def test_receipt_recovery_and_duplicate_collection_do_not_repeat_measurement(bundle, tmp_path):
    frozen = common.api()
    doc = next(row['document'] for row in campaign.task_plan(bundle)
               if row['document']['diagnosis_block'] == 'new_plain_before')
    root, boot = tmp_path / 'run', 'boot-a'
    raw = root / 'raw' / boot / doc['task_id']; performance(raw, doc)
    summary = runtime.validate_summary(raw, doc, bundle)
    receipt = root / 'accepted' / boot / (doc['task_id'] + '.json')
    files = {str(raw.relative_to(root) / name): value for name, value in frozen.inventory(raw).items()}
    frozen.write(receipt, {'task': doc, 'boot_id': boot, 'summary': summary, 'files': files})
    state = {'bundle': str(bundle), 'boot_id': boot, 'frozen_files': {}}
    recorder = campaign.Recorder(root, state)
    value = recorder.saved('case/plain', doc)
    assert recorder.saved('case/plain', doc) == value
    assert len(list((root / 'records').rglob('*.json'))) == 1
    with pytest.raises(ValueError):
        runtime.Evidence(root, bundle).summary(value, doc, 'wrong-boot')
    (raw / 'result.jsonl').write_text('changed')
    with pytest.raises(ValueError):
        recorder.saved('case/plain', doc)


def test_cross_boot_resume_stops_before_deployment(bundle, tmp_path, monkeypatch):
    state = {'bundle': str(bundle), 'boot_id': 'old-boot', 'status': 'INTERRUPTED'}
    monkeypatch.setattr(campaign, 'validate_state', lambda *a, **k: state)
    class OtherBoot:
        def boot_id(self): return 'new-boot'
        def deploy(self, *args): raise AssertionError('must not deploy')
    with pytest.raises(RuntimeError, match='another boot'):
        campaign.run_current(tmp_path / 'run', OtherBoot())
    assert state['status'] == 'STOPPED_BOOT_CHANGED'
