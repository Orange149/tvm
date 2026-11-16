"""Meaningful regression checks for one-boot diagnostics and evidence recovery."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from cpu_policies.diagnosis import campaign, common, runtime
from cpu_policies.diagnosis.domain import configurations


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    frozen = common.api()
    root = tmp_path / 'bundle'
    (root / 'plans').mkdir(parents=True)
    for name in ('programs.json', 'inputs.json', 'executors.json'):
        (root / 'plans' / name).write_bytes((common.DEFAULT_PARENT / 'plans' / name).read_bytes())
    (root / 'manifest.json').write_text(json.dumps({'bundle_id': 'diagnosis-test'}))
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
             'queue_timing_enabled': doc.get('detailed', False), 'defer_result_write_enabled': True,
             'total_latency_ms': 500, 'cpu_stage_gate_mode': doc['configuration']['execution_mode']}
            for i in range(28)]
    (root / 'result.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    (root / 'affinity.json').write_text(json.dumps(affinity(doc['configuration'])))
    return rows


def test_task_plan_is_twenty_distinct_real_sessions_299_frames(bundle):
    plan = campaign.task_plan(bundle)
    assert len({row['document']['task_id'] for row in plan}) == 20
    assert sum(sum(row['document'][k] for k in ('warmup', 'scored', 'tail')) for row in plan) == 299
    assert sum(row['document']['kind'].startswith('qualify') for row in plan) == 7
    assert sum(row['document'].get('detailed') for row in plan) == 3
    docs = {row['name']: row['document'] for row in plan}
    before, after = [docs['static_first/' + label] for label in ('plain_before', 'plain_after')]
    assert before['configuration'] == after['configuration']
    assert before['task_id'] != after['task_id']


def test_detailed_pipeline_flags_do_not_leak_to_serial_components(bundle):
    for item in campaign.task_plan(bundle):
        doc = item['document']
        args, _ = runtime.native_command(doc, bundle, '/b', '/j')
        assert ('--queue-timing' in args) is doc['detailed']
        assert ('--session-profile-window' in args) is doc['detailed']
        assert ('--serial' in args) is (doc['kind'] in ('calibration', 'qualify_component'))
        assert '--process-cpu-affinity' not in args
    cfg = configurations()['components']['cpu05_t1']
    with pytest.raises(ValueError, match='confined'):
        runtime.make_task('calibration', cfg, bundle, detailed=True, diagnosis_detail=True)


def test_runtime_rejects_task_edits_and_diagnosis_detail_mismatch(bundle):
    doc = campaign.task_plan(bundle)[-1]['document']
    edited = deepcopy(doc)
    edited['executor_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='identity'):
        runtime.validate_document(edited, bundle)
    with pytest.raises(ValueError, match='differ'):
        runtime.make_task('performance', doc['configuration'], bundle, scored=16,
                          detailed=True, diagnosis_detail=False)


def test_queue_partial_order_allows_dequeue_before_push_return():
    q = {'edge_index': 0, 'push_request_ms': 1., 'enqueue_ms': 2.,
         'dequeue_ms': 3., 'push_return_ms': 4.}
    runtime.validate_queue_events([{'queue_timings': [q, {**q, 'edge_index': 1}]}], 1)
    q['push_return_ms'] = 2.5
    runtime.validate_queue_events([{'queue_timings': [q, {**q, 'edge_index': 1}]}], 1)
    q['dequeue_ms'] = 1.5
    with pytest.raises(ValueError, match='partial order'):
        runtime.validate_queue_events([{'queue_timings': [q, {**q, 'edge_index': 1}]}], 1)


def test_missing_expected_profile_does_not_pass_as_valid_detail(bundle, tmp_path):
    doc = next(row['document'] for row in campaign.task_plan(bundle) if row['document']['detailed'])
    raw = tmp_path / 'raw'
    performance(raw, doc)
    with pytest.raises(ValueError, match='profile.json'):
        runtime.validate_summary(raw, doc, bundle)


def test_plain_session_cannot_silently_enable_queue_probe(bundle, tmp_path):
    doc = next(row['document'] for row in campaign.task_plan(bundle)
               if row['document']['diagnosis_block'] == 'plain_before')
    raw = tmp_path / 'raw'
    rows = performance(raw, doc)
    rows[0]['queue_timing_enabled'] = True
    (raw / 'result.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError, match='switch differs'):
        runtime.validate_summary(raw, doc, bundle)


def test_duplicate_or_wrong_input_frames_rejected(bundle, tmp_path):
    doc = next(row['document'] for row in campaign.task_plan(bundle)
               if row['document']['diagnosis_block'] == 'plain_before')
    raw = tmp_path / 'raw'
    rows = performance(raw, doc)
    for field, bad in [('frame_id', 0), ('input_index', 1)]:
        changed = deepcopy(rows)
        changed[1][field] = bad
        (raw / 'result.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in changed))
        with pytest.raises(ValueError):
            runtime.validate_summary(raw, doc, bundle)


def test_resume_after_receipt_before_record_does_not_repeat_measurement(bundle, tmp_path):
    frozen = common.api()
    doc = next(row['document'] for row in campaign.task_plan(bundle)
               if row['document']['diagnosis_block'] == 'plain_before')
    root, boot = tmp_path / 'run', 'boot-a'
    raw = root / 'raw' / boot / doc['task_id']
    performance(raw, doc)
    summary = runtime.validate_summary(raw, doc, bundle)
    receipt = root / 'accepted' / boot / (doc['task_id'] + '.json')
    files = {str(raw.relative_to(root) / name): value for name, value in frozen.inventory(raw).items()}
    frozen.write(receipt, {'task': doc, 'boot_id': boot, 'summary': summary, 'files': files})
    state = {'bundle': str(bundle), 'boot_id': boot, 'frozen_files': {}}
    recorder = campaign.Recorder(root, state)
    first = recorder.saved('case/plain_before', doc)
    assert recorder.saved('case/plain_before', doc) == first
    assert len(list((root / 'records').rglob('*.json'))) == 1
    assert len(list((root / 'accepted').rglob('*.json'))) == 1
    with pytest.raises(ValueError):
        runtime.Evidence(root, bundle).summary(first, doc, 'boot-b')
    (raw / 'affinity.json').write_text('{}')
    with pytest.raises(ValueError):
        recorder.saved('case/plain_before', doc)


def test_resume_refuses_a_different_boot_before_deploy(bundle, tmp_path, monkeypatch):
    state = {'bundle': str(bundle), 'boot_id': 'old-boot', 'status': 'INTERRUPTED'}
    monkeypatch.setattr(campaign, 'validate_state', lambda *a, **k: state)
    class DifferentBoot:
        def boot_id(self):
            return 'new-boot'
        def deploy(self, *args):
            raise AssertionError('must not deploy after boot mismatch')
    with pytest.raises(RuntimeError, match='another boot'):
        campaign.run_current(tmp_path / 'run', DifferentBoot())
    assert state['status'] == 'STOPPED_BOOT_CHANGED'


def test_network_interruption_retains_boot_and_allows_resume(bundle, tmp_path, monkeypatch):
    frozen = common.api()
    from cpu_vta_partitioning.transport import TransportError
    state = {'bundle': str(bundle), 'boot_id': 'same-boot', 'status': 'MEASURING'}
    monkeypatch.setattr(campaign, 'validate_state', lambda *a, **k: state)
    class Disconnected:
        def boot_id(self):
            raise TransportError('simulated disconnect')
    with pytest.raises(TransportError):
        campaign.run_current(tmp_path / 'run', Disconnected())
    assert frozen.read(tmp_path / 'run/state.json')['boot_id'] == 'same-boot'
    assert state['status'] == 'INTERRUPTED'


def test_generated_diagnostic_arguments_accepted_by_frozen_native_parser(bundle):
    import shutil
    import subprocess
    emulator = shutil.which('qemu-aarch64-static')
    assert emulator, 'packaged argument acceptance requires QEMU'
    plan = campaign.task_plan(bundle)
    docs = [row['document'] for row in plan if row['document']['diagnosis_block'] == 'component'][:1]
    docs += [next(row['document'] for row in plan if row['document']['detailed'])]
    source = common.DEFAULT_PARENT
    for doc in docs:
        args, _ = runtime.native_command(doc, bundle, '/board/bundle', '/board/job')
        runner = '/board/bundle/' + doc['executor_path']
        result = subprocess.run([emulator, str(source / 'runtime/ld-linux-aarch64.so.1'),
                                 '--library-path', str(source / 'runtime'), str(source / doc['executor_path']),
                                 *args[args.index(runner) + 1:], '--help'],
                                capture_output=True, text=True, timeout=20)
        assert result.returncode == 0, result.stdout + result.stderr
        assert 'Unknown option' not in result.stderr
