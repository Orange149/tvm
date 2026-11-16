"""Pipeline contracts and packaging; physical inference is checked separately."""
from copy import deepcopy
import json
from pathlib import Path
import os
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / 'artifacts/yolo_transfer_01/profile_bundle_bridge02'


def test_equal_predictions_are_not_reported_as_correct_ordering():
    from yolo_transfer.analysis import ranking_metrics
    result = ranking_metrics({'a': 1000000, 'b': 1000000, 'c': 1000000},
                             {'a': {'ii_ms': 1.}, 'b': {'ii_ms': 3.}, 'c': {'ii_ms': 2.}})
    assert result['spearman_rho'] is None
    assert result['predicted_tied_pairs'] == 3
    assert result['comparable_pairs'] == 0 and result['discordant_pairs'] == 0


def test_budget_comparison_uses_frozen_order_and_common_measurement_ids():
    from yolo_transfer.analysis import evaluate
    from yolo_transfer.pipeline import METHODS
    names = ['case' + str(i) for i in range(20)]
    lists = {METHODS[0]: names, METHODS[1]: list(reversed(names)), METHODS[2]: names[5:] + names[:5]}
    measured = {name: {'ii_ms': 20. - i} for i, name in enumerate(names)}
    result = evaluate(lists, measured)
    assert result['methods'][METHODS[0]]['1']['ii_ms'] == 20.
    assert result['methods'][METHODS[1]]['1']['ii_ms'] == 1.
    assert result['methods'][METHODS[2]]['5']['ii_ms'] == 11.
    assert all(row['20']['pool_relative_II_regret_percent'] == 0 for row in result['methods'].values())
    assert result['single_boot_observation_only']
    measured.pop(names[0])
    with pytest.raises(ValueError, match='union'):
        evaluate(lists, measured)


def test_domain_count_is_independent_of_costs_and_matches_small_search_cardinality():
    from yolo_transfer.domain_counts import allocation_counts
    small = allocation_counts(5, 1)
    assert small['topologies'] == 6
    assert small['single_core'] == small['prefix'] == 6 * (5 * 16 + 4 * 9)
    full = allocation_counts()
    assert full['topologies'] == 2509 and full['contexts'] == 9
    assert full['enhanced'] == 472266548
    assert full['single_core'] == full['prefix'] == 3067532
    assert full['costs_or_pipeline_data_used'] is False


def test_freeze_write_is_idempotent_but_refuses_changed_ranking(tmp_path):
    from yolo_transfer.ranking import write_frozen
    target = tmp_path / 'pool.json'
    write_frozen(target, {'candidate': 1})
    write_frozen(target, {'candidate': 1})
    with pytest.raises(ValueError, match='changed'):
        write_frozen(target, {'candidate': 2})


@pytest.fixture
def actual_assets():
    if not (PROFILE / 'manifest.json').is_file():
        pytest.skip('local compiled YOLO assets not available')
    return PROFILE


def test_seven_stage_command_preserves_live_inputs_gate_and_worker_mapping(tmp_path, actual_assets):
    from cpu_vta_partitioning.common import read, write
    from cpu_policies.open_resources.domain import candidate, stage, validate_candidate
    from yolo_transfer import pipeline
    programs = read(actual_assets / 'plans/programs.json')
    rows = [stage('cpu', 0, 1, 2, [0, 1]), stage('vta', 1, 2, 1, [2]),
            stage('cpu', 2, 3, 2, [0, 1]), stage('vta', 3, 4, 1, [2]),
            stage('cpu', 4, 5, 2, [0, 1]), stage('vta', 5, 13, 1, [2]),
            stage('cpu', 13, 14, 2, [2, 3])]
    cfg = candidate(rows, 'host_shared_2', 'resource_group')
    validate_candidate(cfg, programs, unit_count=14, max_vta_islands=3)
    write(tmp_path / 'plans/programs.json', programs)
    write(tmp_path / 'plans/pool.json', {cfg['id']: cfg})
    write(tmp_path / 'plans/inputs.json', read(actual_assets / 'plans/inputs.json'))
    write(tmp_path / 'manifest.json', {'bundle_id': 'fixture', 'runner_sha256': 'runner'})
    write(tmp_path / 'plans/protocol.json', {'experiment': 'fixture',
        'qualification': {'warmup': 0, 'scored': 1, 'tail': 0},
        'performance': {'warmup': 8, 'scored': 16, 'tail': 4}})
    doc = pipeline.make_task(tmp_path, cfg['id'])
    argv, inputs = pipeline.native_command(doc, tmp_path, '/remote', '/job')
    assert argv[argv.index('--cpu-stage-gate') + 1] == 'resource_group'
    assert argv[argv.index('--output-mode') + 1] == 'raw'
    assert argv[argv.index('--runs') + 1] == '28'
    assert '--cpu-worker-pinning' in argv and '--resource-affinity-probe' not in argv
    assert '--process-cpu-affinity' not in argv
    assert 'TVM_EXCLUDE_WORKER0=1' in argv
    for index, row in enumerate(rows[1:], 1):
        previous = programs[rows[index - 1]['program']]['output_node_ids']
        expected = ','.join(f'stage{index-1}:{previous.index(node)}' for node in programs[row['program']]['input_node_ids'])
        assert argv[argv.index(f'--stage{index}-input-sources') + 1] == expected
    assert inputs == '/remote/inputs/000.bin\n'
    assert len(programs[rows[5]['program']]['input_names']) == 2
    assert len(programs[rows[6]['program']]['output_node_ids']) == 8


def test_raw_pipeline_hash_guard_checks_every_detection_output():
    from yolo_transfer.pipeline import check_outputs
    expected = {str(i): {'shape': [2], 'dtype': 'float32', 'nbytes': 8, 'fnv1a64': str(i)} for i in range(8)}
    program = {'output_node_ids': list(range(8))}
    actual = [{'index': i, **expected[str(i)]} for i in range(8)]
    check_outputs(actual, program, expected)
    actual[7]['fnv1a64'] = 'wrong'
    with pytest.raises(ValueError, match='ARM'):
        check_outputs(actual, program, expected)
    with pytest.raises(ValueError, match='incomplete'):
        check_outputs(actual[:-1], program, expected)


def pipeline_raw_fixture(tmp_path):
    from cpu_vta_partitioning.common import write
    from cpu_policies.open_resources.domain import candidate, stage
    cfg = candidate([stage('cpu', 0, 9, 2, [0, 1]), stage('vta', 9, 13, 1, [2]),
                     stage('cpu', 13, 14, 1, [3])], 'host_shared_2')
    expected = {str(i): {'shape': [2], 'dtype': 'float32', 'nbytes': 8, 'fnv1a64': str(i)} for i in range(8)}
    bundle, raw = tmp_path / 'bundle', tmp_path / 'raw'
    write(bundle / 'plans/programs.json', {cfg['stages'][-1]['program']: {'output_node_ids': list(range(8))}})
    write(bundle / 'plans/expected_outputs.json', expected)
    entries = []
    for index, row in enumerate(cfg['stages']):
        tids = [100 * (index + 1) + slot for slot in range(row['threads'])]
        affinities = [{'tid': tid, 'cpus': [row['mask'][slot]], 'cpu_time_delta_ms': 20.}
                      for slot, tid in enumerate(tids)]
        entries.append({'requested_cpus': row['mask'], 'active_runtime_threads': row['threads'],
                        'caller_tid': tids[0], 'thread_ids': tids, 'effective_affinities': affinities})
    report = {'cpu_worker_pinning': True, 'stages': entries,
              'control_threads': [{'role': 'collector', 'tid': 401, 'cpus': [0, 1, 2, 3]},
                                  {'role': 'producer', 'tid': 402, 'cpus': [0, 1, 2, 3]}]}
    write(raw / 'affinity.json', report)
    rows = []
    for i in range(28):
        row = {'frame_id': i, 'completion_index': i, 'input_index': 0,
               'completion_ms': 5. + 10 * i, 'completion_process_cpu_ms': 100. + 20 * i,
               'total_latency_ms': 30., 'cpu_stage_gate_mode': 'off', 'queue_timing_enabled': False,
               'defer_result_write_enabled': True, 'stage_cpu_time_scope': 'disabled_concurrent_pipeline',
               'raw_outputs': [{'index': j, **value} for j, value in enumerate(expected.values())]}
        for index in range(3):
            for op, value in [('set', .1), ('run', 5. - index), ('get', .1)]:
                row[f'stage{index}_{op}_ms'] = value
                row[f'stage{index}_{op}_process_cpu_ms'] = None
        rows.append(row)
    (raw / 'result.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
    document = {'kind': 'performance', 'warmup': 8, 'scored': 16, 'tail': 4, 'input_indices': [0],
                'configuration': cfg, 'stages': cfg['stages']}
    return bundle, raw, document, rows


def test_pipeline_uses_fifteen_intervals_and_one_process_clock_not_summed_stage_clocks(tmp_path):
    from yolo_transfer.pipeline import summary
    bundle, raw, doc, _ = pipeline_raw_fixture(tmp_path)
    result = summary(raw, doc, bundle)
    assert result['gaps'] == 15 and result['kept_timestamps'] == 16
    assert result['ii_ms'] == 10. and result['fps'] == 100.
    assert result['process_core_ms_per_completion'] == 20.
    assert result['average_cpu_cores'] == 2.
    assert result['stage_service_ms'] == pytest.approx([5.2, 4.2, 3.2])


@pytest.mark.parametrize('fault', ['last_output', 'duplicate_frame', 'bad_scope', 'wrong_gate', 'impossible_CPU_clock', 'wrong_caller', 'restricted_shared_host'])
def test_invalid_pipeline_records_are_rejected(tmp_path, fault):
    from yolo_transfer.pipeline import summary
    bundle, raw, doc, rows = pipeline_raw_fixture(tmp_path)
    if fault == 'last_output':
        rows[-1]['raw_outputs'][-1]['fnv1a64'] = 'bad'
    elif fault == 'duplicate_frame':
        rows[-1]['frame_id'] = 0
    elif fault == 'bad_scope':
        rows[0]['stage_cpu_time_scope'] = 'exclusive_process_cpu_time'
    elif fault == 'wrong_gate':
        rows[0]['cpu_stage_gate_mode'] = 'resource_group'
    elif fault == 'impossible_CPU_clock':
        for i, row in enumerate(rows):
            row['completion_process_cpu_ms'] = 100. + 60 * i
    elif fault == 'wrong_caller':
        path = raw / 'affinity.json'; report = json.loads(path.read_text())
        report['stages'][0]['effective_affinities'][0]['cpus'] = [1]
        report['stages'][0]['effective_affinities'][1]['cpus'] = [0]
        path.write_text(json.dumps(report))
    else:
        from cpu_policies.open_resources.domain import candidate, stage
        cfg = candidate([doc['stages'][0], stage('vta', 9, 13, 1, [0, 1, 2, 3]), doc['stages'][2]], 'shared4')
        doc['configuration'] = cfg; doc['stages'] = cfg['stages']
        path = raw / 'affinity.json'; report = json.loads(path.read_text())
        report['stages'][1]['requested_cpus'] = [0, 1, 2, 3]
        # Existing subset checks alone would accept this unintended restriction.
        report['stages'][1]['effective_affinities'][0]['cpus'] = [2]
        path.write_text(json.dumps(report))
    (raw / 'result.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError):
        summary(raw, doc, bundle)


def packaging_fixture(tmp_path, actual_assets):
    from cpu_vta_partitioning.common import read, write, sha256
    from cpu_policies.open_resources.domain import candidate, nonempty_masks, stage
    from yolo_transfer.pipeline import METHODS
    masks = nonempty_masks(); configs = []
    for first in masks:
        for last in masks:
            configs.append(candidate([stage('cpu', 0, 9, len(first), first),
                stage('vta', 9, 13, 1, [0, 1, 2, 3]), stage('cpu', 13, 14, len(last), last)]))
            if len(configs) == 20:
                break
        if len(configs) == 20:
            break
    rows = [{**cfg, 'score_ns': 100000000 + i, 'selected_score_ns': 100000000 + i,
             'original_score_ns': 100000000 + i} for i, cfg in enumerate(configs)]
    pool = {cfg['id']: cfg for cfg in rows}
    ranked = tmp_path / 'ranked'
    write(ranked / 'rankings.json', {'lists': {name: rows for name in METHODS},
        'original_verification': {'passed': True}, 'enhanced_verification': {'passed': True}})
    write(ranked / 'pool.json', pool)
    write(ranked / 'FREEZE.json', {'frozen_at': '2026-11-08T00:00:00+00:00',
        'rankings_sha256': sha256(ranked / 'rankings.json'), 'pool_sha256': sha256(ranked / 'pool.json')})
    write(ranked / 'ddr.json', {'fixture': True})
    write(ranked / 'INPUTS.json', {'profile_capsule_manifest.json': {'sha256': sha256(actual_assets / 'manifest.json')},
                                 'programs.json': {'sha256': sha256(actual_assets / 'plans/programs.json')}})
    for name in ('original', 'enhanced'):
        write(ranked / name / 'fixture.json', {'scope': 'packaging_only_no_full_domain_proof_claim'})
    write(ranked / 'inputs/isolated_costs.json', {'costs': {}, 'fixture': True})
    return ranked


def test_pipeline_capsule_relocates_and_loads_only_its_frozen_controller(tmp_path, monkeypatch, actual_assets):
    from yolo_transfer import pipeline, ranking
    from yolo_transfer.domain_counts import allocation_counts
    ranked = packaging_fixture(tmp_path, actual_assets)
    calls = []
    def packaging_proof(path):
        # Only the expensive proof dependency is mocked in this packaging test.
        # The real complete-domain verifier runs before production preparation.
        calls.append(Path(path))
        return {'passed': True, 'domain_counts': allocation_counts(), 'pipeline_FPS_read': False,
                'scope': 'packaging dependency fixture; not an experimental verification'}
    monkeypatch.setattr(ranking, 'verify_artifacts', packaging_proof)
    bundle = tmp_path / 'bundle'
    pipeline.prepare(actual_assets, ranked, bundle)
    assert calls == [ranked.resolve()]
    moved = tmp_path / 'relocated'; bundle.rename(moved)
    script = moved / 'sources/run_pipeline.py'
    probe = ("import runpy,sys; from pathlib import Path; entry=sys.argv[1]; capsule=sys.argv[2]; "
             "sys.argv=[entry, 'validate', '--bundle', capsule]; "
             "runpy.run_path(sys.argv[0],run_name='__main__'); "
             "import yolo_transfer.pipeline as p; import cpu_policies.open_resources.runtime as r; "
             "assert Path(p.__file__).is_relative_to(Path(capsule)); "
             "assert Path(r.__file__).is_relative_to(Path(capsule)); assert 'tvm' not in sys.modules")
    result = subprocess.run([sys.executable, '-c', probe, str(script), str(moved)],
                            cwd=tmp_path, env=dict(os.environ, PYTHONPATH=''), capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)['validated'] is True
    inputs = json.loads((moved / 'plans/inputs.json').read_text())
    assert all('raw' not in value for value in inputs['records'][0]['frontiers'].values())
    assert (moved / 'inputs/000.bin').is_file()
