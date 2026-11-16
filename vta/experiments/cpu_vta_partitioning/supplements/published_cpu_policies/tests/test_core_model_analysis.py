"""Error metrics, matched mapping interventions and raw-bound report replay."""
from hashlib import sha256
import json
from pathlib import Path

import pytest

from cpu_policies.core_model import analysis as a, common, domain, search


def test_prediction_error_uses_same_measured_pool_and_ties():
    assert a.average_ranks([2., 1., 2.]) == [2.5, 1., 2.5]
    result = a.prediction_metrics([10., 20., 30.], [20., 20., 30.])
    assert result['MAPE_percent'] == pytest.approx(50 / 3)
    assert result['MAE_ms'] == pytest.approx(10 / 3)
    assert 0 < result['spearman_on_measured_pool_only'] <= 1
    assert a.prediction_metrics([1., 1.], [2., 3.])['spearman_on_measured_pool_only'] is None
    for bad in (0., float('nan'), float('inf')):
        with pytest.raises(ValueError): a.prediction_metrics([bad], [1.])


def test_caller_pairing_requires_same_four_core_compute_sets():
    rows = []
    for first, middle, fps in [(0, 0, 10.), (0, 3, 12.), (3, 0, 11.), (3, 3, 10.5)]:
        rows.append({'configuration': str((first, middle)), 'front_caller': first, 'middle_caller': middle,
                     'front_threads': 4, 'middle_threads': 4, 'front_mask': '0,1,2,3',
                     'middle_mask': '0,1,2,3', 'fps': fps, 'ii_ms': 1000 / fps})
    result = a.placement_pairs(rows)
    assert len(result) == 4
    assert result[0]['fps_change_percent'] == pytest.approx(20)
    rows[1]['front_threads'] = 3
    assert len(a.placement_pairs(rows)) == 2


def synthetic_costs():
    records = {}
    for key, cfg in domain.requirements().items():
        stage = cfg['stages'][0]; n = stage['threads']
        total = {'cpu_00_02': 100., 'cpu_05_07': 70., 'cpu_20_20': 1.,
                 'vta_03_04': 3., 'vta_08_19': 5.}[stage['program']]
        weights = [1.] if n == 1 else [.4] + [.6 / (n - 1)] * (n - 1)
        run = max(total / n, max(weights) * total) + .5 if stage['device'] == 'cpu' else 10.
        cost = {'set_ms': .2, 'run_ms': run, 'get_ms': .3, 'service_ms': run + .5,
                'run_core_ms': total, 'set_core_ms': .1, 'get_core_ms': .2, 'boundary_core_ms': .3}
        if stage['device'] == 'cpu':
            cost.update(worker_run_core_ms=[total * w for w in weights], cpu_core_weights=weights)
        records[key] = {'configuration': cfg, 'cost': cost}
    return {'costs': records}


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def _performance(root, boot, cfg, gap, block):
    task_id = sha256((cfg['id'] + block).encode()).hexdigest()[:24]
    doc = {'kind': 'performance', 'configuration': cfg, 'warmup': 8, 'scored': 16, 'tail': 4,
           'run_probe': False, 'task_id': task_id, 'diagnosis_case': cfg['id'],
           'diagnosis_block': block, 'executor_sha256': 'executor'}
    raw = root / 'raw' / boot / task_id; raw.mkdir(parents=True)
    frames = []
    for i in range(28):
        row = {'frame_id': i, 'completion_index': i, 'input_index': 0, 'completion_ms': i * gap}
        for index in range(5):
            row.update({f'stage{index}_ms': 1.2, f'stage{index}_set_ms': .1,
                        f'stage{index}_run_ms': 1., f'stage{index}_get_ms': .1})
        frames.append(row)
    (raw / 'result.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in frames))
    _write(raw / 'affinity.json', {})
    files = {str(path.relative_to(root)): {'sha256': sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
             for path in raw.iterdir()}
    summary = {'ii_ms': gap}
    receipt = root / 'accepted' / boot / (task_id + '.json')
    _write(receipt, {'task': doc, 'boot_id': boot, 'summary': summary, 'files': files})
    summary = {**summary, '_receipt': {'path': str(receipt.relative_to(root)), 'sha256': sha256(receipt.read_bytes()).hexdigest()}}
    _write(root / 'records' / boot / (task_id + '.json'), {'document': doc, 'boot_id': boot, 'summary': summary})
    return {'document': doc, 'boot_id': boot, 'summary': summary}


def test_analysis_replays_frozen_rankings_and_bound_raw_measurements(tmp_path, monkeypatch):
    from cpu_policies.core_model import campaign
    root, bundle = tmp_path / 'run', tmp_path / 'bundle'
    frozen = common.api()
    programs = {name: frozen.read(common.DEFAULT_PARENT / 'plans/programs.json')[name] for name in domain.PROGRAMS}
    data = synthetic_costs(); ddr = {'stage_service_ms': {name: 0. for name in domain.PROGRAMS}}
    ranked = search.rank(data, ddr=ddr, programs=programs)
    _write(bundle / 'plans/programs.json', programs); _write(bundle / 'plans/baseline_ddr.json', ddr)
    state = {'experiment': 'test', 'bundle': str(bundle), 'bundle_id': 'bundle', 'boot_id': 'boot'}
    monkeypatch.setattr(campaign, 'validate_state', lambda *args, **kw: state)
    monkeypatch.setattr(campaign, 'validated_records', lambda *args, **kw: None)
    monkeypatch.setattr(common, 'api', lambda *args, **kw: frozen)
    monkeypatch.setattr(a, '_plot', lambda *args: None)
    _write(root / 'state.json', state); _write(root / 'costs.json', data)
    _write(root / 'rankings.json', ranked); _write(root / 'search_pool.json', ranked['pool'])
    measurements = {}
    for i, (key, cfg) in enumerate(ranked['pool'].items()):
        measurements[key] = _performance(root, 'boot', cfg, 50. + i, 'performance')
    _performance(root, 'boot', domain.baseline_candidate(), 61., 'old_bridge')
    _write(root / 'screening.json', {'boot_id': 'boot', 'rounds': 1, 'order': list(measurements), 'measurements': measurements})
    result = a.analyze(root)
    assert result['pipeline_count'] == len(measurements)
    assert result['method_top1_count'] == 4
    report = json.loads((root / 'analysis/analysis.json').read_text())
    assert report['pipeline_feedback_used_for_costs'] is False
    assert report['full_DP_changed'] is False
    assert len(report['caller_placement_pairs']) == 4
    assert all(row['completion_intervals'] == 15 for row in report['measurements'])
    path = next((root / 'raw').rglob('result.jsonl')); path.write_text('tampered')
    with pytest.raises(ValueError, match='evidence changed'):
        a.analyze(root)
