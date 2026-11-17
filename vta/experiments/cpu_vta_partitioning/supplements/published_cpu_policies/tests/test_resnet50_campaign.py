"""Scope, identity and window checks; never connects to a board."""
import json
from pathlib import Path

import pytest

from cpu_vta_partitioning.common import digest
from cpu_policies.open_resources.domain import candidate, stage
from resnet50_validation.common import METHODS, protocol
from resnet50_validation.runtime import task, summary
from resnet50_validation.analysis import budgets
from resnet50_validation.campaign import run


def capsule(tmp_path):
    cfg = candidate([stage('cpu', 0, 1, 1, [0]), stage('vta', 1, 37, 1, [3]),
                     stage('cpu', 37, 38, 1, [1])], 'host_shared_3', 'off')
    programs = {r['program']: {**r, 'output_node_ids': [r['end']]} for r in cfg['stages']}
    def put(name, value):
        p = tmp_path / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(value))
    put('manifest.json', dict(network='resnet50_v2', unit_count=38, canonical_sha256='graph',
        runner_sha256='runner', runtime_sha256='runtime', bundle_id='bundle'))
    put('plans/protocol.json', protocol()); put('plans/programs.json', programs)
    return cfg


def test_graph_size_38_and_48_frames_are_explicit(tmp_path):
    cfg = capsule(tmp_path)
    doc = task(tmp_path, cfg)
    assert doc['network'] == 'resnet50_v2' and doc['stages'][-1]['end'] == 38
    assert (doc['warmup'], doc['scored'], doc['tail']) == (8, 16, 24)
    assert sum(doc[k] for k in ('warmup', 'scored', 'tail')) == 48
    assert task(tmp_path, cfg, 'qualify')['scored'] == 1
    with pytest.raises(ValueError, match='unrequested'):
        task(tmp_path, cfg, 'long')


def test_task_identity_binds_graph_and_runtime(tmp_path):
    cfg = capsule(tmp_path); a = task(tmp_path, cfg)
    p = tmp_path / 'manifest.json'; data = json.loads(p.read_text())
    data['canonical_sha256'] = 'other_network_graph'; p.write_text(json.dumps(data))
    assert a['task_id'] != task(tmp_path, cfg)['task_id']
    assert a['task_id'] == digest({k: v for k, v in a.items() if k != 'task_id'})[:24]


def test_no_board_without_explicit_flag(tmp_path):
    with pytest.raises(ValueError, match='execute-board'):
        run(tmp_path, tmp_path/'run', tmp_path/'local.json', execute_board=False)


def test_budget_selects_only_each_frozen_prefix():
    lists = {method: [{'id': str(i)} for i in range(20)] for method in METHODS}
    results = {str(i): dict(II_ms=100-i, FPS=1000/(100-i)) for i in range(20)}
    rows = budgets(lists, results)
    assert [r['static_rank'] for r in rows] == [1, 5, 20, 1, 5, 20]
    assert rows[0]['FPS'] < rows[2]['FPS']
    with pytest.raises(ValueError, match='all selected'):
        budgets(lists, {k: v for k, v in results.items() if k != '19'})


def test_early_exhaustion_is_rejected_not_silently_extended(tmp_path, monkeypatch):
    from common_top20 import runtime as original
    import resnet50_validation.runtime as module
    cfg = capsule(tmp_path); document = task(tmp_path, cfg)
    def fake_summary(raw, doc, bundle):
        return {'stage_completion_counts_at_window': [{'stage': 'cpu', 'counts': [12, 48]}]}
    monkeypatch.setattr(original, 'summary', fake_summary)
    with pytest.raises(ValueError, match='exhausted'):
        summary(tmp_path, document, tmp_path)
    assert document['tail'] == 24


def test_inventory_change_is_reported_not_rejected(tmp_path, monkeypatch):
    from common_top20 import runtime as original
    cfg = capsule(tmp_path); document = task(tmp_path, cfg)
    def fake_summary(raw, doc, bundle):
        return {'stage_completion_counts_at_window': [{'stage': 'cpu', 'counts': [11, 27]}]}
    monkeypatch.setattr(original, 'summary', fake_summary)
    result = summary(tmp_path, document, tmp_path)
    assert result['inventory_at_window'] == [{'stage': 'cpu', 'first': 2, 'last': 3}]
    assert result['early_stage_exhaustion'] is False
