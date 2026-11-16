from copy import deepcopy

import pytest

from cpu_policies.core_model import domain as d


def test_frozen_partition_and_legacy_baseline_identity():
    cfg = d.baseline_candidate()
    assert cfg['id'] == 'cfg_b81cf0b8325c03d01989d7a3'
    assert tuple(row['program'] for row in cfg['stages']) == d.PROGRAMS
    assert len(d.requirements()) == 11
    assert len(d.configurations()['components']) == 11


def test_ordered_map_survives_construction_and_changes_identity():
    cfg = d.baseline_candidate()
    rows = deepcopy(cfg['stages'])
    rows[0] = d.stage('cpu', 0, 3, 4, [0, 1, 2, 3], [3, 0, 1, 2])
    changed = d.candidate(rows)
    d.validate_candidate(changed)
    assert changed['stages'][0]['mask'] == [0, 1, 2, 3]
    assert changed['stages'][0]['worker_core_map'] == [3, 0, 1, 2]
    assert changed['stages'][0]['caller_core'] == 3
    assert changed['id'] != cfg['id']
    assert changed['stages'][0]['signature'] != cfg['stages'][0]['signature']
    assert d.canonical_tuple(changed) != d.canonical_tuple(cfg)
    assert d.component_key(d.component(changed['stages'][0])) != d.component_key(d.component(cfg['stages'][0]))


@pytest.mark.parametrize('mask,mapping', [([0, 1], [0, 0]), ([0, 1], [0, 2]),
                                         ([1, 0], [0, 1]), ([0, 1], [False, 1]),
                                         ([0, 1, 2], [2, 1, 0])])
def test_invalid_or_out_of_domain_worker_mapping(mask, mapping):
    with pytest.raises(ValueError):
        d.stage('cpu', 0, 3, len(mask), mask, mapping)


def test_every_225_and_1024_candidate_has_unique_complete_execution_identity():
    fixed = list(d.allocations(False)); movable = list(d.allocations(True))
    assert len(d.worker_options(False)) == 15
    assert len(d.worker_options(True)) == 32
    assert len(fixed) == len({cfg['id'] for cfg in fixed}) == 225
    assert len(movable) == len({cfg['id'] for cfg in movable}) == 1024
    assert {cfg['id'] for cfg in fixed} < {cfg['id'] for cfg in movable}
    for cfg in movable:
        d.validate_candidate(cfg)
        assert cfg['stages'][-1] == d.stage('cpu', 20, 21, 1, [0])
        for row in cfg['stages']:
            if row['device'] == 'cpu':
                assert row['worker_core_map'][1:] == sorted(set(row['mask']) - {row['caller_core']})


def test_candidate_context_tail_and_metadata_cannot_be_changed():
    original = d.baseline_candidate()
    for mutation in ('tail', 'host', 'signature', 'gate'):
        cfg = deepcopy(original)
        if mutation == 'tail': cfg['stages'][-1] = d.stage('cpu', 20, 21, 1, [1])
        if mutation == 'host': cfg['host_mask'] = [2]
        if mutation == 'signature': cfg['stages'][0]['signature'] += '_other'
        if mutation == 'gate': cfg['execution_mode'] = 'resource_group'
        cfg['id'] = 'cfg_' + d.digest(d.configuration(cfg))[:24]
        with pytest.raises(ValueError): d.validate_candidate(cfg)


def test_components_are_canonical_and_vta_has_no_worker_map():
    for cfg in d.requirements().values():
        d.validate_candidate(cfg, component=True)
        row = cfg['stages'][0]
        if row['device'] == 'cpu':
            assert row['worker_core_map'] == list(range(row['threads']))
        else:
            assert row['worker_core_map'] == [] and row['threads'] == 1
            assert cfg['affinity_mode'] == 'group'
    with pytest.raises(ValueError): d.stage('vta', 3, 5, 1, [0, 1, 2, 3], [0])


def test_intervention_grid_changes_mapping_only_and_disjoint_groups_are_exact():
    configs = d.intervention_candidates()
    assert len(configs) == 6
    for label in ('caller_0_0', 'caller_0_3', 'caller_3_0', 'caller_3_3'):
        cfg = configs[label]
        d.validate_candidate(cfg)
        assert cfg['stages'][0]['threads'] == cfg['stages'][2]['threads'] == 4
        assert cfg['stages'][0]['mask'] == cfg['stages'][2]['mask'] == [0, 1, 2, 3]
    assert configs['disjoint_2_2']['stages'][0]['mask'] == [0, 1]
    assert configs['disjoint_2_2']['stages'][2]['mask'] == [2, 3]
    assert configs['disjoint_3_1']['stages'][0]['mask'] == [0, 1, 2]
    assert configs['disjoint_3_1']['stages'][2]['mask'] == [3]
