from copy import deepcopy

import pytest

from cpu_policies.core_model import domain as d, model as m, search as s


def records(uniform=False):
    result = {}
    for key, cfg in d.requirements().items():
        row = cfg['stages'][0]; n = row['threads']
        total = {'cpu_00_02': 100., 'cpu_05_07': 70., 'cpu_20_20': 1.,
                 'vta_03_04': 3., 'vta_08_19': 5.}[row['program']]
        weights = ([1. / n] * n if uniform or n == 1 else [.4] + [.6 / (n - 1)] * (n - 1))
        run = max(total / n, max(weights) * total) + .5 if row['device'] == 'cpu' else 10.
        cost = {'set_ms': .2, 'run_ms': run, 'get_ms': .3, 'service_ms': run + .5,
                'run_core_ms': total, 'set_core_ms': .1, 'get_core_ms': .2, 'boundary_core_ms': .3}
        if row['device'] == 'cpu':
            cost.update(worker_run_core_ms=[total * w for w in weights], cpu_core_weights=weights)
        result[key] = {'configuration': cfg, 'cost': cost}
    return {'costs': result}


def replace_cost(data, program, n, run, service, boundary=0., vector=None):
    for item in data['costs'].values():
        row = item['configuration']['stages'][0]
        if (row['program'], row['threads']) != (program, n): continue
        item['cost'] = {'service_ms': service, 'set_ms': boundary, 'run_ms': service - boundary,
                        'get_ms': 0., 'run_core_ms': run, 'set_core_ms': boundary,
                        'get_core_ms': 0., 'boundary_core_ms': boundary}
        if row['device'] == 'cpu':
            item['cost'].update(worker_run_core_ms=vector if vector is not None else [run / n] * n,
                                cpu_core_weights=[1. / n] * n)


def zero_data():
    data = records()
    for item in list(data['costs'].values()):
        row = item['configuration']['stages'][0]
        replace_cost(data, row['program'], row['threads'], 0., 0.)
    return data


def test_boundary_work_goes_to_explicit_caller_once():
    data = zero_data()
    replace_cost(data, 'cpu_00_02', 4, 40., 12., boundary=2.)
    cfg = d.intervention_candidates()['caller_3_0']
    score = m.score_configuration(cfg, m.Costs(data, distribution='weighted'))
    assert score['cpu_core_demand_ms'] == [10., 10., 10., 12.]
    assert score['score_ms'] == 12.
    assert sum(score['cpu_core_demand_ns']) == score['cpu_compute_core_ns'] + 2_000_000


def test_single_physical_vta_service_and_host_work_are_accumulated():
    data = zero_data()
    replace_cost(data, 'vta_03_04', 1, 4., 13., boundary=2.)
    replace_cost(data, 'vta_08_19', 1, 6., 17., boundary=3.)
    score = m.score_configuration(d.baseline_candidate(), m.Costs(data))
    assert score['score_ms'] == score['vta_service_sum_ms'] == 30.
    assert score['host_core_ms'] == 15.
    assert score['cpu_core_demand_ms'] == [0., 0., 0., 0.]
    assert score['subset_capacity_ms'][-1] == 3.75


def test_ddr_bound_is_maximum_not_added_to_stage_service():
    data = records()
    ddr = {'stage_service_ms': {program: {'1': 1000., '2': 1000., '3': 1000., '4': 1000.}
                               for program in d.PROGRAMS}}
    score = m.score_configuration(d.baseline_candidate(), m.Costs(data, ddr=ddr))
    assert score['score_ms'] == score['ddr_service_ms'] == 5000.


def test_nanosecond_remainder_follows_logical_slot_before_physical_core():
    data = zero_data()
    replace_cost(data, 'cpu_00_02', 3, .000001, 0., vector=[.000001 / 3] * 3)
    rows = d.baseline_candidate()['stages']
    rows[0] = d.stage('cpu', 0, 3, 3, [0, 1, 2], [2, 0, 1])
    cfg = d.candidate(rows)
    score = m.score_configuration(cfg, m.Costs(data, distribution='weighted'))
    assert score['cpu_compute_core_ns'] == sum(score['cpu_core_demand_ns']) == 1
    assert score['cpu_core_demand_ns'] == [0, 0, 1, 0]


def test_nonuniform_cost_moves_with_logical_workers_and_keeps_common_total():
    data = records()
    first = d.intervention_candidates()['caller_0_0']
    second = d.intervention_candidates()['caller_3_3']
    balanced = m.score_configuration(first, m.Costs(data, distribution='balanced'))
    weighted = m.score_configuration(first, m.Costs(data, distribution='weighted'))
    moved = m.score_configuration(second, m.Costs(data, distribution='weighted'))
    assert weighted['cpu_core_demand_ns'][0] > balanced['cpu_core_demand_ns'][0]
    assert moved['cpu_core_demand_ns'][3] > moved['cpu_core_demand_ns'][0]
    assert balanced['cpu_compute_core_ns'] == weighted['cpu_compute_core_ns'] == moved['cpu_compute_core_ns']
    assert balanced['vta_service_sum_ns'] == weighted['vta_service_sum_ns']


def test_bad_isolated_vector_is_rejected_instead_of_uniform_fallback():
    data = records()
    key = d.component_key(d.component(d.stage('cpu', 0, 3, 4, [0, 1, 2, 3])))
    data['costs'][key]['cost']['worker_run_core_ms'][0] += 1
    with pytest.raises(ValueError, match='conserve'):
        m.score_configuration(d.baseline_candidate(), m.Costs(data, distribution='weighted'))
    data = records(); del data['costs'][key]['cost']['worker_run_core_ms']; del data['costs'][key]['cost']['cpu_core_weights']
    with pytest.raises(ValueError, match='requires isolated'):
        m.score_configuration(d.baseline_candidate(), m.Costs(data, distribution='weighted'))


def test_all_four_complete_rankings_verified_and_pool_frozen_before_performance():
    result = s.rank(records())
    assert result['verification']['passed']
    assert result['verification']['compared_scores'] == 2048
    assert result['verification']['domain_counts'] == {
        'balanced_fixed': 225, 'weighted_fixed': 225,
        'balanced_movable': 1024, 'weighted_movable': 1024}
    assert len(result['pool']) <= 12
    assert set(result['selection']['order']) == set(result['pool'])
    assert len(result['selection']['order']) == len(set(result['selection']['order']))
    for method in s.METHODS:
        assert result['selection']['top1'][method] in result['pool']
        rows = result['methods'][method]
        assert [row['rank'] for row in rows] == list(range(1, len(rows) + 1))
        assert rows == sorted(rows, key=lambda row: (row['score_ns'], d.canonical_tuple(row['configuration'])))
    assert result['ranking_uses_pipeline_feedback'] is False
    assert result['selection']['selection_uses_pipeline_measurements'] is False


def test_independent_oracle_detects_a_one_nanosecond_scoring_bug(monkeypatch):
    original = s.score_configuration
    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        result['score_ns'] += 1
        return result
    monkeypatch.setattr(s, 'score_configuration', corrupt)
    with pytest.raises(ValueError, match='independent per-core score'):
        s.rank(records())


def test_equal_worker_data_has_no_disagreement_fillers_and_deduplicates_selections():
    result = s.rank(records(uniform=True))
    assert len(result['pool']) < 12
    assert not any('score_disagreement' in reason for reasons in result['selection']['reasons'].values() for reason in reasons)
    assert any(len(reasons) > 1 for reasons in result['selection']['reasons'].values())
    assert result['selection']['fill_duplicate_slots'] is False


def test_complete_pipeline_performance_metadata_never_changes_ranking():
    data = records()
    expected = s.rank(data)
    data['complete_pipeline_fps'] = {'arbitrary': 1000000.}
    actual = s.rank(data)
    assert actual['verification'] == expected['verification']
    assert actual['selection'] == expected['selection']
