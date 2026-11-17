"""Analysis accepts only new identity-bound receipts and preserves measured scope."""
from copy import deepcopy

import pytest

from cpu_vta_partitioning.common import digest, inventory, write
from cpu_policies.open_resources.domain import candidate, component, component_key, requirements, stage
from resnet50_validation import analysis


def test_analysis_checks_raw_boot_and_task_even_when_inventory_matches(tmp_path):
    doc = {'task_id': 'one', 'kind': 'performance'}
    raw = tmp_path / 'raw'; raw.mkdir()
    (raw / 'boot_id').write_text('other-boot')
    (raw / 'task_hash').write_text(digest(doc))
    path = tmp_path / 'receipt.json'
    item = dict(task=doc, boot_id='expected-boot', raw=str(raw), files=inventory(raw), summary={})
    write(path, item)
    with pytest.raises(ValueError, match='raw task or boot'):
        analysis.receipt(path, doc, 'expected-boot')
    (raw / 'boot_id').write_text('expected-boot')
    (raw / 'task_hash').write_text('wrong-task')
    item['files'] = inventory(raw); write(path, item, replace=True)
    with pytest.raises(ValueError, match='raw task or boot'):
        analysis.receipt(path, doc, 'expected-boot')
    (raw / 'task_hash').write_text(digest(doc))
    item['files'] = inventory(raw); write(path, item, replace=True)
    assert analysis.receipt(path, doc, 'expected-boot') == item


def curves():
    records = {}
    for threads, service in enumerate((8., 4., 5., 3.), 1):
        cfg = component(stage('cpu', 0, 1, threads, list(range(threads))))
        records[component_key(cfg)] = dict(configuration=cfg,
            cost=dict(service_ms=service, set_ms=.1, run_ms=service - .2, get_ms=.1,
                      run_core_ms=7. + threads, boundary_core_ms=.05,
                      set_core_ms=.03, get_core_ms=.02),
            source=dict(boot_id='boot', task_id='task' + str(threads)))
    return records


def test_thread_curves_keep_nonmonotonic_points_and_do_not_invent_participation():
    rows = analysis.thread_curve_rows(curves())
    assert len(rows) == 4 and rows[2]['slower_than_previous_threads']
    assert rows[2]['marginal_service_reduction_ms'] == -1.
    assert rows[2]['speedup_over_one_thread'] == 1.6
    assert rows[2]['total_core_ms'] == 10.05
    assert all(r['compute_participating_threads'] is None for r in rows)
    assert all('approximation' in r['other_masks'] for r in rows)
    with pytest.raises(ValueError, match='all four'):
        analysis.thread_curve_rows(dict(list(curves().items())[:3]))


def test_call_accounting_keeps_component_dma_and_network_scopes_separate():
    def item(kind, key, warmup, scored, tail, stages=1):
        return dict(task=dict(task_id=key, kind=kind, warmup=warmup, scored=scored, tail=tail,
                              stages=[{}] * stages), measurement_and_collection_seconds=2.)
    assert analysis.measurement_cost([item('calibration', 'c', 2, 8, 0)])['scheduled_stage_calls_including_warmup'] == 10
    dma = analysis.measurement_cost([item('traffic', 'd', 1, 1, 0)])
    assert dma['scheduled_stage_calls_including_warmup'] == 2 and dma['complete_network_inferences'] is None
    items = [item('qualify_pipeline', 'q', 0, 1, 0, 5), item('performance', 'p', 8, 16, 24, 5)]
    result = analysis.measurement_cost(items)
    assert result['complete_network_inferences'] == 49
    assert result['scheduled_stage_calls_including_warmup'] == 245
    assert result['measurement_and_collection_seconds'] == 4.
    assert result['initialization_and_empty_callback_probe_calls'] is None
    with pytest.raises(ValueError, match='separately'):
        analysis.measurement_cost([item('traffic', 'd', 1, 1, 0), item('calibration', 'c', 2, 8, 0)])
    with pytest.raises(ValueError, match='duplicate'):
        analysis.measurement_cost([items[0], items[0]])
    missing = deepcopy(items); missing[0].pop('measurement_and_collection_seconds')
    assert analysis.measurement_cost(missing)['measurement_and_collection_seconds'] is None


def stage_fixture():
    cfg = candidate([stage('cpu', 0, 1, 1, [2]), stage('vta', 1, 2, 1, [0, 1, 2, 3]),
                     stage('cpu', 2, 3, 1, [3])], 'shared4', 'resource_group')
    programs = {r['program']: dict(name=r['program'], device=r['device'], begin=r['begin'], end=r['end'])
                for r in cfg['stages']}
    records = {}
    for key, value in requirements(programs).items():
        records[key] = dict(configuration=value,
            cost=dict(service_ms=5., run_core_ms=3., boundary_core_ms=.5,
                      set_ms=.5, run_ms=4., get_ms=.5), source=dict(boot_id='boot'))
    results = {cfg['id']: dict(II_ms=9., FPS=1000 / 9., stage_service_ms=[6., 7., 8.])}
    return cfg, programs, records, results


def test_stage_comparison_marks_homogeneous_aliases_and_host_cpu_work():
    cfg, programs, records, results = stage_fixture()
    rows = analysis.stage_rows({cfg['id']: cfg}, results, records, programs, None)
    assert rows[0]['pipeline_minus_isolated_ms'] == 1.
    assert 'homogeneous_core_permutation_not_per_mask_measurement' in rows[0]['assumptions']
    assert rows[1]['VTA_host_core_ms'] == 3.5
    assert rows[0]['VTA_host_core_ms'] is None
    results[cfg['id']]['stage_service_ms'].pop()
    with pytest.raises(ValueError, match='coverage'):
        analysis.stage_rows({cfg['id']: cfg}, results, records, programs, None)


def test_score_explanation_replays_frozen_formula_and_detects_changed_score():
    from cpu_policies.open_resources.costs import Costs
    from resnet50_validation.ranking import scalar_score
    cfg, programs, records, results = stage_fixture()
    lists, scores = {}, {}
    for method in ('tarnawski', 'complete'):
        costs = Costs(records, programs=programs,
                      distribution='balanced' if method == 'tarnawski' else 'session_weights')
        value = scalar_score(cfg, costs, method)
        lists[method] = [{**cfg, 'score_ns': value['score_ns']}]
        scores[method] = {cfg['id']: value}
    rows, terms = analysis.score_rows(lists, {cfg['id']: cfg}, results, records, programs, None, scores)
    assert len(rows) == 2 and any(r['term'] == 'host_concentration' for r in terms)
    assert all(r['binding_terms'] for r in rows)
    scores['complete'][cfg['id']]['score_ns'] += 1
    with pytest.raises(ValueError, match='frozen'):
        analysis.score_rows(lists, {cfg['id']: cfg}, results, records, programs, None, scores)


def test_cost_report_uses_recorded_build_and_deployment_time_not_guessed_values(tmp_path):
    output, run = tmp_path / 'output', tmp_path / 'run'
    write(output / 'profile_capsule/plans/board_payload.json', {'unpacked_bytes': 100})
    write(output / 'pipeline_bundle/plans/board_payload.json', {'unpacked_bytes': 80})
    write(output / 'build/programs/cpu/record.json', {'compile_seconds': 3.})
    write(output / 'build/build_reports/one.json', {'wall_seconds': 2.})
    write(output / 'build/build_reports/two.json', {'wall_seconds': 1.})
    write(run / 'components/deployments/0000.json', {'status': 'COMPLETE', 'elapsed_seconds': 4.})
    report = analysis.cost_report(output, run, [], [], [])
    assert report['compilation']['successful_program_worker_seconds'] == 3.
    assert report['compilation']['preparation_invocation_wall_seconds'] == 3.
    assert report['deployment']['elapsed_seconds'] == 4.
    assert report['search']['last_invocation_elapsed_seconds'] is None
    assert report['components']['initialization_and_empty_callback_probe_calls'] is None
    assert report['local_reference']['complete_graph_calls'] is None


def test_thread_plot_smoke_retains_measured_curves(tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    analysis.plot_thread_curves(analysis.thread_curve_rows(curves()), ['cpu_00_00'], tmp_path)
    assert (tmp_path / 'all_thread_curves.pdf').stat().st_size > 100
    assert (tmp_path / 'selected_thread_curves_1.svg').stat().st_size > 100


def test_reused_receipt_is_counted_once_and_stopped_work_stays_visible(tmp_path):
    output, run = tmp_path / 'output', tmp_path / 'run'
    write(output / 'profile_capsule/plans/board_payload.json', {'unpacked_bytes': 100})
    write(output / 'profile_capsule/manifest.json', {'bundle_id': 'new'})
    write(output / 'pipeline_bundle/plans/board_payload.json', {'unpacked_bytes': 80})
    def accepted(name, bundle):
        doc = dict(task_id=name, kind='calibration', bundle_id=bundle, warmup=2, scored=8, tail=0, stages=[{}])
        raw = tmp_path / name; raw.mkdir()
        (raw / 'boot_id').write_text('same-boot'); (raw / 'task_hash').write_text(digest(doc))
        item = dict(task=doc, boot_id='same-boot', raw=str(raw), files=inventory(raw), summary={},
                    measurement_and_collection_seconds=2.)
        path = tmp_path / (name + '.json'); write(path, item)
        return item, {'receipt': str(path)}
    reused, old_path = accepted('old-reused', 'old')
    unused, unused_path = accepted('old-unused', 'old')
    fresh, _ = accepted('fresh', 'new')
    write(run / 'components/imported_components.json', {'archive_audit': {'records': [old_path, unused_path]}})
    result = analysis.cost_report(output, run, [reused, fresh], [], [])
    assert result['components']['accepted_native_sessions'] == 2
    assert result['new_component_measurements']['accepted_native_sessions'] == 1
    assert result['reused_component_sources']['accepted_native_sessions'] == 1
    assert result['cumulative_unique_component_work']['accepted_native_sessions'] == 3
    assert result['cumulative_unique_component_work']['scheduled_stage_calls_including_warmup'] == 30
    assert result['archived_compiler_regression']['additional_stopped_sessions_not_used_for_costs'] == 1
