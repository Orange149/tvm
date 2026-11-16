from cpu_policies.open_resources.costs import Costs
from cpu_policies.open_resources.domain import candidate, component, component_key, stage
from cpu_policies.open_resources.sensitivity import evaluate


def test_thread_weight_diagnostic_reorders_only_its_frozen_pool():
    a, b = stage('cpu', 0, 1, 2, [0, 1]), stage('cpu', 2, 3, 2, [0, 1])
    v = stage('vta', 1, 2, 1, [0, 1, 2, 3])
    records = {}
    for row, weights in [(a, [.9, .1]), (b, [.1, .9]), (v, [1.])]:
        cfg = component(row)
        records[component_key(cfg)] = {'kind': 'synthetic', 'configuration': cfg,
            'cost': {'service_ms': 100. if row['device'] == 'cpu' else 1.,
                     'run_core_ms': 200. if row['device'] == 'cpu' else 1.,
                     'boundary_core_ms': 0.}, 'session_thread_weights': weights}
    cross = candidate([a, v, stage('cpu', 2, 3, 2, [1, 2])])
    separate = candidate([a, v, stage('cpu', 2, 3, 2, [2, 3])])
    pool = {c['id']: c for c in [cross, separate]}
    ddr = {'stage_service_ms': {r['program']: {str(r['threads']): 1.} for r in [a, b, v]}}
    report = evaluate(pool, records, ddr)
    x = next(r for r in report['configurations'] if r['id'] == cross['id'])
    assert x['balanced_ms'] == 200.
    assert x['session_weighted_ms'] == 180.
    assert report['scope'].startswith('frozen_measurement_pool_only')
    assert report['uses_pipeline_fps'] is False and report['changes_frozen_selection'] is False
    assert len(pool) == 2
