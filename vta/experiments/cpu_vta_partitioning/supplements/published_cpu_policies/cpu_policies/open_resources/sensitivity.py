"""Offline sensitivity to CPU work splitting on a frozen shortlist.

Session thread counters include warmup, copies and coarse /proc ticks.  They
are alternative model weights, never measured scored-window per-thread costs.
This diagnostic neither edits selection nor reads pipeline performance.
"""
from .costs import Costs, score_configuration
from .domain import canonical_tuple


def evaluate(configurations, records, ddr):
    balanced = Costs(records, ddr=ddr, distribution='balanced')
    session = Costs(records, ddr=ddr, distribution='session_weights')
    rows = []
    source = configurations.values() if isinstance(configurations, dict) else configurations
    for config in source:
        a = score_configuration(config, balanced)
        b = score_configuration(config, session)
        rows.append({'id': config['id'], 'context_id': config['context_id'],
                     'balanced_ms': a['score_ms'], 'session_weighted_ms': b['score_ms'],
                     'relative_score_change': b['score_ms'] / a['score_ms'] - 1 if a['score_ms'] else None,
                     'balanced_per_core_ms': a['cpu_core_demand_ms'],
                     'session_weighted_per_core_ms': b['cpu_core_demand_ms'],
                     '_canonical': canonical_tuple(config)})
    baseline = sorted(rows, key=lambda r: (r['balanced_ms'], r['_canonical']))
    alternative = sorted(rows, key=lambda r: (r['session_weighted_ms'], r['_canonical']))
    for i, row in enumerate(baseline, 1):
        row['balanced_pool_rank'] = i
    for i, row in enumerate(alternative, 1):
        row['session_weighted_pool_rank'] = i
    for row in rows:
        del row['_canonical']
    return {'scope': 'frozen_measurement_pool_only_not_global_domain_reranking',
            'uses_pipeline_fps': False, 'changes_frozen_selection': False,
            'counter_scope': 'whole_session_thread_CPU_shares_rescaled_to_scored_total_compute_work',
            'limitations': ['warmup/tail and caller copy work included in session weights',
                            'coarse per-thread CPU ticks are not scored-window CPU times',
                            'this does not establish overlap of two full-domain Top-20 lists'],
            'configurations': sorted(rows, key=lambda r: r['id']),
            'pool_order_changed': [r['id'] for r in baseline] != [r['id'] for r in alternative],
            'maximum_absolute_relative_score_change': max((abs(r['relative_score_change'])
                for r in rows if r['relative_score_change'] is not None), default=0.)}
