"""Separate resource strategies before spending the five resource slots.

Tiny thread-count changes alone must not crowd out fixed resource budgets or
host placement. Input is the complete-domain bounded static pool; FPS is not
an input, and the 15 percent static guard is retained.
"""
from copy import deepcopy
from itertools import combinations

from .domain import canonical_tuple, compatible_groups, context


STRATEGIES = ('fixed_groups_serial', 'disjoint_groups_open', 'overlap_open',
              'single_host_shared', 'reserved_host')
LABELS = {'fixed_groups_serial': '固定核组、同组串行',
          'disjoint_groups_open': '分开核组、开放并发',
          'overlap_open': '共享或部分重叠、开放并发',
          'single_host_shared': '主机固定单核、仍与计算共享',
          'reserved_host': '为主机及控制保留核心'}


def strategies(row):
    masks = [tuple(s['mask']) for s in row['stages'] if s['device'] == 'cpu']
    ctx = context(row['context_id'])
    result = []
    if row['execution_mode'] == 'resource_group' and compatible_groups(masks):
        result.append('fixed_groups_serial')
    if row['execution_mode'] == 'off':
        if len(set(masks)) >= 2 and compatible_groups(masks):
            result.append('disjoint_groups_open')
        if any(set(a).intersection(b) for a, b in combinations(masks, 2)):
            result.append('overlap_open')
    if ctx['family'] == 'pinned_shared':
        result.append('single_host_shared')
    elif ctx['family'] == 'reserved':
        result.append('reserved_host')
    return result


def _topology(row):
    return tuple((s['device'], s['begin'], s['end']) for s in row['stages'])


def _resource(row):
    masks = [set(s['mask']) for s in row['stages'] if s['device'] == 'cpu']
    relations = tuple('same' if a == b else 'disjoint' if not a & b else
                      'containment' if a <= b or b <= a else 'crossing_overlap'
                      for a, b in combinations(masks, 2))
    return (tuple(s['threads'] for s in row['stages'] if s['device'] == 'cpu'),
            relations, context(row['context_id'])['family'], row['execution_mode'])


def select(pure20, static_pool):
    if len(pure20) != 20:
        raise ValueError('strategy selection requires an exact pure Top-20')
    key = lambda r: (r['score_ns'], canonical_tuple(r))
    pure = sorted(pure20, key=key)
    chosen, identifiers = [], set()
    def add(row, reason):
        if row['id'] in identifiers:
            return False
        value = deepcopy(row)
        value['selection_reason'] = reason
        value['resource_strategies'] = strategies(row)
        chosen.append(value)
        identifiers.add(row['id'])
        return True
    for row in pure[:10]:
        add(row, 'pure_static_top10')
    eligible = sorted((r for r in static_pool if r['score_ns'] * 100 <= pure[0]['score_ns'] * 115), key=key)
    seen_topologies = {_topology(r) for r in chosen}
    for row in eligible:
        if len(chosen) >= 15:
            break
        t = _topology(row)
        if t not in seen_topologies and add(row, 'new_topology_within_15pct'):
            seen_topologies.add(t)
    cut_count = len(chosen) - 10
    for row in pure:
        if len(chosen) >= 15:
            break
        add(row, 'pure_static_fallback_for_unavailable_cut')
    coverage = []
    for family in STRATEGIES:
        seen_resources = {_resource(r) for r in chosen}
        candidates = [r for r in eligible if r['id'] not in identifiers and family in strategies(r)]
        if candidates:
            novel = [r for r in candidates if _resource(r) not in seen_resources]
            best = (novel or candidates)[0]
            add(best, 'resource_strategy:' + family)
            coverage.append({'strategy': family, 'label': LABELS[family], 'selected': best['id']})
        else:
            coverage.append({'strategy': family, 'label': LABELS[family], 'selected': None,
                             'reason': 'no_unselected_representative_within_static_15pct_guard'})
    for row in eligible:
        if len(chosen) >= 20:
            break
        if _resource(row) not in {_resource(r) for r in chosen}:
            add(row, 'new_resource_fallback_within_15pct')
    for row in pure:
        if len(chosen) >= 20:
            break
        add(row, 'pure_static_fallback')
    if len(chosen) != 20:
        raise ValueError('strategy selection could not fill twenty distinct identities')
    for i, row in enumerate(chosen, 1):
        row['selection_order'] = i
    for entry in coverage:
        entry['covered_by'] = [row['id'] for row in chosen
                               if entry['strategy'] in strategies(row)]
        entry['covered'] = bool(entry['covered_by'])
        if entry['selected'] is None and entry['covered']:
            entry['reason'] = 'already_covered_by_another_selection_slot'
    return chosen, {'pure': 10, 'new_topologies_requested': 5, 'new_topologies_selected': cut_count,
                    'resource_strategy_slots': 5, 'resource_strategy_coverage': coverage,
                    'maximum_score_ratio': 1.15, 'uses_pipeline_fps': False,
                    'physical_core_mirrors_are_not_distinct_strategies': True,
                    'fallback': 'new_resource_within_guard_then_pure_static_order',
                    'optimal_diversity_claimed': False}
