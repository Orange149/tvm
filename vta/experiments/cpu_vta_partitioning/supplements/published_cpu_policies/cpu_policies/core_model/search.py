"""Exhaust the small caller-choice domain, then freeze prospective tests.

The scalar oracle has its own arithmetic and raw-cost lookup. It deliberately
does not call the scoring implementation or its allocation/capacity helpers.
No complete-pipeline result is accepted by any ranking interface.
"""
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
from hashlib import sha256
from itertools import combinations
import json

from .domain import (allocations, baseline_candidate, canonical_tuple, component,
                     component_key, intervention_candidates, requirements)
from .model import Costs, score_configuration

METHODS = ('balanced_fixed', 'weighted_fixed', 'balanced_movable', 'weighted_movable')
METHOD_LABELS = {'balanced_fixed': '核时均分／调用线程固定最小核心',
                 'weighted_fixed': '隔离逐线程核时／调用线程固定最小核心',
                 'balanced_movable': '核时均分／允许选择调用线程',
                 'weighted_movable': '隔离逐线程核时／允许选择调用线程'}


def _hash(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _reference_ns(value):
    return int((Decimal(str(value)) * 1_000_000).to_integral_value(rounding=ROUND_HALF_UP))


def _raw_table(records):
    records = records.get('costs', records)
    table = {}
    for key, cfg in requirements().items():
        if key not in records:
            raise ValueError('independent oracle lacks required component: ' + key)
        item = records[key]
        if item.get('configuration', cfg) != cfg:
            raise ValueError('independent oracle source configuration differs')
        row = cfg['stages'][0]
        table[row['program'], row['threads']] = item['cost']
    return table


def reference_score(cfg, records, distribution='balanced', ddr=None, *, table=None):
    """Independent integer scalar score, restricted to the fixed experiment."""
    table = table or _raw_table(records)
    core = {number: 0 for number in range(4)}
    host = accelerator = memory = cpu_service = 0
    for row in cfg['stages']:
        cost = table[row['program'], row['threads']]
        service = _reference_ns(cost['service_ms'])
        work = _reference_ns(cost['run_core_ms'])
        boundary = _reference_ns(cost['boundary_core_ms'])
        if ddr is not None:
            if hasattr(ddr, 'stage'):
                item = ddr.stage(row)
            else:
                item = ddr.get('stage_service_ms', ddr)[row['program']]
                if isinstance(item, dict):
                    item = item[str(row['threads'])]
            memory += _reference_ns(item)
        if row['device'] == 'vta':
            accelerator += service
            host += work + boundary
            continue
        cpu_service = max(cpu_service, service)
        n = row['threads']
        if distribution == 'balanced':
            weights = [1] * n
        elif distribution == 'weighted':
            vector = cost.get('worker_run_core_ms')
            if n == 1:
                fractions = [1.]
            elif vector is not None and sum(vector) > 0:
                fractions = [value / sum(vector) for value in vector]
            else:
                fractions = cost['cpu_core_weights']
            weights = [int((Decimal(str(value)) * 1_000_000_000_000).to_integral_value(
                rounding=ROUND_HALF_UP)) for value in fractions]
        else:
            raise ValueError('invalid oracle distribution')
        denominator = sum(weights)
        portions = [(work * weight) // denominator for weight in weights]
        leftovers = work - sum(portions)
        order = sorted(range(n), key=lambda i: (-(work * weights[i] % denominator), i))
        for index in order[:leftovers]:
            portions[index] += 1
        for slot, physical_core in enumerate(row['worker_core_map']):
            core[physical_core] += portions[slot]
        core[row['caller_core']] += boundary
    possible = [cpu_service, accelerator, memory]
    # Enumerate subset tuples instead of copying the scorer's bitmask loop.
    for size in range(1, 5):
        for subset in combinations(range(4), size):
            demand = sum(core[number] for number in subset)
            if size == 4:
                demand += host
            possible.append((demand + size - 1) // size)
    return max(possible)


def _sorting_key(item):
    return item['score_ns'], canonical_tuple(item['configuration'])


def verify(records, ddr=None, programs=None, scored=None):
    """Compare every score and all four complete ranked lists independently."""
    all_configs = list(allocations(True))
    fixed_ids = {cfg['id'] for cfg in allocations(False)}
    table = _raw_table(records)
    compared, independent = 0, {}
    for distribution in ('balanced', 'weighted'):
        costs = Costs(records, programs=programs, ddr=ddr, distribution=distribution)
        values = []
        for cfg in all_configs:
            reference = reference_score(cfg, records, distribution, ddr, table=table)
            actual = (scored[distribution][cfg['id']]['score_ns'] if scored is not None
                      else score_configuration(cfg, costs)['score_ns'])
            if actual != reference:
                raise ValueError('independent per-core score disagrees: ' + cfg['id'])
            values.append({'configuration': cfg, 'score_ns': reference})
            compared += 1
        for movable in (False, True):
            method = distribution + ('_movable' if movable else '_fixed')
            rows = values if movable else [row for row in values if row['configuration']['id'] in fixed_ids]
            independent[method] = [(row['configuration']['id'], row['score_ns']) for row in sorted(rows, key=_sorting_key)]
    return {'passed': True, 'independent_implementation': 'raw_cost_scalar_subset_tuple_oracle',
            'compared_scores': compared, 'domain_counts': {key: len(value) for key, value in independent.items()},
            'ranking_hashes': {key: _hash(value) for key, value in independent.items()},
            'rankings': independent, 'synthetic_only': False}


def select_pool(methods, scores):
    pool, reasons, additions = {}, {}, []

    def add(cfg, reason):
        key = cfg['id']
        if key not in pool:
            pool[key] = cfg
            additions.append(key)
        reasons.setdefault(key, []).append(reason)

    for method in METHODS:
        add(methods[method][0]['configuration'], 'static_first:' + method)
    for name, cfg in intervention_candidates().items():
        add(cfg, name)
    fractions = {}
    for key, balanced in scores['balanced'].items():
        left = scores['weighted'][key]['score_ns']
        right = balanced['score_ns']
        if right <= 0:
            raise ValueError('disagreement selection requires a positive predicted interval')
        fractions[key] = Fraction(left - right, right)
    for sign, label in ((1, 'largest_positive_score_disagreement'), (-1, 'largest_negative_score_disagreement')):
        eligible = [key for key, value in fractions.items() if value * sign > 0 and key not in pool]
        eligible.sort(key=lambda key: (-sign * fractions[key], canonical_tuple(scores['balanced'][key]['configuration'])))
        if eligible:
            add(scores['balanced'][eligible[0]]['configuration'], label)
    if len(pool) > 12:
        raise ValueError('prospective performance pool exceeds twelve configurations')
    return pool, {'order': additions, 'reasons': reasons, 'maximum_pool': 12,
                  'actual_pool': len(pool), 'duplicates_share_measurement': True,
                  'fill_duplicate_slots': False, 'selection_uses_pipeline_measurements': False,
                  'score_disagreement': 'weighted_over_balanced_minus_one_exact_fraction',
                  'top1': {method: methods[method][0]['configuration']['id'] for method in METHODS}}


def rank(records, ddr=None, programs=None):
    all_configs = list(allocations(True))
    fixed_ids = {cfg['id'] for cfg in allocations(False)}
    if len(all_configs) != 1024 or len(fixed_ids) != 225:
        raise ValueError('caller-choice domain coverage differs')
    scored = {}
    methods = {}
    for distribution in ('balanced', 'weighted'):
        costs = Costs(records, programs=programs, ddr=ddr, distribution=distribution)
        scored[distribution] = {cfg['id']: {'configuration': cfg, **score_configuration(cfg, costs)} for cfg in all_configs}
        for movable in (False, True):
            method = distribution + ('_movable' if movable else '_fixed')
            rows = [dict(row) for key, row in scored[distribution].items() if movable or key in fixed_ids]
            methods[method] = [{**row, 'rank': index} for index, row in enumerate(sorted(rows, key=_sorting_key), 1)]
    acceptance = verify(records, ddr, programs, scored)
    for method in METHODS:
        rows = [(row['configuration']['id'], row['score_ns']) for row in methods[method]]
        if rows != acceptance['rankings'][method]:
            raise ValueError('independent full ranking disagrees: ' + method)
    acceptance.pop('rankings')
    pool, selection = select_pool(methods, scored)
    return {'methods': methods, 'method_labels': METHOD_LABELS, 'pool': pool,
            'selection': selection, 'verification': acceptance,
            'ranking_uses_pipeline_feedback': False,
            'fixed_graph': list(baseline_candidate()['stages'][i]['program'] for i in range(5)),
            'worker_domain': 'caller_choice_then_remaining_helpers_ascending',
            'scope': 'fixed_partition_resource_enumeration_not_full_cut_DP',
            'profile_scope': 'same_fresh_isolated_component_costs_for_all_four_methods'}
