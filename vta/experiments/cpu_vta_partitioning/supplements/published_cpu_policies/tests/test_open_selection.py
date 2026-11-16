from copy import deepcopy
import random

from cpu_policies.open_resources.domain import candidate, contexts, nonempty_masks, stage
from cpu_policies.open_resources.selection import select, strategies, STRATEGIES


def fixture():
    pool = []
    for ctx in contexts():
        for a in nonempty_masks(ctx['compute_pool']):
            for b in nonempty_masks(ctx['compute_pool']):
                for gate in ('off', 'resource_group'):
                    if gate == 'resource_group' and set(a) & set(b) and a != b:
                        continue
                    row = candidate([stage('cpu', 0, 3, len(a), a),
                                     stage('vta', 3, 18, 1, ctx['host_mask']),
                                     stage('cpu', 18, 21, len(b), b)], ctx, gate)
                    row.update(score_ns=105_000_000, score_ms=105.)
                    pool.append(row)
    # Simulate a strict Top-20 crowded with overlapping shared-core variants.
    pure = [r for r in pool if r['context_id'] == 'shared4' and r['execution_mode'] == 'off'
            and 'overlap_open' in strategies(r)][:20]
    for i, row in enumerate(pure):
        row.update(score_ns=100_000_000 + i, score_ms=100 + i / 1e6, static_rank=i + 1)
    return pure, pool


def test_light_thread_variants_cannot_crowd_out_resource_strategies():
    pure, pool = fixture()
    selected, policy = select(pure, pool)
    assert [r['id'] for r in selected[:10]] == [r['id'] for r in pure[:10]]
    assert len(selected) == len({r['id'] for r in selected}) == 20
    assert [c['strategy'] for c in policy['resource_strategy_coverage']] == list(STRATEGIES)
    by_id = {r['id']: r for r in selected}
    for c in policy['resource_strategy_coverage']:
        assert c['selected'] is not None
        assert c['strategy'] in strategies(by_id[c['selected']])


def test_guard_prevents_forced_out_of_range_family_and_selection_ignores_fps():
    pure, pool = fixture()
    for row in pool:
        if row['reserved_cores']:
            row.update(score_ns=120_000_000, score_ms=120.)
    a, policy = select(pure, pool)
    missing = next(c for c in policy['resource_strategy_coverage'] if c['strategy'] == 'reserved_host')
    assert missing['selected'] is None
    assert not any(r['reserved_cores'] for r in a)
    shuffled = deepcopy(pool)
    random.Random(20261107).shuffle(shuffled)
    for i, row in enumerate(shuffled):
        row['measured_fps'] = 1e9 - i
    b, _ = select(pure, shuffled)
    assert [r['id'] for r in a] == [r['id'] for r in b]
    assert policy['uses_pipeline_fps'] is False


def test_family_selected_in_another_slot_is_reported_as_covered():
    pure, pool = fixture()
    reserved = next(r for r in pool if r['reserved_cores']
                    and 'fixed_groups_serial' in strategies(r))
    reserved.update(score_ns=100_100_000, score_ms=100.1)
    pool = [r for r in pool if not r['reserved_cores']] + [reserved]
    selected, policy = select(pure, pool)
    entry = next(c for c in policy['resource_strategy_coverage']
                 if c['strategy'] == 'reserved_host')
    assert reserved['id'] in {r['id'] for r in selected}
    assert entry['selected'] is None
    assert entry['covered'] is True
    assert entry['covered_by'] == [reserved['id']]
    assert entry['reason'] == 'already_covered_by_another_selection_slot'
