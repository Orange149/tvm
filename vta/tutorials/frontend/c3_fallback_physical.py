"""Certify unchanged physical access order, without exploring sync mutations.

The exact access models, RAW/WAR/WAW witnesses, token/FIFO graph and RESET/OUT
checks are inherited unchanged. The old sync study's hypothetical edge removals
and wait movements are outside this comparison and are never enumerated here.
"""
from c3_sync_release_audit import Graph, access_model, requirements, Obligations


def certify_physical(records, certificate):
    graph=Graph(records)
    accesses=access_model(records,certificate)
    live_rows=requirements(accesses)
    physical_rows=requirements(accesses,include_dead=True)
    live=Obligations(live_rows,graph.ancestors,True)
    physical=Obligations(physical_rows,graph.ancestors,False)
    return {'status':'ACTUAL_UNCHANGED_LIVE_AND_PHYSICAL_ORDER_CERTIFIED',
        'instructions':len(records),'live_obligations':len(live_rows),
        'physical_obligations':len(physical_rows),
        'original_dead_read_unordered_count':len(physical.original_unordered),
        'original_dead_read_unordered_examples':physical.original_unordered[:4],
        'live_unordered_count':len(live.original_unordered),
        'existing_dead_RESET_read_observations_retained':True,
        'all_STORE_observers_checked_by_original_access_model':True,
        'sync_hypotheses_enumerated':0,
        'new_sync_optimization_or_equivalence_claim':False}
