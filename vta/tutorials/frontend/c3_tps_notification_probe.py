"""Local TPS notification-carrier proposal and exact dependency comparison.

This is a bounded diagnostic, not a production optimizer. No board transport.
UOP payload LOADs stay in their original order and are never recombined here.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import re

from c3_five_tradeoff_commands import completion_dag


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def material(records):
    return [r for r in records if not r['description'].startswith('NOP-')
            and r['description'] not in ('LOAD UOP', 'FINISH')]


def relations(records):
    ancestors = completion_dag(records)
    payload = material(records)
    return {(i, j) for i, left in enumerate(payload) for j, right in enumerate(payload)
            if ancestors[right['index']] >> left['index'] & 1}


def token_counts(records):
    counts = Counter()
    for i, row in enumerate(records):
        require(row['index'] == i and row['stage'] in (1, 2, 3), 'INVALID_INSTRUCTION')
        s = row['stage']
        for k, (bit, lane) in enumerate(zip(row['bits'],
                ((s-1, s), (s+1, s), (s, s-1), (s, s+1)))):
            require(bit in (0, 1), 'INVALID_BIT')
            if bit:
                require(lane in ((1, 2), (2, 1), (2, 3), (3, 2)), 'INVALID_LANE')
                counts[(('pop' if k < 2 else 'push'), lane)] += 1
    return counts


def finite_batch_progress(records, command_depths, token_depths):
    """Acyclic full batch fits all queues; fetch and token writes cannot stall.

    With the entire batch fetchable, the earliest unfinished node in any
    topological order has its FIFO and token predecessors completed. Induct
    over these nodes. No cycle latency or overlap utilization is asserted.
    """
    count = token_counts(records)
    stages = Counter(r['stage'] for r in records)
    for s in (1, 2, 3):
        require(stages[s] <= command_depths[s], 'FULL_BATCH_FIFO_BOUND_UNPROVED')
    for lane in ((1, 2), (2, 1), (2, 3), (3, 2)):
        require(count['push', lane] == count['pop', lane], 'TOKEN_COUNT_MISMATCH')
        require(count['push', lane] <= token_depths[lane], 'TOKEN_WRITE_BOUND_UNPROVED')
    completion_dag(records)  # Reject circular waits, including backward edges.
    return {'status': 'FINITE_FULL_BATCH_PROGRESS_PROVED',
            'method': 'all_commands_fit_and_total_token_writes_fit_and_acyclic',
            'stage_counts': dict(stages), 'command_depths': command_depths,
            'token_pairs': sum(n for (kind, _), n in count.items() if kind == 'push'),
            'finite_instruction_completion_assumed': True, 'cycle_timing_proved': False}


def propose(records):
    """Two fixed rules, evaluated before execution; no token deletion."""
    result = deepcopy(records)
    removed, edits = set(), []
    for row in result:
        i = row['index']
        if row['description'] != 'NOP-COMPUTE-STAGE':
            continue
        tail = [r for r in result[i+1:] if r['stage'] == 2]
        if (row['bits'] == (1, 0, 0, 0) and len(tail) >= 2
                and tail[0]['description'] == 'LOAD UOP'
                and tail[0]['bits'] == (1, 0, 0, 0)
                and tail[1]['description'] == 'GEMM'
                and 'reset_out: 0' in tail[1]['detail']
                and tail[1]['bits'] in ((0, 0, 0, 0), (0, 0, 1, 0))):
            target = tail[1]
            target['bits'] = (1, *target['bits'][1:])
            rule = 'two_ready_pops_on_uop_load_and_gemm'
        elif (row['bits'] == (0, 0, 1, 0) and tail
              and tail[0]['description'] == 'LOAD ACC' and tail[0]['bits'] == (0, 0, 0, 0)):
            target = tail[0]
            target['bits'] = (0, 0, 1, 0)
            rule = 'second_release_on_next_acc_load'
        else:
            continue
        removed.add(i)
        edits.append({'removed_original_instruction': i,
                      'target_original_instruction': target['index'], 'rule': rule})
    result = [r for r in result if r['index'] not in removed]
    for i, row in enumerate(result):
        row['index'] = i
    return result, edits


def canonical_payload(row):
    """Keep every encoded payload field, omit printed token running totals."""
    detail = re.sub(r'^\s*(?:dep -|l2g_queue|s2g_queue).*$', '', row['detail'], flags=re.M)
    return row['description'], ' '.join(detail.split())


def certify_rewrite(before, after, command_depths, token_depths):
    expected, edits = propose(before)
    require(len(expected) == len(after), 'REWRITE_INSTRUCTION_CENSUS')
    for left, right in zip(expected, after):
        require((left['stage'], left['bits'], canonical_payload(left)) ==
                (right['stage'], right['bits'], canonical_payload(right)), 'UNEXPLAINED_ENCODING_CHANGE')
    require(token_counts(before) == token_counts(after), 'TOKEN_IDENTITY_OR_MULTIPLICITY_CHANGED')
    # All non-NOP payloads, including UOP LOADs, stay byte-field equivalent and ordered.
    payload = lambda rs: [canonical_payload(r) for r in rs if not r['description'].startswith('NOP-')]
    require(payload(before) == payload(after), 'PAYLOAD_OR_UOP_LOAD_ORDER_CHANGED')
    old, new = relations(before), relations(after)
    require(old == new, 'MATERIAL_REACHABILITY_CHANGED')
    return {'status': 'CARRIER_REWRITE_EXACTLY_CERTIFIED', 'edits': edits,
            'before_instructions': len(before), 'after_instructions': len(after),
            'material_reachability_exactly_equal': True,
            'all_non_nop_payload_and_order_equal': True,
            'token_multiplicity_equal': True,
            'progress': finite_batch_progress(after, command_depths, token_depths)}


def organization(records):
    count = Counter(r['description'] for r in records)
    return {'instructions': len(records), 'compute_nops': count['NOP-COMPUTE-STAGE'],
            'uop_loads': count['LOAD UOP'],
            'mac_gemm': sum(r['description'] == 'GEMM' and 'reset_out: 0' in r['detail'] for r in records),
            'reset': sum(r['description'] == 'GEMM' and 'reset_out: 1' in r['detail'] for r in records)}
