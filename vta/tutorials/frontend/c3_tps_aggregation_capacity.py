"""Read-only organization report over an already certified complete native call.

This is a diagnostic projection, not a new optimization or certification basis.
Kernel allocations and merged device LOAD instructions are counted separately.
"""
from collections import Counter
import hashlib
import re

from c3_thread_sync_audit import instruction_dump
from c3_five_tradeoff_commands import decode_submissions


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _allocation_projection(native):
    """Frozen queue's allocation/eviction projection, checked against references.

    No timing, greedy LOAD placement or new lifetime proof is inferred here.
    Warm origin and successful-submit reset are certified caller contracts.
    """
    definitions = native['cache_definitions']['definitions']
    warm_origin = native['cache_definitions']['origin_dump_sha256'] != native['dump_sha256']
    seen, events = set(), []
    reasons = Counter({'first_load': 0, 'capacity_eviction_reload': 0, 'submission_reset_reload': 0})
    total_wraps, total_serialized, alias_hits, cached_hits = 0, 0, 0, 0
    for submit in native['submissions']:
        ordinal = submit['ordinal']
        objects = {key: {'identity': key, 'values': values, 'begin': 0, 'end': 0}
                   for key, values in definitions.items()}
        cache, dram = [], []
        cache_idx, sram_end, wraps = 0, 0, 0
        invalidated = set()
        refs = [row for row in native['uop_values']['checks'] if row['submission'] == ordinal]
        for ref in refs:
            identity = ref['kernel_identity']
            obj = objects[identity]
            event = {'submission': ordinal, 'compute_index': ref['compute_index'],
                     'kernel_identity': identity, 'base_uops': len(obj['values'])}
            if obj['begin'] != obj['end']:
                event['reason'] = 'already_resident_cache_hit'
                cached_hits += 1
            else:
                previous = cache[cache_idx - 1] if cache_idx else None
                if previous and previous['begin'] != previous['end'] and previous['values'] == obj['values']:
                    obj['begin'], obj['end'] = previous['begin'], previous['end']
                    cache.insert(cache_idx, obj)
                    cache_idx += 1
                    alias_hits += 1
                    event['reason'] = 'identical_content_residency_alias'
                else:
                    reason = ('capacity_eviction_reload' if identity in invalidated else
                              'submission_reset_reload' if warm_origin or identity in seen else
                              'first_load')
                    reasons[reason] += 1
                    if not any(item is obj for item in dram):
                        dram.append(obj)
                    size = len(obj['values'])
                    begin = 0
                    wrapped = sram_end + size > 8192
                    if wrapped:
                        cache_idx, sram_end = 0, size
                        wraps += 1
                    else:
                        begin = sram_end
                        sram_end += size
                    evict_begin, evicted = cache_idx, []
                    while cache_idx < len(cache):
                        old = cache[cache_idx]
                        if old['begin'] >= sram_end:
                            break
                        if old['begin'] != old['end']:
                            evicted.append(old['identity'])
                            invalidated.add(old['identity'])
                        old['begin'] = old['end'] = 0
                        cache_idx += 1
                    obj['begin'], obj['end'] = begin, sram_end
                    cache.insert(cache_idx, obj)
                    del cache[evict_begin:cache_idx]
                    cache_idx = evict_begin + 1
                    invalidated.discard(identity)
                    event.update(reason=reason, wrapped=wrapped, evicted_identities=evicted)
            seen.add(identity)
            event['uop_begin'], event['uop_end'] = obj['begin'], obj['end']
            _require([obj['begin'], obj['end']] == [ref['uop_begin'], ref['uop_end']],
                     'DIAGNOSTIC_ALLOCATION_REFERENCE_MISMATCH')
            events.append(event)
        serialized = 4 * sum(len(obj['values']) for obj in dram)
        _require(wraps == submit['uop_sram_wraps'] and serialized == submit['serialized_uop_bytes'],
                 'DIAGNOSTIC_ALLOCATION_STATISTICS_MISMATCH')
        total_wraps += wraps
        total_serialized += serialized
    _require(total_wraps == native['uop_values']['uop_sram_wraps'] and
             total_serialized == native['uop_values']['serialized_uop_bytes'],
             'DIAGNOSTIC_TOTAL_STATISTICS_MISMATCH')
    return {'status': 'ALLOCATION_PROJECTION_MATCHES_CERTIFIED_REFERENCES_AND_STATISTICS',
            'allocation_events': sum(reasons.values()), 'reason_counts': dict(reasons),
            'already_resident_cache_hits': cached_hits, 'identical_content_residency_aliases': alias_hits,
            'warm_host_definitions': warm_origin, 'events': events,
            'reason_basis': 'frozen source-certified queue/Sync lifetime; not a hardware timing observation',
            'allocation_event_is_not_a_device_LOAD_instruction': True}


def organization(dump, native):
    """Return exact native counts plus conservative checked allocation reasons."""
    _require(native['status'] == 'ACTUAL_BOUNDED_NATIVE_FIFO_TOKEN_UOP_WITH_NOP_BARRIERS_CERTIFIED',
             'ORGANIZATION_REQUIRES_EXISTING_NATIVE_CERTIFICATE')
    _require(hashlib.sha256(dump.encode()).hexdigest() == native['dump_sha256'],
             'ORGANIZATION_DUMP_CERTIFICATE_MISMATCH')
    parsed = instruction_dump(dump)
    records = decode_submissions(dump)
    _require(parsed['submission_count'] == native['submission_count'] == len(native['submissions']),
             'ORGANIZATION_SUBMISSION_CENSUS')
    totals = Counter()
    for report, block, certified in zip(parsed['submissions'], records, native['submissions']):
        _require(report['instructions'] == certified['instruction_count'], 'ORGANIZATION_INSN_CENSUS')
        semantic = report['semantic']
        totals.update({'MAC_GEMM': semantic.get('MAC_GEMM', 0),
                       'RESET_GEMM': semantic.get('RESET_GEMM', 0), 'ALU': semantic.get('ALU', 0),
                       'compute_NOP': semantic.get('NOP-COMPUTE-STAGE', 0),
                       'all_instructions': report['instructions'], 'UOP_LOAD': semantic.get('LOAD_UOP', 0),
                       'UOP_LOAD_bytes': report['load_bytes'].get('UOP', 0)})
        _require(semantic.get('LOAD_UOP', 0) == certified['uop_loads'], 'ORGANIZATION_UOP_LOAD_CENSUS')
        actual_refs = [row for row in block if row['description'] == 'GEMM' or row['description'].startswith('ALU -')]
        certified_refs = [row for row in native['uop_values']['checks'] if row['submission'] == certified['ordinal']]
        _require(len(actual_refs) == len(certified_refs), 'ORGANIZATION_COMPUTE_CENSUS')
        for actual, reference in zip(actual_refs, certified_refs):
            extent = re.search(r'range \((\d+), (\d+)\)', actual['detail'])
            _require(extent and list(map(int, extent.groups())) ==
                     [reference['uop_begin'], reference['uop_end']], 'ORGANIZATION_ACTUAL_REFERENCE_RANGE')
    values = native['uop_values']
    definitions = native['cache_definitions']['definitions']
    _require(totals['UOP_LOAD'] == values['uop_loads'], 'ORGANIZATION_TOTAL_UOP_LOADS')
    return {'schema': 'c3_tps_aggregation_organization_v1', 'status': 'CERTIFIED_CALL_DIAGNOSTIC_PROJECTION',
            **dict(totals), 'single_kernel_max_base_uops': max(map(len, definitions.values())),
            'host_identity_base_uops_total': sum(map(len, definitions.values())),
            'host_content_unique_base_uops_total': sum(len(value) for value in
                                                       {tuple(tuple(word) for word in v)
                                                        for v in definitions.values()}),
            'serialized_uop_bytes': values['serialized_uop_bytes'],
            'UOP_SRAM_wraps': values['uop_sram_wraps'], 'submissions': native['submission_count'],
            'allocation_projection': _allocation_projection(native),
            'not_new_certification_basis': True, 'performance_attribution': 'not inferred',
            'unknowns': ['cycle cost of notification, LOAD splitting, cache packing and overlap are not separated']}
