"""Premeasurement interval domain; no costs, performance results, or TVM imports."""
from itertools import combinations


def topologies(unit_count=14, max_islands=3):
    if type(unit_count) is not int or unit_count < 3 or type(max_islands) is not int or max_islands < 1:
        raise ValueError('invalid interval domain bounds')
    result = []
    for islands in range(1, min(max_islands, (unit_count - 1) // 2) + 1):
        for transitions in combinations(range(1, unit_count), 2 * islands):
            boundaries = (0, *transitions, unit_count)
            result.append(tuple(('cpu' if index % 2 == 0 else 'vta', begin, end)
                                for index, (begin, end) in enumerate(zip(boundaries, boundaries[1:]))))
    return tuple(result)


def programs(unit_count=14, max_islands=3, *, include_atomic_cpu=True):
    reachable = {edge for topology in topologies(unit_count, max_islands) for edge in topology}
    edges = set(reachable)
    if include_atomic_cpu:
        edges.update(('cpu', i, i + 1) for i in range(unit_count))
    return [{'name': f'{device}_{begin:02}_{end - 1:02}', 'device': device,
             'begin': begin, 'end': end, 'kind': 'interval',
             'eligible_search': (device, begin, end) in reachable,
             'purpose': 'search_stage' if (device, begin, end) in reachable else 'atomic_thread_curve_only'}
            for device, begin, end in sorted(edges)]


def summary(unit_count=14, max_islands=3):
    rows = programs(unit_count, max_islands)
    cpu = [r for r in rows if r['device'] == 'cpu']
    vta = [r for r in rows if r['device'] == 'vta']
    return {'unit_count': unit_count, 'max_vta_islands': max_islands,
            'topology_count': len(topologies(unit_count, max_islands)),
            'search_CPU_stages': sum(r['eligible_search'] for r in cpu),
            'CPU_atomic_only_extras': sum(not r['eligible_search'] for r in cpu),
            'VTA_stages': len(vta), 'CPU_signature_count': len(cpu) * 4,
            'VTA_signature_count': len(vta) * 5,
            'total_basic_signature_count': len(cpu) * 4 + len(vta) * 5,
            'domain_limit': 'fixed 14 units, first and final units remain CPU; no primitive cuts or alternative fusion search',
            'pipeline_costs_or_FPS_used': False}
