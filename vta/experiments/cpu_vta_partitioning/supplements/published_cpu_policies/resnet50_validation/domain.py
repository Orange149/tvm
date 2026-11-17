"""Pure legal-interval DAG and independent cardinality; no TVM or cost inputs."""
from __future__ import annotations

from functools import lru_cache
from itertools import combinations, product


def _units(value):
    rows = value['units'] if isinstance(value, dict) else value
    if not isinstance(rows, (list, tuple)) or len(rows) < 3:
        raise ValueError('unit metadata is required')
    if [row['index'] for row in rows] != list(range(len(rows))):
        raise ValueError('units must be consecutively indexed')
    starts = tuple(row['index'] for row in rows if row.get('kind') == 'main_preadd')
    if not starts or starts[0] == 0 or starts[-1] == len(rows) - 1:
        raise ValueError('main-path VTA starts must leave CPU endpoints')
    return len(rows), starts


def legal_edges(units):
    n, starts = _units(units)
    return tuple((device, begin, end) for device in ('cpu', 'vta')
                 for begin in range(n) for end in range(begin + 1, n + 1)
                 if device == 'cpu' or begin in starts)


def _path_table(units, max_islands):
    n, starts = _units(units)
    if type(max_islands) is not int or max_islands < 1:
        raise ValueError('invalid maximum VTA islands')
    @lru_cache(None)
    def count(position, device, used):
        if position == n:
            return int(device == 'vta' and 1 <= used <= max_islands)
        if device == 'vta' and position not in starts:
            return 0
        if used + (device == 'vta') > max_islands:
            return 0
        return sum(count(end, 'vta' if device == 'cpu' else 'cpu', used + (device == 'vta'))
                   for end in range(position + 1, n + 1))
    return n, starts, count


def programs(units, max_islands=3):
    n, starts, suffixes = _path_table(units, max_islands)
    reached, edges = set(), set()
    def visit(position, device, used):
        key = position, device, used
        if key in reached or position == n:
            return
        reached.add(key)
        if device == 'vta' and position not in starts:
            return
        next_used = used + (device == 'vta')
        if next_used > max_islands:
            return
        other = 'vta' if device == 'cpu' else 'cpu'
        for end in range(position + 1, n + 1):
            if suffixes(end, other, next_used):
                edges.add((device, position, end))
                visit(end, other, next_used)
    visit(0, 'cpu', 0)
    return [{'name': f'{device}_{begin:02}_{end - 1:02}', 'device': device,
             'begin': begin, 'end': end, 'kind': 'interval',
             'eligible_search': True, 'purpose': 'search_stage'}
            for device, begin, end in sorted(edges)]


def topology_counts(units, max_islands=3):
    n, starts = _units(units)
    def exact(required):
        @lru_cache(None)
        def count(position, device, used):
            if position == n:
                return int(device == 'vta' and used == required)
            if device == 'vta' and position not in starts:
                return 0
            next_used = used + (device == 'vta')
            if next_used > required:
                return 0
            return sum(count(end, 'vta' if device == 'cpu' else 'cpu', next_used)
                       for end in range(position + 1, n + 1))
        return count(0, 'cpu', 0)
    return {str(i): exact(i) for i in range(1, max_islands + 1)}


@lru_cache(None)
def _allocation_coefficients(cpu_stages):
    complete = single = 0
    for cores, contexts in ((4, 5), (3, 4)):
        masks = [frozenset(group) for width in range(1, cores + 1)
                 for group in combinations(range(cores), width)]
        serial = sum(all(a == b or not a.intersection(b)
                         for a, b in combinations(groups, 2))
                     for groups in product(masks, repeat=cpu_stages))
        complete += contexts * (len(masks) ** cpu_stages + serial)
        single += contexts * cores ** cpu_stages
    return complete, single


def summary(units, max_islands=3):
    n, starts = _units(units)
    rows = programs(units, max_islands)
    cpu = sum(row['device'] == 'cpu' for row in rows)
    vta = len(rows) - cpu
    topologies = topology_counts(units, max_islands)
    complete = single = 0
    for islands, count in topologies.items():
        factor, one = _allocation_coefficients(int(islands) + 1)
        complete += factor * count
        single += one * count
    return {'unit_count': n, 'max_vta_islands': max_islands,
            'VTA_entry_units': list(starts), 'topologies_per_island_count': topologies,
            'topology_count': sum(topologies.values()), 'search_CPU_stages': cpu,
            'VTA_stages': vta, 'program_count': len(rows),
            'CPU_signature_count': cpu * 4, 'VTA_signature_count': vta * 5,
            'total_basic_signature_count': cpu * 4 + vta * 5,
            'DMA_signature_count': vta, 'complete_resource_configurations': complete,
            'tarnawski_resource_configurations': single,
            'costs_or_pipeline_FPS_used': False,
            'domain_limit': 'CPU endpoints, VTA entry only at residual main_preadd, at most three VTA islands; no atomic-only additions'}
