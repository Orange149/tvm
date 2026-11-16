"""Independent domain cardinality; separate from the frozen profile controller."""
from itertools import combinations, product
from math import comb
from .domain import topologies


def allocation_counts(unit_count=14, max_islands=3):
    """Independent cardinality proof, without costs or the vectorized search."""
    enhanced = single = prefix = 0
    per_cpu_count = {}
    for islands in range(1, min(max_islands, (unit_count - 1) // 2) + 1):
        cpu_count = islands + 1
        topology_count = comb(unit_count - 1, 2 * islands)
        choices = {}
        for cores, contexts in ((4, 5), (3, 4)):
            masks = [frozenset(m) for width in range(1, cores + 1) for m in combinations(range(cores), width)]
            serial = sum(all(a == b or not a.intersection(b) for a, b in combinations(groups, 2))
                         for groups in product(masks, repeat=cpu_count))
            choices[str(cores)] = {'open': len(masks) ** cpu_count, 'serial_groups': serial,
                                  'contexts': contexts}
            enhanced += topology_count * contexts * (len(masks) ** cpu_count + serial)
            single += topology_count * contexts * cores ** cpu_count
            prefix += topology_count * contexts * cores ** cpu_count
        per_cpu_count[str(cpu_count)] = {'topologies': topology_count, 'allocations_per_context': choices}
    return {'topologies': len(topologies(unit_count, max_islands)), 'contexts': 9,
            'enhanced': enhanced, 'single_core': single, 'prefix': prefix,
            'detail': per_cpu_count, 'costs_or_pipeline_data_used': False}
