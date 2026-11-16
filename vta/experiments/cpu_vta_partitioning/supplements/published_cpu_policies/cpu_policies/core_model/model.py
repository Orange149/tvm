"""Same resource formula, with uniform or isolated logical-worker CPU demand."""
from copy import deepcopy
from math import isfinite

from ..open_resources.costs import (quantize_ns, allocate_core_ns, ceil_div_ns,
                                    _validate_cost, NS_PER_MS)
from .domain import CORES, component, component_key, stage, validate_candidate


class Costs:
    def __init__(self, records, programs=None, ddr=None, distribution='balanced'):
        if distribution not in ('balanced', 'weighted'):
            raise ValueError('unknown core-demand distribution')
        self.records = records.get('costs', records)
        self.programs, self.ddr, self.distribution = programs, ddr, distribution

    def get(self, row, context_value=None):
        canonical = stage(row['device'], row['begin'], row['end'], row['threads'],
                          CORES[:row['threads']] if row['device'] == 'cpu' else CORES)
        source_cfg = component(canonical)
        key = component_key(source_cfg)
        if key not in self.records:
            raise ValueError('missing canonical measured component: ' + key)
        record = self.records[key]
        if 'configuration' in record and record['configuration'] != source_cfg:
            raise ValueError('component configuration does not match source key')
        value = deepcopy(_validate_cost(record['cost']))
        weights = []
        if row['device'] == 'cpu':
            n = row['threads']
            observed = value.get('worker_run_core_ms')
            if observed is not None:
                if (not isinstance(observed, (list, tuple)) or len(observed) != n
                        or any(type(v) not in (int, float) or not isfinite(v) or v < 0 for v in observed)
                        or abs(sum(observed) - value['run_core_ms']) > 1e-8 * max(1., value['run_core_ms'])):
                    raise ValueError('worker run increments do not conserve common CPU core work')
            measured = value.get('cpu_core_weights')
            if measured is not None and (
                    not isinstance(measured, (list, tuple)) or len(measured) != n
                    or any(type(v) not in (int, float) or not isfinite(v) or v < 0 for v in measured)
                    or abs(sum(measured) - 1.) > 1e-8):
                raise ValueError('invalid isolated worker distribution')
            if observed is not None and sum(observed) > 0:
                normalized = [v / sum(observed) for v in observed]
                if measured is not None and any(abs(a - b) > 1e-8 for a, b in zip(measured, normalized)):
                    raise ValueError('declared worker shares disagree with absolute run increments')
                measured = normalized
            if n == 1:
                measured = [1.]
            if measured is None:
                if self.distribution == 'weighted':
                    raise ValueError('weighted scoring requires isolated per-worker increments')
            elif (not isinstance(measured, (list, tuple)) or len(measured) != n
                    or any(type(v) not in (int, float) or not isfinite(v) or v < 0 for v in measured)
                    or abs(sum(measured) - 1.) > 1e-8):
                raise ValueError('invalid isolated worker distribution')
            weights = list(measured) if self.distribution == 'weighted' else [1. / n] * n
        assumptions = []
        if row['device'] == 'cpu' and (row['mask'] != canonical['mask']
                                       or row['worker_core_map'] != canonical['worker_core_map']):
            assumptions.append('homogeneous_core_permutation_of_measured_logical_worker_costs')
        value.update(source_key=key, requested_key=component_key(component(row)),
                     cpu_core_weights=weights, assumptions=assumptions)
        return value


def ddr_stage_service(ddr, row):
    if ddr is None:
        return 0.
    if hasattr(ddr, 'stage'):
        value = ddr.stage(row)
    else:
        item = ddr.get('stage_service_ms', ddr)[row['program']]
        value = item[str(row['threads'])] if isinstance(item, dict) else item
    if type(value) not in (int, float) or not isfinite(value) or value < 0:
        raise ValueError('invalid DDR service')
    return float(value)


def score_configuration(configuration, costs, ddr=None):
    if not isinstance(costs, Costs):
        costs = Costs(costs)
    validate_candidate(configuration, costs.programs)
    ddr = costs.ddr if ddr is None else ddr
    demand = [0] * 4
    cpu_max = vta = cpu_work = host_work = boundary_work = ddr_service = 0
    sources = []
    for row in configuration['stages']:
        value = costs.get(row)
        service, run, boundary = (quantize_ns(value[k]) for k in ('service_ms', 'run_core_ms', 'boundary_core_ms'))
        boundary_work += boundary
        ddr_service += quantize_ns(ddr_stage_service(ddr, row))
        sources.append({'program': row['program'], 'source_key': value['source_key'],
                        'assumptions': value['assumptions']})
        if row['device'] == 'cpu':
            cpu_max = max(cpu_max, service)
            cpu_work += run
            portions = allocate_core_ns(run, value['cpu_core_weights'], threads=row['threads'])
            for core, portion in zip(row['worker_core_map'], portions):
                demand[core] += portion
            demand[row['caller_core']] += boundary
        else:
            vta += service
            host_work += run + boundary
    capacities = []
    for bits in range(1, 16):
        fixed = sum(demand[c] for c in CORES if bits & (1 << c))
        capacities.append(ceil_div_ns(fixed + (host_work if bits == 15 else 0), bits.bit_count()))
    score = max(cpu_max, vta, ddr_service, *capacities)
    return {'score_ns': score, 'score_ms': score / NS_PER_MS,
            'cpu_core_demand_ns': demand, 'cpu_core_demand_ms': [x / NS_PER_MS for x in demand],
            'cpu_service_max_ns': cpu_max, 'cpu_service_max_ms': cpu_max / NS_PER_MS,
            'vta_service_sum_ns': vta, 'vta_service_sum_ms': vta / NS_PER_MS,
            'cpu_compute_core_ns': cpu_work, 'cpu_compute_core_ms': cpu_work / NS_PER_MS,
            'host_core_ns': host_work, 'host_core_ms': host_work / NS_PER_MS,
            'boundary_core_ns': boundary_work, 'boundary_core_ms': boundary_work / NS_PER_MS,
            'subset_capacity_ns': capacities, 'subset_capacity_ms': [x / NS_PER_MS for x in capacities],
            'ddr_service_ns': ddr_service, 'ddr_service_ms': ddr_service / NS_PER_MS,
            'ddr_available': ddr is not None, 'cpu_distribution': costs.distribution,
            'component_sources': sources, 'control_cost_model_status': 'not_separately_profiled',
            'host_migration_model': 'ideal_movable_work_within_shared_four_cores',
            'model_scope': 'same_frozen_capacity_formula_changed_CPU_distribution_and_mapping_only',
            'accounting': {'unit': 'integer_nanosecond', 'CPU_total_conserved': True,
                           'allocation': 'largest_remainder_ties_logical_slot',
                           'boundary': 'caller_core_only', 'VTA': 'single_physical_device_sum'}}
