"""Exclude host CPU demand without altering any recorded component costs."""
from .common import settings
from resource_envelope.model import score as reference_score

def score(configuration, costs, key):
    config = settings(key)
    if costs.distribution != 'session_weights':
        raise ValueError('host ablation requires the frozen isolated thread shares')
    details = reference_score(configuration, costs)
    original = details['original_components']
    observed_host = original['host_core_ns']
    effective_host = observed_host if config['host_cpu_demand'] else 0
    demand = original['cpu_core_demand_ns']
    host_bits = sum(1 << c for c in configuration['host_mask'])
    capacities = {}
    for subset in range(1, 16):
        work = sum(v for c, v in enumerate(demand) if subset & (1 << c))
        if subset & host_bits == host_bits:
            work += effective_host
        width = subset.bit_count()
        capacities['CPU_subset_' + str(subset)] = (work + width - 1) // width
    base_terms = dict(CPU_stage=original['cpu_service_max_ns'],
        VTA_sum=original['vta_service_sum_ns'], DDR_sum=original['ddr_service_ns'],
        **capacities, **{'serial_group_'+g: v for g,v in original['group_services_ns'].items()})
    capacity = max(base_terms.values())
    terms = dict(capacity=capacity, team_service=details['CPU_team_guard_ns'])
    if config['host_guard']:
        terms['host_concentration'] = max(v + (effective_host if c in configuration['host_mask'] else 0)
                                          for c,v in enumerate(demand))
    value = max(terms.values())
    return dict(score_ns=value, terms_ns=terms,
        binding_terms=[t for t,v in terms.items() if v==value],
        capacity_terms_ns=base_terms,
        capacity_binding_terms=[t for t,v in base_terms.items() if v==capacity],
        observed_host_cpu_ns=observed_host, scored_host_cpu_ns=effective_host,
        excluded_host_cpu_ns=observed_host-effective_host,
        CPU_core_demand_ns=demand, VTA_service_sum_ns=original['vta_service_sum_ns'],
        DDR_service_ns=original['ddr_service_ns'],
        planning_proxy_not_a_latency_bound=True, component_costs_modified=False,
        pipeline_FPS_used=False)
