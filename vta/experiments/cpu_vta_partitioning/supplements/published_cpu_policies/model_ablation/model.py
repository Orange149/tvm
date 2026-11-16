"""Exactly the existing formulas, with independently declared switches."""
from .common import settings
from resource_envelope.model import score as original_score

def score(configuration, costs, key):
    config = settings(key)
    expected = 'session_weights' if config['thread_weights'] else 'balanced'
    if costs.distribution != expected:
        raise ValueError('CPU distribution does not match ablation switch')
    details = original_score(configuration, costs)
    terms = {'capacity': details['original']}
    if config['host_guard']:
        terms['host_concentration'] = details['host_concentration_guard_ns']
    if config['team_guard']:
        terms['team_service'] = details['CPU_team_guard_ns']
    value = max(terms.values())
    return dict(score_ns=value, terms_ns=terms,
        binding_terms=[term for term, demand in terms.items() if demand == value],
        planning_proxy_not_a_latency_bound=True, pipeline_FPS_used=False)

def capacity_binding(configuration,costs):
    """Post-ranking explanation, kept outside the frozen score document."""
    details=original_score(configuration,costs);original=details['original_components']
    components={'CPU_stage':original['cpu_service_max_ns'],
        'VTA_sum':original['vta_service_sum_ns'],'DDR_sum':original['ddr_service_ns'],
        **{'CPU_subset_'+str(i+1):v for i,v in enumerate(original['subset_capacity_ns'])},
        **{'serial_group_'+k:v for k,v in original['group_services_ns'].items()}}
    return [term for term,demand in components.items() if demand==details['original']]
