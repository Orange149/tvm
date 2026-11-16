"""Scalar accounting independent of both vectorized search and C++ oracle."""
from .common import settings
from cpu_policies.open_resources.costs import quantize_ns
from resource_envelope.model import score as reference_score

def score(configuration,costs,key):
    config=settings(key)
    distribution='session_weights' if config['thread_weights'] else 'balanced'
    if costs.distribution!=distribution:raise ValueError('worker distribution differs from model')
    details=reference_score(configuration,costs)
    original=details['original_components'];demand=original['cpu_core_demand_ns']
    observed=original['host_core_ns'];host=observed if config['host_cpu_demand'] else 0
    cpu_device=[0]*4
    for row in configuration['stages']:
        if row['device']=='cpu':
            if key=='tarnawski' and (row['threads']!=1 or len(row['mask'])!=1):
                raise ValueError('Tarnawski CPU devices must be physical single cores')
            if key=='tarnawski':cpu_device[row['mask'][0]]+=quantize_ns(costs.get(row,configuration)['service_ms'])
    bits=sum(1<<c for c in configuration['host_mask']);capacities={}
    for subset in range(1,16):
        work=sum(v for c,v in enumerate(demand) if subset&(1<<c))
        if subset&bits==bits:work+=host
        width=subset.bit_count();capacities['CPU_subset_'+str(subset)]=(work+width-1)//width
    common=dict(VTA_sum=original['vta_service_sum_ns'],DDR_sum=original['ddr_service_ns'])
    if key=='tarnawski':
        if configuration['execution_mode']!='resource_group':raise ValueError('reused CPU devices require serial execution')
        base=dict(common,**{'CPU_device_'+str(c):v for c,v in enumerate(cpu_device)})
    else:
        base=dict(common,CPU_stage=original['cpu_service_max_ns'],**capacities,
            **{'serial_group_'+g:v for g,v in original['group_services_ns'].items()})
    terms={'capacity':max(base.values())}
    if config['team_guard']:terms['team_service']=details['CPU_team_guard_ns']
    if config['host_guard']:
        terms['host_concentration']=max(v+(host if c in configuration['host_mask'] else 0) for c,v in enumerate(demand))
    result=max(terms.values())
    return dict(score_ns=result,terms_ns=terms,binding_terms=[k for k,v in terms.items() if v==result],
        capacity_terms_ns=base,capacity_binding_terms=[k for k,v in base.items() if v==terms['capacity']],
        observed_host_cpu_ns=observed,scored_host_cpu_ns=host,excluded_host_cpu_ns=observed-host,
        CPU_core_demand_ns=demand,CPU_device_service_ns=cpu_device,
        VTA_service_sum_ns=original['vta_service_sum_ns'],DDR_service_ns=original['ddr_service_ns'],
        planning_proxy_not_a_latency_bound=True,component_costs_modified=False,pipeline_FPS_used=False)
