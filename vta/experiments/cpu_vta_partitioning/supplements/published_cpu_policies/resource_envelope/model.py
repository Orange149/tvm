"""Static resource envelopes with distinct work and thread-team budgets.

The host concentration guard robustifies unknown OS placement over an allowed
set. CPU team occupancy adapts the fixed-team reservation contract of Pipe-it
and Graphi; on overlapping open groups it is a conservative planning proxy,
NOT a proven upper/lower bound on actual Linux scheduling or response time.
"""
from cpu_policies.open_resources.costs import Costs,score_configuration,quantize_ns

VARIANTS=('original','host_concentration','team_occupancy','combined')

def score(configuration,costs,ddr=None):
    if not isinstance(costs,Costs):costs=Costs(costs)
    original=score_configuration(configuration,costs,ddr)
    demand=original['cpu_core_demand_ns'];host=original['host_core_ns']
    allowed=set(configuration['host_mask'])
    # Unknown host fractions p lie in a simplex supported on the allowed set.
    # For every core require II >= D[c] + H*p[c] for every such p.
    host_guard=max(demand[c]+(host if c in allowed else 0) for c in range(4))
    reservation=[0]*4
    for row in configuration['stages']:
        if row['device']!='cpu':continue
        service=quantize_ns(costs.get(row,configuration)['service_ms'])
        for core in row['mask']:reservation[core]+=service
    team_guard=max(reservation)
    base=original['score_ns']
    return {'original':base,'host_concentration':max(base,host_guard),
        'team_occupancy':max(base,team_guard),'combined':max(base,host_guard,team_guard),
        'cpu_team_reservation_ns':reservation,'host_concentration_guard_ns':host_guard,
        'CPU_team_guard_ns':team_guard,'original_components':original,
        'planning_proxy_not_a_latency_bound':True,'pipeline_FPS_used':False}
