"""Profile-based behavior grouping, independent of all measured pipeline FPS.

No host/core/category quotas. Rank representatives by the static score and
keep one per behavior family. This compresses the bounded terminal pool, not
the complete DP domain; physical mirrors and light-stage alternatives may be
lost, and no equivalence or globally optimal family-retention is claimed.
"""
from itertools import permutations
from cpu_policies.open_resources.domain import CORES,context,canonical_tuple
from cpu_policies.open_resources.costs import Costs,quantize_ns

PERMUTATIONS=tuple(permutations(CORES))

def profile_scales(programs,costs):
    from cpu_policies.open_resources.domain import stage
    ctx=context('shared4');result={}
    for name,p in programs.items():
        if p['device']!='cpu':continue
        values=[costs.get(stage('cpu',p['begin'],p['end'],n,CORES[:n]),ctx) for n in range(1,5)]
        result[name]=(max(quantize_ns(v['service_ms']) for v in values),
            max(quantize_ns(v['run_core_ms'])+quantize_ns(v['boundary_core_ms']) for v in values))
    return result

def behavior_key(configuration,scales,*,preserve_worker_roles=False):
    cpu=[s for s in configuration['stages'] if s['device']=='cpu']
    max_wall=max(scales[s['program']][0] for s in cpu)
    max_work=max(scales[s['program']][1] for s in cpu)
    # Hardware-derived 1/P contribution threshold; fixed before evaluation.
    # Both service and on-core work matter. Tiny intervals are not declared
    # equivalent; their alternatives are explicitly compressed for the budget.
    critical=[s for s in cpu if scales[s['program']][0]*len(CORES)>=max_wall
                                or scales[s['program']][1]*len(CORES)>=max_work]
    value=[]
    for perm in PERMUTATIONS:
        remap=lambda mask:tuple(sorted(perm[c] for c in mask))
        stages=tuple((s['begin'],s['end'],s['threads'],remap(s['mask']),perm[s['caller_core']],
                      tuple(perm[c] for c in s['worker_core_map']) if preserve_worker_roles else ()) for s in critical)
        value.append((stages,remap(configuration['compute_pool']),remap(configuration['host_mask']),
            remap(configuration['control_mask']),remap(configuration['reserved_cores']),
            configuration['execution_mode'],sum(s['device']=='vta' for s in configuration['stages'])))
    return min(value)

def select(records,expand,scales,count=20,*,preserve_worker_roles=False):
    if count!=20:raise ValueError('this protocol freezes twenty candidates')
    chosen=[];families=set();skipped=0;examined=0
    for record in sorted(records,key=lambda r:(r['score_ns'],r['canonical'])):
        cfg=expand(record);key=behavior_key(cfg,scales,preserve_worker_roles=preserve_worker_roles);examined+=1
        if key in families:skipped+=1;continue
        families.add(key);cfg.update(selection_order=len(chosen)+1,
            selection_reason='best_static_representative_of_new_profile_behavior_family')
        chosen.append(cfg)
        if len(chosen)==count:break
    if len(chosen)!=count:raise ValueError('not enough distinct behavior families in retained terminal pool')
    return chosen,{'family_count':len(chosen),'representatives_examined':examined,'repeated_family_candidates_skipped':skipped,
        'criticality_rule':'max isolated wall or CPU work across measured 1..4 threads reaches 1/physical_core_count of this topology maximum',
        'physical_permutation':'caller role and allowed/reserved/control relations preserved; physical indices canonicalized',
        'logical_worker_roles_preserved':preserve_worker_roles,
        'compression_losses':['physical core mirrors','resource alternatives of stages below the contribution threshold',
            'families not retained by the original Top-2 topology/resource collectors'],
        'full_DP_domain_changed':False,'globally_optimal_family_Top20_claimed':False,'pipeline_FPS_used':False}
