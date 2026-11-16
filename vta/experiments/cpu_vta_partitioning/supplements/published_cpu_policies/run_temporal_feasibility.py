#!/usr/bin/env python3
"""Read-only full-domain cost planning around static scores, with no board/FPS."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))

if __name__=='__main__':
    import json
    import time
    import numpy as np
    from cpu_vta_partitioning.common import read,sha256,write
    from cpu_policies.open_resources.costs import Costs
    from cpu_policies.open_resources import domain
    from policy_branch_eval.ranking import inputs,engine_from
    from policy_branch_eval.common import isolated_inputs_only
    from staged_ablation.model import score
    from workload_model.common import ROOT,OUTPUT,freeze,require_space
    require_space();target=OUTPUT/'search_feasibility';target.mkdir(exist_ok=True)
    source=ROOT/'artifacts/staged_ablation_01/weights_only/proof.json'
    proof=read(source);results={};ratios=[(1,1),(23,20),(13,10),(3,2)]
    with isolated_inputs_only():
        for network in ('resnet18','yolov3_tiny416'):
            destination=target/(network+'.json')
            if destination.exists():
                result=read(destination)
                if result['engine_implementation']!=proof['implementation']:raise ValueError('feasibility engine changed')
                results[network]=result;continue
            programs,records,ddr,hashes,prior,units=inputs(network)
            engine=engine_from(proof,network);costs=Costs(records,programs=programs,ddr=ddr,distribution='session_weights')
            graphs=engine._topologies(programs,units,3)
            reference=read(prior/'rankings.json')['methods']['enhanced']['rankings'][0]['score_ns']
            limits=[(reference*a+b-1)//b for a,b in ratios]
            summaries=[];started=time.monotonic();global_total=0;global_min=None
            for context in domain.contexts():
                table=engine._prepare(programs,costs,context,ddr)
                counts=[0]*len(limits);cpu=[set() for _ in limits];vta=[set() for _ in limits]
                total=0;minimum=None
                for graph_index,graph in enumerate(graphs):
                    names=tuple(name for name in graph if table[name]['cpu'])
                    allocation=engine._allocation_domain(tuple(context['compute_pool']),len(names))
                    values=engine._join_values(names,graph,table,context,allocation)
                    for gate,key in [('off','enhanced'),('resource_group','serial')]:
                        eligible=np.ones(len(allocation['allocations']),dtype=bool) if gate=='off' else allocation['compatible']
                        total+=int(eligible.sum())
                        local_min=int(values[key][eligible].min());minimum=local_min if minimum is None else min(minimum,local_min)
                        for j,threshold in enumerate(limits):
                            chosen=np.flatnonzero(eligible&(values[key]<=threshold));counts[j]+=len(chosen)
                            if not len(chosen):continue
                            for name in graph:
                                if not table[name]['cpu']:vta[j].add(name)
                            for position,name in enumerate(names):
                                for index in np.unique(allocation['allocations'][chosen,position]):
                                    mask=allocation['options'][int(index)]
                                    cpu[j].add((name,len(mask)))
                    if graph_index in (0,len(graphs)//2,len(graphs)-1):
                        contexts={c['id']:c for c in domain.contexts()}
                        for index in (0,len(allocation['allocations'])//2,len(allocation['allocations'])-1):
                            terminal=engine.Terminal(int(values['enhanced'][index]),graph_index,
                                tuple(map(int,allocation['allocations'][index])),context['id'],'off')
                            row=engine._configuration(terminal,graphs,programs,contexts)
                            if score(row,costs,'weights_only')['score_ns']!=terminal.score:
                                raise ValueError('capacity-only scalar/complete-domain vector mismatch')
                summaries.append(dict(context=context['id'],full_domain_configurations=total,minimum_ns=minimum,
                    thresholds=[dict(limit_ns=t,configurations=n,CPU_signatures=sorted(cpu[j]),
                                     VTA_programs=sorted(vta[j])) for j,(t,n) in enumerate(zip(limits,counts))]))
                global_total+=total;global_min=minimum if global_min is None else min(global_min,minimum)
                print('全域可行性计数',network,context['id'],total,counts,flush=True)
            expected=966240306 if units==21 else 472266548
            if global_total!=expected:raise ValueError('full-domain coverage differs')
            result=dict(network=network,contexts=summaries,full_domain_configurations=global_total,
                global_capacity_only_minimum_ns=global_min,current_main_static_reference_ns=reference,
                threshold_ratios=[f'{a}/{b}' for a,b in ratios],
                planning_counts_only=True,search_domain_filtered=False,pipeline_FPS_read=False,board_access=False,
                engine_implementation=proof['implementation'],component_hashes=hashes,
                scalar_probes_passed=True,seconds=time.monotonic()-started)
            freeze(destination,result);results[network]=result
    freeze(target/'FEASIBILITY_FREEZE.json',dict(source_sha256=sha256(Path(__file__)),
        files={n:sha256(target/(n+'.json')) for n in results},no_profile_expansion_started=True,
        no_new_rankings_or_board_measurement=True))
    print(json.dumps({n:dict(configurations=r['full_domain_configurations'],seconds=r['seconds']) for n,r in results.items()}))
