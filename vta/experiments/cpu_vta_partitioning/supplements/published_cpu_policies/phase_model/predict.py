"""Freeze a bounded temporal hypothesis using isolation inputs exclusively."""
from copy import deepcopy
from pathlib import Path
import time

from cpu_vta_partitioning.common import read,sha256,write
from cpu_policies.open_resources.costs import Costs,quantize_ns
from phase_profile.prepare import configurations
from policy_branch_eval.common import isolated_inputs_only
from policy_branch_eval.ranking import inputs
from staged_ablation.model import score
from workload_model.common import freeze
from .simulator import Replay


def build_stages(network, configuration, profiles, costs):
    stages=[]; assumptions=[]
    for row in configuration['stages']:
        cost=costs.get(row, configuration)
        if row['device']=='cpu':
            found=[v for v in profiles.values() if v['identity']['network']==network
                   and v['identity']['program']==row['program']
                   and v['identity']['threads']==row['threads']]
            if len(found)!=1:
                raise ValueError('missing/ambiguous real isolated phase profile')
            profile=found[0]
            if not profile['no_pipeline_performance']:
                raise ValueError('non-isolation input is forbidden')
            # These new component clocks are not scaled to old measurements.
            cost=profile['clean_cost']
            value=dict(device='cpu',mask=row['mask'],worker_core_map=row['worker_core_map'],
                       profile=deepcopy(profile['average']))
            if profile['identity']['mask']!=row['mask']:
                assumptions.append(dict(program=row['program'],
                    assumption='homogeneous_core_permutation_of_isolated_phase_work',
                    measured=profile['identity']['mask'],requested=row['mask']))
        else:
            value=dict(device='vta',mask=row['mask'])
            assumptions.append(dict(program=row['program'],
                assumption='existing_isolated_VTA_cost_retains_original_boot_and_executor',
                source_key=cost['source_key']))
        for op in ('set','get'):
            cpu=quantize_ns(cost[op+'_core_ms']);wall=quantize_ns(cost[op+'_ms'])
            value[op+'_cpu_ns']=cpu
            value[op+'_nonrunning_ns']=max(0,wall-cpu)
            if cpu>wall:
                assumptions.append(dict(program=row['program'],operation=op,
                    assumption='process_CPU_marginally_exceeds_wall_offCPU_interval_clipped',
                    difference_ns=cpu-wall))
        if row['device']=='vta':
            value['run_cpu_ns']=quantize_ns(cost['run_core_ms'])
            value['run_nonrunning_ns']=max(0,quantize_ns(cost['run_ms'])-value['run_cpu_ns'])
        stages.append(value)
    return stages,assumptions


def predict(output):
    output=Path(output).resolve();target=output/'temporal_development'
    started=time.monotonic()
    with isolated_inputs_only():
        projected=read(output/'model_inputs/ISOLATED_PHASES.json')
        if not projected['no_pipeline_FPS']:
            raise ValueError('phase inputs contain pipeline performance')
        result={};source_hashes={str(output/'model_inputs/ISOLATED_PHASES.json'):
                               sha256(output/'model_inputs/ISOLATED_PHASES.json')}
        for network in ('resnet18','yolov3_tiny416'):
            programs,records,ddr,hashes,_,_=inputs(network)
            costs=Costs(records,programs=programs,ddr=ddr,distribution='session_weights')
            result[network]={}
            for case,configuration in configurations(network)['pipelines'].items():
                stages,assumptions=build_stages(network,configuration,projected['profiles'],costs)
                base=score(configuration,costs,'fine_host')
                simulations={}
                for runnable in (True,False):
                    for order in ('before','after'):
                        name=('runnable' if runnable else 'blocked')+'_host_'+order
                        replay=Replay(stages,runnable_completion_wait=runnable,host_order=order).run()
                        simulations[name]={**replay,'combined_score_ns':max(base['score_ns'],replay['interval_ns'])}
                result[network][case]=dict(configuration=configuration,stages=stages,
                    original_score=base,simulations=simulations,assumptions=assumptions,
                    component_hashes=hashes)
        hypothesis=dict(
            name='bounded_fork_join_shared_core_replay',status='DEVELOPMENT_HYPOTHESIS',
            source='https://www.cs.toronto.edu/~pekhimenko/Papers/Daydream-ATC_20.pdf',
            literature_scope='dependency/task replay representation only; CPU scheduler is our hypothesis',
            primary='runnable_host_before',
            ablation='blocked_host_before; changes predicted completion waiting only, not actual execution',
            host_sensitivity='compare before/after placement of unlocalized host CPU work; no selection by FPS',
            evaluation='compare four preregistered predictions with new same-boot clean pipeline sessions; '
                       'report all, reject unsupported factors, do not choose a favorable host order after measurement',
            prior_known_configurations=True,not_unseen_validation=True,
            primary_score='max(original model, replay interval)',
            unknown=['Linux time slices, wakeup/dispatch latency and yield fairness',
                     'noncaller runtime overhead outside callbacks',
                     'VTA poll runnable timing and actual host migration',
                     'producer, collector and queue CPU work',
                     'cache/DDR contention effects on compute demand'],
            excluded_wait='isolated caller completion tail is not added as fixed compute work',
            scheduler='equal sharing of runnable fixed-core tasks; migratable host assigned least-loaded '
                      'allowed core at event boundaries; no Linux scheduler reconstruction',
            old_or_new_pipeline_FPS_read=False,full_domain_model=False,
            no_claim_of_strict_bound=True)
        sources={str(Path(__file__)):sha256(Path(__file__)),
                 str(Path(__file__).with_name('simulator.py')):sha256(Path(__file__).with_name('simulator.py'))}
        freeze(target/'HYPOTHESIS.json',hypothesis)
        freeze(target/'PREDICTIONS.json',dict(networks=result,input_hashes=source_hashes,model_sources=sources,
                                            pipeline_performance_read=False))
        seal=dict(files={str(p.relative_to(output)):sha256(p) for p in
                         (target/'HYPOTHESIS.json',target/'PREDICTIONS.json',output/'model_inputs/ISOLATED_PHASES.json')},
                  scope='four known development configurations; no full-domain claim',
                  primary='runnable_host_before',pipeline_performance_read=False,
                  source_hashes=sources)
        freeze(output/'PIPELINE_PREDICTIONS_FREEZE.json',seal)
    write(target/'LOCAL_COST.json',dict(seconds=time.monotonic()-started),replace=True)
    return {network:{case:dict(original_ms=v['original_score']['score_ns']/1e6,
            predicted_ms={k:r['combined_score_ns']/1e6 for k,r in v['simulations'].items()})
            for case,v in cases.items()} for network,cases in result.items()}
