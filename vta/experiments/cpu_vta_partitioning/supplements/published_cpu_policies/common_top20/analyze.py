"""Every ranked item must have new common evidence before budget reporting."""
from pathlib import Path
import statistics

from cpu_vta_partitioning.common import read,sha256,verify_inventory
from cpu_policies.open_resources.costs import Costs
from cpu_policies.open_resources.domain import validate_candidate
from policy_branch_eval.analysis import csv_write
from policy_branch_eval.ranking import inputs
from policy_branch_eval.selection import fixed_family
from staged_ablation.model import score
from workload_model.common import freeze
from .runtime import task,summary


def load_result(run,bundle,cfg,phase,boot):
    doc=task(bundle,cfg,phase);path=Path(run)/'accepted'/boot/(doc['task_id']+'.json')
    if not path.exists():raise ValueError('required formal item has not been measured: '+cfg['id']+' '+phase)
    item=read(path)
    if item['task']!=doc or item['boot_id']!=boot:raise ValueError('comparison receipt identity differs')
    verify_inventory(item['raw'],item['files'])
    if summary(item['raw'],doc,bundle)!=item['summary']:raise ValueError('comparison receipt does not replay')
    return item,dict(path=str(path),sha256=sha256(path))


def analyze(output,run):
    output,run=Path(output).resolve(),Path(run).resolve();state=read(run/'state.json')
    if state['status'] not in ('SHORT_COMPLETE','LONG_COMPLETE'):
        raise ValueError('complete the whole frozen short pool before reporting method budgets')
    boot=state['boot_id'];networks={};all_rows=[];evidence={};winners={};seconds=0;total=0
    for network in ('resnet18','yolov3_tiny416'):
        bundle=output/network/'capsule';rank=read(bundle/'plans/rankings.json')
        programs,records,ddr,hashes,_,units=inputs(network)
        costs=Costs(records,programs=programs,ddr=ddr,distribution='session_weights')
        balanced=Costs(records,programs=programs,ddr=ddr,distribution='balanced')
        if hashes!=rank['component_hashes']:raise ValueError('analysis component identity differs')
        measured={}
        for identity,cfg in rank['pool'].items():
            validate_candidate(cfg,programs,unit_count=units)
            qual,qsource=load_result(run,bundle,cfg,'qualify',boot)
            short,source=load_result(run,bundle,cfg,'short',boot)
            if short['summary']['output_fingerprints']!=qual['summary']['output_fingerprints']:
                raise ValueError('short result differs from full-output qualification')
            measured[identity]=short['summary'];evidence[network+'/'+identity]=dict(qualify=qsource,short=source)
            seconds+=qual.get('measurement_and_collection_seconds',0)+short.get('measurement_and_collection_seconds',0)
            row=dict(network=network,configuration_id=identity,context=cfg['context_id'],CPU_gate=cfg['execution_mode'],
                stages=[dict(program=s['program'],threads=s['threads'],mask=s['mask']) for s in cfg['stages']],
                host_mask=cfg['host_mask'],control_mask=cfg['control_mask'],
                FPS=short['summary']['FPS'],II_ms=short['summary']['II_ms'],
                process_CPU_ms_per_completion=short['summary']['process_CPU_ms_per_completion'],
                pipeline_stage_service_ms=short['summary']['stage_service_ms'],
                isolated_stage_service_ms=[costs.get(s,cfg)['service_ms'] for s in cfg['stages']],
                fixed_group_family=fixed_family(cfg),window_stage_counts=short['summary']['stage_completion_counts_at_window'])
            for method,rows in rank['lists'].items():
                positions=[i for i,r in enumerate(rows,1) if r['id']==identity]
                row[method+'_rank']=positions[0] if positions else None
            for label,key in [('current','fine_host'),('semantics_revision','weights_only')]:
                prediction=score(cfg,costs,key);row[label+'_predicted_ms']=prediction['score_ns']/1e6
                row[label+'_absolute_percentage_error']=100*abs(row[label+'_predicted_ms']/row['II_ms']-1)
                row[label+'_binding_terms']=prediction['binding_terms']
            applicable=cfg['execution_mode']=='resource_group' and all(s['threads']==1 for s in cfg['stages'] if s['device']=='cpu')
            row['tarnawski_predicted_ms']=score(cfg,balanced,'tarnawski')['score_ns']/1e6 if applicable else None
            all_rows.append(row)
        reports={};winner_ids=[]
        for method,rows in rank['lists'].items():
            if len(rows)!=20 or any(r['id'] not in measured for r in rows):raise ValueError('twenty complete ranked results are required')
            budgets={}
            for budget in (1,5,20):
                chosen=max(rows[:budget],key=lambda r:measured[r['id']]['FPS']);item=measured[chosen['id']]
                budgets[str(budget)]=dict(FPS=item['FPS'],II_ms=item['II_ms'],configuration_id=chosen['id'],
                    rank=next(i for i,r in enumerate(rows,1) if r['id']==chosen['id']),all_prefix_measured=True)
            reports[method]=dict(budgets=budgets,selected_configurations=[r['id'] for r in rows])
            winner_ids.append(budgets['20']['configuration_id'])
        winners[network]=dict(configurations=sorted(set(winner_ids)),method_winners={k:v['budgets']['20']['configuration_id'] for k,v in reports.items()})
        pool_rows=[r for r in all_rows if r['network']==network]
        networks[network]=dict(methods=reports,pool_count=len(measured),
            best_in_union=max(measured,key=lambda i:measured[i]['FPS']),
            union_MAPE={k:statistics.mean(r[k+'_absolute_percentage_error'] for r in pool_rows)
                        for k in ('current','semantics_revision')},
            component_hashes=hashes,single_boot=True)
        total+=len(measured)
    result=dict(networks=networks,rows=all_rows,source_receipts=evidence,boot_id=boot,
        new_common_short_configurations=total,new_short_native_sessions=2*total,new_short_DNN_inferences=29*total,
        measurement_and_collection_seconds=seconds,old_pipeline_FPS_reused=False,
        pipeline_FPS_used_for_static_rank=False,new_component_profiles=0,
        full_global_lists_verified=True,statistical_significance_claimed=False,goal_complete=False)
    freeze(output/'SHORT_RESULTS.json',result)
    freeze(output/'LONG_SELECTION_FREEZE.json',dict(networks=winners,boot_id=boot,
        short_results_sha256=sha256(output/'SHORT_RESULTS.json'),selection='each method budget20 short-window best, deduplicate',
        no_reselection_after_long_review=True,window=dict(warmup=8,scored=64,tail=32)))
    csv_write(output/'common_short_candidates.csv',all_rows)
    if state['status']=='LONG_COMPLETE':
        long_rows=[];long_sources={}
        for network,selection in winners.items():
            bundle=output/network/'capsule';pool=read(bundle/'plans/rankings.json')['pool']
            for identity in selection['configurations']:
                item,source=load_result(run,bundle,pool[identity],'long',boot)
                short=next(r for r in all_rows if r['network']==network and r['configuration_id']==identity)
                long_rows.append(dict(network=network,configuration_id=identity,short_FPS=short['FPS'],long_FPS=item['summary']['FPS'],
                    short_II_ms=short['II_ms'],long_II_ms=item['summary']['II_ms'],
                    stage_service_ms=item['summary']['stage_service_ms'],
                    window_stage_counts=item['summary']['stage_completion_counts_at_window']))
                long_sources[network+'/'+identity]=source
        freeze(output/'LONG_RESULTS.json',dict(rows=long_rows,sources=long_sources,boot_id=boot,
            frozen_selection_sha256=sha256(output/'LONG_SELECTION_FREEZE.json'),DNN_inferences=104*len(long_rows),
            native_sessions=len(long_rows),same_boot_review_not_independent_start=True))
        csv_write(output/'common_long_selected.csv',long_rows)
    return dict(pool_count=total,networks=networks,long_selection_count=sum(len(v['configurations']) for v in winners.values()),
                full_timing_model_validated=False,goal_complete=False)
