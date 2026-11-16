"""Read pipeline evidence only after the isolation-based prediction seal."""
import json
from pathlib import Path
import statistics

from cpu_vta_partitioning.common import read,sha256,verify_inventory,write
from phase_profile.campaign import specs
from phase_profile.runtime import make_task,summary
from policy_branch_eval.analysis import csv_write
from workload_model.common import freeze
from .extract import project_call


def mean(values):
    return statistics.mean(values)


def measured_call_metrics(projected,launches,callbacks):
    kernels=projected['kernels']
    return dict(known_CPU_ms=projected['known_CPU_ns']/1e6,
        serial_CPU_ms=sum(projected['serial_cpu_segments_ns'])/1e6,
        callback_CPU_ms=sum(w['callback_cpu_ns'] for k in kernels for w in k['workers'])/1e6,
        completion_tail_CPU_ms=sum(k['caller_completion_wait_and_return_cpu_ns'] for k in kernels)/1e6,
        run_wall_ms=projected['run_wall_ns']/1e6,
        callback_cover_ms=sum((max(w['wall_end_ns'] for w in callbacks[k['launch']])-
                              min(w['wall_start_ns'] for w in callbacks[k['launch']]))/1e6
                             for k in launches),
        worker_start_spread_ms=sum((max(w['wall_start_ns'] for w in callbacks[k['launch']])-
                                   min(w['wall_start_ns'] for w in callbacks[k['launch']]))/1e6
                                  for k in launches))


def analyze(output,run_dir):
    output,run_dir=Path(output).resolve(),Path(run_dir).resolve()
    seal=read(output/'PIPELINE_PREDICTIONS_FREEZE.json')
    for leaf,expected in seal['files'].items():
        if sha256(output/leaf)!=expected:
            raise ValueError('prediction seal changed')
    for path,expected in seal['source_hashes'].items():
        if sha256(path)!=expected:
            raise ValueError('frozen predictor implementation changed')
    state=read(run_dir/'state.json')
    if state['status']!='PIPELINES_COMPLETE':
        raise ValueError('all preregistered pipelines and bridges must finish')
    frozen=read(output/'temporal_development/PREDICTIONS.json')
    isolated=read(output/'model_inputs/ISOLATED_PHASES.json')['profiles']
    boot=state['boot_id'];evidence={};records={};rows=[];stage_rows=[]
    for spec in specs(output,'pipelines'):
        bundle=output/spec['network']/'capsule';doc=make_task(bundle,spec)
        receipt=run_dir/'accepted'/boot/(doc['task_id']+'.json');item=read(receipt)
        if item['boot_id']!=boot or item['task']!=doc:
            raise ValueError('pipeline identity differs')
        verify_inventory(item['raw'],item['files'])
        if summary(item['raw'],doc,bundle)!=item['summary']:
            raise ValueError('pipeline evidence no longer replays')
        evidence[str(receipt)]=sha256(receipt)
        records[spec['network'],spec['case'],spec['phase'],spec['runner']]=item
    for network,cases in frozen['networks'].items():
        for case,prediction in cases.items():
            clean=records[network,case,'clean','phase'];observed=records[network,case,'observed','phase']
            measured=clean['summary']['ii_ms']
            row=dict(network=network,case=case,configuration_id=prediction['configuration']['id'],
                original_predicted_ms=prediction['original_score']['score_ns']/1e6,
                measured_II_ms=measured,FPS=clean['summary']['fps'],
                observed_II_ms=observed['summary']['ii_ms'],
                observer_interval_change_percent=100*(observed['summary']['ii_ms']/measured-1))
            for name,value in prediction['simulations'].items():
                row[name+'_predicted_ms']=value['combined_score_ns']/1e6
                row[name+'_absolute_percentage_error']=100*abs(value['combined_score_ns']/1e6/measured-1)
            row['original_absolute_percentage_error']=100*abs(row['original_predicted_ms']/measured-1)
            old=records.get((network,case,'clean','old'))
            if old:
                row.update(original_executor_FPS=old['summary']['fps'],
                    original_executor_II_ms=old['summary']['ii_ms'],
                    new_executor_II_change_percent=100*(measured/old['summary']['ii_ms']-1))
            rows.append(row)
            raw=Path(observed['raw'])
            phase=[json.loads(s) for s in (raw/'phase_trace.jsonl').read_text().splitlines()]
            trace=[json.loads(s) for s in (raw/'wait_trace.jsonl').read_text().splitlines()]
            launches=[r for r in trace if r['kind']==0];callbacks={};barriers={}
            for value in trace:
                if value['kind'] in (1,2):
                    (callbacks if value['kind']==1 else barriers).setdefault(value['launch'],[]).append(value)
            for i,stage in enumerate(observed['task']['stages']):
                if stage['device']!='cpu':
                    continue
                profile=next(v for v in isolated.values() if v['identity']['network']==network
                    and v['identity']['program']==stage['program'] and v['identity']['threads']==stage['threads'])
                metrics=[]
                for f in range(observed['task']['warmup'],observed['task']['warmup']+observed['task']['scored']):
                    span=next(r for r in phase if r['stage']==stage['program'] and r['call']==f and r['operation']=='run')
                    selected=[r for r in launches if r['caller']==span['tid']
                        and span['wall_start_ns']<=r['wall_start_ns'] and r['wall_end_ns']<=span['wall_end_ns']]
                    projected=project_call(span,launches,callbacks,barriers)
                    metrics.append(measured_call_metrics(projected,selected,callbacks))
                value=profile['average'];ns=lambda v:v['numerator']/v['denominator']/1e6
                stage_rows.append(dict(network=network,case=case,program=stage['program'],threads=stage['threads'],
                    mask=stage['mask'],isolated_clean_service_ms=profile['clean_cost']['service_ms'],
                    pipeline_clean_service_ms=clean['summary']['stage_service_ms'][i],
                    pipeline_observed_service_ms=observed['summary']['stage_service_ms'][i],
                    isolated_observed_profile_service_ms=prediction['stages'][i]['profile']['run_wall']['rounded_up_ns']/1e6
                        +profile['clean_cost']['set_ms']+profile['clean_cost']['get_ms'],
                    primary_replayed_stage_service_ms=prediction['simulations']['runnable_host_before']['stage_service_ns'][i]/1e6,
                    isolated_known_CPU_ms=ns(value['known_CPU']),
                    isolated_serial_CPU_ms=sum(ns(v) for v in value['serial_cpu_segments']),
                    isolated_callback_CPU_ms=sum(ns(s) for k in value['kernels'] for w in k['workers'] for s in w['compute_segments']),
                    isolated_completion_tail_CPU_ms=sum(ns(k['isolated_completion_wait_and_return_cpu']) for k in value['kernels']),
                    **{'pipeline_observed_'+name:mean([m[name] for m in metrics]) for name in metrics[0]}))
    findings={}
    for network in frozen['networks']:
        pool=[r for r in rows if r['network']==network]
        original=min(pool,key=lambda r:(r['original_predicted_ms'],r['configuration_id']))
        proposed=min(pool,key=lambda r:(r['runnable_host_before_predicted_ms'],r['configuration_id']))
        best=min(pool,key=lambda r:r['measured_II_ms'])
        findings[network]=dict(original_choice=original['case'],proposed_choice=proposed['case'],
            measured_pool_best=best['case'],original_selected_FPS=original['FPS'],
            proposed_selected_FPS=proposed['FPS'],measured_best_FPS=best['FPS'],
            selected_FPS_change_percent=100*(proposed['FPS']/original['FPS']-1),
            original_mean_absolute_percentage_error=mean([r['original_absolute_percentage_error'] for r in pool]),
            replay_mean_absolute_percentage_error=mean([r['runnable_host_before_absolute_percentage_error'] for r in pool]),
            host_order_changes_choice=min(pool,key=lambda r:(r['runnable_host_after_predicted_ms'],r['configuration_id']))['case']!=proposed['case'])
    decision=dict(status='DEVELOPMENT_ONLY_NOT_ADOPTED',
        explanation='no full-domain promotion from four known cases; scheduler/host ordering and '
                    'executor bridge uncertainty must be reported; selection benefit is assessed separately from error',
        findings=findings,global_profile_expansion_authorized_by_this_result=False,
        existing_FPScosts_used_for_prediction=False,old_profiles_not_rescaled=True,
        not_unseen_validation=True,goal_complete=False)
    target=output/'temporal_development'
    freeze(target/'RESULTS.json',dict(rows=rows,CPU_stages=stage_rows,findings=findings,
        provenance=evidence,prediction_seal_sha256=sha256(output/'PIPELINE_PREDICTIONS_FREEZE.json'),
        boot_id=boot,component_calls=230,pipeline_DNN_inferences=state['DNN_inferences'],
        native_sessions=32+state['accepted_sessions'],single_boot=True,
        full_domain_validation_done=False))
    freeze(target/'DECISION.json',decision)
    csv_write(target/'pipeline_comparison.csv',rows)
    csv_write(target/'CPU_stages.csv',stage_rows)
    return decision
