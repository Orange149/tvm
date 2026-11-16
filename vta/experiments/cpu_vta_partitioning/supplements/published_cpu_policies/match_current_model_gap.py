"""Match existing device models to the exact traced layout, not a new model.

Only VTA isolated inputs change relative to the old finite replay. CPU work,
mapping, queues, scheduling approximations and waiting remain unchanged.
Predictions are frozen before diagnostic traces are read by the analyzer.
"""
from copy import deepcopy
from pathlib import Path
import json
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,write,sha256
from finite_service.common import freeze
from device_completion_replay import CompletionFinite
from policy_branch_eval.common import isolated_inputs_only

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/current_model_gap_01'


def project(case,donor):
    if case['configuration']['execution_mode']!='off':raise ValueError('diagnostic layout requires gate off')
    result=deepcopy(case)
    programs={s['program']:p for s,p in zip(donor['configuration']['stages'],donor['stages']) if s['device']=='vta'}
    for cfg,stage in zip(result['configuration']['stages'],result['stages']):
        if cfg['device']!='vta':continue
        source=programs[cfg['program']]
        if cfg['mask']!=[0,1,2,3] or stage['mask']!=source['mask']:
            raise ValueError('host placement profile differs')
        stage.clear();stage.update(deepcopy(source))
    if result['configuration']!=case['configuration']:
        raise ValueError('execution layout changed during projection')
    if [s for s in result['stages'] if s['device']=='cpu']!=[s for s in case['stages'] if s['device']=='cpu']:
        raise ValueError('CPU input changed during VTA cost refresh')
    return result


def run():
    dev=ROOT/'artifacts/finite_service_01/development'
    current=ROOT/'artifacts/device_completion_replay_01'
    old=read(dev/'INPUTS.json');latest=read(current/'INPUTS.json')
    if sha256(dev/'INPUTS.json')!=read(dev/'PLAN.json')['inputs_sha256']:
        raise ValueError('old isolated inputs changed')
    if sha256(current/'INPUTS.json')!=read(current/'PLAN.json')['inputs_sha256']:
        raise ValueError('new isolated inputs changed')
    profiles_path=ROOT/'artifacts/completion_profile_01/analysis/RESULTS.json'
    if sha256(profiles_path)!=read(current/'PLAN.json')['profile_sha256']:
        raise ValueError('isolated completion profile provenance differs')
    profiles=read(profiles_path)['profiles']
    native_programs=read(ROOT/'artifacts/yolo_scheduler_trace_repair_01/capsule/plans/programs.json')
    for name in ('vta_03_06','vta_09_12'):
        if profiles['yolov3_tiny416/'+name]['module_sha256']!=native_programs[name]['files']['module.so']['sha256']:
            raise ValueError('VTA program changed relative to native trace')
    if old['quanta_ns']!=latest['quanta_ns']:raise ValueError('CPU scheduler calibration changed')
    if latest['contains_pipeline_performance']:raise ValueError('donor includes pipeline measurements')
    for path,h in read(current/'PLAN.json')['sources'].items():
        if sha256(path)!=h:raise ValueError('existing device model source changed')
    base=old['isolated']['cases']['yolov3_tiny416/overlap']
    donor=next(v for k,v in latest['cases'].items() if k.startswith('yolov3_tiny416/'))
    case=project(base,donor)
    modes=['aggregate','independent']
    freeze(OUT/'INPUTS.json',dict(case=case,window=old['windows']['yolov3_tiny416/overlap'],
        quanta_ns=old['quanta_ns'],modes=modes,point='midpoint',placement='wake_boundary',
        native_trace_or_performance_included=False,
        CPU_source=str(dev/'INPUTS.json'),CPU_source_sha256=sha256(dev/'INPUTS.json'),
        VTA_source=str(current/'INPUTS.json'),VTA_source_sha256=sha256(current/'INPUTS.json'),
        profile_boot=latest['profile_boot'],cross_boot_projection_is_development_only=True))
    freeze(OUT/'PLAN.json',dict(input_sha256=sha256(OUT/'INPUTS.json'),
        original_model_sources=read(current/'PLAN.json')['sources'],source_sha256=sha256(__file__),
        new_model=False,new_parameters=False,configuration_changed=False,new_DNN=0,
        measurement_comparison_not_used_to_select_variant=True,
        objective='locate stage/worker errors of both existing models on the exact already-traced layout'))
    files={};started=time.monotonic()
    for mode in modes:
        path=OUT/(mode+'.json')
        if not path.exists():
            with isolated_inputs_only():
                result=CompletionFinite(case['stages'],quanta=old['quanta_ns'],placement='wake_boundary',
                    completion_model=mode,completion_point='midpoint',runnable_completion_wait=True,controls=None,
                    gate=case['configuration']['execution_mode'],host_order='before',
                    **old['windows']['yolov3_tiny416/overlap']).run()
            freeze(path,dict(mode=mode,result=result,input_sha256=sha256(OUT/'INPUTS.json'),
                no_pipeline_performance_read=True))
        if read(path)['input_sha256']!=sha256(OUT/'INPUTS.json'):raise ValueError('cached replay input differs')
        files[path.name]=sha256(path)
    freeze(OUT/'REPLAY_FREEZE.json',dict(files=files,plan_sha256=sha256(OUT/'PLAN.json'),
        no_native_performance_read=True))
    write(OUT/'COST.json',dict(replay_seconds=time.monotonic()-started),replace=True)

    # No diagnostic native input is read until both predictions are sealed.
    trace_path=ROOT/'artifacts/yolo_scheduler_trace_repair_01/analysis/RESULTS.json'
    trace=read(trace_path)
    if sha256(trace_path)!=read(trace_path.parent.parent/'DELIVERY.json')['results_sha256']:
        raise ValueError('accepted native trace changed')
    from audit_finite_service_alignment import summarize_model,summarize_native
    frames,native=summarize_native(trace);rows=[];worker_rows=[]
    for mode in modes:
        model=read(OUT/(mode+'.json'))['result'];summary=summarize_model(model,frames)
        task={(r['frame'],r['kernel'],r['slot']):r for r in model['tasks'] if r['stage']==0 and r['operation']=='worker_compute' and r['generation']==0}
        joins={(r['frame'],r['kernel']):r for r in model['kernel_joins'] if r['stage']==0}
        agreements=0;startup=0
        for w in trace['critical_rows']:
            key=w['frame'],w['kernel'];j=joins[key];caller=task[key+(0,)];target=task[key+(w['slot'],)]
            agreement=w['slot'] in j['last_slots'];agreements+=agreement
            startup+=max(0,target['first_service_ns']-caller['first_service_ns'])
            worker_rows.append(dict(mode=mode,frame=key[0],kernel=key[1],native_last_slot=w['slot'],
                model_last_slots=j['last_slots'],last_slot_agreement=agreement))
        rows.append(dict(mode=mode,
            stage_services_ms=[n/1e6 for n in model['replay']['stage_service_ns']],
            native_clean_stage_services_ms=trace['clean']['stage_service_ms'],
            predicted_II_ms=model['replay']['interval_ns']/1e6,native_clean_II_ms=trace['clean']['ii_ms'],
            model_same_native_critical_slot_post_caller_start_ms_per_frame=startup/len(frames)/1e6,
            last_slot_agreement_fraction=agreements/len(trace['critical_rows']),
            same_layout=True,only_VTA_costs_changed_relative_to_old_finite=True,
            diagnostic_not_independent_validation=True))
    result=dict(rows=rows,worker_rows=worker_rows,source_trace_sha256=sha256(trace_path),
        native_boot=trace['boot_id'],new_VTA_profile_boot=latest['profile_boot'],
        different_boot_development_only=True,no_model_promoted=True,new_DNN=0,
        new_search_or_Top20=False,source_sha256=sha256(__file__))
    write(OUT/'analysis/RESULTS.json',result,replace=True)
    return result


if __name__=='__main__':
    r=run();print(json.dumps({k:v for k,v in r.items() if k!='worker_rows'},ensure_ascii=False,indent=2))
