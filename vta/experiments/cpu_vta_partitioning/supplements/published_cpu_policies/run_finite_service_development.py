#!/usr/bin/env python3
"""Freeze isolated-input finite-service predictions, then compare known cases."""
from pathlib import Path
from statistics import mean
import argparse,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,write,digest,sha256,now
from model_semantics_audit.replay import validate_isolated
from phase_model.simulator import rational
from finite_service.common import ROOT,OUTPUT,freeze,sources as platform_sources
from finite_service_replay import FiniteReplay
from thread_history_audit.simulator import HistoryReplay
from scheduler_gap_audit.curves import csv_write


def code_identity():
    files=[Path(__file__),ROOT/'finite_service_replay.py',ROOT/'tests/test_finite_service_replay.py',
        ROOT/'thread_history_audit/simulator.py',ROOT/'model_semantics_audit/simulator.py',
        ROOT/'static_revision/trace_replay.py',ROOT/'phase_model/simulator.py',ROOT/'pool_temporal/simulator.py']
    return {**platform_sources(),**{str(p):sha256(p) for p in files}}


def prepare(output=OUTPUT/'development'):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if not read(OUTPUT/'MIXED_VALIDATION.json')['finite_service_platform_check_passed']:
        raise ValueError('platform held-out check failed; DNN development stopped')
    source=ROOT/'artifacts/model_semantics_audit_01';seal=read(source/'INPUT_FREEZE.json')
    for p,h in seal['files'].items():
        if sha256(source/p)!=h:raise ValueError('isolated projection changed')
    iso=validate_isolated(read(source/'ISOLATED_INPUTS.json'));cal=read(OUTPUT/'CALIBRATION.json')
    if cal['sources']!=platform_sources():raise ValueError('platform source changed')
    windows={k:dict(frames=104,warmup=8,scored=64,queue_depth=2) for k in iso['cases']}
    windows['yolov3_tiny416/overlap']=dict(frames=48,warmup=8,scored=16,queue_depth=2)
    freeze(output/'INPUTS.json',dict(isolated=iso,quanta_ns=cal['quanta_ns'],windows=windows,
        origin='isolated_projection_plus_zero_DNN_platform_profile',contains_native_DNN_performance=False))
    plan=dict(sources=code_identity(),inputs_sha256=sha256(output/'INPUTS.json'),
        calibration_sha256=sha256(OUTPUT/'CALIBRATION.json'),
        primary_placement='wake_boundary',sensitivity_placement='retain',
        scenarios=[dict(case=k,placement=p) for k in sorted(iso['cases']) for p in ('wake_boundary','retain')],
        fluid_baselines=['wake_boundary','retain','instant'],new_DNN_inferences=0,
        development_known_cases_only=True,platform_boot=read(OUTPUT/'PLATFORM_RECEIPT.json')['boot_id'],
        old_DNN_boot_not_relabelled=True,cross_boot_platform_transfer_is_development_only=True,
        no_case_specific_parameter_selection=True,no_new_score_or_DP_change=True)
    freeze(output/'PLAN.json',plan);return dict(frozen=True,finite_scenarios=6)


def replay(output=OUTPUT/'development'):
    output=Path(output);plan=read(output/'PLAN.json')
    if plan['sources']!=code_identity() or plan['inputs_sha256']!=sha256(output/'INPUTS.json'):
        raise ValueError('development code/input changed')
    doc=read(output/'INPUTS.json');iso=validate_isolated(doc['isolated']);files={};started=time.monotonic()
    specs=[dict(case=k,placement=p,mode='fluid') for k in sorted(iso['cases']) for p in plan['fluid_baselines']]
    specs += [dict(**s,mode='finite') for s in plan['scenarios']]
    for spec in specs:
        path=output/'replays'/(digest(spec)[:24]+'.json');identity=dict(spec=spec,plan_sha256=sha256(output/'PLAN.json'))
        if path.exists():
            item=read(path)
            if item['input_identity']!=digest(identity) or item['content_sha256']!=digest({k:v for k,v in item.items() if k!='content_sha256'}):
                raise ValueError('development cache changed')
        else:
            case=iso['cases'][spec['case']]
            args=dict(placement=spec['placement'],runnable_completion_wait=True,controls=None,
                gate=case['configuration']['execution_mode'],host_order='before',**doc['windows'][spec['case']])
            if spec['mode']=='finite':result=FiniteReplay(case['stages'],quanta=doc['quanta_ns'],**args).run()
            else:result=HistoryReplay(case['stages'],**args).run()
            item=dict(spec=spec,result=result,input_identity=digest(identity),native_DNN_inputs_read=False,
                known_configuration_development=True)
            item['content_sha256']=digest(item);freeze(path,item)
        files[str(path.relative_to(output))]=sha256(path)
        print('有限服务开发回放',spec['case'],spec['mode'],spec['placement'],flush=True)
    freeze(output/'REPLAY_FREEZE.json',dict(files=files,sources=code_identity(),
        plan_sha256=sha256(output/'PLAN.json'),before_native_analysis=True))
    write(output/'COST.json',dict(seconds=time.monotonic()-started,new_DNN_inferences=0),replace=True)
    return dict(replays=len(files),DNN_inferences=0)


def analyze(output=OUTPUT/'development'):
    output=Path(output);seal=read(output/'REPLAY_FREEZE.json');plan=read(output/'PLAN.json')
    if seal['sources']!=code_identity() or seal['plan_sha256']!=sha256(output/'PLAN.json'):
        raise ValueError('finite prediction seal changed')
    old_path=ROOT/'artifacts/scheduler_gap_audit_01/analysis/RESULTS.json'
    if sha256(old_path)!=read(ROOT/'artifacts/scheduler_gap_audit_01/DELIVERY.json')['results_sha256']:
        raise ValueError('old stage reference changed')
    old=read(old_path);native={}
    for c in old['cases']:
        key=c['network']+'/'+c['case']
        front=next(s for s in old['stages'] if s['network']==c['network'] and s['case']==c['case'] and s['stage']=='cpu_00_02')
        native[key]=dict(stage_ms=front['clean_pipeline_service_ms'],ii_ms=c['clean_interval_ms'],
            source=str(old_path),boot_id=old['boot_id'],window=dict(frames=104,warmup=8,scored=64))
    new_path=ROOT/'artifacts/yolo_scheduler_trace_repair_01/analysis/RESULTS.json'
    if sha256(new_path)!=read(ROOT/'artifacts/yolo_scheduler_trace_repair_01/DELIVERY.json')['results_sha256']:
        raise ValueError('corrected scheduler reference changed')
    newer=read(new_path)
    native['yolov3_tiny416/overlap']=dict(stage_ms=newer['clean']['stage_service_ms'][0],ii_ms=newer['clean']['ii_ms'],
        source=str(new_path),boot_id=newer['boot_id'],window=dict(frames=48,warmup=8,scored=16))
    rows=[];work={}
    for name,h in seal['files'].items():
        if sha256(output/name)!=h:raise ValueError('finite replay changed')
        item=read(output/name);spec=item['spec'];r=item['result'];v=r['replay'];ref=native[spec['case']]
        amount=v['useful_CPU_ns'];key=spec['case']
        if key in work and amount!=work[key]:raise ValueError('scheduler changed finite CPU work')
        work[key]=amount
        front_actor=[t for t in r['actor_totals'] if t['role']=='stage' and t['stage']==0 and t['operation']=='completion_wait']
        rows.append(dict(**spec,predicted_front_ms=v['stage_service_ns'][0]/1e6,
            native_front_ms=ref['stage_ms'],front_absolute_error_ms=abs(v['stage_service_ns'][0]/1e6-ref['stage_ms']),
            predicted_II_ms=v['interval_ns']/1e6,native_II_ms=ref['ii_ms'],
            II_absolute_error_ms=abs(v['interval_ns']/1e6-ref['ii_ms']),
            completion_wait_CPU_ms=sum(float(rational(t['CPU'])) for t in front_actor)/ref['window']['scored']/1e6,
            mean_model_first_service_skew_ms=mean(k['first_service_skew_ns'] for k in r['kernel_joins'] if k['stage']==0)/1e6,
            native_source=ref['source'],native_boot=ref['boot_id'],matched_window=ref['window']))
    pairs=[]
    for case in sorted(native):
        for placement in ('wake_boundary','retain'):
            f=next(r for r in rows if r['case']==case and r['mode']=='finite' and r['placement']==placement)
            b=next(r for r in rows if r['case']==case and r['mode']=='fluid' and r['placement']==placement)
            pairs.append(dict(case=case,placement=placement,front_error_change_ms=f['front_absolute_error_ms']-b['front_absolute_error_ms'],
                II_error_change_ms=f['II_absolute_error_ms']-b['II_absolute_error_ms']))
    result=dict(rows=rows,matched_pairs=pairs,known_case_development=True,no_unseen_validation_claim=True,
        no_pipeline_data_fitted=True,no_new_DNN_inferences=True,primary_placement=plan['primary_placement'],
        no_automatic_promotion_to_DP=True,platform_calibration_from_different_boot=True,
        native_references=native,recorded_at=now())
    dest=output/'analysis';dest.mkdir(exist_ok=True);write(dest/'RESULTS.json',result)
    csv_write(dest/'predictions.csv',rows);csv_write(dest/'matched_scheduler_effect.csv',pairs)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','replay','analyze','run'))
    p.add_argument('--output',type=Path,default=OUTPUT/'development');a=p.parse_args()
    if a.command=='run':
        result={}
        for op in ('prepare','replay','analyze'):result[op]=globals()[op](a.output)
        result=result['analyze']
    else:result=globals()[a.command](a.output)
    print(json.dumps(result,ensure_ascii=False))
