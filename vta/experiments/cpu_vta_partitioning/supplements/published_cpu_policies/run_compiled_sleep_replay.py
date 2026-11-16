"""Four fixed rule combinations; cached old results, six new offline replays."""
from pathlib import Path
from types import FunctionType
import json
import shutil
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,write,sha256
from finite_service.common import freeze
from finite_service.synthetic import mixed_actions,run as synthetic
from policy_branch_eval.common import isolated_inputs_only
from compiled_sleep_model import CompiledFinite,CompiledScheduler,make_select
from scheduler_gap_audit.curves import csv_write

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/compiled_sleeper_01'
PARENT=ROOT/'artifacts/three_thread_host_gap_01'
VARIANTS=dict(original=(False,False),wake_only=(True,False),sleep_only=(False,True),both=(True,True))
LABELS=dict(original='原近似',wake_only='仅修正唤醒粒度',sleep_only='仅修正休眠补偿',both='两项都按内核规则修正')


def run():
    started=time.monotonic();source=PARENT/'existing_model_diagnostic/INPUTS.json'
    parameter=OUT/'COMPILED_SLEEPER.json'
    with isolated_inputs_only():
        x=read(source);p=read(parameter)
        if p['DNN_FPS_used'] or p['sleeper_credit_ns']!=9000000 or p['wakeup_granularity_ns']!=3000000:
            raise ValueError('independently audited parameters changed')
        for path,h in x['replay_sources'].items():
            if sha256(path)!=h:raise ValueError('frozen model source changed')
        old=read(PARENT/'existing_model_diagnostic/REPLAY_FREEZE.json')
        wake_root=ROOT/'artifacts/compiled_wakeup_01/three_thread'
        wake=read(wake_root/'REPLAY_FREEZE.json');wake_plan=read(wake_root/'PLAN.json')
        if old['input_sha256']!=sha256(source) or wake_plan['input_sha256']!=sha256(source):
            raise ValueError('cached replay uses a different isolated input')
        plan=dict(source_input_sha256=sha256(source),parameter_sha256=sha256(parameter),
            source_hashes={str(f):sha256(f) for f in (Path(__file__),ROOT/'compiled_sleep_model.py',ROOT/'tests/test_compiled_sleep_model.py')},
            variants={k:dict(wake=v[0],sleeper=v[1],proof=make_select(wake=v[0],sleeper=v[1])[1]) for k,v in VARIANTS.items()},
            primary_variant='both',primary_not_chosen_after_errors=True,
            parameters_not_searched_or_FPS_fitted=True,all_three_host_positions_required=True,
            other_scheduler_rules_and_isolated_profiles_unchanged=True,
            known_configuration_development_only=True,new_native_sessions=0,
            no_DP_or_Top20_promotion_from_local_error_only=True)
        freeze(OUT/'PLAN.json',plan);paths={};platform={}
        for variant,(w,s) in VARIANTS.items():
            for cid,stages in x['cases'].items():
                if variant in ('original','wake_only'):
                    item=(old['outputs'] if variant=='original' else wake['paths'])[cid]
                    path=Path(item['path'])
                    if sha256(path)!=item['sha256']:raise ValueError('cached baseline changed')
                else:
                    path=OUT/'replays'/(variant+'_'+cid+'.json')
                    if not path.exists():
                        result=CompiledFinite(stages,quanta=x['quanta_ns'],wake=w,sleeper=s,
                            wakeup_granularity_ns=p['wakeup_granularity_ns'],sleeper_credit_ns=p['sleeper_credit_ns'],
                            placement='wake_boundary',completion_model='independent',completion_point='midpoint',
                            runnable_completion_wait=True,controls=None,gate='off',host_order='before',**x['window']).run()
                        freeze(path,dict(result=result,plan_sha256=sha256(OUT/'PLAN.json'),no_native_performance_read=True))
                    if read(path)['plan_sha256']!=sha256(OUT/'PLAN.json'):raise ValueError('cached new replay differs')
                paths[variant+'/'+cid]=dict(path=str(path),sha256=sha256(path),reused=variant in ('original','wake_only'))
            factory=lambda q:CompiledScheduler(q,wake=w,sleeper=s,wakeup_granularity_ns=p['wakeup_granularity_ns'],sleeper_credit_ns=p['sleeper_credit_ns'])
            f=FunctionType(synthetic.__code__,{**synthetic.__globals__,'Scheduler':factory},synthetic.__name__,synthetic.__defaults__)
            platform[variant]=f(mixed_actions(),x['quanta_ns'],fluid=False)
        freeze(OUT/'REPLAY_FREEZE.json',dict(paths=paths,platform_predictions=platform,plan_sha256=sha256(OUT/'PLAN.json'),
            native_data_not_read=True,new_offline_DNN_replays=6,cached_offline_DNN_replays=6))
    # No measured phase timings/FPS are available to the preceding code.
    native=read(PARENT/'analysis/GAP_COMPARISON.json');rows=[];ordering={}
    for variant in VARIANTS:
        group=[]
        for n in native['rows']:
            r=read(paths[variant+'/'+n['configuration_id']]['path'])['result']['replay']
            row=dict(variant=variant,variant_zh=LABELS[variant],role=n['role'],configuration_id=n['configuration_id'],
                predicted_front_ms=r['stage_service_ns'][0]/1e6,native_front_ms=n['pipeline_front_ms'],
                front_error_ms=r['stage_service_ns'][0]/1e6-n['pipeline_front_ms'],
                predicted_II_ms=r['interval_ns']/1e6,native_II_ms=n['pipeline_II_ms'],native_FPS=n['pipeline_FPS'])
            rows.append(row);group.append(row)
        group.sort(key=lambda r:(r['predicted_II_ms'],r['configuration_id']))
        ordering[variant]=dict(order=[r['role'] for r in group],selected_native_FPS=group[0]['native_FPS'])
    old_platform=read(ROOT/'artifacts/finite_service_01/MIXED_VALIDATION.json');platform_rows=[]
    for variant,pred in platform.items():
        for a in old_platform['results'][0]['actors']:
            wall=pred['wall_ns'][str(a['actor'])];err=abs(wall-a['observed_wall_ns'])
            platform_rows.append(dict(variant=variant,actor=a['actor'],predicted_wall_ns=wall,
                observed_wall_ns=a['observed_wall_ns'],absolute_error_ns=err,
                original_allowed_error_ns=a['allowed_error_ns'],passed=err<=a['allowed_error_ns']))
    result=dict(rows=rows,ordering=ordering,platform_regression=platform_rows,
        primary_variant='both',no_choice_of_favorable_variant_after_measurement=True,
        source_sha256=sha256(__file__),plan_sha256=sha256(OUT/'PLAN.json'),
        native_source_sha256=sha256(PARENT/'analysis/GAP_COMPARISON.json'),
        new_native_sessions=0,new_DNN=0,known_case_development_only=True,
        no_new_Top20_or_FPS_gain_claim=True,full_CFS_emulator_not_implemented=True,
        compute_seconds=time.monotonic()-started)
    write(OUT/'RESULTS.json',result,replace=True);csv_write(OUT/'phase_errors.csv',rows)
    csv_write(OUT/'platform_regression.csv',platform_rows)
    report=ROOT.parents[4]/'paper/my_paper/analysis/20261111_compiled_sleeper';report.mkdir(parents=True,exist_ok=True)
    for name in ('RESULTS.json','PLAN.json','COMPILED_SLEEPER.json','phase_errors.csv','platform_regression.csv'):
        shutil.copy2(OUT/name,report/name)
    return result


if __name__=='__main__':
    r=run();print(json.dumps({k:r[k] for k in ('rows','ordering','platform_regression','compute_seconds')},ensure_ascii=False,indent=2))
