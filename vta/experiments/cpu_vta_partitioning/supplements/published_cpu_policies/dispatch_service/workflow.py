from pathlib import Path
from statistics import mean
from types import FunctionType
import time

from cpu_vta_partitioning.common import digest, now, read, sha256, write
from finite_service.common import freeze
from finite_service.synthetic import mixed_actions, run as original_synthetic
from finite_service_replay import FiniteReplay
from scheduler_gap_audit.curves import csv_write
from .model import DispatchReplay, DispatchScheduler

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'artifacts/dispatch_service_01'


def sources():
    files=[ROOT/'run_dispatch_service.py',ROOT/'tests/test_dispatch_service.py',
           *sorted(Path(__file__).parent.glob('*.py')),ROOT/'finite_service_replay.py',
           ROOT/'finite_service/scheduler.py',ROOT/'finite_service/synthetic.py']
    return {str(p):sha256(p) for p in files}


def prepare(output=OUTPUT):
    output=Path(output)
    calibration=ROOT/'artifacts/finite_service_01/CALIBRATION.json'
    original_inputs=ROOT/'artifacts/finite_service_01/development/INPUTS.json'
    for path,expected in read(calibration)['sources'].items():
        if sha256(path)!=expected:raise ValueError('old calibrated model changed')
    plan=dict(experiment='dispatch_service_01',sources=sources(),
        source_reference='Linux v5.4 fair.c check_preempt_tick, hrtick_start_fair and set_next_entity',
        source_url='https://raw.githubusercontent.com/torvalds/linux/v5.4/kernel/sched/fair.c',
        change='retain CPU consumed since dispatch even while a task is alone; recompute remaining empirical service on runnable count changes',
        unchanged=['isolated work','host wake-boundary placement','empirical granules','wake credit','CPU affinity','waiting policy','DP domain'],
        calibration_sha256=sha256(calibration),isolated_input_sha256=sha256(original_inputs),
        platform_criteria='same existing per-thread wall error <= max(20 percent observed wall, largest measured service granule)',
        platform_pass_required_before_DNN_replay=True,
        native_sessions=0,DNN_inferences=0,board_connections=0,
        all_current_DNN_examples_are_known_development_cases=True,
        old_synthetic_observations_known_not_new_independent_validation=True,
        no_pipeline_slowdown_or_FPS_fitted=True,no_vendor_CFS_reproduction_claim=True,
        no_automatic_native_validation_or_full_DP=True)
    freeze(output/'PLAN.json',plan)
    return dict(frozen=True,native_sessions=0,DNN_inferences=0)


def assert_plan(output):
    output=Path(output);p=read(output/'PLAN.json')
    if p['sources']!=sources():raise ValueError('dispatch model identity changed')
    if p['calibration_sha256']!=sha256(ROOT/'artifacts/finite_service_01/CALIBRATION.json'):
        raise ValueError('platform calibration changed')
    if p['isolated_input_sha256']!=sha256(ROOT/'artifacts/finite_service_01/development/INPUTS.json'):
        raise ValueError('isolated development inputs changed')
    return p


def synthetic(actions, quanta):
    # Thin adaptation of an already frozen, independently checked event loop.
    # Only its scheduler constructor changes; all work and sleep semantics stay.
    ns=dict(original_synthetic.__globals__,Scheduler=DispatchScheduler)
    fn=FunctionType(original_synthetic.__code__,ns,original_synthetic.__name__,original_synthetic.__defaults__)
    result=fn(actions,quanta,fluid=False)
    result['model']='finite_service_dispatch_elapsed'
    return result


def platform(output=OUTPUT):
    output=Path(output);assert_plan(output)
    calibration=read(ROOT/'artifacts/finite_service_01/CALIBRATION.json')
    quanta=calibration['quanta_ns']
    # Prediction is sealed before this function reads native mixed observations.
    predicted=synthetic(mixed_actions(),quanta)
    freeze(output/'PLATFORM_PREDICTION.json',predicted)
    validation=read(ROOT/'artifacts/finite_service_01/MIXED_VALIDATION.json')
    baseline=next(x for x in validation['results'] if x['model']=='FINITE_VIRTUAL_RUNTIME')
    rows=[]
    for row in baseline['actors']:
        actor=row['actor'];observed=row['observed_wall_ns'];value=predicted['wall_ns'][str(actor)]
        error=abs(value-observed);allowed=row['allowed_error_ns']
        rows.append(dict(actor=actor,observed_wall_ns=observed,predicted_wall_ns=value,
            old_predicted_wall_ns=row['predicted_wall_ns'],old_absolute_error_ns=row['absolute_error_ns'],
            absolute_error_ns=error,allowed_error_ns=allowed,passed=error<=allowed))
    result=dict(passed=all(x['passed'] for x in rows),actors=rows,
        old_mean_absolute_error_ns=mean(x['old_absolute_error_ns'] for x in rows),
        new_mean_absolute_error_ns=mean(x['absolute_error_ns'] for x in rows),
        platform_observations_not_used_to_tune_revision=True,
        known_controls_not_independent_validation=True,prediction_sha256=sha256(output/'PLATFORM_PREDICTION.json'),
        source_validation_sha256=sha256(ROOT/'artifacts/finite_service_01/MIXED_VALIDATION.json'),
        no_DNN_data_used=True)
    freeze(output/'PLATFORM_VALIDATION.json',result)
    return result


def replay(output=OUTPUT):
    output=Path(output);assert_plan(output)
    if not read(output/'PLATFORM_VALIDATION.json')['passed']:
        raise ValueError('dispatch correction fails platform controls; no DNN replay allowed')
    inputs=read(ROOT/'artifacts/finite_service_01/development/INPUTS.json')
    started=time.monotonic();files={}
    for key,case in inputs['isolated']['cases'].items():
        for mode,cls in [('original',FiniteReplay),('dispatch_elapsed',DispatchReplay)]:
            spec=dict(case=key,mode=mode,placement='wake_boundary')
            file=output/'replays'/(digest(spec)[:24]+'.json')
            if not file.exists():
                result=cls(case['stages'],quanta=inputs['quanta_ns'],placement='wake_boundary',
                    runnable_completion_wait=True,controls=None,
                    gate=case['configuration']['execution_mode'],host_order='before',**inputs['windows'][key]).run()
                freeze(file,dict(spec=spec,result=result,plan_sha256=sha256(output/'PLAN.json'),
                    no_native_DNN_inputs=True,known_development_only=True))
            else:
                doc=read(file)
                if doc['spec']!=spec or doc['plan_sha256']!=sha256(output/'PLAN.json'):
                    raise ValueError('dispatch replay cache differs')
            files[str(file.relative_to(output))]=sha256(file)
            print('调度额度核查',key,mode,flush=True)
    freeze(output/'REPLAY_FREEZE.json',dict(files=files,sources=sources(),
        new_DNN_inferences=0,no_DNN_performance_read=True))
    write(output/'COST.json',dict(seconds=time.monotonic()-started),replace=True)
    return dict(replays=len(files),DNN_inferences=0)


def analyze(output=OUTPUT):
    output=Path(output);assert_plan(output)
    seal=read(output/'REPLAY_FREEZE.json')
    previous=read(ROOT/'artifacts/finite_service_01/development/analysis/RESULTS.json')
    if seal['sources']!=sources():raise ValueError('dispatch result source changed')
    rows=[];work={}
    for name,expected in seal['files'].items():
        if sha256(output/name)!=expected:raise ValueError('dispatch result changed')
        doc=read(output/name);spec=doc['spec'];value=doc['result']['replay'];case=spec['case']
        source=next(r for r in previous['rows'] if r['case']==case and r['mode']=='finite' and r['placement']=='wake_boundary')
        if case in work and value['useful_CPU_ns']!=work[case]:raise ValueError('model changed useful CPU work')
        work[case]=value['useful_CPU_ns']
        rows.append(dict(**spec,predicted_front_ms=value['stage_service_ns'][0]/1e6,
            predicted_II_ms=value['interval_ns']/1e6,native_front_ms=source['native_front_ms'],native_II_ms=source['native_II_ms'],
            front_absolute_error_ms=abs(value['stage_service_ns'][0]/1e6-source['native_front_ms']),
            II_absolute_error_ms=abs(value['interval_ns']/1e6-source['native_II_ms']),
            synthetic_wait_CPU_ns=value['synthetic_wait_CPU_ns']))
    result=dict(rows=rows,platform=read(output/'PLATFORM_VALIDATION.json'),known_development_only=True,
        no_independent_DNN_validation_claim=True,no_automatic_promotion=True,
        formula_or_full_DP_changed=False,source_sha256=sha256(__file__),recorded_at=now())
    write(output/'analysis/RESULTS.json',result,replace=True)
    csv_write(output/'analysis/predictions.csv',rows)
    return result
