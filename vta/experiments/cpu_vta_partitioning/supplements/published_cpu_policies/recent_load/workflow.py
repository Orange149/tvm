from pathlib import Path
import time
from fractions import Fraction

from cpu_vta_partitioning.common import digest, now, read, sha256, write
from finite_service.common import freeze
from finite_service_replay import FiniteReplay
from dispatch_service.model import DispatchReplay
from policy_branch_eval.common import isolated_inputs_only
from .model import RecentLoadReplay,HALF_LIFE_NS
from scheduler_gap_audit.curves import csv_write

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'artifacts/recent_load_01'
MODES=('original','dispatch_elapsed','recent_load','recent_load_and_dispatch')


def sources():
    files=[ROOT/'run_recent_load.py',ROOT/'tests/test_recent_load.py',
        *sorted(Path(__file__).parent.glob('*.py')),ROOT/'finite_service_replay.py',
        ROOT/'finite_service/scheduler.py',ROOT/'dispatch_service/model.py',
        ROOT/'thread_history_audit/simulator.py',ROOT/'model_semantics_audit/simulator.py']
    return {str(p):sha256(p) for p in files}


def prepare(output=OUTPUT):
    output=Path(output);input_path=ROOT/'artifacts/finite_service_01/development/INPUTS.json'
    freeze(output/'PLAN.json',dict(experiment='recent_load_01',sources=sources(),
        isolated_inputs_sha256=sha256(input_path),
        source_urls=['https://raw.githubusercontent.com/torvalds/linux/v5.4/kernel/sched/pelt.c',
                     'https://raw.githubusercontent.com/torvalds/linux/v5.4/kernel/sched/fair.c'],
        source_functions=['accumulate_sum','___update_load_sum','wake_affine_weight','find_idlest_group_cpu'],
        primary_change='movable host work uses recent runnable-load memory at existing modeled wake boundaries',
        recent_load_half_life_ns=HALF_LIFE_NS,
        rule='allowed idle cores first, then recent load, current count, previous core and core index',
        primary_mode='recent_load',secondary_factor='dispatch_elapsed',modes=list(MODES),
        no_pipeline_CPU_FPS_or_core_histogram_as_inputs=True,
        no_parameter_sweep_or_case_specific_weights=True,
        no_runtime_or_execution_configuration_change=True,
        vendor_CFS_or_PELT_reproduced=False,
        all_examples_are_known_development=True,
        native_sessions=0,DNN_inferences=0,board_connections=0,
        no_automatic_full_DP_or_native_trial=True))
    return dict(frozen=True,offline_replays=12,DNN_inferences=0)


def assert_plan(output):
    p=read(Path(output)/'PLAN.json')
    if p['sources']!=sources() or p['isolated_inputs_sha256']!=sha256(ROOT/'artifacts/finite_service_01/development/INPUTS.json'):
        raise ValueError('recent-load source/input identity changed')
    return p


def evaluate(stages,quanta,mode,**kwargs):
    if mode not in MODES:raise ValueError('unknown load model')
    if mode=='original':return FiniteReplay(stages,quanta=quanta,**kwargs).run()
    if mode=='dispatch_elapsed':return DispatchReplay(stages,quanta=quanta,**kwargs).run()
    return RecentLoadReplay(stages,quanta=quanta,dispatch_elapsed=mode.endswith('_and_dispatch'),**kwargs).run()


def replay(output=OUTPUT):
    output=Path(output);assert_plan(output);started=time.monotonic()
    inputs=read(ROOT/'artifacts/finite_service_01/development/INPUTS.json');files={}
    for case,item in inputs['isolated']['cases'].items():
        for mode in MODES:
            spec=dict(case=case,mode=mode);path=output/'replays'/(digest(spec)[:24]+'.json')
            if not path.exists():
                with isolated_inputs_only():
                    result=evaluate(item['stages'],inputs['quanta_ns'],mode,placement='wake_boundary',
                        runnable_completion_wait=True,controls=None,gate=item['configuration']['execution_mode'],
                        host_order='before',**inputs['windows'][case])
                freeze(path,dict(spec=spec,result=result,plan_sha256=sha256(output/'PLAN.json'),
                    no_native_performance_read=True))
            elif read(path)['plan_sha256']!=sha256(output/'PLAN.json'):
                raise ValueError('load-model cache changed')
            files[str(path.relative_to(output))]=sha256(path)
            print('近期负载选核回放',case,mode,flush=True)
    freeze(output/'REPLAY_FREEZE.json',dict(files=files,sources=sources(),no_DNN_performance_read=True))
    write(output/'COST.json',dict(seconds=time.monotonic()-started),replace=True)
    return dict(replays=len(files),DNN_inferences=0)


def analyze(output=OUTPUT):
    output=Path(output);assert_plan(output);seal=read(output/'REPLAY_FREEZE.json')
    old=read(ROOT/'artifacts/finite_service_01/development/analysis/RESULTS.json')
    rows=[];work={}
    for leaf,expected in seal['files'].items():
        path=output/leaf
        if sha256(path)!=expected:raise ValueError('recent-load replay changed')
        item=read(path);spec=item['spec'];value=item['result']['replay']
        reference=next(x for x in old['rows'] if x['case']==spec['case'] and x['mode']=='finite' and x['placement']=='wake_boundary')
        if spec['case'] in work and work[spec['case']]!=value['useful_CPU_ns']:
            raise ValueError('placement changed finite CPU work')
        work[spec['case']]=value['useful_CPU_ns']
        host=[Fraction(0) for _ in range(4)]
        for row in item['result']['window_cpu']['rows']:
            stage=row['stage']
            config=read(ROOT/'artifacts/finite_service_01/development/INPUTS.json')['isolated']['cases'][spec['case']]['stages']
            if row['role']=='stage' and config[stage]['device']=='vta':
                for c,v in enumerate(row['core_CPU']):host[c]+=Fraction(v['numerator'],v['denominator'])
        gaps=item['result']['window_cpu']['gaps']
        rows.append(dict(**spec,predicted_front_ms=value['stage_service_ns'][0]/1e6,
            predicted_II_ms=value['interval_ns']/1e6,native_front_ms=reference['native_front_ms'],native_II_ms=reference['native_II_ms'],
            front_absolute_error_ms=abs(value['stage_service_ns'][0]/1e6-reference['native_front_ms']),
            II_absolute_error_ms=abs(value['interval_ns']/1e6-reference['native_II_ms']),
            modeled_host_CPU_core_ms_per_gap=[float(v/gaps/1_000_000) for v in host]))
    result=dict(rows=rows,known_development_only=True,no_independent_validation_claim=True,
        new_native_sessions=0,new_DNN_inferences=0,primary_mode='recent_load',
        no_automatic_full_DP_or_native_trial=True,source_sha256=sha256(__file__),recorded_at=now())
    write(output/'analysis/RESULTS.json',result,replace=True)
    csv_write(output/'analysis/predictions.csv',rows)
    return result
