"""Frozen driver-chronology development replay over six known configurations."""
from copy import deepcopy
from pathlib import Path
from fractions import Fraction
import argparse,json,sys,time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import digest,now,read,sha256,write
from finite_service.common import freeze
from policy_branch_eval.common import isolated_inputs_only
from scheduler_gap_audit.curves import csv_write
from polling_replay import PollingFinite,PollingFluid
from phase_model.simulator import rational

ROOT=Path(__file__).resolve().parent
OUTPUT=ROOT/'artifacts/polling_replay_01'
PROFILE=ROOT/'artifacts/polling_profile_02/analysis/RESULTS.json'
POOL=ROOT/'artifacts/finite_service_01/prospective'
MODES=[(s,t) for s in ('finite','fluid') for t in ('aggregate','periodic')]


def exact(v):
    v=Fraction(v);return dict(numerator=v.numerator,denominator=v.denominator)


def sources():
    names=['run_polling_replay.py','polling_replay.py','tests/test_polling_replay.py',
        'finite_service_replay.py','finite_service/scheduler.py','static_revision/trace_replay.py',
        'thread_history_audit/simulator.py','model_semantics_audit/simulator.py']
    return {str(ROOT/n):sha256(ROOT/n) for n in names}


def prepare(output=OUTPUT):
    output=Path(output);profiles=read(PROFILE)
    if not all(p['probe_quality_passed'] for p in profiles['profiles'].values()):
        raise ValueError('component probe quality does not support all five shapes')
    if sha256(PROFILE)!=read(PROFILE.parent.parent/'DELIVERY.json')['results_sha256']:
        raise ValueError('isolated driver projection changed')
    selected=read(POOL/'SELECTION_FREEZE.json');pool=read(POOL/'ISOLATED_POOL.json');cases={}
    for network,identities in selected['networks'].items():
        for identity in identities['selected_ids']:
            item=deepcopy(pool['networks'][network][identity])
            for cfg,stage in zip(item['configuration']['stages'],item['stages']):
                if cfg['device']!='vta':continue
                if stage['mask']!=[0,1,2,3]:raise ValueError('no isolated polling profile for this host mask')
                p=profiles['profiles'][network+'/'+cfg['program']]
                program=read(POOL/network/'capsule/plans/programs.json')[cfg['program']]
                if program['files']['module.so']['sha256']!=p['module_sha256']:
                    raise ValueError('driver profile belongs to another module')
                profile=deepcopy(p['stages'])
                observed=sum((rational(x) for x in profile['host_serial_CPU']),Fraction(0))
                observed+=sum((rational(x['CPU']) for x in profile['devices']),Fraction(0))
                clean=Fraction(str(p['clean_cost']['run_core_ms']))*1_000_000
                if clean<=0 or observed<=0:raise ValueError('driver CPU normalization undefined')
                factor=clean/observed
                profile['host_serial_CPU']=[exact(rational(x)*factor) for x in profile['host_serial_CPU']]
                for d in profile['devices']:
                    original_CPU=rational(d['CPU']);off=rational(d['wall'])-original_CPU
                    d['CPU']=exact(original_CPU*factor);d['wall']=exact(original_CPU*factor+off)
                    d['submit_mmio_ns']=exact(rational(d['submit_mmio_ns'])*factor)
                stage['device_profile']=profile
                for op in ('set','get'):
                    CPU=Fraction(str(p['clean_cost'][op+'_core_ms']))*1_000_000
                    wall=Fraction(str(p['clean_cost'][op+'_ms']))*1_000_000
                    stage[op+'_cpu_ns']=exact(CPU);stage[op+'_nonrunning_ns']=exact(max(0,wall-CPU))
                item.setdefault('profile_assumptions',[]).append(dict(program=cfg['program'],
                    normalized_run_CPU=exact(clean),normalization_factor=exact(factor),
                    observed_nonrunning_device_time_preserved=True,
                    serial_nonrunning_unmodeled=True,submit_wall_used_as_CPU_proxy_capped_at_total=True,
                    poll_CPU_slices_assumed_equal=True,not_a_hardware_completion_predictor=True))
            item['network']=network;cases[network+'/'+identity]=item
    freeze(output/'INPUTS.json',dict(cases=cases,window=pool['window'],quanta_ns=pool['quanta_ns'],
        origin='isolated_CPU_phases_plus_current_isolated_VTA_poll_counts',contains_pipeline_performance=False))
    freeze(output/'PLAN.json',dict(sources=sources(),inputs_sha256=sha256(output/'INPUTS.json'),
        polling_profile_sha256=sha256(PROFILE),known_configuration_selection_sha256=sha256(POOL/'SELECTION_FREEZE.json'),
        profile_boot=profiles['native_state']['boot_id'],
        modes=[dict(CPU=s,driver=t) for s,t in MODES],primary=dict(CPU='finite',driver='periodic'),
        matched_baseline=dict(CPU='finite',driver='aggregate'),
        all_six_cases_are_known_development=True,
        interpretation='same CPU work and nonrunning duration; change driver chronology only, within-call host core retained',
        native_sessions=0,DNN_inferences=0,no_automatic_new_pipeline_or_full_DP=True,
        residual_serial_nonrunning_and_actual_hardware_completion_not_recovered=True))
    return dict(cases=len(cases),replays=len(cases)*len(MODES),new_DNN_inferences=0)


def assert_plan(output):
    p=read(Path(output)/'PLAN.json')
    if p['sources']!=sources() or p['inputs_sha256']!=sha256(Path(output)/'INPUTS.json'):
        raise ValueError('polling replay sources/input changed')
    return p


def replay(output=OUTPUT):
    output=Path(output);assert_plan(output);inputs=read(output/'INPUTS.json');start=time.monotonic();files={}
    for key,item in inputs['cases'].items():
        for CPU,timing in MODES:
            spec=dict(case=key,CPU=CPU,driver=timing);path=output/'replays'/(digest(spec)[:24]+'.json')
            if not path.exists():
                cls=PollingFinite if CPU=='finite' else PollingFluid
                kwargs=dict(quanta=inputs['quanta_ns']) if CPU=='finite' else {}
                with isolated_inputs_only():
                    result=cls(item['stages'],placement='wake_boundary',driver_timing=timing,
                        runnable_completion_wait=True,controls=None,gate=item['configuration']['execution_mode'],
                        host_order='before',**kwargs,**inputs['window']).run()
                freeze(path,dict(spec=spec,result=result,configuration=item['configuration'],
                    static_score_ns=item['static_score_ns'],
                    combined_score_ns=max(item['static_score_ns'],result['replay']['interval_ns']),
                    plan_sha256=sha256(output/'PLAN.json'),no_native_performance_read=True))
            else:
                if read(path)['plan_sha256']!=sha256(output/'PLAN.json'):raise ValueError('poll replay cache changed')
            files[str(path.relative_to(output))]=sha256(path)
            print('轮询时序回放',key,CPU,timing,flush=True)
    freeze(output/'REPLAY_FREEZE.json',dict(files=files,sources=sources(),native_results_read=False))
    write(output/'COST.json',dict(seconds=time.monotonic()-start),replace=True)
    return dict(replays=len(files),new_DNN_inferences=0)


def analyze(output=OUTPUT):
    output=Path(output);plan=assert_plan(output);seal=read(output/'REPLAY_FREEZE.json')
    native=read(POOL/'analysis/RESULTS.json')
    if native['native_boot']!=plan['profile_boot']:raise ValueError('development native/profile cohorts differ in boot')
    rows=[];work={}
    for leaf,h in seal['files'].items():
        if sha256(output/leaf)!=h:raise ValueError('polling prediction changed')
        doc=read(output/leaf);spec=doc['spec'];network,cid=spec['case'].split('/')
        reference=next(r for r in native['candidates'] if r['network']==network and r['configuration_id']==cid)
        pred=doc['result']['replay'];key=spec['case']
        if key in work and work[key]!=pred['useful_CPU_ns']:raise ValueError('chronology changed CPU work')
        work[key]=pred['useful_CPU_ns']
        rows.append(dict(network=network,configuration_id=cid,CPU=spec['CPU'],driver=spec['driver'],
            predicted_II_ms=pred['interval_ns']/1e6,combined_score_ms=doc['combined_score_ns']/1e6,
            native_II_ms=reference['ii_ms'],native_FPS=reference['fps'],
            predicted_front_ms=pred['stage_service_ns'][0]/1e6,native_front_ms=reference['stage_service_ms'][0],
            front_absolute_error_ms=abs(pred['stage_service_ns'][0]/1e6-reference['stage_service_ms'][0]),
            interval_absolute_error_ms=abs(pred['interval_ns']/1e6-reference['ii_ms']),events=pred['events']))
    result=dict(rows=rows,known_development_not_independent_validation=True,no_new_DNN_inferences=True,
        model_work_conserved=True,profile_boot=plan['profile_boot'],no_automatic_promotion=True,
        source_sha256=sha256(__file__),recorded_at=now())
    write(output/'analysis/RESULTS.json',result,replace=True);csv_write(output/'analysis/predictions.csv',rows)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','replay','analyze'))
    p.add_argument('--output',type=Path,default=OUTPUT);a=p.parse_args()
    print(json.dumps(globals()[a.command](a.output),ensure_ascii=False,indent=2))
