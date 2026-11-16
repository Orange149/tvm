"""Frozen known-case replay of independently progressing VTA completion."""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import argparse,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import digest,now,read,sha256,write
from finite_service.common import freeze
from phase_model.simulator import rational
from policy_branch_eval.common import isolated_inputs_only
from scheduler_gap_audit.curves import csv_write
from device_completion_replay import CompletionFinite,CompletionFluid
from cpu_policies.open_resources.domain import canonical_tuple

ROOT=Path(__file__).resolve().parent
OUTPUT=ROOT/'artifacts/device_completion_replay_01'
PROFILE=ROOT/'artifacts/completion_profile_01/analysis/RESULTS.json'
POOL=ROOT/'artifacts/finite_service_01/prospective'
MODES=[dict(CPU='finite',completion='aggregate',point='midpoint')]+[
    dict(CPU='finite',completion='independent',point=p) for p in ('lower','midpoint','upper')]+[
    dict(CPU='fluid',completion=c,point='midpoint') for c in ('aggregate','independent')]


def exact(x):
    x=Fraction(x);return dict(numerator=x.numerator,denominator=x.denominator)


def sources():
    names=['run_device_completion_replay.py','device_completion_replay.py','tests/test_device_completion_replay.py',
        'tests/test_device_completion_contract.py',
        'finite_service_replay.py','finite_service/scheduler.py','polling_replay.py',
        'static_revision/trace_replay.py','thread_history_audit/simulator.py','model_semantics_audit/simulator.py',
        'cpu_policies/open_resources/domain.py']
    return {str(ROOT/n):sha256(ROOT/n) for n in names}


def prepare(output=OUTPUT):
    output=Path(output);profiles=read(PROFILE)
    if not all(p['probe_quality_passed'] for p in profiles['profiles'].values()):
        raise ValueError('completion probe quality insufficient; diagnosis only, no automatic repeat')
    pool=read(POOL/'ISOLATED_POOL.json');selected=read(POOL/'SELECTION_FREEZE.json');cases={}
    for network,selection in selected['networks'].items():
        for cid in selection['selected_ids']:
            item=deepcopy(pool['networks'][network][cid])
            for cfg,s in zip(item['configuration']['stages'],item['stages']):
                if cfg['device']!='vta':continue
                p=profiles['profiles'][network+'/'+cfg['program']]
                actual=read(POOL/network/'capsule/plans/programs.json')[cfg['program']]['files']['module.so']['sha256']
                if actual!=p['module_sha256'] or s['mask']!=[0,1,2,3]:raise ValueError('completion profile program/host mismatch')
                profile=deepcopy(p['stages'])
                observed=sum((rational(x) for x in profile['host_serial_CPU']),Fraction(0))+sum((rational(d['CPU']) for d in profile['devices']),Fraction(0))
                clean=Fraction(str(p['clean_cost']['run_core_ms']))*1_000_000
                if clean<=0 or observed<=0:raise ValueError('driver CPU normalization undefined')
                factor=clean/observed
                profile['host_serial_CPU']=[exact(rational(x)*factor) for x in profile['host_serial_CPU']]
                for d in profile['devices']:
                    CPU=rational(d['CPU']);off=rational(d['wall'])-CPU
                    d['CPU']=exact(CPU*factor);d['wall']=exact(CPU*factor+off)
                    d['submit_mmio_ns']=exact(rational(d['submit_mmio_ns'])*factor)
                s['device_profile']=profile
                for op in ('set','get'):
                    CPU=Fraction(str(p['clean_cost'][op+'_core_ms']))*1_000_000
                    wall=Fraction(str(p['clean_cost'][op+'_ms']))*1_000_000
                    s[op+'_cpu_ns']=exact(CPU);s[op+'_nonrunning_ns']=exact(max(0,wall-CPU))
                item.setdefault('completion_assumptions',[]).append(dict(program=cfg['program'],
                    normalization_factor=exact(factor),host_serial_nonrunning_unmeasured=True,
                    poll_CPU_equal_share_proxy=True,submit_wall_capped_CPU_proxy=True,
                    hardware_service_from_isolated_completion_bracket=True,
                    hardware_interference_not_predicted=True))
            item['network']=network;cases[network+'/'+cid]=item
    freeze(output/'INPUTS.json',dict(cases=cases,window=pool['window'],quanta_ns=pool['quanta_ns'],
        contains_pipeline_performance=False,profile_boot=profiles['native_state']['boot_id']))
    freeze(output/'PLAN.json',dict(sources=sources(),inputs_sha256=sha256(output/'INPUTS.json'),
        profile_sha256=sha256(PROFILE),profile_boot=profiles['native_state']['boot_id'],modes=MODES,
        primary=dict(CPU='finite',completion='independent',point='midpoint'),
        baseline=dict(CPU='finite',completion='aggregate',point='midpoint'),
        all_six_cases_known_development=True,no_new_native_sessions=True,no_new_DNN=True,
        no_automatic_DP_or_pipeline_expansion=True,
        success_before_native_reference='both-network top1 not worse and at least one improved; then fresh validation required',
        CPU_scheduler_parameters_unchanged=True,within_device_host_core_retained=True,
        completion_lower_upper_sensitivity_not_performance_bounds=True,
        source='GCAPS section4 resource separation adapted; original driver exits on actual completion flag'))
    return dict(cases=len(cases),replays=len(cases)*len(MODES))


def assert_plan(output):
    p=read(output/'PLAN.json')
    if p['sources']!=sources() or p['inputs_sha256']!=sha256(output/'INPUTS.json') or p['profile_sha256']!=sha256(PROFILE):
        raise ValueError('device completion plan/source/input changed')
    return p


def replay(output=OUTPUT):
    output=Path(output);plan=assert_plan(output);inputs=read(output/'INPUTS.json');files={};start=time.monotonic()
    for case,item in inputs['cases'].items():
        for mode in MODES:
            spec=dict(case=case,**mode);path=output/'replays'/(digest(spec)[:24]+'.json')
            if not path.exists():
                cls=CompletionFinite if mode['CPU']=='finite' else CompletionFluid
                extra=dict(quanta=inputs['quanta_ns']) if mode['CPU']=='finite' else {}
                with isolated_inputs_only():
                    value=cls(item['stages'],placement='wake_boundary',completion_model=mode['completion'],
                        completion_point=mode['point'],runnable_completion_wait=True,controls=None,
                        gate=item['configuration']['execution_mode'],host_order='before',**extra,**inputs['window']).run()
                freeze(path,dict(spec=spec,result=value,configuration=item['configuration'],
                    static_score_ns=item['static_score_ns'],combined_score_ns=max(item['static_score_ns'],value['replay']['interval_ns']),
                    plan_sha256=sha256(output/'PLAN.json'),no_pipeline_performance_read=True))
            elif read(path)['plan_sha256']!=sha256(output/'PLAN.json'):raise ValueError('cached completion replay differs')
            files[str(path.relative_to(output))]=sha256(path)
            print('独立设备完成回放',case,mode,flush=True)
    freeze(output/'REPLAY_FREEZE.json',dict(sources=sources(),files=files,native_performance_read=False))
    write(output/'COST.json',dict(seconds=time.monotonic()-start),replace=True)
    return dict(replays=len(files),new_DNN=0)


def analyze(output=OUTPUT):
    output=Path(output);plan=assert_plan(output);seal=read(output/'REPLAY_FREEZE.json')
    native=read(POOL/'analysis/RESULTS.json')
    if native['native_boot']!=plan['profile_boot']:raise ValueError('known native and completion cohorts boot mismatch')
    rows=[];configurations={}
    for leaf,h in seal['files'].items():
        if sha256(output/leaf)!=h:raise ValueError('completion prediction changed')
        d=read(output/leaf);spec=d['spec'];network,cid=spec['case'].split('/')
        configurations[network,cid]=d['configuration']
        n=next(r for r in native['candidates'] if r['network']==network and r['configuration_id']==cid)
        p=d['result']['replay'];calls=d['result']['driver_completion']
        rows.append(dict(network=network,configuration_id=cid,**{k:v for k,v in spec.items() if k!='case'},
            predicted_II_ms=p['interval_ns']/1e6,combined_score_ms=d['combined_score_ns']/1e6,
            predicted_front_ms=p['stage_service_ns'][0]/1e6,native_front_ms=n['stage_service_ms'][0],
            native_II_ms=n['ii_ms'],native_FPS=n['fps'],
            front_absolute_error_ms=abs(p['stage_service_ns'][0]/1e6-n['stage_service_ms'][0]),
            interval_absolute_error_ms=abs(p['interval_ns']/1e6-n['ii_ms']),
            modeled_driver_checks=sum(c['checks'] for c in calls),useful_CPU_ms=p['useful_CPU_ns']/1e6))
    choices=[]
    for network in ('resnet18','yolov3_tiny416'):
        for mode in MODES:
            group=[r for r in rows if r['network']==network and all(r[k]==v for k,v in mode.items())]
            group.sort(key=lambda r:(r['combined_score_ms'],canonical_tuple(configurations[network,r['configuration_id']])))
            choices.append(dict(network=network,**mode,selected_id=group[0]['configuration_id'],
                selected_native_FPS=group[0]['native_FPS'],
                known_pool_best_FPS=max(r['native_FPS'] for r in group)))
    result=dict(rows=rows,choices=choices,known_development_not_independent=True,
        model_not_promoted=True,no_new_pipeline_measurement=True,recorded_at=now())
    write(output/'analysis/RESULTS.json',result,replace=True);csv_write(output/'analysis/predictions.csv',rows)
    csv_write(output/'analysis/choices.csv',choices)
    return dict(choices=choices,rows=len(rows))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','replay','analyze'))
    p.add_argument('--output',type=Path,default=OUTPUT);a=p.parse_args()
    print(json.dumps(globals()[a.command](a.output),ensure_ascii=False,indent=2))
