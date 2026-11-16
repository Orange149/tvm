"""Current layouts, existing replay: isolated CPU/host/device dependencies.

This is a known-case feasibility test, not a new full-domain model. Eight
predeclared replays reuse the finite/fluid schedulers without fitting settings.
"""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import argparse
import json
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,write,sha256,digest,verify_inventory
from cpu_policies.open_resources.costs import quantize_ns
from model_ablation.model import score
from phase_model.simulator import rational
from device_completion_replay import CompletionFinite,CompletionFluid
from policy_branch_eval.common import isolated_inputs_only
from run_scope_rank import assert_plan,inputs_for,freeze

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/current_host_lock_replay_20261112_01'
NEW=ROOT/'artifacts/current_isolated_phases_20261112_01/projection/ISOLATED_PHASES.json'
OLD=ROOT/'artifacts/cross_network_model_01/phase_validation/ISOLATED_CPU_INPUTS.json'
POOL=ROOT/'artifacts/window_tail_check_20261112_01/POOL.json'
MODES=[dict(CPU=cpu,completion=completion) for cpu in ('finite','fluid')
       for completion in ('aggregate','independent')]


def exact(v):
    value=rational(v);return dict(numerator=value.numerator,denominator=value.denominator)


def sources():
    names=['run_current_host_lock_replay.py','device_completion_replay.py','polling_replay.py',
        'finite_service_replay.py','finite_service/scheduler.py','phase_model/simulator.py',
        'model_semantics_audit/simulator.py','thread_history_audit/simulator.py','static_revision/trace_replay.py',
        'policy_branch_eval/common.py','tests/test_current_host_lock_replay.py']
    return {str(ROOT/n):sha256(ROOT/n) for n in names}


def normalize_device(profile):
    """Reuse the prior isolated clean/observer CPU normalization; never FPS."""
    stages=deepcopy(profile['stages'])
    observed=sum((rational(x) for x in stages['host_serial_CPU']),Fraction(0))
    observed+=sum((rational(d['CPU']) for d in stages['devices']),Fraction(0))
    clean=Fraction(str(profile['clean_cost']['run_core_ms']))*1_000_000
    if observed<=0 or clean<=0:raise ValueError('invalid isolated host CPU normalization')
    factor=clean/observed
    stages['host_serial_CPU']=[exact(rational(x)*factor) for x in stages['host_serial_CPU']]
    for d in stages['devices']:
        cpu=rational(d['CPU']);off=rational(d['wall'])-cpu
        if off<0:raise ValueError('isolated device CPU exceeds wall')
        d['CPU']=exact(cpu*factor);d['wall']=exact(cpu*factor+off)
        d['submit_mmio_ns']=exact(rational(d['submit_mmio_ns'])*factor)
    return stages,exact(factor)


def build_stages(cfg,programs,costs,new,old):
    stages=[];assumptions=[]
    for row in cfg['stages']:
        module=programs[row['program']]['files']['module.so']['sha256']
        value=dict(device=row['device'],mask=row['mask']);cost=costs.get(row,cfg)
        if row['device']=='vta':
            profile=new['profiles'][row['program']]
            if profile['module_sha256']!=module or profile['mask']!=row['mask']:
                raise ValueError('VTA host/module profile mismatch')
            value['device_profile'],factor=normalize_device(profile);cost=profile['clean_cost']
            assumptions.append(dict(program=row['program'],kind='clean_isolated_CPU_normalization',factor=factor,
                poll_CPU_equal_share_proxy=True,hardware_duration_from_isolated_bracket=True))
        else:
            value['worker_core_map']=row['worker_core_map']
            matched=new['profiles'].get(row['program'])
            if matched and matched['device']=='cpu' and matched['threads']==row['threads']:
                if matched['module_sha256']!=module:raise ValueError('new CPU module differs')
                value['profile']=deepcopy(matched['average']);cost=matched['clean_cost']
                if matched['worker_core_map']!=row['worker_core_map']:
                    assumptions.append(dict(program=row['program'],kind='homogeneous_core_permutation',
                        measured=matched['worker_core_map'],requested=row['worker_core_map']))
            else:
                matches=[p for p in old['profiles'].values() if p['identity']['network']=='yolov3_tiny416'
                    and p['identity']['program']==row['program'] and p['identity']['threads']==row['threads']
                    and p['identity']['module_sha256']==module]
                if len(matches)==1:
                    profile=matches[0];value['profile']=deepcopy(profile['average']);cost=profile['clean_cost']
                    assumptions.append(dict(program=row['program'],kind='old_isolated_structure_preserved_original_boot',
                        old_module_exact=True,requested_mapping=row['worker_core_map'],
                        measured_mapping=profile['configuration']['stages'][0]['worker_core_map'],
                        old_source_boot=profile['provenance']['boot_id']))
                elif row['threads']==1:
                    value['profile']=dict(kernels=[],serial_cpu_segments=[quantize_ns(cost['run_core_ms'])])
                    assumptions.append(dict(program=row['program'],kind='single_thread_run_CPU_aggregate',
                        unlocalized_nonrunning_ns=max(0,quantize_ns(cost['run_ms'])-quantize_ns(cost['run_core_ms'])),
                        not_a_fine_trace=True))
                else:raise ValueError('missing genuine multithread profile; no macro fork fabricated')
        for op in ('set','get'):
            cpu=quantize_ns(cost[op+'_core_ms']);wall=quantize_ns(cost[op+'_ms'])
            value[op+'_cpu_ns']=cpu;value[op+'_nonrunning_ns']=max(0,wall-cpu)
        stages.append(value)
    return stages,assumptions


def prepare():
    if (OUT/'PLAN.json').exists():check();return read(OUT/'PLAN.json')
    # Check original cost/CPU sources first; the guarded projection below reads
    # only the isolated inputs and execution configurations, never receipts.
    plan=assert_plan();programs,costs,_=inputs_for(plan,'yolov3_tiny416','original')
    new=read(NEW);old=read(OLD)
    if not new['no_pipeline_performance_input'] or not new['all_probe_quality_passed'] or not old['no_pipeline_FPS']:
        raise ValueError('usable isolated input required')
    for value in (new,old):
        for p,h in value['sources'].items():
            if sha256(p)!=h:raise ValueError('isolated provenance changed')
    with isolated_inputs_only():
        cfgs=read(POOL)['yolov3_tiny416'];cases={}
        for key,cfg in cfgs.items():
            stages,assumptions=build_stages(cfg,programs,costs,new,old)
            cases[key]=dict(configuration=cfg,stages=stages,assumptions=assumptions,
                static_score=score(cfg,costs,'complete'))
        quanta=read(ROOT/'artifacts/device_completion_replay_01/INPUTS.json')['quanta_ns']
        freeze(OUT/'ISOLATED_INPUTS.json',dict(cases=cases,quanta_ns=quanta,
            window=dict(frames=48,warmup=8,scored=16,queue_depth=2),
            no_pipeline_performance_input=True,known_reference_layouts_not_unseen=True))
    return freeze(OUT/'PLAN.json',dict(sources=sources(),input_sha256=sha256(OUT/'ISOLATED_INPUTS.json'),
        isolated_sources={str(p):sha256(p) for p in (NEW,OLD,POOL)},modes=MODES,
        primary=dict(CPU='finite',completion='independent'),
        reference=dict(CPU='finite',completion='aggregate'),replays=8,
        native_sessions=0,DNN_inferences=0,no_new_scheduler_code=True,
        purpose='Does existing CPU/host/device ordering explain current serialized VTA service inflation?',
        decision='Do not choose a scheduler by known FPS. Report both; large differences mean scheduling remains unresolved. This development comparison cannot authorize full-domain promotion.',
        unknowns=['producer/collector CPU work','single-thread run nonrunning order',
            'Linux scheduling history','interference-dependent CPU/DDR demands'],
        GCAPS_scope='Section4 distinguishes host CPU operations and independent device work; not its priority scheduler.',
        no_DP_or_Top20_modified=True))


def check():
    p=read(OUT/'PLAN.json')
    if p['sources']!=sources() or p['input_sha256']!=sha256(OUT/'ISOLATED_INPUTS.json'):
        raise ValueError('frozen replay sources/inputs changed')
    for path,h in p['isolated_sources'].items():
        if sha256(path)!=h:raise ValueError('isolated source changed')
    return p


def replay():
    plan=check();inputs=read(OUT/'ISOLATED_INPUTS.json');files={};started=time.monotonic()
    for key,case in inputs['cases'].items():
        for mode in MODES:
            spec=dict(configuration=key,**mode);path=OUT/'replays'/(digest(spec)[:24]+'.json')
            if not path.exists():
                cls=CompletionFinite if mode['CPU']=='finite' else CompletionFluid
                extra=dict(quanta=inputs['quanta_ns']) if mode['CPU']=='finite' else {}
                with isolated_inputs_only():
                    value=cls(case['stages'],placement='wake_boundary',completion_model=mode['completion'],
                        completion_point='midpoint',runnable_completion_wait=True,controls=None,
                        gate=case['configuration']['execution_mode'],host_order='before',
                        **extra,**inputs['window']).run()
                freeze(path,dict(spec=spec,result=value,plan_sha256=sha256(OUT/'PLAN.json'),
                    no_pipeline_performance_read=True,not_a_new_full_domain_predictor=True))
            elif read(path)['plan_sha256']!=sha256(OUT/'PLAN.json'):raise ValueError('cached replay changed')
            files[str(path.relative_to(OUT))]=sha256(path)
            print('Current VTA lock replay',key,mode,flush=True)
    freeze(OUT/'REPLAY_FREEZE.json',dict(files=files,sources=sources(),no_pipeline_performance_read=True))
    freeze(OUT/'COST.json',dict(elapsed_seconds=time.monotonic()-started,offline_replays=len(files),new_DNN=0))
    return dict(replays=len(files),new_native_sessions=0)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','replay','validate'))
    a=p.parse_args();v=prepare() if a.command=='prepare' else replay() if a.command=='replay' else check()
    print(json.dumps(v,ensure_ascii=False,indent=2))
