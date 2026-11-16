"""Same-clock isolation projection with explicit unknown worker overhead."""
from fractions import Fraction
import json
from pathlib import Path
import statistics

from cpu_vta_partitioning.common import digest,read,sha256,verify_inventory,write
from phase_profile.runtime import summary
from workload_model.common import freeze


def average(values):
    value=Fraction(sum(values),len(values))
    return dict(numerator=value.numerator,denominator=value.denominator,
                rounded_up_ns=(value.numerator+value.denominator-1)//value.denominator)


def worker_segments(callback,barriers):
    """Split actual compute around observed barriers; barrier waiting is separate."""
    ordered=sorted(barriers,key=lambda r:r['wall_start_ns'])
    result=[];cpu=callback['cpu_start_ns'];wall=callback['wall_start_ns']
    for barrier in ordered:
        if (barrier['wall_start_ns']<wall or barrier['wall_end_ns']>callback['wall_end_ns']
                or barrier['cpu_start_ns']<cpu or barrier['cpu_end_ns']>callback['cpu_end_ns']):
            raise ValueError('barrier escaped its worker callback or moved backwards')
        result.append(barrier['cpu_start_ns']-cpu)
        cpu=barrier['cpu_end_ns'];wall=barrier['wall_end_ns']
    if callback['cpu_end_ns']<cpu:raise ValueError('negative final worker work')
    result.append(callback['cpu_end_ns']-cpu)
    return result


def project_call(phase,launches,callbacks,barriers):
    selected=sorted([r for r in launches if r['caller']==phase['tid']
        and phase['wall_start_ns']<=r['wall_start_ns'] and r['wall_end_ns']<=phase['wall_end_ns']],
        key=lambda r:r['wall_start_ns'])
    kernels=[];last_cpu=phase['cpu_start_ns'];serial=[];known_cpu=0
    for launch in selected:
        if launch['cpu_start_ns']<last_cpu:raise ValueError('caller CPU clock went backwards')
        gap=launch['cpu_start_ns']-last_cpu;serial.append(gap);known_cpu+=gap
        group=sorted(callbacks.get(launch['launch'],[]),key=lambda r:r['task'])
        if [r['task'] for r in group]!=list(range(launch['tasks'])):
            raise ValueError('missing or duplicate compute task')
        if group[0]['tid']!=phase['tid']:raise ValueError('logical worker zero is not the caller')
        pre=group[0]['cpu_start_ns']-launch['cpu_start_ns']
        tail=launch['cpu_end_ns']-group[0]['cpu_end_ns']
        if pre<0 or tail<0:raise ValueError('caller callback CPU outside launch')
        workers=[];barrier_counts=[]
        for row in group:
            if not launch['wall_start_ns']<=row['wall_start_ns']<=row['wall_end_ns']<=launch['wall_end_ns']:
                raise ValueError('callback escaped its launch')
            bs=[v for v in barriers.get(launch['launch'],[]) if v['task']==row['task']]
            parts=worker_segments(row,bs);barrier_counts.append(len(bs))
            workers.append(dict(logical_worker=row['task'],compute_segments_ns=parts,
                callback_cpu_ns=row['cpu_end_ns']-row['cpu_start_ns'],
                barrier_cpu_ns=sum(v['cpu_end_ns']-v['cpu_start_ns'] for v in bs)))
        if len(set(barrier_counts))!=1:raise ValueError('workers disagree on barrier generations')
        kernel=dict(module=launch['module'],function_offset=launch['function_offset'],tasks=launch['tasks'],
            launch_prefix_cpu_ns=pre,caller_completion_wait_and_return_cpu_ns=tail,
            workers=workers,barrier_generations=barrier_counts[0])
        kernels.append(kernel)
        known_cpu+=pre+tail+sum(v['callback_cpu_ns'] for v in workers)
        last_cpu=launch['cpu_end_ns']
    suffix=phase['cpu_end_ns']-last_cpu
    if suffix<0:raise ValueError('run end precedes last launch CPU time')
    serial.append(suffix);known_cpu+=suffix
    caller_cpu=phase['cpu_end_ns']-phase['cpu_start_ns']
    noncaller=sum(w['callback_cpu_ns'] for k in kernels for w in k['workers'][1:])
    if known_cpu!=caller_cpu+noncaller:
        raise ValueError('caller/worker CPU accounting does not conserve work')
    return dict(serial_cpu_segments_ns=serial,kernels=kernels,known_CPU_ns=known_cpu,
        caller_CPU_ns=caller_cpu,noncaller_callback_CPU_ns=noncaller,
        run_wall_ns=phase['wall_end_ns']-phase['wall_start_ns'])


def signature(call):
    return [(k['module'],k['function_offset'],k['tasks'],k['barrier_generations']) for k in call['kernels']]


def average_calls(calls):
    if any(signature(call)!=signature(calls[0]) for call in calls):
        raise ValueError('isolation run changed kernel sequence; cannot fabricate alignment')
    result=dict(serial_cpu_segments=[average([c['serial_cpu_segments_ns'][i] for c in calls])
                                    for i in range(len(calls[0]['serial_cpu_segments_ns']))],kernels=[])
    for i,base in enumerate(calls[0]['kernels']):
        k=dict(module=base['module'],function_offset=base['function_offset'],tasks=base['tasks'],
            barrier_generations=base['barrier_generations'],
            launch_prefix_cpu=average([c['kernels'][i]['launch_prefix_cpu_ns'] for c in calls]),
            isolated_completion_wait_and_return_cpu=average([c['kernels'][i]['caller_completion_wait_and_return_cpu_ns'] for c in calls]),workers=[])
        for j,w in enumerate(base['workers']):
            k['workers'].append(dict(logical_worker=j,compute_segments=[average([
                c['kernels'][i]['workers'][j]['compute_segments_ns'][q] for c in calls])
                for q in range(len(w['compute_segments_ns']))]))
        result['kernels'].append(k)
    result.update(known_CPU=average([c['known_CPU_ns'] for c in calls]),
                  caller_CPU=average([c['caller_CPU_ns'] for c in calls]),
                  run_wall=average([c['run_wall_ns'] for c in calls]))
    return result


def extract(output,run_dir):
    output,run_dir=Path(output).resolve(),Path(run_dir).resolve()
    state=read(run_dir/'state.json')
    if state['status'] not in ('COMPONENTS_COMPLETE','PIPELINES_COMPLETE'):
        raise ValueError('component collection must finish before projection')
    boot=state['boot_id'];profiles={};rows=[];source_receipts={}
    for receipt in sorted((run_dir/'accepted'/boot).glob('*.json')):
        item=read(receipt);task=item['task'];spec=task['spec']
        if spec['scope']!='component' or spec['phase']!='observed':continue
        bundle=output/spec['network']/'capsule'
        verify_inventory(item['raw'],item['files'])
        if summary(item['raw'],task,bundle)!=item['summary']:raise ValueError('phase receipt no longer replays')
        raw=Path(item['raw']);phase=[json.loads(s) for s in (raw/'phase_trace.jsonl').read_text().splitlines()]
        trace=[json.loads(s) for s in (raw/'wait_trace.jsonl').read_text().splitlines()]
        launches=[r for r in trace if r['kind']==0];callbacks={};barriers={}
        for row in trace:
            if row['kind'] in (1,2):
                (callbacks if row['kind']==1 else barriers).setdefault(row['launch'],[]).append(row)
        stage=task['stages'][0];calls=[]
        for call in range(task['warmup'],task['warmup']+task['scored']):
            span=next(r for r in phase if r['stage']==stage['program'] and r['call']==call and r['operation']=='run')
            calls.append(project_call(span,launches,callbacks,barriers))
        averaged=average_calls(calls)
        clean_spec={**spec,'phase':'clean','probe':False}
        from phase_profile.runtime import make_task
        clean_task=make_task(bundle,clean_spec);clean_receipt=run_dir/'accepted'/boot/(clean_task['task_id']+'.json')
        clean=read(clean_receipt);verify_inventory(clean['raw'],clean['files'])
        if summary(clean['raw'],clean_task,bundle)!=clean['summary']:raise ValueError('clean cost source changed')
        programs=read(bundle/'plans/programs.json');identity=dict(network=spec['network'],
            canonical_sha256=programs[stage['program']]['canonical_sha256'],program=stage['program'],
            threads=stage['threads'],mask=stage['mask'],module_sha256=programs[stage['program']]['files']['module.so']['sha256'])
        key='phase_'+digest(identity)[:24]
        profile=dict(identity=identity,configuration=task['configuration'],average=averaged,
            clean_cost=clean['summary']['cost'],observed_cost=item['summary']['cost'],
            observed_service_ratio=item['summary']['cost']['service_ms']/clean['summary']['cost']['service_ms'],
            caller_completion_wait_is_not_intrinsic_compute=True,
            missing_noncaller_runtime_CPU_not_distributed=True,
            no_pipeline_performance=True,
            provenance=dict(boot_id=boot,observed_receipt=str(receipt),observed_sha256=sha256(receipt),
                clean_receipt=str(clean_receipt),clean_sha256=sha256(clean_receipt),
                runtime_sha256=task['runtime_sha256'],executor_sha256=task['executor_sha256'],adapter_sha256=task['adapter_sha256']))
        profiles[key]=profile;source_receipts[str(receipt)]=sha256(receipt);source_receipts[str(clean_receipt)]=sha256(clean_receipt)
        known=averaged['known_CPU']['numerator']/averaged['known_CPU']['denominator']/1e6
        observed=item['summary']['cost']['run_core_ms']
        rows.append(dict(network=spec['network'],program=stage['program'],threads=stage['threads'],
            kernel_launches_per_frame=len(averaged['kernels']),
            internal_barriers_per_frame=sum(k['barrier_generations'] for k in averaged['kernels']),
            clean_run_ms=clean['summary']['cost']['run_ms'],observed_run_ms=item['summary']['cost']['run_ms'],
            clean_service_ms=clean['summary']['cost']['service_ms'],observed_service_ratio=profile['observed_service_ratio'],
            known_run_CPU_ms=known,observed_process_run_CPU_ms=observed,
            unlocated_run_CPU_ms=observed-known))
    if len(profiles)!=10:raise ValueError('expected ten complete isolated profiles')
    target=output/'model_inputs';freeze(target/'ISOLATED_PHASES.json',dict(schema='isolated-phase-model-v1',
        profiles=profiles,no_pipeline_FPS=True,no_VTA_runnable_timing_available=True,
        clock_and_scope='averages over scored calls; integer/rational nanoseconds; no post-hoc scaling',
        sources=source_receipts,extractor_sha256=sha256(Path(__file__))))
    from policy_branch_eval.analysis import csv_write
    csv_write(target/'component_accounting.csv',rows)
    return dict(profiles=len(profiles),source_receipts=len(source_receipts),
        model_input=str(target/'ISOLATED_PHASES.json'),accounting=rows,
        full_pipeline_prediction_validated=False)
