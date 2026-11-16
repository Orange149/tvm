"""Project the five sealed isolated inputs; never read a pipeline receipt."""
from pathlib import Path
from fractions import Fraction
import json

import run_current_isolated_phases as native
from cpu_vta_partitioning.common import read,write,sha256,digest,verify_inventory
from phase_model.extract import project_call,average_calls,average
from analyze_polling_profile import project_calls
from completion_profile.workflow import completion_bounds
from scheduler_gap_audit.curves import csv_write

OUT=native.OUT/'projection'


def CPU_profile(doc,spans,trace):
    launches=[r for r in trace if r['kind']==0];callbacks={};barriers={}
    for r in trace:
        if r['kind'] in (1,2):(callbacks if r['kind']==1 else barriers).setdefault(r['launch'],[]).append(r)
    calls=[]
    for frame in range(doc['warmup'],doc['warmup']+doc['scored']):
        phase=[r for r in spans if r['call']==frame and r['operation']=='run']
        if len(phase)!=1:raise ValueError('missing measured CPU phase')
        calls.append(project_call(phase[0],launches,callbacks,barriers))
    return average_calls(calls)


def VTA_profile(doc,spans,events):
    calls=project_calls(spans,events,doc['warmup'],doc['scored'])
    grouped=[sorted([e for e in events if e['call']==f],key=lambda e:e['wall_start_ns'])
        for f in range(doc['warmup'],doc['warmup']+doc['scored'])]
    result=dict(host_serial_CPU=[average([c['host_serial_CPU_ns'][i] for c in calls])
        for i in range(len(calls[0]['host_serial_CPU_ns']))],devices=[])
    for i,example in enumerate(calls[0]['devices']):
        group=[g[i] for g in grouped];old=[c['devices'][i] for c in calls]
        bounds=[completion_bounds(e) for e in group]
        entry={key:average([e[key] for e in old]) for key in
            ('CPU_ns','wall_ns','poll_checks','submit_mmio_ns','post_start_sleep_ns','poll_wait_ns')}
        entry['CPU']=entry.pop('CPU_ns');entry['wall']=entry.pop('wall_ns');entry['instructions']=example['instructions']
        entry['completion_lower_ns']=average([b['lower_ns'] for b in bounds])
        entry['completion_upper_ns']=average([b['upper_ns'] for b in bounds])
        result['devices'].append(entry)
    return result


def changes(clean,observed):
    result={k:100*(observed[k]/clean[k]-1) for k in ('run_ms','run_core_ms')}
    return dict(changes_percent=result,passed=all(abs(v)<=20 for v in result.values()))


def run():
    if OUT.exists():raise FileExistsError('preserve the isolated input projection')
    native.validate();state=read(native.RUN/'state.json')
    if state['status']!='COMPLETE' or state['native_sessions']!=15 or state['component_calls']!=105:
        raise ValueError('five complete matched components required')
    profiles={};sources={};table=[];count=0
    programs=read(native.BUNDLE/'plans/programs.json')
    for case in native.CASES:
        receipts={}
        for mode in ('qualify','clean','observed'):
            doc=native.task(dict(scope='component',case=case,mode=mode))
            p=native.RUN/'accepted'/state['boot_id']/(doc['task_id']+'.json');receipt=read(p)
            if receipt['task']!=doc or receipt['boot_id']!=state['boot_id']:raise ValueError('isolated task/boot differs')
            verify_inventory(receipt['raw'],receipt['files'])
            raw=Path(receipt['raw'])
            if (raw/'task_hash').read_text().strip()!=digest(doc):raise ValueError('isolated raw task differs')
            if digest(native.summarize(raw,doc,native.BUNDLE))!=digest(receipt['summary']):raise ValueError('isolated record cannot be rebuilt')
            receipts[mode]=receipt;sources[str(p)]=sha256(p)
            count+=sum(doc[k] for k in ('warmup','scored','tail'))
        fingerprints=[r['summary']['output_fingerprints'] for r in receipts.values()]
        if any(v!=fingerprints[0] for v in fingerprints):raise ValueError('phase probe changed complete output')
        observed=receipts['observed'];doc=observed['task'];raw=Path(observed['raw']);s=doc['stages'][0]
        spans=[json.loads(line) for line in (raw/'phase_trace.jsonl').read_text().splitlines()]
        common=dict(program=case,device=s['device'],threads=s['threads'],mask=s['mask'],
            worker_core_map=s.get('worker_core_map'),configuration=doc['configuration'],
            module_sha256=programs[case]['files']['module.so']['sha256'],runtime_sha256=doc['runtime_sha256'],
            executor_sha256=doc['executor_sha256'],source_boot=state['boot_id'],
            clean_cost=receipts['clean']['summary']['cost'],observed_cost=observed['summary']['cost'],
            no_pipeline_performance_input=True,quality=changes(receipts['clean']['summary']['cost'],observed['summary']['cost']))
        if s['device']=='cpu':
            trace=[json.loads(line) for line in (raw/'wait_trace.jsonl').read_text().splitlines()]
            common['average']=CPU_profile(doc,spans,trace)
            common['caller_wait_excluded_by_existing_replay_not_redistributed']=True
            common['noncaller_runtime_work_outside_callbacks_not_fabricated']=True
        else:
            common['stages']=VTA_profile(doc,spans,observed['summary']['absolute_device']['mapped'])
        profiles[case]=common
        table.append(dict(program=case,device=s['device'],threads=s['threads'],mask=s['mask'],
            clean_run_ms=common['clean_cost']['run_ms'],clean_run_CPU_ms=common['clean_cost']['run_core_ms'],
            **common['quality']['changes_percent'],probe_quality_passed=common['quality']['passed']))
    if count!=105:raise ValueError('component invocation count differs')
    sources[str(native.OUT/'PLAN.json')]=sha256(native.OUT/'PLAN.json')
    sources[str(native.OUT/'TASKS.json')]=sha256(native.OUT/'TASKS.json')
    source_files=[Path(__file__),native.ROOT/'phase_model/extract.py',native.ROOT/'analyze_polling_profile.py',
        native.ROOT/'device_phase/extract.py',native.ROOT/'completion_profile/workflow.py']
    sources.update({str(p):sha256(p) for p in source_files})
    OUT.mkdir();write(OUT/'ISOLATED_PHASES.json',dict(profiles=profiles,sources=sources,
        no_pipeline_performance_input=True,component_calls=105,DNN_inferences=0,
        all_probe_quality_passed=all(v['quality']['passed'] for v in profiles.values()),
        native_state=state,projection_schema='existing_CPU_fork_join_and_independent_VTA_completion'))
    csv_write(OUT/'components.csv',table)
    write(OUT/'FREEZE.json',dict(files={str(p):sha256(p) for p in (OUT/'ISOLATED_PHASES.json',OUT/'components.csv')},
        sources=sources,no_pipeline_results_read=True,not_yet_a_new_predictor_or_DP=True))
    print(json.dumps(dict(rows=table,all_probe_quality_passed=all(v['quality']['passed'] for v in profiles.values())),ensure_ascii=False))


if __name__=='__main__':run()
