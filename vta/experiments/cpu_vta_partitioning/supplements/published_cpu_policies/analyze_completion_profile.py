"""Audit isolated completion brackets; never read pipeline FPS."""
from pathlib import Path
from statistics import mean
import json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import now,read,sha256,verify_inventory,write
from completion_profile import workflow as w
from analyze_polling_profile import project_calls
from phase_model.extract import average
from scheduler_gap_audit.curves import csv_write

ROOT=w.ROOT


def analyze():
    plan=read(w.OUTPUT/'PLAN.json');state=read(w.RUN/'state.json')
    if state['status']!='COMPLETE' or state['boot_id']!=plan['expected_boot_id']:
        raise ValueError('completion component batch incomplete or boot differs')
    profiles={};sources={};table=[];devices=[]
    for spec in plan['specs']:
        if spec['mode']!='observed':continue
        bundle=w.OUTPUT/spec['network']/'capsule';w.validate(bundle);items={}
        for mode in ('qualify','clean','observed'):
            doc=w.task(bundle,{**spec,'mode':mode});path=w.RUN/'accepted'/state['boot_id']/(doc['task_id']+'.json')
            item=read(path);verify_inventory(item['raw'],item['files'])
            if item['task']!=doc or item['boot_id']!=state['boot_id'] or w.summarize(item['raw'],doc,bundle)!=item['summary']:
                raise ValueError('completion evidence identity/revalidation failed')
            items[mode]=item;sources[str(path)]=sha256(path)
        prints=[i['summary']['output_fingerprints'] for i in items.values()]
        if any(p!=prints[0] for p in prints):raise ValueError('complete component outputs disagree')
        observed=items['observed']['summary'];clean=items['clean']['summary'];raw=Path(items['observed']['raw'])
        spans=[json.loads(s) for s in (raw/'phase_trace.jsonl').read_text().splitlines()]
        doc=items['observed']['task'];mapped=observed['device']['mapped']
        calls=project_calls(spans,mapped,doc['warmup'],doc['scored'])
        grouped=[]
        for frame in range(doc['warmup'],doc['warmup']+doc['scored']):
            grouped.append(sorted([e for e in mapped if e['call']==frame],key=lambda e:e['wall_start_ns']))
        stage=dict(host_serial_CPU=[average([c['host_serial_CPU_ns'][i] for c in calls])
            for i in range(len(calls[0]['host_serial_CPU_ns']))],devices=[])
        for i,example in enumerate(calls[0]['devices']):
            group=[g[i] for g in grouped];old=[c['devices'][i] for c in calls]
            bounds=[w.completion_bounds(e) for e in group]
            entry={key:average([e[key] for e in old]) for key in ('CPU_ns','wall_ns','poll_checks','submit_mmio_ns','post_start_sleep_ns','poll_wait_ns')}
            entry['CPU']=entry.pop('CPU_ns');entry['wall']=entry.pop('wall_ns');entry['instructions']=example['instructions']
            entry['completion_lower_ns']=average([b['lower_ns'] for b in bounds])
            entry['completion_upper_ns']=average([b['upper_ns'] for b in bounds])
            entry['poll_read_envelope_ns']=average([e['poll_read_envelope_ns'] for e in group])
            stage['devices'].append(entry)
            devices.append(dict(network=spec['network'],program=spec['case'],device_index=i,
                instructions=example['instructions'],mean_checks=mean(e['observed_poll_reads'] for e in group),
                completion_lower_ms=mean(b['lower_ns'] for b in bounds)/1e6,
                completion_upper_ms=mean(b['upper_ns'] for b in bounds)/1e6,
                mean_bracket_width_us=mean(b['width_ns'] for b in bounds)/1000,
                max_bracket_width_us=max(b['width_ns'] for b in bounds)/1000,
                first_poll_complete_calls=sum(b['first_poll_completed'] for b in bounds),
                observed_driver_wall_ms=mean(e['wall_end_ns']-e['wall_start_ns'] for e in group)/1e6))
        cpu=100*(observed['cost']['run_core_ms']/clean['cost']['run_core_ms']-1)
        wall=100*(observed['cost']['run_ms']/clean['cost']['run_ms']-1)
        quality=abs(cpu)<=20 and abs(wall)<=20
        profiles[spec['network']+'/'+spec['case']]=dict(network=spec['network'],program=spec['case'],
            stages=stage,clean_cost=clean['cost'],observed_cost=observed['cost'],
            probe_CPU_change_percent=cpu,probe_wall_change_percent=wall,probe_quality_passed=quality,
            module_sha256=read(bundle/'plans/programs.json')[spec['case']]['files']['module.so']['sha256'],
            source_boot=state['boot_id'],no_pipeline_FPS_read=True,
            completion_interval_not_exact_hardware_duration=True,
            completion_under_CPU_DDR_interference_still_unmeasured=True)
        table.append(dict(network=spec['network'],program=spec['case'],device_calls=len(stage['devices']),
            clean_run_ms=clean['cost']['run_ms'],clean_run_CPU_ms=clean['cost']['run_core_ms'],
            probe_CPU_change_percent=cpu,probe_wall_change_percent=wall,probe_quality_passed=quality,
            total_lower_ms=sum(d['completion_lower_ms'] for d in devices if d['network']==spec['network'] and d['program']==spec['case']),
            total_upper_ms=sum(d['completion_upper_ms'] for d in devices if d['network']==spec['network'] and d['program']==spec['case'])))
    result=dict(profiles=profiles,component_table=table,device_table=devices,source_receipts=sources,
        native_state=state,analysis_source_sha256=sha256(__file__),recorded_at=now(),
        no_pipeline_FPS_read=True,no_DP_change=True,extra_MMIO_reads=0)
    dest=w.OUTPUT/'analysis';write(dest/'RESULTS.json',result,replace=True)
    csv_write(dest/'components.csv',table);csv_write(dest/'devices.csv',devices)
    return dict(components=table,device_count=len(devices),native_state=state)

if __name__=='__main__':print(json.dumps(analyze(),ensure_ascii=False,indent=2))
