"""Locate host CPU work relative to actual isolated device calls."""
import json
from pathlib import Path

from cpu_vta_partitioning.common import read,sha256,verify_inventory
from phase_model.extract import average
from policy_branch_eval.analysis import csv_write
from workload_model.common import freeze
from .collect import summarize,task


def split_run(run,events):
    selected=sorted([r for r in events if r['call']==run['call']],key=lambda r:r['wall_start_ns'])
    if not selected:raise ValueError('run has no mapped hardware call')
    cursor=run['cpu_start_ns'];host=[];devices=[]
    for event in selected:
        serial=event['cpu_start_ns']-cursor
        if serial<0:raise ValueError('device sequence CPU clock moves backwards')
        host.append(serial)
        devices.append(dict(wall_ns=event['wall_end_ns']-event['wall_start_ns'],
            CPU_ns=event['cpu_end_ns']-event['cpu_start_ns'],instructions=event['instructions']))
        cursor=event['cpu_end_ns']
    host.append(run['cpu_end_ns']-cursor)
    if host[-1]<0 or sum(host)+sum(e['CPU_ns'] for e in devices)!=run['cpu_end_ns']-run['cpu_start_ns']:
        raise ValueError('host/device CPU accounting does not conserve work')
    return dict(host_serial_CPU_ns=host,devices=devices,
                run_CPU_ns=run['cpu_end_ns']-run['cpu_start_ns'],
                run_wall_ns=run['wall_end_ns']-run['wall_start_ns'])


def extract(output,run_dir):
    output,run_dir=Path(output).resolve(),Path(run_dir).resolve()
    state=read(run_dir/'state.json')
    if state['status']!='COMPLETE':raise ValueError('bounded device component batch is incomplete')
    profiles={};sources={};table=[]
    for spec in read(output/'PLAN.json')['specs']:
        if spec['mode']!='observed':continue
        bundle=output/spec['network']/'capsule';doc=task(bundle,spec)
        path=run_dir/'accepted'/state['boot_id']/(doc['task_id']+'.json');item=read(path)
        verify_inventory(item['raw'],item['files'])
        if item['task']!=doc or summarize(item['raw'],doc,bundle)!=item['summary']:
            raise ValueError('device component evidence no longer replays')
        raw=Path(item['raw']);spans=[json.loads(s) for s in (raw/'phase_trace.jsonl').read_text().splitlines()]
        calls=[]
        for f in range(doc['warmup'],doc['warmup']+doc['scored']):
            run=next(s for s in spans if s['call']==f and s['operation']=='run')
            calls.append(split_run(run,item['summary']['device']['mapped']))
        signature=lambda c:[d['instructions'] for d in c['devices']]
        if any(signature(c)!=signature(calls[0]) for c in calls):
            raise ValueError('device instruction sequence changes; cannot fabricate average alignment')
        stages=dict(host_serial_CPU=[average([c['host_serial_CPU_ns'][i] for c in calls])
                                    for i in range(len(calls[0]['host_serial_CPU_ns']))],
                    devices=[dict(wall=average([c['devices'][i]['wall_ns'] for c in calls]),
                                  CPU=average([c['devices'][i]['CPU_ns'] for c in calls]),
                                  instructions=d['instructions']) for i,d in enumerate(calls[0]['devices'])])
        clean_doc=task(bundle,{**spec,'mode':'clean'});clean_path=run_dir/'accepted'/state['boot_id']/(clean_doc['task_id']+'.json')
        clean=read(clean_path);verify_inventory(clean['raw'],clean['files'])
        if summarize(clean['raw'],clean_doc,bundle)!=clean['summary']:
            raise ValueError('clean VTA profile evidence changed')
        value=dict(network=spec['network'],program=spec['case'],stages=stages,
            clean_cost=clean['summary']['cost'],observed_cost=item['summary']['cost'],
            no_pipeline_performance=True,boot_id=state['boot_id'],
            module_sha256=read(bundle/'plans/programs.json')[spec['case']]['files']['module.so']['sha256'],
            observed_receipt=str(path),clean_receipt=str(clean_path))
        profiles[spec['network']+'/'+spec['case']]=value
        val=lambda a:a['numerator']/a['denominator']/1e6
        table.append(dict(network=spec['network'],program=spec['case'],device_calls_per_frame=len(stages['devices']),
            clean_service_ms=value['clean_cost']['service_ms'],
            observed_service_ms=value['observed_cost']['service_ms'],
            probe_service_change_percent=100*(value['observed_cost']['service_ms']/value['clean_cost']['service_ms']-1),
            run_CPU_before_first_device_ms=val(stages['host_serial_CPU'][0]),
            run_CPU_between_devices_ms=sum(val(a) for a in stages['host_serial_CPU'][1:-1]),
            run_CPU_after_last_device_ms=val(stages['host_serial_CPU'][-1]),
            CPU_inside_device_calls_ms=sum(val(d['CPU']) for d in stages['devices']),
            wall_inside_device_calls_ms=sum(val(d['wall']) for d in stages['devices']),
            run_CPU_ms=mean([c['run_CPU_ns'] for c in calls])/1e6,
            run_wall_ms=mean([c['run_wall_ns'] for c in calls])/1e6))
        sources[str(path)]=sha256(path);sources[str(clean_path)]=sha256(clean_path)
    if len(profiles)!=4:raise ValueError('four genuine component profiles required')
    freeze(output/'ISOLATED_DEVICE_PHASES.json',dict(profiles=profiles,sources=sources,
        extractor_sha256=sha256(Path(__file__)),no_pipeline_FPS=True,
        CPU_inside_device_is_not_device_compute=True,
        poll_runnable_times_within_device_call_unknown=True))
    csv_write(output/'device_phase_accounting.csv',table)
    return dict(profiles=4,accounting=table)


def mean(values):return sum(values)/len(values)
