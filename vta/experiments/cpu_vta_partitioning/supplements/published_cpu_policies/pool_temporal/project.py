"""Project real isolated calls; never replace the frozen macro cost tables."""
import json
from pathlib import Path

from cpu_vta_partitioning.common import read,sha256,verify_inventory
from phase_model.extract import project_call,average_calls,average
from device_phase.extract import split_run
from workload_model.common import OUTPUT,freeze
from .runtime import make_task,summary


def project(output,run):
    output,run=Path(output).resolve(),Path(run).resolve();state=read(run/'state.json')
    if state['status']!='COMPLETE':raise ValueError('candidate intrinsic collection incomplete')
    old_cpu_path=OUTPUT/'phase_validation/ISOLATED_CPU_INPUTS.json'
    old_device_path=OUTPUT/'device_phase_02/ISOLATED_DEVICE_PHASES.json'
    CPU=read(old_cpu_path)['profiles'].copy();device={};sources={str(p):sha256(p) for p in (old_cpu_path,old_device_path)}
    for value in read(old_device_path)['profiles'].values():
        device[value['network']+'/'+value['program']+'/a0-1-2-3']={**value,'mask':[0,1,2,3]}
    observed=[s for s in read(output/'TASKS.json') if s['mode']=='observed']
    for spec in observed:
        bundle=output/spec['network']/'capsule';doc=make_task(bundle,spec)
        path=run/'accepted'/state['boot_id']/(doc['task_id']+'.json');item=read(path)
        if item['task']!=doc or item['boot_id']!=state['boot_id']:raise ValueError('candidate projection source identity differs')
        verify_inventory(item['raw'],item['files'])
        if summary(item['raw'],doc,bundle)!=item['summary']:raise ValueError('candidate intrinsic receipt does not replay')
        sources[str(path)]=sha256(path)
        raw=Path(item['raw']);phase=[json.loads(s) for s in (raw/'phase_trace.jsonl').read_text().splitlines()]
        row=doc['stages'][0];p=read(bundle/'plans/programs.json')[row['program']]
        common=dict(network=spec['network'],program=row['program'],threads=row['threads'],mask=row['mask'],
            module_sha256=p['files']['module.so']['sha256'],no_pipeline_performance=True,
            provenance=dict(receipt=str(path),sha256=sha256(path),boot_id=state['boot_id'],
                executor_sha256=doc['executor_sha256'],observer_sha256=doc['observer_sha256'],
                runtime_sha256=doc['runtime_sha256']),observed_cost=item['summary']['cost'])
        runs=[next(r for r in phase if r['operation']=='run' and r['call']==i)
              for i in range(doc['warmup'],doc['warmup']+doc['scored'])]
        if row['device']=='cpu':
            events=[json.loads(s) for s in (raw/'wait_trace.jsonl').read_text().splitlines()]
            launches=[r for r in events if r['kind']==0];callbacks={};barriers={}
            for r in events:
                if r['kind'] in (1,2):(callbacks if r['kind']==1 else barriers).setdefault(r['launch'],[]).append(r)
            calls=[project_call(r,launches,callbacks,barriers) for r in runs]
            identity={k:common[k] for k in ('network','program','threads','mask','module_sha256')}
            key=spec['network']+'/'+spec['case']
            CPU[key]=dict(identity=identity,average=average_calls(calls),no_pipeline_performance=True,
                observed_cost=common['observed_cost'],provenance=common['provenance'],
                no_new_clean_profile=True,whole_phase_cost_not_used_to_rescale_worker_work=True)
        else:
            calls=[split_run(r,item['summary']['device']['mapped']) for r in runs]
            sig=lambda r:[d['instructions'] for d in r['devices']]
            if any(sig(r)!=sig(calls[0]) for r in calls):raise ValueError('device instruction sequence changed across scored calls')
            stages=dict(host_serial_CPU=[average([c['host_serial_CPU_ns'][i] for c in calls])
                        for i in range(len(calls[0]['host_serial_CPU_ns']))],
                devices=[dict(wall=average([c['devices'][i]['wall_ns'] for c in calls]),
                              CPU=average([c['devices'][i]['CPU_ns'] for c in calls]),instructions=d['instructions'])
                         for i,d in enumerate(calls[0]['devices'])])
            key=spec['network']+'/'+row['program']+'/a'+'-'.join(map(str,row['mask']))
            if key in device:raise ValueError('new device signature duplicates a reused one')
            device[key]=dict(**common,stages=stages,no_new_clean_profile=True)
    if len(CPU)!=52 or len(device)!=50:raise ValueError('candidate intrinsic coverage incomplete')
    result=dict(CPU=CPU,VTA=device,sources=sources,macro_costs_modified=False,pipeline_FPS_read=False,
                projector_sha256=sha256(Path(__file__)),input_scope='isolated scored calls only; old exact signatures retain provenance')
    freeze(output/'INTRINSICS.json',result)
    return dict(CPU_profiles=len(CPU),VTA_profiles=len(device),macro_costs_modified=False)
