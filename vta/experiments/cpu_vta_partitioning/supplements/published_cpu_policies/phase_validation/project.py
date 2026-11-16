"""Verified isolation-only projection, including new whole-session CPU shares."""
import json
from pathlib import Path

from cpu_vta_partitioning.common import digest,read,sha256,verify_inventory
from phase_model.extract import project_call,average_calls
from phase_profile.runtime import make_task,summary
from workload_model.common import OUTPUT,ROOT,freeze


def thread_weights(clean,profile):
    threads=profile['identity']['threads'];program=profile['identity']['program']
    entries=[r for r in clean['summary']['thread_ledger'] if r.get('stage')==program]
    amounts=[]
    for core in profile['identity']['mask']:
        slot=[r for r in entries if r['actual_mask']==[core]]
        if len(slot)!=1:raise ValueError('isolated logical thread ownership is ambiguous')
        amounts.append(slot[0]['cpu_time_delta_ms'])
    total=sum(amounts)
    return [a/total for a in amounts] if total>0 else [1/threads]*threads


def project(output,run_dir):
    output,run_dir=Path(output).resolve(),Path(run_dir).resolve()
    state=read(run_dir/'state.json')
    if state['status']!='COMPONENTS_COMPLETE':raise ValueError('new isolated CPU components are incomplete')
    old_path=OUTPUT/'phase_profile/model_inputs/ISOLATED_PHASES.json';old=read(old_path)
    profiles=old['profiles'].copy();sources={str(old_path):sha256(old_path)}
    observed=[s for s in read(output/'COMPONENT_TASKS.json') if s['phase']=='observed']
    if len(observed)!=4:raise ValueError('exactly four missing CPU signatures expected')
    for spec in observed:
        bundle=output/spec['network']/'capsule';doc=make_task(bundle,spec)
        receipt=run_dir/'accepted'/state['boot_id']/(doc['task_id']+'.json');item=read(receipt)
        if item['task']!=doc or item['boot_id']!=state['boot_id']:raise ValueError('isolated CPU identity differs')
        verify_inventory(item['raw'],item['files'])
        if summary(item['raw'],doc,bundle)!=item['summary']:raise ValueError('isolated CPU receipt does not replay')
        raw=Path(item['raw']);phase=[json.loads(s) for s in (raw/'phase_trace.jsonl').read_text().splitlines()]
        events=[json.loads(s) for s in (raw/'wait_trace.jsonl').read_text().splitlines()]
        launches=[r for r in events if r['kind']==0];callbacks={};barriers={}
        for r in events:
            if r['kind'] in (1,2):(callbacks if r['kind']==1 else barriers).setdefault(r['launch'],[]).append(r)
        stage=doc['stages'][0];calls=[]
        for frame in range(doc['warmup'],doc['warmup']+doc['scored']):
            span=next(r for r in phase if r['stage']==stage['program'] and r['operation']=='run' and r['call']==frame)
            calls.append(project_call(span,launches,callbacks,barriers))
        clean_task=make_task(bundle,{**spec,'phase':'clean','probe':False})
        clean_path=run_dir/'accepted'/state['boot_id']/(clean_task['task_id']+'.json');clean=read(clean_path)
        verify_inventory(clean['raw'],clean['files'])
        if summary(clean['raw'],clean_task,bundle)!=clean['summary']:raise ValueError('clean isolated cost does not replay')
        program=read(bundle/'plans/programs.json')[stage['program']]
        identity=dict(network=spec['network'],canonical_sha256=program['canonical_sha256'],
            program=stage['program'],threads=stage['threads'],mask=stage['mask'],
            module_sha256=program['files']['module.so']['sha256'])
        key='phase_'+digest(identity)[:24]
        if key in profiles:raise ValueError('new isolated component duplicates an old measured signature')
        profiles[key]=dict(identity=identity,configuration=doc['configuration'],average=average_calls(calls),
            clean_cost=clean['summary']['cost'],observed_cost=item['summary']['cost'],
            no_pipeline_performance=True,
            provenance=dict(boot_id=state['boot_id'],observed_receipt=str(receipt),observed_sha256=sha256(receipt),
                clean_receipt=str(clean_path),clean_sha256=sha256(clean_path),
                runtime_sha256=doc['runtime_sha256'],executor_sha256=doc['executor_sha256'],adapter_sha256=doc['adapter_sha256']))
        sources[str(receipt)]=sha256(receipt);sources[str(clean_path)]=sha256(clean_path)
    if len(profiles)!=14:raise ValueError('fourteen isolated CPU profiles are required')
    for profile in profiles.values():
        path=Path(profile['provenance']['clean_receipt'])
        if sha256(path)!=profile['provenance']['clean_sha256']:raise ValueError('prior clean isolated source changed')
        clean=read(path);verify_inventory(clean['raw'],clean['files'])
        profile['fresh_session_thread_weights']=thread_weights(clean,profile)
        sources[str(path)]=sha256(path)
    freeze(output/'ISOLATED_CPU_INPUTS.json',dict(profiles=profiles,sources=sources,
        no_pipeline_FPS=True,thread_shares_scope='whole isolated session including warmup and startup empty probe',
        extractor_sha256=sha256(Path(__file__))))
    return dict(profiles=len(profiles),new_profiles=4,sources=len(sources))
