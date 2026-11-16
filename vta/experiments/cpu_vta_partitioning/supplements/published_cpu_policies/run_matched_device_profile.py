"""Current-YOLO VTA completion diagnosis with the original frozen executor.

Three isolated programs (qualify/clean/observed), then one qualified observed
pipeline. Its clean48-frame reference is already accepted and never repeated.
"""
from copy import deepcopy
from pathlib import Path
from types import FunctionType
import argparse
import json
import shutil
import statistics
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,write,sha256,digest,inventory,verify_inventory,locked,now,sync_tree
from cpu_vta_partitioning.measurement import frames,input_sequence,correctness,calibration
from cpu_vta_partitioning.measurement.tasks import native_command as base_command
from cpu_policies.open_resources.domain import component,validate_candidate
from cpu_policies.open_resources.runtime import _affinity
from phase_profile.prepare import link_checked
from common_top20.runtime import summary as pipeline_summary
from common_top20.campaign import accept as original_accept
from board_transport_without_reprogram import transport
from run_scope_rank import freeze
from matched_device_profile.records import validate_events,device_projection

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/matched_device_profile_20261112_01'
RUN=ROOT/'runs/matched_device_profile_20261112_01'
PARENT=ROOT/'artifacts/window_tail_check_20261112_01'
BUNDLE=OUT/'capsule'
CONFIG=ROOT/'artifacts/cross_network_model_01/board.local.json'


def sources():
    project=ROOT.parents[1]/'src/cpu_vta_partitioning'
    paths=[Path(__file__),ROOT/'matched_device_profile/records.py',ROOT/'matched_device_profile/__init__.py',
        ROOT/'tests/test_matched_device_profile_records.py',ROOT/'tests/test_matched_device_profile_runtime.py',
        ROOT/'board_transport_without_reprogram.py',ROOT/'common_top20/runtime.py',ROOT/'common_top20/campaign.py',
        ROOT/'completion_profile/workflow.py',ROOT/'completion_profile/observer.cc',
        ROOT/'cpu_policies/open_resources/domain.py',ROOT/'cpu_policies/open_resources/runtime.py',
        *[project/p for p in ['common.py','measurement/__init__.py','measurement/tasks.py','transport/__init__.py','transport/scripts.py']]]
    return {str(p):sha256(p) for p in paths}


def specifications():
    return ([dict(scope='component',case=p,mode=m) for p in ('vta_01_01','vta_05_06','vta_09_11')
             for m in ('qualify','clean','observed')]
            +[dict(scope='pipeline',case='static_first',mode=m) for m in ('qualify','observed')])


def prepare():
    if (OUT/'PLAN.json').exists():validate();return read(OUT/'PLAN.json')
    import run_window_tail_check as parent
    parent.validate(PARENT/'yolov3_tiny416/capsule')
    next_path=PARENT/'NEXT_DEVICE_PROFILE_PLAN.json';intent=read(next_path)
    source=PARENT/'yolov3_tiny416/capsule';old=read(source/'manifest.json')
    programs=read(source/'plans/programs.json')
    for name,program in intent['programs'].items():
        if programs[name]!=program:raise ValueError('current module/frontier identity differs')
    observer=Path(intent['existing_observer_path'])
    if sha256(observer)!=intent['existing_observer_sha256']:raise ValueError('frozen observer changed')
    if sha256(source/'runtime/libvta.so')!=intent['existing_runtime_sha256']:
        raise ValueError('observer driver ABI belongs to a different runtime')
    for leaf,info in old['files'].items():
        if leaf.startswith(('programs/','runtime/','firmware/','inputs/','references/')) or leaf=='runner/vta_stage_pipeline_runner':
            link_checked(source/leaf,BUNDLE/leaf,info['sha256'])
    link_checked(observer,BUNDLE/'runner/libdevice_phase.so',intent['existing_observer_sha256'])
    cfg=read(PARENT/'POOL.json')['yolov3_tiny416'][intent['target_configuration']]
    protocol=read(PARENT/'PROTOCOL.json')
    protocol.update(experiment='matched_device_profile_20261112_01',component=dict(warmup=2,scored=8,tail=0),
        pipeline=dict(warmup=8,scored=16,tail=24),qualification=dict(warmup=0,scored=1,tail=0),
        specs=specifications(),native_sessions=11,VTA_component_calls=63,DNN_inferences=49,
        no_extra_MMIO_reads=True,no_new_executable_runtime_modules=True,
        current_state='FROZEN_SMALL_MATCHED_PROFILE',maximum_distinct_configurations=4,
        additional_native_campaign_requires_user_approval=False,
        user_authorized_scope='Existing goal explicitly permits necessary small-scale diagnosis; no adaptive retries.',
        component_probe_quality_limit_percent=20,observer_capacity=4096,new_component_profiles=3,
        execution_order='component range order with qualify/clean/observed, then qualified observed pipeline')
    protocol.pop('seed',None)
    clean=next(r for r in read(PARENT/'analysis/RESULTS.json')['rows'] if r['configuration']==cfg['id'])
    if sha256(clean['receipt'])!=clean['receipt_sha256']:raise ValueError('accepted clean pipeline changed')
    for leaf,value in [('programs.json',programs),('inputs.json',read(source/'plans/inputs.json')),
                       ('board_execution.json',read(source/'plans/board_execution.json')),
                       ('protocol.json',protocol),('configuration.json',cfg),
                       ('preparation.json',dict(sources=sources(),source_manifest=str(source/'manifest.json'),
                            source_manifest_sha256=sha256(source/'manifest.json'),next_plan_sha256=sha256(next_path),
                            observer_sha256=intent['existing_observer_sha256'],clean_pipeline_reference=clean))]:
        freeze(BUNDLE/'plans'/leaf,value)
    for i,(path,expected) in enumerate(sorted(sources().items())):
        target=BUNDLE/'sources'/f'{i:02}_{Path(path).name}'
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
        if sha256(target)!=expected:raise ValueError('controller snapshot changed while copying')
    assets=inventory(BUNDLE)
    payload={k:v for k,v in assets.items() if k.startswith(('programs/','runtime/','firmware/','inputs/','references/','runner/'))}
    payload['plans/board_execution.json']=assets['plans/board_execution.json']
    freeze(BUNDLE/'plans/board_payload.json',dict(files=payload,unpacked_bytes=sum(v['bytes'] for v in payload.values())))
    manifest=dict(experiment=protocol['experiment'],network='yolov3_tiny416',
        runner_sha256=old['runner_sha256'],runtime_sha256=old['runtime_sha256'],
        observer_sha256=intent['existing_observer_sha256'],files=inventory(BUNDLE))
    manifest['bundle_id']=digest(manifest)[:24];freeze(BUNDLE/'manifest.json',manifest)
    freeze(OUT/'PLAN.json',dict(sources=sources(),bundle_id=manifest['bundle_id'],protocol=protocol,
        expected_boot=intent['expected_boot'],parent_plan_sha256=sha256(next_path),
        clean_reference=clean,native_session_limit=11,VTA_component_calls_limit=63,DNN_limit=49,
        no_reboot=True,no_reprogram=True,new_compilation=False))
    validate()
    freeze(OUT/'TASKS.json',dict(plan_sha256=sha256(OUT/'PLAN.json'),tasks=[make_task(s) for s in specifications()]))
    return read(OUT/'PLAN.json')


def validate():
    m=read(BUNDLE/'manifest.json')
    if digest({k:v for k,v in m.items() if k!='bundle_id'})[:24]!=m['bundle_id']:
        raise ValueError('matched capsule identity changed')
    verify_inventory(BUNDLE,m['files'])
    payload=read(BUNDLE/'plans/board_payload.json')
    if (not payload['files'] or payload['unpacked_bytes']!=sum(v['bytes'] for v in payload['files'].values())
            or any(m['files'].get(k)!=v for k,v in payload['files'].items())
            or not {'runner/vta_stage_pipeline_runner','runner/libdevice_phase.so','plans/board_execution.json'}.issubset(payload['files'])):
        raise ValueError('deployable board payload incomplete or changed')
    proof=read(BUNDLE/'plans/preparation.json')
    if proof['sources']!=sources():raise ValueError('matched profile controller changed')
    if sha256(proof['source_manifest'])!=proof['source_manifest_sha256']:
        raise ValueError('clean reference capsule changed')
    if sha256(BUNDLE/'runner/libdevice_phase.so')!=m['observer_sha256']:
        raise ValueError('completion observer identity changed')
    return m


def make_task(spec,bundle=BUNDLE):
    bundle=Path(bundle);m=read(bundle/'manifest.json');p=read(bundle/'plans/protocol.json')
    if spec not in p['specs']:raise ValueError('task outside the frozen11-session scope')
    pipe=read(bundle/'plans/configuration.json')
    if spec['scope']=='component':
        row=next(s for s in pipe['stages'] if s['program']==spec['case'])
        cfg=component(row,pipe)
    else:cfg=pipe
    validate_candidate(cfg,read(bundle/'plans/programs.json'),component=spec['scope']=='component',unit_count=14)
    qualified=spec['mode']=='qualify'
    window=p['qualification'] if qualified else p[spec['scope']]
    kind=('qualify_component' if spec['scope']=='component' else 'qualify_pipeline') if qualified else (
        'calibration' if spec['scope']=='component' else 'performance')
    doc=dict(kind=kind,spec=deepcopy(spec),configuration=cfg,stages=cfg['stages'],environment=p['environment'],
        **window,input_indices=[0],network=m['network'],experiment=p['experiment'],
        executor_sha256=m['runner_sha256'],runtime_sha256=m['runtime_sha256'],bundle_id=m['bundle_id'],
        observer=spec['mode']!='clean',observer_sha256=m['observer_sha256'] if spec['mode']!='clean' else None,
        detailed=False,startup_affinity_probe=True)
    doc['task_id']=digest(doc)[:24];return doc


def command(doc,bundle,remote,job):
    if doc!=make_task(doc['spec'],bundle):raise ValueError('device task/window identity differs')
    argv,inputs=base_command(doc,bundle,remote,job)
    at=argv.index(remote+'/runtime/ld-linux-aarch64.so.1')
    argv[at:at]=['TVM_BIND_THREADS=1','TVM_EXCLUDE_WORKER0=1']
    if doc['configuration']['affinity_mode']=='pinned':argv.append('--cpu-worker-pinning')
    argv+=['--allow-cpu-affinity-superset','--resource-affinity-probe',
           '--control-cpu-affinity',','.join(map(str,doc['configuration']['control_mask']))]
    if not doc['kind'].startswith('qualify'):argv[argv.index('--output-mode')+1]='raw'
    if doc['observer']:
        index=next(i for i,s in enumerate(argv) if s.startswith('LD_PRELOAD='))
        argv[index]='LD_PRELOAD='+remote+'/runner/libdevice_phase.so:'+remote+'/runtime/libvta.so'
        at=argv.index(remote+'/runtime/ld-linux-aarch64.so.1')
        argv[at:at]=['VTA_DEVICE_PHASE_TRACE='+job+'/device_trace.jsonl']
    if '--process-cpu-affinity' in argv or '--queue-timing' in argv:raise ValueError('unrequested execution change')
    return argv,inputs


def summarize(raw,doc,bundle):
    if doc!=make_task(doc['spec'],bundle):raise ValueError('raw task identity differs')
    raw=Path(raw);total=sum(doc[k] for k in ('warmup','scored','tail'))
    rows=frames(raw/'result.jsonl',total);input_sequence(rows,doc)
    affinity=read(raw/'affinity.json');ledger,idle=_affinity(affinity,doc['configuration'],True)
    wanted=len(read(Path(bundle)/'plans/programs.json')[doc['stages'][-1]['program']]['output_node_ids'])
    fingerprints=[]
    for row in rows:
        outputs=row.get('raw_outputs',[])
        if len(outputs)!=wanted or sorted(r['index'] for r in outputs)!=list(range(wanted)):
            raise ValueError('missing or duplicated frontier output')
        fingerprints.append([[v['index'],v['shape'],v['dtype'],v['fnv1a64']] for v in outputs])
    if any(f!=fingerprints[0] for f in fingerprints):raise ValueError('fixed-input output changed')
    value=dict(correct=True,output_fingerprints=fingerprints[0],thread_ledger=ledger,idle_helpers=idle)
    if doc['kind'].startswith('qualify'):correctness(rows,doc,bundle,raw)
    elif doc['spec']['scope']=='component':
        cost=calibration(rows,doc['warmup'],doc['scored']);kept=rows[doc['warmup']:doc['warmup']+doc['scored']]
        for op in ('set','run','get'):
            cost[op+'_ms']=statistics.mean(r['stage0_'+op+'_ms'] for r in kept)
            cost[op+'_core_ms']=statistics.mean(r['stage0_'+op+'_process_cpu_ms'] for r in kept)
        value['cost']=cost
    else:
        kept=rows[doc['warmup']:doc['warmup']+doc['scored']]
        value.update(II_ms=(kept[-1]['completion_ms']-kept[0]['completion_ms'])/(len(kept)-1),
            stage_service_ms=[statistics.mean(r[f'stage{i}_ms'] for r in kept) for i in range(len(doc['stages']))])
        value['observed_FPS_not_formal_performance']=1000/value['II_ms']
    trace=raw/'device_trace.jsonl'
    if doc['observer']:
        lines=[json.loads(s) for s in trace.read_text().splitlines()]
        checked=validate_events(lines,doc,affinity,rows)
        value['device']=checked;value['projection']=device_projection(checked,rows,doc)
    elif trace.exists():raise ValueError('unrequested MMIO probe present')
    return value


def qemu():
    validate();old=ROOT/'artifacts/completion_profile_01/build';runtime=BUNDLE/'runtime';dest=OUT/'qemu';dest.mkdir(exist_ok=True)
    args=[shutil.which('qemu-aarch64-static'),'-E','LD_PRELOAD='+str(BUNDLE/'runner/libdevice_phase.so'),
          '-E','VTA_DEVICE_PHASE_TRACE='+str(dest/'device_trace.jsonl'),str(runtime/'ld-linux-aarch64.so.1'),
          '--library-path',str(old)+':'+str(runtime),str(old/'harness')]
    p=subprocess.run(args,capture_output=True,text=True,timeout=30);(dest/'qemu.log').write_text(p.stdout+p.stderr)
    if p.returncode:raise ValueError('MMIO forwarding harness failed')
    events=[json.loads(s) for s in (dest/'device_trace.jsonl').read_text().splitlines()]
    if events[0]['events']!=3 or events[0]['overflow']:raise ValueError('MMIO forwarding count failed')
    for e,count,result in zip(events[1:],[5,1,3],[0,0,1]):
        if (e['observed_poll_reads']!=count or e['driver_poll_iters']!=count or e['result']!=result
                or e['MMIO_error'] or e['start_writes']!=1 or e['auto_writes']!=3):
            raise ValueError('MMIO original read/write/timeout forwarding changed')
    dis=subprocess.run(['aarch64-linux-gnu-objdump','-d','--disassemble=_ZNSt6chrono3_V212steady_clock3nowEv',
                        str(runtime/'libstdc++.so.6')],capture_output=True,text=True,check=True).stdout
    if 'clock_gettime@plt' not in dis or 'mov\tw0, #0x1' not in dis:
        raise ValueError('native steady clock is not the audited CLOCK_MONOTONIC')
    (dest/'steady_clock_disassembly.txt').write_text(dis)
    value=dict(passed=True,bundle_id=read(BUNDLE/'manifest.json')['bundle_id'],fake_device_calls=3,DNN_inferences=0,
        MMIO_first_poll_and_timeout_forwarding=True,no_extra_MMIO_reads=True,
        native_and_observer_clock_same_scale_verified=True,unknown_session_origin_still_retained=True,
        observer_sha256=sha256(BUNDLE/'runner/libdevice_phase.so'))
    freeze(OUT/'QEMU.json',value);return value


def accept(ssh,doc,remote,boot,expected=None):
    function=FunctionType(original_accept.__code__,{**original_accept.__globals__,'summary':summarize})
    return function(ssh,doc,BUNDLE,remote,RUN,boot,expected)


def freeze_components(items):
    profile={};quality={};references={};max_events=0
    for name in ('vta_01_01','vta_05_06','vta_09_11'):
        clean,observed=[items['component',name,mode] for mode in ('clean','observed')]
        c,o=clean['summary']['cost'],observed['summary']['cost']
        changes={k:100*(o[k]/c[k]-1) for k in ('run_ms','run_core_ms')}
        projection=observed['summary']['projection'];eligible=projection['per_frame_projection_supported']
        quality[name]=dict(changes_percent=changes,passed=eligible and all(abs(v)<=20 for v in changes.values()))
        profile[name]=projection
        for mode in ('qualify','clean','observed'):
            doc=items['component',name,mode]['task'];path=RUN/'accepted'/read(RUN/'run.json')['boot_id']/(doc['task_id']+'.json')
            references[str(path)]=sha256(path)
        if eligible:
            counts=[len([e for e in observed['summary']['device']['mapped'] if e['frame']==frame]) for frame in range(10)]
            max_events+=max(counts)*48
    value=dict(profiles=profile,quality=quality,source_receipts=references,
        pipeline_trace_events_planning_estimate=max_events,
        pipeline_allowed=all(r['passed'] for r in quality.values()) and max_events<=4096,
        no_pipeline_trace_read=True,no_FPS_fit=True,capacity_estimate_not_a_hardware_bound=True)
    freeze(OUT/'COMPONENT_FREEZE.json',value);return value


def board(*,execute_board=False):
    if not execute_board:raise ValueError('--execute-board is required')
    validate();plan=read(OUT/'PLAN.json');proof=read(OUT/'QEMU.json')
    if not proof['passed'] or proof['bundle_id']!=plan['bundle_id']:raise ValueError('matching QEMU proof required')
    ssh=transport(read(CONFIG)['board'],command,execute_board=True);boot=ssh.boot_id()
    if boot!=plan['expected_boot']:raise ValueError('boot changed; do not merge this current-layout diagnosis')
    planned=read(OUT/'TASKS.json')
    if planned['plan_sha256']!=sha256(OUT/'PLAN.json') or planned['tasks']!=[make_task(s) for s in specifications()]:
        raise ValueError('frozen native task scope changed')
    started=time.monotonic();items={};completed=0
    with locked(RUN):
        freeze(RUN/'run.json',dict(boot_id=boot,bundle_id=plan['bundle_id'],plan_sha256=sha256(OUT/'PLAN.json'),
                                 task_list_sha256=sha256(OUT/'TASKS.json')))
        def state(status,**extra):
            write(RUN/'state.json',dict(status=status,boot_id=boot,accepted_sessions=completed,
                total_session_limit=11,updated_at=now(),**extra),replace=True)
        try:
            remote=ssh.deploy(BUNDLE,RUN);ssh.qualify_environment(remote,boot,RUN,BUNDLE)
            for doc in planned['tasks']:
                spec=doc['spec'];key=spec['scope'],spec['case'],spec['mode']
                if spec['scope']=='pipeline' and spec['mode']=='qualify':
                    profile=freeze_components(items)
                    if not profile['pipeline_allowed']:
                        state('COMPONENTS_LIMITED_NO_PIPELINE',new_DNN=0,VTA_component_calls=63)
                        sync_tree(RUN)
                        return read(RUN/'state.json')
                before=None if spec['mode']=='qualify' else items[spec['scope'],spec['case'],'qualify']['summary']['output_fingerprints']
                state('RUNNING',current_task=doc['task_id'],current_spec=spec)
                item=accept(ssh,doc,remote,boot,before);items[key]=item;completed+=1
                print('Matched device profile',completed,'/11',spec,flush=True)
            ssh.idle_command('cd '+__import__('shlex').quote(remote)+' && sha256sum -c board.sha256 > matched_post_verify.log && sync',boot=boot,timeout=600)
            if ssh.boot_id()!=boot:raise ValueError('boot changed after diagnosis')
            state('COMPLETE',native_sessions=11,VTA_component_calls=63,new_DNN=49,
                  seconds=time.monotonic()-started,no_reboot=True,no_reprogram=True,no_extra_measurement=True)
            sync_tree(RUN)
        except Exception as error:
            state('STOPPED_ON_ERROR',error_type=type(error).__name__,error=str(error));raise
    return read(RUN/'state.json')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','validate','qemu','board','status'))
    p.add_argument('--execute-board',action='store_true');a=p.parse_args()
    if a.command=='prepare':value=prepare()
    elif a.command=='validate':value=validate()
    elif a.command=='qemu':value=qemu()
    elif a.command=='board':value=board(execute_board=a.execute_board)
    else:value=read(RUN/'state.json') if (RUN/'state.json').exists() else dict(status='NOT_STARTED')
    print(json.dumps(value,ensure_ascii=False,indent=2))
