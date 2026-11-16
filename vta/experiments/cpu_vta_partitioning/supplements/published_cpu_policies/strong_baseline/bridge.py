"""Same historical configuration, separate executor and output-accounting factors."""
from copy import deepcopy
from pathlib import Path
from types import FunctionType
import json,random,shlex,shutil,subprocess,time
import numpy as np
from cpu_vta_partitioning.common import read,write,sha256,digest,inventory,verify_inventory,locked,now,sync_tree
from cpu_vta_partitioning.transport import SSH
from cpu_vta_partitioning.measurement import frames,input_sequence,correctness
from cpu_policies.open_resources.runtime import _affinity
from phase_profile.prepare import link_checked
from phase_profile import runtime as phase
from phase_profile.campaign import accept as original_accept
from host_placement_check.prepare import validate_input_paths
from yolo_validation.components import hardware_check
from finite_service.common import freeze
from .audit import ROOT,OUTPUT,CURRENT

DEST=OUTPUT/'bridge'
RUN=ROOT/'runs/resnet18_strong_baseline_bridge_01'
CONFIG=ROOT/'artifacts/cross_network_model_01/board.local.json'
VARIANTS=(('historical_classification','old','classification'),
          ('current_classification','phase','classification'),('current_raw','phase','raw'))


def protocol():
    return dict(experiment='resnet18_strong_baseline_bridge_01',input_index=0,
        qualification=dict(warmup=0,scored=1,tail=0),pipeline=dict(warmup=8,scored=16,tail=4),
        environment=dict(cores=4,frequency_khz=1066666,poll_sleep_ns=1000,post_start_sleep_ns=1000,
            spin_count=0,exclude_worker0=1,bind_threads=1),variants=VARIANTS,
        max_native_sessions=6,max_DNN_inferences=87,seed=20261111,
        hypothesis='separate historical/current executor identity from classification/raw result processing on the exact 13FPS layout',
        no_reboot=True,no_anchor=True,no_drift_rejection=True,no_automatic_repeat=True,
        no_runtime_module_or_score_change=True,no_new_profile=True,
        qualification_full_tensor=True,classification_performance_full_hash_unavailable=True,
        qualification_logical_worker_probe=True,performance_logical_probe=False,
        no_performance_acceptance_threshold=True)


def sources():
    paths=[Path(__file__),Path(__file__).with_name('audit.py'),ROOT/'run_strong_baseline_bridge.py',
        ROOT/'tests/test_strong_baseline_bridge.py',ROOT/'phase_profile/runtime.py',ROOT/'phase_profile/campaign.py',
        ROOT.parents[1]/'src/cpu_vta_partitioning/measurement/tasks.py']
    return {str(p):sha256(p) for p in paths}


def baseline():
    p=read(OUTPUT/'BASELINE_FREEZE.json')
    if p['source_sha256']!=sha256(Path(__file__).with_name('audit.py')):
        raise ValueError('baseline auditor changed after evidence freeze')
    if sha256(p['current_parent']+'/manifest.json')!=p['current_parent_sha256']:
        raise ValueError('bridge assets parent differs')
    return p


def specs():
    variants=list(VARIANTS);random.Random(protocol()['seed']).shuffle(variants)
    return [dict(network='resnet18',scope='pipeline',case=baseline()['configuration']['id'],
        variant=v,runner=r,phase=p,probe=False,performance_output=mode)
        for v,r,mode in variants for p in ('qualify','clean')]


def prepare():
    b=baseline();target=DEST/'capsule'
    if (target/'manifest.json').exists():return validate(target)
    source=CURRENT;m=read(source/'manifest.json');verify_inventory(source,m['files'])
    cfg=b['configuration'];needed={s['program'] for s in cfg['stages']}
    for leaf,info in m['files'].items():
        if leaf.startswith(('runtime/','runner/','firmware/','inputs/','references/')) or leaf.startswith('programs/') and leaf.split('/')[1] in needed:
            link_checked(source/leaf,target/leaf,info['sha256'])
    programs=read(source/'plans/programs.json')
    write(target/'plans/programs.json',{n:programs[n] for n in needed})
    write(target/'plans/inputs.json',read(source/'plans/inputs.json'))
    write(target/'plans/configurations.json',dict(pipelines={cfg['id']:cfg},components={}))
    write(target/'plans/protocol.json',protocol())
    write(target/'plans/board_execution.json',read(source/'plans/board_execution.json'))
    write(target/'plans/preparation.json',dict(sources=sources(),baseline_sha256=sha256(OUTPUT/'BASELINE_FREEZE.json'),
        current_parent_sha256=b['current_parent_sha256'],new_modules=0,new_runtime=0,new_executor=0))
    assets=inventory(target);payload={k:v for k,v in assets.items() if k.startswith(('runtime/','runner/','firmware/','inputs/','references/','programs/'))}
    payload['plans/board_execution.json']=assets['plans/board_execution.json']
    write(target/'plans/board_payload.json',dict(files=payload,unpacked_bytes=sum(v['bytes'] for v in payload.values())))
    new=dict(experiment=protocol()['experiment'],network='resnet18',
        **{k:m[k] for k in ('phase_runner_sha256','old_runner_sha256','adapter_sha256','runtime_sha256')},files=inventory(target))
    new['bundle_id']=digest(new)[:24];write(target/'manifest.json',new)
    previous=read(ROOT/'runs/device_completion_holdout_01/state.json')
    freeze(DEST/'BOARD_PLAN.json',dict(sources=sources(),protocol=protocol(),specs=specs(),
        baseline_sha256=sha256(OUTPUT/'BASELINE_FREEZE.json'),bundle_sha256=sha256(target/'manifest.json'),
        expected_boot_id=previous['boot_id'],sessions=6,DNN_inferences=87,
        classification_full_output_checked_in_qualification_only=True))
    return validate(target)


def validate(bundle):
    bundle=Path(bundle);m=read(bundle/'manifest.json');p=read(bundle/'plans/preparation.json')
    if digest({k:v for k,v in m.items() if k!='bundle_id'})[:24]!=m['bundle_id']:raise ValueError('bridge manifest differs')
    verify_inventory(bundle,m['files']);validate_input_paths(bundle,m)
    if p['sources']!=sources() or p['baseline_sha256']!=sha256(OUTPUT/'BASELINE_FREEZE.json'):
        raise ValueError('bridge source or baseline changed')
    return dict(validated=True,bundle_id=m['bundle_id'])


def make_task(bundle,spec):
    if (spec['variant'],spec['runner'],spec['performance_output']) not in VARIANTS or spec['probe']:
        raise ValueError('unknown bridge execution variant')
    m=read(Path(bundle)/'manifest.json');cfg=baseline()['configuration'];q=spec['phase']=='qualify'
    if spec['case']!=cfg['id'] or spec['scope']!='pipeline' or spec['phase'] not in ('qualify','clean'):
        raise ValueError('bridge configuration or window changed')
    doc=dict(kind='qualify_pipeline' if q else 'performance',configuration=deepcopy(cfg),stages=deepcopy(cfg['stages']),
        spec=deepcopy(spec),environment=protocol()['environment'],**protocol()['qualification' if q else 'pipeline'],
        output_mode='raw' if q else spec['performance_output'],input_indices=[0],
        executor_sha256=m['old_runner_sha256' if spec['runner']=='old' else 'phase_runner_sha256'],
        adapter_sha256=None,runtime_sha256=m['runtime_sha256'],bundle_id=m['bundle_id'],
        experiment=protocol()['experiment'],startup_affinity_probe=q)
    doc['task_id']=digest(doc)[:24];return doc


def native_command(doc,bundle,remote,job):
    fn=FunctionType(phase.native_command.__code__,{**phase.native_command.__globals__,'make_task':make_task})
    argv,data=fn(doc,bundle,remote,job)
    argv[argv.index('--output-mode')+1]=doc['output_mode']
    if not doc['startup_affinity_probe']:argv.remove('--resource-affinity-probe')
    return argv,data


def summary(raw,doc,bundle):
    raw,bundle=Path(raw),Path(bundle);rows=frames(raw/'result.jsonl',sum(doc[k] for k in ('warmup','scored','tail')))
    input_sequence(rows,doc);q=doc['kind'].startswith('qualify')
    ledger,idle=_affinity(read(raw/'affinity.json'),doc['configuration'],q)
    if any(r.get('output_mode')!=doc['output_mode'] or r.get('queue_timing_enabled') is not False
        or r.get('defer_result_write_enabled') is not True or r.get('stage_count')!=7 for r in rows):
        raise ValueError('bridge output/timing scope differs')
    record=read(bundle/'plans/inputs.json')['records'][0]
    final=doc['stages'][-1]['program'];node=str(read(bundle/'plans/programs.json')[final]['output_node_ids'][0])
    expected_top1=int(np.argmax(np.load(bundle/record['frontiers'][node]['npy'])))
    fingerprint=None
    if doc['output_mode']=='raw':
        signatures=[[[o['index'],o['shape'],o['dtype'],o['fnv1a64']] for o in r.get('raw_outputs',[])] for r in rows]
        if not signatures or any(len(s)!=1 or s!=signatures[0] for s in signatures):raise ValueError('bridge full output fingerprint missing/changes')
        fingerprint=signatures[0]
    elif any(type(r.get('top1')) is not int or r['top1']!=expected_top1 for r in rows):
        raise ValueError('historical classification output differs from shared reference')
    if q:correctness(rows,doc,bundle,raw);value=dict(correct=True)
    else:
        kept=rows[doc['warmup']:doc['warmup']+doc['scored']];times=[r['completion_ms'] for r in kept]
        if any(b<=a for a,b in zip(times,times[1:])):raise ValueError('bridge completion timestamps invalid')
        interval=(times[-1]-times[0])/(len(times)-1)
        value=dict(correct=True,ii_ms=interval,fps=1000/interval,kept_timestamps=len(times),gaps=len(times)-1,
            process_core_ms_per_completion=(kept[-1]['completion_process_cpu_ms']-kept[0]['completion_process_cpu_ms'])/(len(times)-1),
            stage_service_ms=[sum(r[f'stage{i}_ms'] for r in kept)/len(kept) for i in range(7)])
    return dict(**value,output_fingerprints=fingerprint,expected_top1=expected_top1,
        full_tensor_output_observed=doc['output_mode']=='raw',classification_only_when_requested=doc['output_mode']=='classification',
        thread_ledger=ledger,idle_auxiliary_threads=idle)


def qemu():
    from cpu_vta_partitioning.measurement.tasks import native_command as base
    bundle=DEST/'capsule';validate(bundle);results={}
    for runner in ('old','phase'):
        spec=next(s for s in specs() if s['runner']==runner and s['phase']=='qualify');doc=make_task(bundle,spec)
        doc=deepcopy(doc);doc['kind']='qualify_component';doc['stages']=doc['stages'][:1];doc['configuration']['stages']=doc['stages']
        raw=DEST/('qemu_'+runner);raw.mkdir(exist_ok=True)
        argv,data=base(doc,bundle,str(bundle),str(raw));(raw/'inputs.txt').write_text(data)
        if runner=='phase':argv[argv.index(str(bundle/'runner/vta_stage_pipeline_runner'))]=str(bundle/'runner/vta_stage_pipeline_runner_phase')
        at=argv.index(str(bundle/'runtime/ld-linux-aarch64.so.1'));args=[shutil.which('qemu-aarch64-static')]
        for x in argv[2:at]:args+=['-E',x]
        args+=argv[at:]+['--allow-cpu-affinity-superset','--cpu-worker-pinning','--resource-affinity-probe']
        p=subprocess.run(args,capture_output=True,text=True,timeout=180);(raw/'qemu.log').write_text(p.stdout+p.stderr)
        if p.returncode:raise ValueError('bridge ARM loading/CPU output failed')
        correctness(frames(raw/'result.jsonl',1),doc,bundle,raw);results[runner]=dict(passed=True,CPU_output=True)
    write(DEST/'QEMU.json',results,replace=True);return results


def board(*,execute_board=False):
    if not execute_board:raise ValueError('--execute-board required')
    plan=read(DEST/'BOARD_PLAN.json');bundle=DEST/'capsule';validate(bundle)
    if plan['sources']!=sources() or not all(v['passed'] for v in read(DEST/'QEMU.json').values()):raise ValueError('bridge qualification changed')
    measure=FunctionType(SSH.measure.__code__,{**SSH.measure.__globals__,'native_command':native_command})
    ssh=type('StrongBaselineSSH',(SSH,),{'measure':measure})(read(CONFIG)['board'],execute_board=True)
    boot=ssh.boot_id()
    if boot!=plan['expected_boot_id']:raise ValueError('boot changed; stop before mixing batches')
    accept=FunctionType(original_accept.__code__,{**original_accept.__globals__,'summary':summary})
    start=time.monotonic()
    with locked(RUN):
        freeze(RUN/'run.json',dict(boot_id=boot,plan_sha256=sha256(DEST/'BOARD_PLAN.json')))
        remote=ssh.deploy(bundle,RUN);ssh.qualify_environment(remote,boot,RUN,bundle)
        write(RUN/'hardware.json',hardware_check(ssh,bundle,boot),replace=True)
        expected={};count=0
        for i,spec in enumerate(plan['specs'],1):
            doc=make_task(bundle,spec)
            fp=None if spec['phase']=='qualify' or doc['output_mode']=='classification' else expected[spec['variant']]
            item=accept(ssh,doc,bundle,remote,RUN,boot,fp)
            if spec['phase']=='qualify':expected[spec['variant']]=item['summary']['output_fingerprints']
            count+=sum(doc[k] for k in ('warmup','scored','tail'))
            write(RUN/'state.json',dict(status='RUNNING',accepted_sessions=i,total_sessions=6,DNN_inferences=count,boot_id=boot),replace=True)
            print('13FPS同配置桥接',i,'/6',spec['variant'],spec['phase'],flush=True)
        if count!=87:raise ValueError('bridge inference budget differs')
        ssh.idle_command('cd '+shlex.quote(remote)+' && sha256sum -c board.sha256 > bridge_post_verify.log && sync',boot=boot,timeout=180)
        if ssh.boot_id()!=boot:raise ValueError('boot changed after bridge')
        state=dict(status='COMPLETE',accepted_sessions=6,DNN_inferences=count,boot_id=boot,
            seconds=time.monotonic()-start,no_reboot=True,no_automatic_repeat=True)
        write(RUN/'state.json',state,replace=True);sync_tree(RUN);return state
