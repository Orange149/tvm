"""Prospective small-pool verification using an unchanged frozen executor."""
from copy import deepcopy
from pathlib import Path
from types import FunctionType
import random,shlex,shutil,subprocess,time
from cpu_vta_partitioning.common import read,write,sha256,digest,inventory,verify_inventory,locked,now,sync_tree
from cpu_vta_partitioning.transport import SSH
from phase_profile.prepare import link_checked
from phase_profile import runtime as phase
from phase_profile.campaign import accept as original_accept
from host_placement_check.prepare import validate_input_paths
from yolo_validation.components import hardware_check
from finite_service.common import ROOT,OUTPUT,CONFIG,freeze

POOL=OUTPUT/'prospective'
RUN=ROOT/'runs/finite_service_holdout_01'


def protocol():
    return dict(experiment='finite_service_holdout_01',input_index=0,
        qualification=dict(warmup=0,scored=1,tail=0),pipeline=dict(warmup=8,scored=16,tail=4),
        environment=dict(cores=4,frequency_khz=1066666,poll_sleep_ns=1000,post_start_sleep_ns=1000,
            spin_count=0,exclude_worker0=1,bind_threads=1),seed=20261111,
        no_reboot=True,no_observer=True,no_anchor=True,no_drift_rejection=True,no_automatic_repeat=True,
        runtime_changed=False,executor_changed=False,only_local_pool_validation=True)


def sources():
    files=[Path(__file__),ROOT/'run_finite_service_holdout.py',ROOT/'tests/test_finite_service_holdout_board.py',
        ROOT/'phase_profile/runtime.py',ROOT/'phase_profile/campaign.py',
        ROOT.parents[1]/'src/cpu_vta_partitioning/transport/__init__.py',
        ROOT.parents[1]/'src/cpu_vta_partitioning/measurement/tasks.py']
    return {str(p):sha256(p) for p in files}


def selection():
    seal=read(POOL/'SELECTION_FREEZE.json');plan=read(POOL/'PLAN.json')
    if sha256(POOL/'PLAN.json')!=seal['plan_sha256'] or sha256(ROOT/'prepare_finite_service_holdout.py')!=plan['source_sha256']:
        raise ValueError('prospective selection identity changed')
    for network,row in seal['networks'].items():
        for p,h in row['prediction_files'].items():
            if sha256(POOL/p)!=h:raise ValueError('prospective predictions changed')
    if sum(len(v) for v in seal['selected_configurations'].values())>8:raise ValueError('native pool exceeds frozen limit')
    return seal


def specs():
    selected=selection()['selected_configurations'];pairs=[(n,c['id']) for n,rows in selected.items() for c in rows]
    random.Random(protocol()['seed']).shuffle(pairs)
    return [dict(network=n,case=cid,scope='pipeline',phase=p,runner='phase',probe=False)
            for n,cid in pairs for p in ('qualify','clean')]


def prepare():
    selected=selection()['selected_configurations'];valid={}
    boot=read(OUTPUT/'PLATFORM_RECEIPT.json')['boot_id']
    freeze(POOL/'BOARD_PLAN.json',dict(sources=sources(),protocol=protocol(),specs=specs(),expected_boot_id=boot,
        selection_sha256=sha256(POOL/'SELECTION_FREEZE.json'),
        DNN_inferences=sum(len(v) for v in selected.values())*29))
    for network,configs in selected.items():
        target=POOL/network/'capsule'
        if (target/'manifest.json').exists():valid[network]=validate(target);continue
        source=ROOT/'artifacts/scheduler_gap_audit_01'/network/'capsule';m=read(source/'manifest.json')
        if digest({k:v for k,v in m.items() if k!='bundle_id'})[:24]!=m['bundle_id']:raise ValueError('frozen source capsule differs')
        verify_inventory(source,m['files'])
        needed={s['program'] for c in configs for s in c['stages']};programs=read(source/'plans/programs.json')
        for rel,row in m['files'].items():
            use=rel.startswith(('runtime/','runner/','firmware/','inputs/','references/'))
            use=use or (rel.startswith('programs/') and rel.split('/')[1] in needed)
            if use:link_checked(source/rel,target/rel,row['sha256'])
        write(target/'plans/programs.json',{name:programs[name] for name in sorted(needed)})
        write(target/'plans/inputs.json',read(source/'plans/inputs.json'))
        write(target/'plans/configurations.json',dict(pipelines={c['id']:c for c in configs},components={}))
        write(target/'plans/protocol.json',protocol())
        write(target/'plans/board_execution.json',dict(boot_policy='one_boot_no_automatic_reboot',
            udmabuf_bytes=201326592,storage_margin_bytes=268435456))
        write(target/'plans/preparation.json',dict(sources=sources(),source_manifest=str(source/'manifest.json'),
            source_manifest_sha256=sha256(source/'manifest.json'),selection_sha256=sha256(POOL/'SELECTION_FREEZE.json'),
            new_modules=0,new_runtimes=0,new_executors=0))
        assets=inventory(target);payload={n:v for n,v in assets.items() if n.startswith(('runtime/','runner/','firmware/','inputs/','references/','programs/'))}
        payload['plans/board_execution.json']=assets['plans/board_execution.json']
        write(target/'plans/board_payload.json',dict(files=payload,unpacked_bytes=sum(v['bytes'] for v in payload.values())))
        new=dict(experiment=protocol()['experiment'],network=network,
            **{k:m[k] for k in ('phase_runner_sha256','old_runner_sha256','adapter_sha256','runtime_sha256')},files=inventory(target))
        new['bundle_id']=digest(new)[:24];write(target/'manifest.json',new);valid[network]=validate(target)
    return valid


def validate(bundle):
    bundle=Path(bundle);m=read(bundle/'manifest.json')
    if digest({k:v for k,v in m.items() if k!='bundle_id'})[:24]!=m['bundle_id']:raise ValueError('holdout capsule seal differs')
    verify_inventory(bundle,m['files']);validate_input_paths(bundle,m)
    p=read(bundle/'plans/preparation.json')
    if p['sources']!=sources() or read(bundle/'plans/protocol.json')!=protocol():raise ValueError('holdout controller changed')
    if sha256(p['source_manifest'])!=p['source_manifest_sha256'] or sha256(POOL/'SELECTION_FREEZE.json')!=p['selection_sha256']:
        raise ValueError('holdout source capsule/selection changed')
    return dict(validated=True,bundle_id=m['bundle_id'],network=m['network'])


def make_task(bundle,spec):
    bundle=Path(bundle);m=read(bundle/'manifest.json')
    if spec['network']!=m['network'] or spec['scope']!='pipeline' or spec['runner']!='phase' or spec['probe']:
        raise ValueError('unexpected holdout execution mode')
    if spec['phase'] not in ('qualify','clean'):raise ValueError('unexpected holdout phase')
    cfg=read(bundle/'plans/configurations.json')['pipelines'][spec['case']];q=spec['phase']=='qualify'
    doc=dict(kind='qualify_pipeline' if q else 'performance',configuration=deepcopy(cfg),stages=deepcopy(cfg['stages']),
        spec=deepcopy(spec),environment=protocol()['environment'],**protocol()['qualification' if q else 'pipeline'],input_indices=[0],
        executor_sha256=m['phase_runner_sha256'],adapter_sha256=None,runtime_sha256=m['runtime_sha256'],
        bundle_id=m['bundle_id'],experiment=protocol()['experiment'],startup_affinity_probe=True)
    doc['task_id']=digest(doc)[:24];return doc


def native_command(doc,bundle,remote_bundle,remote_job):
    fn=FunctionType(phase.native_command.__code__,{**phase.native_command.__globals__,'make_task':make_task})
    return fn(doc,bundle,remote_bundle,remote_job)


summary=phase.summary
accept=original_accept


def matched_ssh(config,*,execute_board=False):
    fn=FunctionType(SSH.measure.__code__,{**SSH.measure.__globals__,'native_command':native_command})
    return type('FiniteHoldoutSSH',(SSH,),{'measure':fn})(config,execute_board=execute_board)


def qemu():
    from cpu_vta_partitioning.measurement.tasks import native_command as base
    from cpu_vta_partitioning.measurement import frames,correctness
    q=shutil.which('qemu-aarch64-static')
    if not q:raise ValueError('QEMU required')
    results={}
    for network in selection()['selected_configurations']:
        bundle=POOL/network/'capsule';validate(bundle)
        spec=next(s for s in specs() if s['network']==network and s['phase']=='qualify')
        task=make_task(bundle,spec);task['kind']='qualify_component';task['stages']=task['stages'][:1]
        task['configuration']=deepcopy(task['configuration']);task['configuration']['stages']=task['stages']
        raw=POOL/network/'qemu';raw.mkdir(exist_ok=True)
        argv,inputs=base(task,bundle,str(bundle),str(raw));(raw/'inputs.txt').write_text(inputs)
        loader=str(bundle/'runtime/ld-linux-aarch64.so.1');at=argv.index(loader);args=[q]
        for env in argv[2:at]:args+=['-E',env]
        argv[argv.index(str(bundle/'runner/vta_stage_pipeline_runner'))]=str(bundle/'runner/vta_stage_pipeline_runner_phase')
        args+=argv[at:]+['--allow-cpu-affinity-superset','--cpu-worker-pinning','--resource-affinity-probe']
        p=subprocess.run(args,capture_output=True,text=True,timeout=180);(raw/'qemu.log').write_text(p.stdout+p.stderr)
        if p.returncode:raise ValueError('holdout QEMU CPU output failed')
        correctness(frames(raw/'result.jsonl',1),task,bundle,raw);results[network]=dict(passed=True,CPU_output_passed=True)
    write(POOL/'QEMU.json',results);return results


def board(*,execute_board=False):
    if not execute_board:raise ValueError('--execute-board is required')
    plan=read(POOL/'BOARD_PLAN.json')
    if plan['sources']!=sources() or plan['selection_sha256']!=sha256(POOL/'SELECTION_FREEZE.json'):raise ValueError('holdout board plan changed')
    checks=read(POOL/'QEMU.json')
    if not all(v['passed'] for v in checks.values()):raise ValueError('QEMU required')
    ssh=matched_ssh(read(CONFIG)['board'],execute_board=True);boot=ssh.boot_id()
    if boot!=plan['expected_boot_id']:raise ValueError('holdout boot changed; do not mix measurements')
    started=time.monotonic()
    with locked(RUN):
        freeze(RUN/'run.json',dict(boot_id=boot,board_plan_sha256=sha256(POOL/'BOARD_PLAN.json'),protocol=protocol()))
        def state(status,**kw):write(RUN/'state.json',dict(status=status,boot_id=boot,updated_at=now(),**kw),replace=True)
        try:
            remote={}
            for network in selection()['selected_configurations']:
                (RUN/network).mkdir(parents=True,exist_ok=True)
                b=POOL/network/'capsule';validate(b);remote[network]=ssh.deploy(b,RUN/network)
                ssh.qualify_environment(remote[network],boot,RUN/network,b)
                write(RUN/network/'hardware.json',hardware_check(ssh,b,boot),replace=True)
            expected={};count=0;todo=specs()
            for index,spec in enumerate(todo,1):
                n=spec['network'];b=POOL/n/'capsule';doc=make_task(b,spec)
                state('RUNNING',accepted_sessions=index-1,total_sessions=len(todo),current_spec=spec)
                if spec['phase']=='clean' and (n,spec['case']) not in expected:raise ValueError('performance before qualification')
                item=accept(ssh,doc,b,remote[n],RUN,boot,expected.get((n,spec['case'])))
                if spec['phase']=='qualify':expected[n,spec['case']]=item['summary']['output_fingerprints']
                count+=sum(doc[k] for k in ('warmup','scored','tail'))
                print('未见配置共同实测',index,'/',len(todo),n,spec['phase'],flush=True)
            if count!=plan['DNN_inferences']:raise ValueError('holdout inference budget mismatch')
            for n,r in remote.items():
                ssh.idle_command('cd '+shlex.quote(r)+' && sha256sum -c board.sha256 > holdout_post_verify.log && sync',boot=boot,timeout=300)
            if ssh.boot_id()!=boot:raise ValueError('holdout boot changed after execution')
            state('COMPLETE',accepted_sessions=len(todo),total_sessions=len(todo),DNN_inferences=count,
                seconds=time.monotonic()-started,no_reboot=True,no_automatic_extra_measurement=True);sync_tree(RUN)
        except Exception as e:
            state('STOPPED_ON_ERROR',error_type=type(e).__name__,error=str(e));raise
    return read(RUN/'state.json')
