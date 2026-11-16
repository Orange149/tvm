from copy import deepcopy
from pathlib import Path
import os,shutil,subprocess,time
from cpu_vta_partitioning.common import digest,inventory,read,sha256,verify_inventory,write
from phase_profile.prepare import link_checked
from scheduler_gap_audit.prepare import configurations
from .common import ROOT,PARENT,OUTPUT,freeze,protocol,sources,specs


def build(output):
    output=Path(output);dest=output/'build';src=dest/'sources';src.mkdir(parents=True,exist_ok=True)
    old=read(PARENT/'build/BUILD.json')
    for p,h in old['source_files'].items():
        if sha256(p)!=h:raise ValueError('frozen observer source changed')
        shutil.copy2(p,src/Path(p).name)
    # Only the already opt-in session counter implementation changes.
    shutil.copy2(Path(__file__).with_name('session_profile.h'),src/'vta_session_profile.h')
    runner_source=src/'vta_stage_pipeline_runner_phase.cc';text=runner_source.read_text()
    a='if (profile_completion_index == args.session_profile_first_completion) session_profile.Start();'
    b='          session_profile.Stop();\n        }\n        vta_workload_phase::Span collector_phase'
    if text.count(a)!=1 or text.count(b)!=1:raise ValueError('completion anchor source changed')
    text=text.replace(a,'if (profile_completion_index == args.session_profile_first_completion) session_profile.StartAt(frame->done_ms);',1)
    text=text.replace(b,'          session_profile.StopAt(frame->done_ms);\n        }\n        vta_workload_phase::Span collector_phase',1)
    runner_source.write_text(text)
    tc=old['toolchain']
    if sha256(tc['sdk_setup'])!=tc['sdk_setup_sha256']:raise ValueError('SDK changed')
    for row in list(tc['tools'].values())+list(tc.get('setup_files',{}).values()):
        if sha256(row['path'])!=row['sha256']:raise ValueError('SDK dependency changed')
    env=dict(os.environ);env.pop('LD_LIBRARY_PATH',None)
    recipe=read(ROOT/'artifacts/cross_network_model_01/phase_profile/build/BUILD.json')['compiler_recipe']
    tick=time.monotonic();binary=dest/'vta_stage_pipeline_runner_phase'
    cmd=['bash','-c','source "$1" || exit $?; shift; exec "$@"','placement-sdk',tc['sdk_setup'],
         'bash','-c',recipe,'placement-compile',str(src/'vta_stage_pipeline_runner_phase.cc'),str(binary),str(ROOT.parents[4]),str(PARENT/'yolov3_tiny416/capsule/runtime')]
    p=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=180)
    (dest/'compile.log').write_text(p.stdout+p.stderr)
    if p.returncode:raise ValueError('placement observer compile failed: '+str(dest/'compile.log'))
    value=dict(runner_sha256=sha256(binary),old_runner_sha256=old['runner_sha256'],
        source_files={str(p):sha256(p) for p in src.iterdir()},toolchain=tc,seconds=time.monotonic()-tick,
        runtime_rebuilt=False,stage_modules_recompiled=False,changed_header='vta_session_profile.h',
        runner_change='two opt-in session counter calls also record existing completion timestamps')
    write(dest/'BUILD.json',value);return value


def prepare(output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    if (output/'capsule/manifest.json').exists():return validate(output)
    built=read(output/'build/BUILD.json') if (output/'build/BUILD.json').exists() else build(output)
    parent=PARENT/'yolov3_tiny416/capsule';m=read(parent/'manifest.json');dest=output/'capsule'
    if digest({k:v for k,v in m.items() if k!='bundle_id'})[:24]!=m['bundle_id']:raise ValueError('parent capsule changed')
    verify_inventory(parent,m['files'])
    for name,row in m['files'].items():
        if name.startswith(('runtime/','firmware/','inputs/','references/','programs/')):
            link_checked(parent/name,dest/name,row['sha256'])
    for leaf in ('programs.json','inputs.json'):
        write(dest/'plans'/leaf,read(parent/'plans'/leaf))
    cfg=configurations('yolov3_tiny416')['pipelines']['overlap']
    write(dest/'plans/configurations.json',dict(pipelines=dict(overlap=cfg),components={}))
    link_checked(output/'build/vta_stage_pipeline_runner_phase',dest/'runner/vta_stage_pipeline_runner_phase',built['runner_sha256'])
    link_checked(parent/'runner/vta_stage_pipeline_runner_phase',dest/'runner/vta_stage_pipeline_runner',built['old_runner_sha256'])
    write(dest/'plans/protocol.json',protocol());write(dest/'plans/tasks.json',specs())
    write(dest/'plans/board_execution.json',dict(boot_policy='same_boot_no_reboot',udmabuf_bytes=201326592,storage_margin_bytes=268435456))
    write(dest/'plans/preparation.json',dict(sources=sources(),parent_manifest_sha256=sha256(parent/'manifest.json'),build=built))
    assets=inventory(dest);payload={n:v for n,v in assets.items() if n.startswith(('runtime/','firmware/','inputs/','references/','programs/','runner/'))}
    payload['plans/board_execution.json']=assets['plans/board_execution.json']
    write(dest/'plans/board_payload.json',dict(files=payload,unpacked_bytes=sum(v['bytes'] for v in payload.values())))
    manifest=dict(experiment=protocol()['experiment'],network='yolov3_tiny416',phase_runner_sha256=built['runner_sha256'],
        old_runner_sha256=built['old_runner_sha256'],adapter_sha256=None,
        runtime_sha256=sha256(dest/'runtime/libtvm_runtime.so'),files=inventory(dest))
    manifest['bundle_id']=digest(manifest)[:24];write(dest/'manifest.json',manifest)
    freeze(output/'PREPARATION_FREEZE.json',dict(bundle_id=manifest['bundle_id'],sources=sources(),protocol=protocol()))
    return validate(output)


def validate_input_paths(bundle,manifest):
    bundle=Path(bundle)
    for record in read(bundle/'plans/inputs.json')['records']:
        paths=[record['input']]+[v[k] for v in record['frontiers'].values() for k in ('raw','npy')]
        for name in paths:
            if name not in manifest['files'] or not (bundle/name).is_file():raise ValueError('input/reference not packaged: '+name)


def validate(output):
    output=Path(output);bundle=output/'capsule';m=read(bundle/'manifest.json')
    if digest({k:v for k,v in m.items() if k!='bundle_id'})[:24]!=m['bundle_id']:raise ValueError('capsule seal changed')
    verify_inventory(bundle,m['files'])
    validate_input_paths(bundle,m)
    seal=read(output/'PREPARATION_FREEZE.json')
    if seal['sources']!=sources() or seal['protocol']!=protocol() or seal['bundle_id']!=m['bundle_id']:raise ValueError('controller or protocol changed')
    return dict(validated=True,bundle_id=m['bundle_id'],native_sessions=5,DNN_inferences=314,runtime_rebuilt=False,module_rebuilt=False)


def qemu(output):
    output=Path(output).resolve();bundle=output/'capsule';validate(output)
    q=shutil.which('qemu-aarch64-static')
    if not q:raise ValueError('QEMU required')
    runtime=bundle/'runtime';loader=runtime/'ld-linux-aarch64.so.1'
    results=[]
    for name in ('vta_stage_pipeline_runner','vta_stage_pipeline_runner_phase'):
        p=subprocess.run([q,str(loader),'--library-path',str(runtime),str(bundle/'runner'/name),'--help'],capture_output=True,text=True,timeout=30)
        if p.returncode:raise ValueError('ARM loader check failed')
        results.append(dict(runner=name,load_passed=True))
    # Actual CPU numerical check with the new binary, no VTA execution.
    from cpu_vta_partitioning.measurement.tasks import native_command as base
    from .runtime import make_task
    spec=dict(network='yolov3_tiny416',scope='pipeline',case='overlap',**specs()[0])
    task=make_task(bundle,spec);task['kind']='qualify_component';task['stages']=task['stages'][:1]
    task['configuration']=deepcopy(task['configuration']);task['configuration']['stages']=task['stages']
    raw=output/'qemu';raw.mkdir(exist_ok=True)
    argv,inputs=base(task,bundle,str(bundle),str(raw));(raw/'inputs.txt').write_text(inputs)
    at=argv.index(str(loader));envs=argv[2:at]
    argv[argv.index(str(bundle/'runner/vta_stage_pipeline_runner'))]=str(bundle/'runner/vta_stage_pipeline_runner_phase')
    argv+=['--allow-cpu-affinity-superset','--cpu-worker-pinning','--resource-affinity-probe']
    args=[q]
    for e in envs:args+=['-E',e]
    args+=argv[at:]
    p=subprocess.run(args,capture_output=True,text=True,timeout=180);(raw/'qemu.log').write_text(p.stdout+p.stderr)
    if p.returncode:raise ValueError('QEMU CPU output run failed: '+str(raw/'qemu.log'))
    from cpu_vta_partitioning.measurement import frames,correctness
    rows=frames(raw/'result.jsonl',1);correctness(rows,task,bundle,raw)
    value=dict(passed=True,runners=results,CPU_output_passed=True,native_perf_not_verified_by_QEMU=True)
    write(output/'QEMU.json',value);return value
