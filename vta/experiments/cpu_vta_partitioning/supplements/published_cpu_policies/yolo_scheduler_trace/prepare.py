from pathlib import Path
from copy import deepcopy
import os,shutil,subprocess,time
from cpu_vta_partitioning.common import digest,inventory,read,sha256,verify_inventory,write
from phase_profile.prepare import link_checked
from scheduler_gap_audit.prepare import configurations
from host_placement_check.prepare import validate_input_paths
from .common import ROOT,PARENT,protocol,sources,specs,freeze


def build(output):
    output=Path(output);dest=output/'build';src=dest/'sources';src.mkdir(parents=True,exist_ok=True)
    original=read(PARENT/'build/BUILD.json')
    for p,h in original['source_files'].items():
        if sha256(p)!=h:raise ValueError('original phase observer source changed')
        shutil.copy2(p,src/Path(p).name)
    shutil.copy2(Path(__file__).with_name('session_profile.h'),src/'vta_session_profile.h')
    p=src/'vta_stage_pipeline_runner_phase.cc';text=p.read_text()
    a='if (profile_completion_index == args.session_profile_first_completion) session_profile.Start();'
    b='          session_profile.Stop();\n        }\n        vta_workload_phase::Span collector_phase'
    if text.count(a)!=1 or text.count(b)!=1:raise ValueError('existing completion profile hook changed')
    new='''if (profile_completion_index == args.session_profile_first_completion) {
          std::vector<long> scheduler_targets;
          for (size_t j = 0; j < stage_affinities.size(); ++j) {
            if (args.stages[j].device == "cpu") {
              for (const auto& slot : stage_affinities[j].policy_slots) scheduler_targets.push_back(slot.tid);
            } else scheduler_targets.push_back(stage_affinities[j].caller_tid);
          }
          for (const auto& slot : policy_control_slots) if (slot.tid) scheduler_targets.push_back(slot.tid);
          session_profile.StartAt(frame->done_ms, scheduler_targets);
        }'''
    text=text.replace(a,new,1).replace(b,'          session_profile.StopAt(frame->done_ms);\n        }\n        vta_workload_phase::Span collector_phase',1)
    p.write_text(text);tc=original['toolchain']
    if sha256(tc['sdk_setup'])!=tc['sdk_setup_sha256']:raise ValueError('SDK setup changed')
    for t in list(tc['tools'].values())+list(tc.get('setup_files',{}).values()):
        if sha256(t['path'])!=t['sha256']:raise ValueError('SDK dependency changed')
    recipe=read(ROOT/'artifacts/cross_network_model_01/phase_profile/build/BUILD.json')['compiler_recipe']
    env=dict(os.environ);env.pop('LD_LIBRARY_PATH',None);tick=time.monotonic()
    argv=['bash','-c','source "$1" || exit $?; shift; exec "$@"','trace-sdk',tc['sdk_setup'],
        'bash','-c',recipe,'trace-compile',str(p),str(dest/'vta_stage_pipeline_runner_phase'),str(ROOT.parents[4]),str(PARENT/'yolov3_tiny416/capsule/runtime')]
    done=subprocess.run(argv,env=env,capture_output=True,text=True,timeout=180);(dest/'compile.log').write_text(done.stdout+done.stderr)
    if done.returncode:raise ValueError('optional scheduler observer build failed: '+str(dest/'compile.log'))
    proof=dict(runner_sha256=sha256(dest/'vta_stage_pipeline_runner_phase'),old_runner_sha256=original['runner_sha256'],
        adapter_sha256=original['adapter_sha256'],source_files={str(p):sha256(p) for p in src.iterdir()},toolchain=tc,
        runtime_rebuilt=False,modules_recompiled=False,seconds=time.monotonic()-tick,
        runner_change='existing optional session hooks record scheduler events, selected TIDs and existing completion anchors')
    write(dest/'BUILD.json',proof);return proof


def prepare(output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    if (output/'capsule/manifest.json').exists():return validate(output)
    built=read(output/'build/BUILD.json') if (output/'build/BUILD.json').exists() else build(output)
    parent=PARENT/'yolov3_tiny416/capsule';dest=output/'capsule';m=read(parent/'manifest.json')
    if digest({k:v for k,v in m.items() if k!='bundle_id'})[:24]!=m['bundle_id']:raise ValueError('parent capsule seal changed')
    verify_inventory(parent,m['files'])
    for name,row in m['files'].items():
        if name.startswith(('runtime/','firmware/','inputs/','references/','programs/')):link_checked(parent/name,dest/name,row['sha256'])
    for leaf in ('programs.json','inputs.json'):write(dest/'plans'/leaf,read(parent/'plans'/leaf))
    cfg=configurations('yolov3_tiny416')['pipelines']['overlap']
    write(dest/'plans/configurations.json',dict(pipelines=dict(overlap=cfg),components={}))
    for name,source,key in [('vta_stage_pipeline_runner_phase',output/'build/vta_stage_pipeline_runner_phase','runner_sha256'),
                          ('vta_stage_pipeline_runner',parent/'runner/vta_stage_pipeline_runner_phase','old_runner_sha256'),
                          ('libwait_adapter.so',parent/'runner/libwait_adapter.so','adapter_sha256')]:link_checked(source,dest/'runner'/name,built[key])
    write(dest/'plans/protocol.json',protocol());write(dest/'plans/tasks.json',specs())
    write(dest/'plans/board_execution.json',dict(boot_policy='same_boot_no_reboot',udmabuf_bytes=201326592,storage_margin_bytes=268435456))
    write(dest/'plans/preparation.json',dict(sources=sources(),build=built,parent_manifest_sha256=sha256(parent/'manifest.json')))
    assets=inventory(dest);payload={n:v for n,v in assets.items() if n.startswith(('runtime/','firmware/','inputs/','references/','programs/','runner/'))}
    payload['plans/board_execution.json']=assets['plans/board_execution.json']
    write(dest/'plans/board_payload.json',dict(files=payload,unpacked_bytes=sum(v['bytes'] for v in payload.values())))
    manifest=dict(experiment=protocol()['experiment'],network='yolov3_tiny416',phase_runner_sha256=built['runner_sha256'],old_runner_sha256=built['old_runner_sha256'],
        adapter_sha256=built['adapter_sha256'],runtime_sha256=sha256(dest/'runtime/libtvm_runtime.so'),files=inventory(dest))
    manifest['bundle_id']=digest(manifest)[:24];write(dest/'manifest.json',manifest)
    freeze(output/'PREPARATION_FREEZE.json',dict(bundle_id=manifest['bundle_id'],sources=sources(),protocol=protocol()))
    return validate(output)


def validate(output):
    output=Path(output);bundle=output/'capsule';m=read(bundle/'manifest.json')
    if digest({k:v for k,v in m.items() if k!='bundle_id'})[:24]!=m['bundle_id']:raise ValueError('scheduler capsule changed')
    verify_inventory(bundle,m['files']);validate_input_paths(bundle,m)
    seal=read(output/'PREPARATION_FREEZE.json')
    if seal['sources']!=sources() or seal['protocol']!=protocol() or seal['bundle_id']!=m['bundle_id']:raise ValueError('scheduler controller or protocol changed')
    return dict(validated=True,bundle_id=m['bundle_id'],native_sessions=5,DNN_inferences=146)


def qemu(output):
    output=Path(output).resolve();bundle=output/'capsule';validate(output)
    q=shutil.which('qemu-aarch64-static');runtime=bundle/'runtime';loader=runtime/'ld-linux-aarch64.so.1'
    if not q:raise ValueError('QEMU required')
    for name in ('vta_stage_pipeline_runner','vta_stage_pipeline_runner_phase'):
        p=subprocess.run([q,str(loader),'--library-path',str(runtime),str(bundle/'runner'/name),'--help'],capture_output=True,text=True,timeout=30)
        if p.returncode:raise ValueError('AArch64 loader check failed')
    from .runtime import make_task
    from cpu_vta_partitioning.measurement.tasks import native_command
    from cpu_vta_partitioning.measurement import frames,correctness
    task=make_task(bundle,specs()[0]);task['kind']='qualify_component';task['stages']=task['stages'][:1]
    task['configuration']=deepcopy(task['configuration']);task['configuration']['stages']=task['stages']
    raw=output/'qemu';raw.mkdir(exist_ok=True);argv,inputs=native_command(task,bundle,str(bundle),str(raw));(raw/'inputs.txt').write_text(inputs)
    at=argv.index(str(loader));args=[q]
    for value in argv[2:at]:args+=['-E',value]
    argv[argv.index(str(bundle/'runner/vta_stage_pipeline_runner'))]=str(bundle/'runner/vta_stage_pipeline_runner_phase')
    args+=argv[at:]+['--allow-cpu-affinity-superset','--cpu-worker-pinning','--resource-affinity-probe']
    p=subprocess.run(args,capture_output=True,text=True,timeout=180);(raw/'qemu.log').write_text(p.stdout+p.stderr)
    if p.returncode:raise ValueError('QEMU CPU execution failed')
    correctness(frames(raw/'result.jsonl',1),task,bundle,raw)
    proof=dict(passed=True,load_passed=True,CPU_output_passed=True,native_scheduler_events_not_tested_by_QEMU=True)
    write(output/'QEMU.json',proof);return proof
