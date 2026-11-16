"""Small self-contained capsules; build only optional observation binaries."""
from copy import deepcopy
from pathlib import Path
import os
import random
import shutil
import subprocess
import time
from cpu_vta_partitioning.common import digest,inventory,read,sha256,verify_inventory,write
from cpu_policies.open_resources import domain
from common_top20.prepare import runtime_program
from phase_profile.prepare import COMPONENT_PARENTS,link_checked,configurations as old_configurations
from phase_profile.patch import patch as phase_patch
from .common import ROOT,PARENT,freeze,protocol,sources
from .observer_patch import adapter_source,phase_header,control_patch


def configurations(network):
    if network=='yolov3_tiny416':
        pipes=old_configurations(network)['pipelines']
        components={f'front_t{n}':domain.component(domain.stage('cpu',0,3,n,list(range(n)))) for n in (3,4)}
    else:
        ranking=read(PARENT/'common_top20/resnet18/RANKINGS.json')
        first=ranking['lists']['current'][0]
        pipes={'static_first':{k:deepcopy(v) for k,v in first.items() if k not in ('score_ms','score_ns','static_rank')}}
        components={'front_t3':domain.component(domain.stage('cpu',0,3,3,[0,1,2]))}
    for cfg in list(pipes.values())+list(components.values()):
        domain.validate_candidate(cfg,component=len(cfg['stages'])==1,unit_count=21 if network=='resnet18' else 14)
    return dict(components=components,pipelines=pipes)


def specs():
    result=[]
    for scope in ('component','pipeline'):
        cases=[(n,k) for n in ('resnet18','yolov3_tiny416') for k in configurations(n)['components' if scope=='component' else 'pipelines']]
        rng=random.Random(protocol()['seed']+(scope=='pipeline'));rng.shuffle(cases)
        for n,k in cases:
            phases=['clean','observed'];rng.shuffle(phases)
            for phase in ['qualify']+phases:
                result.append(dict(network=n,scope=scope,case=k,phase=phase,probe=phase!='clean',runner='phase'))
    if len(result)!=18:raise ValueError('session bound changed')
    return result


def build(output):
    output=Path(output);dest=output/'build';dest.mkdir(parents=True,exist_ok=True)
    tc=read(PARENT/'phase_profile/build/BUILD.json')['toolchain']
    if sha256(tc['sdk_setup'])!=tc['sdk_setup_sha256']:raise ValueError('SDK changed')
    for v in tc['tools'].values():
        if sha256(v['path'])!=v['sha256']:raise ValueError('cross compiler changed')
    src=dest/'sources';src.mkdir(exist_ok=True)
    pinned=ROOT/'artifacts/bundles/abbcccc62af230ee844745ac/sources/vta/apps/native_deploy'
    for f in pinned.glob('*.h'):shutil.copy2(f,src/f.name)
    (src/'vta_stage_pipeline_runner_phase.cc').write_text(control_patch(phase_patch((pinned/'vta_stage_pipeline_runner.cc').read_text())))
    (src/'phase_probe.h').write_text(phase_header())
    (src/'adapter.cc').write_text(adapter_source())
    shutil.copy2(Path(__file__).with_name('taskstats_client.h'),src/'taskstats_client.h')
    runtime=PARENT/'phase_profile/resnet18/capsule/runtime'
    env=dict(os.environ);env.pop('LD_LIBRARY_PATH',None);started=time.monotonic()
    recipe=read(PARENT/'phase_profile/build/BUILD.json')['compiler_recipe']
    commands=[('runner',recipe,[str(src/'vta_stage_pipeline_runner_phase.cc'),str(dest/'vta_stage_pipeline_runner_phase'),str(ROOT.parents[4]),str(runtime)]),
              ('adapter','exec $CXX -std=c++17 -O2 -shared -fPIC "$1" -o "$2" -ldl -pthread',
               [str(src/'adapter.cc'),str(dest/'libwait_adapter.so')])]
    for name,command,args in commands:
        p=subprocess.run(['bash','-c','source "$1" || exit $?; shift; exec "$@"','scheduler-sdk',tc['sdk_setup'],
                          'bash','-c',command,'scheduler-compile',*args],env=env,capture_output=True,text=True,timeout=180)
        (dest/(name+'_compile.log')).write_text(p.stdout+p.stderr)
        if p.returncode:raise RuntimeError('optional '+name+' build failed: '+str(dest/(name+'_compile.log')))
    proof=dict(seconds=time.monotonic()-started,runner_sha256=sha256(dest/'vta_stage_pipeline_runner_phase'),
        adapter_sha256=sha256(dest/'libwait_adapter.so'),source_files={str(p):sha256(p) for p in src.iterdir()},
        old_runner_sha256='25833e7f8b47fc8824ff9460afc0b968f24934dcee41c3b1f28619a99b6b6fe6',
        runtime_rebuilt=False,stage_modules_recompiled=False,toolchain=tc)
    write(dest/'BUILD.json',proof,replace=True);return proof


def prepare(output):
    output=Path(output).resolve();freeze(output/'TASKS.json',specs())
    if not (output/'build/BUILD.json').exists():build(output)
    built=read(output/'build/BUILD.json');controller=sources();bundles={}
    for network,parent in COMPONENT_PARENTS.items():
        dest=output/network/'capsule'
        if (dest/'manifest.json').exists():bundles[network]=validate_bundle(dest);continue
        m=read(parent/'manifest.json');catalog=read(parent/'plans/programs.json');cfg=configurations(network)
        programs={s['program']:runtime_program(catalog[s['program']]) for section in cfg.values() for c in section.values() for s in c['stages']}
        record=deepcopy(read(parent/'plans/inputs.json')['records'][0])
        needed={str(n) for p in programs.values() for n in p['input_node_ids']+p['output_node_ids']}
        record['frontiers']={n:v for n,v in record['frontiers'].items() if n in needed}
        files={record['input']}
        for v in record['frontiers'].values():files.update(v[k] for k in ('raw','npy'))
        for name in files:link_checked(parent/name,dest/name,m['files'][name]['sha256'])
        for name,p in programs.items():
            for leaf,value in p['files'].items():
                rel='programs/'+name+'/'+leaf;link_checked(parent/rel,dest/rel,value['sha256'])
        for name,v in m['files'].items():
            if name.startswith(('runtime/','firmware/')):link_checked(parent/name,dest/name,v['sha256'])
        old=PARENT/'common_top20'/network/'capsule/runner/vta_stage_pipeline_runner'
        link_checked(old,dest/'runner/vta_stage_pipeline_runner',built['old_runner_sha256'])
        for name,expected in [('vta_stage_pipeline_runner_phase',built['runner_sha256']),('libwait_adapter.so',built['adapter_sha256'])]:
            link_checked(output/'build'/name,dest/'runner'/name,expected)
        write(dest/'plans/programs.json',programs);write(dest/'plans/configurations.json',cfg)
        write(dest/'plans/inputs.json',dict(records=[record],available_indices=[0]))
        write(dest/'plans/protocol.json',protocol());write(dest/'plans/tasks.json',specs())
        write(dest/'plans/board_execution.json',dict(boot_policy='one_boot_no_reboot',udmabuf_bytes=201326592,storage_margin_bytes=268435456))
        write(dest/'plans/preparation.json',dict(controllers=controller,build=built,
            parent_manifest_sha256=sha256(parent/'manifest.json'),runtime_rebuilds=0,module_rebuilds=0))
        for path in controller:
            src=Path(path);target=dest/'sources/scheduler_gap_audit'/src.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target)
        assets=inventory(dest);payload={n:v for n,v in assets.items() if n.startswith(('runtime/','runner/','firmware/','inputs/','programs/'))}
        payload['plans/board_execution.json']=assets['plans/board_execution.json']
        write(dest/'plans/board_payload.json',dict(files=payload,unpacked_bytes=sum(v['bytes'] for v in payload.values())))
        manifest=dict(experiment=protocol()['experiment'],network=network,phase_runner_sha256=built['runner_sha256'],
            adapter_sha256=built['adapter_sha256'],old_runner_sha256=built['old_runner_sha256'],
            runtime_sha256=sha256(dest/'runtime/libtvm_runtime.so'),files=inventory(dest))
        manifest['bundle_id']=digest(manifest)[:24];write(dest/'manifest.json',manifest)
        bundles[network]=validate_bundle(dest)
    freeze(output/'PREPARATION_FREEZE.json',dict(bundles={n:v['bundle_id'] for n,v in bundles.items()},sources=controller,
        protocol=protocol(),task_file_sha256=sha256(output/'TASKS.json')))
    from .predictions import freeze_predictions
    freeze_predictions(output)
    return dict(bundles={n:v['bundle_id'] for n,v in bundles.items()},sessions=18)


def validate_bundle(bundle):
    bundle=Path(bundle);m=read(bundle/'manifest.json')
    if digest({k:v for k,v in m.items() if k!='bundle_id'})[:24]!=m['bundle_id']:raise ValueError('capsule seal differs')
    verify_inventory(bundle,m['files'])
    if read(bundle/'plans/protocol.json')!=protocol():raise ValueError('protocol changed')
    for path,value in read(bundle/'plans/preparation.json')['controllers'].items():
        if sha256(path)!=value:raise ValueError('controller changed; new batch required')
    return m


def validate(output):
    output=Path(output)
    if read(output/'TASKS.json')!=specs():raise ValueError('task selection changed')
    return {n:validate_bundle(output/n/'capsule')['bundle_id'] for n in ('resnet18','yolov3_tiny416')}


def qemu(output):
    from .runtime import make_task,native_command,summary
    output=Path(output).resolve();results=[]
    for spec in specs():
        if spec['phase']!='qualify':continue
        bundle=output/spec['network']/'capsule';doc=make_task(bundle,spec)
        raw=output/'qemu'/doc['task_id'];raw.mkdir(parents=True,exist_ok=True)
        argv,blobs=native_command(doc,bundle,str(bundle),str(raw))
        (raw/'inputs.txt').write_text(blobs)
        env_end=next(i for i,x in enumerate(argv) if x==str(bundle)+'/runtime/ld-linux-aarch64.so.1')
        envs=[e for e in argv[2:env_end] if not e.startswith('VTA_QUEUE_COUNTERS=')];args=[shutil.which('qemu-aarch64-static')]
        for e in envs:args+=['-E',e]
        args+=['-E','VTA_QUEUE_COUNTERS=0',*argv[env_end:]]
        if spec['scope']=='pipeline':
            # QEMU has no VTA device. Load/help validates the same binary;
            # full CPU numeric tests below validate real CPU modules.
            args=args[:args.index(str(bundle)+'/runner/vta_stage_pipeline_runner_phase')+1]+['--help']
        p=subprocess.run(args,capture_output=True,text=True,timeout=180)
        (raw/'qemu.log').write_text(p.stdout+p.stderr)
        if p.returncode:raise ValueError('QEMU ABI/output check failed: '+str(raw/'qemu.log'))
        if spec['scope']=='component':
            from phase_profile.qemu import qemu_summary
            qemu_summary(raw,doc,bundle)
        results.append(dict(spec=spec,passed=True,counters_native_test_required=True))
    write(output/'QEMU.json',dict(cases=results,passed=True),replace=True);return results
