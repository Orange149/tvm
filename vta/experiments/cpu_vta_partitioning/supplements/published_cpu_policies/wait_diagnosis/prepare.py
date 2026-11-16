"""Small self-contained capsule; old program, runtime and runner bytes are copied."""
from pathlib import Path
import random
import shutil
import subprocess
import time
from cpu_vta_partitioning.common import digest,inventory,now,read,sha256,sync_tree,verify_inventory,write
from cpu_vta_partitioning.prepare.build import copy_checked
from cpu_policies.open_resources.domain import validate_candidate
from .common import ROOT,PARENT,PROFILE,BUILD_RECORD,protocol,configurations,sources,frozen,assert_prepared


def audit(output):
    output=Path(output)
    for b in (PARENT,PROFILE):verify_inventory(b,read(b/'manifest.json')['files'])
    cfg=configurations()
    for c in cfg['pipelines'].values():validate_candidate(c,read(PARENT/'plans/programs.json'),unit_count=14)
    runtime=PARENT/'runtime/libtvm_runtime.so'
    if sha256(runtime)!=read(BUILD_RECORD)['runtime_sha256']:raise ValueError('runtime differs from recorded ABI')
    nm=subprocess.run(['aarch64-linux-gnu-nm','-D',str(runtime)],capture_output=True,text=True,check=True).stdout
    symbols=['TVMBackendParallelLaunch','TVMBackendParallelBarrier','_ZN3tvm7runtime9threading5YieldEv']
    if not all(' T '+s+'\n' in nm for s in symbols):raise ValueError('required dynamic ABI not exported')
    output.mkdir(parents=True,exist_ok=True)
    value=dict(status='AUDITED',runtime_sha256=sha256(runtime),runner_sha256=read(PARENT/'manifest.json')['runner_sha256'],
        exported_symbols=symbols,configs=cfg,old_sources_read_only=True,ranking_FPS_read=False)
    frozen(output/'AUDIT.json',value);return value


def task_order():
    # Freeze BEFORE measurements; stage 2 is present but gated by stage-1 evidence.
    rows=[]
    for stage,policy,scope in [(1,'original','component'),(1,'original','pipeline'),(2,'blocking','pipeline')]:
        group=[dict(stage=stage,policy=policy,scope=scope,case=case,probe=probe)
               for case in ('overlap4','split3') for probe in (False,True)]
        random.Random(protocol()['seed']+stage+(scope=='pipeline')*10).shuffle(group);rows+=group
    return rows


def prepare(output):
    from .build import compile_native,self_test
    output=Path(output).resolve();started=time.monotonic()
    if (output/'capsule/manifest.json').exists():return validate(output)
    if not (output/'AUDIT.json').exists():audit(output)
    build=output/'build';built=compile_native(build);self_test(build)
    dest=output/'capsule';cfg=configurations();parent=read(PARENT/'manifest.json');profile=read(PROFILE/'manifest.json')
    for name,info in parent['files'].items():
        if name.startswith(('runtime/','runner/','firmware/','references/')) or name=='inputs/000.bin':
            copy_checked(PARENT/name,dest/name,info['sha256'])
    full=read(PROFILE/'plans/programs.json');programs={s['program']:full[s['program']]
        for c in cfg['pipelines'].values() for s in c['stages']}
    for name in programs:
        for leaf in ('graph.json','module.so','params.bin','record.json'):
            copy_checked(PROFILE/'programs'/name/leaf,dest/'programs'/name/leaf)
    copy_checked(build/'libwait_adapter.so',dest/'adapter/libwait_adapter.so',built['adapter_sha256'])
    for leaf in ('inputs.json','expected_outputs.json'):copy_checked(PROFILE/'plans'/leaf,dest/'plans'/leaf)
    write(dest/'plans/programs.json',programs);write(dest/'plans/configurations.json',cfg)
    write(dest/'plans/protocol.json',protocol());write(dest/'plans/task_order.json',task_order())
    write(dest/'plans/adapter_build.json',built)
    copy_checked(build/'ARM_SELF_TEST.json',dest/'plans/ARM_SELF_TEST.json')
    copy_checked(output/'AUDIT.json',dest/'plans/AUDIT.json')
    # All selected intervals start at zero; no isolated nonzero frontier is requested.
    prep=dict(protocol=protocol(),sources=sources(),source_files={str(p.resolve()):sha256(p) for p in
        [PARENT/'manifest.json',PROFILE/'manifest.json',BUILD_RECORD,PARENT/'plans/pool.json']},
        adapter_sha256=built['adapter_sha256'],source_programs=programs,prepared_at=now())
    frozen(output/'PREPARATION.json',prep);write(dest/'plans/PREPARATION.json',prep)
    import cpu_vta_partitioning
    base=Path(cpu_vta_partitioning.__file__).parent
    for name,value in prep['sources'].items():
        src=base/name[6:] if name.startswith('@base/') else ROOT/name
        copy_checked(src,dest/'sources'/name.replace('@base/','base/'),value)
    write(dest/'plans/board_execution.json',dict(boot_policy='one_boot_no_automatic_reboot',udmabuf_bytes=201326592,
        storage_margin_bytes=268435456))
    files={n:i for n,i in inventory(dest).items() if n.startswith(('programs/','runtime/','runner/','firmware/','adapter/'))
           and not n.endswith('/record.json')}
    for n in ('inputs/000.bin','plans/board_execution.json'):
        files[n]=dict(bytes=(dest/n).stat().st_size,sha256=sha256(dest/n))
    write(dest/'plans/board_payload.json',dict(files=files,unpacked_bytes=sum(v['bytes'] for v in files.values())))
    write(dest/'plans/preparation_cost.json',dict(seconds=time.monotonic()-started,new_stage_compilations=0,
        runtime_rebuilds=0,executor_rebuilds=0,adapter_compiled=True,payload_bytes=sum(v['bytes'] for v in files.values())))
    m=dict(experiment=protocol()['experiment'],runner_sha256=parent['runner_sha256'],
           runtime_sha256=sha256(dest/'runtime/libtvm_runtime.so'),adapter_sha256=built['adapter_sha256'],files=inventory(dest))
    m['bundle_id']=digest(m)[:24];write(dest/'manifest.json',m);sync_tree(dest)
    return validate(output)


def validate_bundle(bundle):
    bundle=Path(bundle);m=read(bundle/'manifest.json')
    if digest({k:v for k,v in m.items() if k!='bundle_id'})[:24]!=m['bundle_id']:raise ValueError('capsule seal changed')
    verify_inventory(bundle,m['files'])
    if read(bundle/'plans/protocol.json')!=protocol():raise ValueError('measurement protocol changed')
    if read(bundle/'plans/PREPARATION.json')['sources']!=sources():raise ValueError('controller source changed')
    if not read(bundle/'plans/ARM_SELF_TEST.json')['passed']:raise ValueError('ARM ABI qualification missing')
    return m


def validate(output):
    output=Path(output);assert_prepared(output);m=validate_bundle(output/'capsule')
    return dict(validated=True,bundle_id=m['bundle_id'],runtime_unchanged=True,executor_unchanged=True,
                adapter_sha256=m['adapter_sha256'],max_sessions=24)
