from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import time
from .common import ROOT, REPO, FULL_PARENT, FRAME_PARENT, STATIC_SOURCE, api, identity, protocol
from .domain import pool_from_rankings, groups
from cpu_policies.diagnosis.prepare import _copy

def compile_runner(parent,work):
    from .native_patch import prepare_source
    from cpu_policies.run_scope.prepare import _sdk_checked
    c=api(parent);native=prepare_source(parent,work/'sources/top20_native')
    record,env=_sdk_checked(parent,work);output=work/'runner/vta_stage_pipeline_runner_frame_stream_top20'
    output.parent.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    result=subprocess.run(['bash','-c','source "$1" || exit $?; shift; exec "$@"','top20-sdk',
        record['toolchain']['sdk_setup'],'bash','-c',record['runner_compiler_recipe'],'top20-runner',
        native['source_path'],str(output),str(REPO),str(work/'runtime')],
        env=env,capture_output=True,text=True,timeout=300)
    (work/'build_logs/top20_compile.log').write_text(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError('native compiler failed: '+str(work/'build_logs/top20_compile.log'))
    native.update(runner_path=str(output.relative_to(work)),runner_sha256=c.sha256(output),
                  compile_seconds=time.monotonic()-start,runtime_rebuilt=False,stage_modules_recompiled=False)
    native['source_path']=str(Path(native['source_path']).relative_to(work))
    c.write(work/'plans/frame_stream_build.json',native)
    return native

def validate_bundle(bundle):
    bundle=Path(bundle).resolve();c=api(bundle);m=c.read(bundle/'manifest.json')
    if c.digest({k:v for k,v in m.items() if k!='bundle_id'})[:24]!=m['bundle_id']:raise ValueError('manifest identity differs')
    if m['controller_identity']!=identity():raise ValueError('controller changed; use frozen top20_controller')
    c.verify_inventory(bundle,m['files'])
    if c.read(bundle/'plans/top20_protocol.json')!=protocol():raise ValueError('protocol differs')
    ranked=c.read(bundle/'plans/static_rankings.json');pool=pool_from_rankings(ranked)
    if pool!=c.read(bundle/'plans/static_pool.json'):raise ValueError('static candidate pool differs')
    source=c.read(bundle/'plans/source_provenance.json')
    if (source['isolated_costs_replayed']!=775 or source['traffic_records_replayed']!=87
            or source['full_domain_counts']['enhanced']!=966240306 or not source['verification']['passed']
            or not source['same_pure20_and_diverse20_as_original']):raise ValueError('static coverage/proof incomplete')
    programs=c.read(bundle/'plans/programs.json');inputs=c.read(bundle/'plans/inputs.json')
    expected_names={s['program'] for cfg in pool.values() for s in cfg['stages']}
    if set(programs)!=expected_names:raise ValueError('program closure differs')
    executor=c.read(bundle/'plans/executors.json')['frame_stream']
    if groups(pool,programs,inputs,executor)!=c.read(bundle/'plans/cpu_groups.json'):raise ValueError('CPU equivalence groups differ')
    from cpu_vta_partitioning.prepare.payload import execution_files
    if execution_files(bundle)!=c.read(bundle/'plans/board_payload.json')['files']:raise ValueError('execution closure differs')
    if c.read(bundle/'OFFLINE_READINESS.json')['status']!='PREPARED_OFFLINE':raise ValueError('not prepared offline')
    return m

def prepare(base_bundle=None,frame_bundle=None,static_source=None,source_run=None):
    full=Path(base_bundle or FULL_PARENT).resolve();frame=Path(frame_bundle or FRAME_PARENT).resolve()
    if static_source is None:
        from .static import build
        static_source=build(full,source_run or ROOT/'runs/resnet18_open_cpu_resource_02',STATIC_SOURCE)
    static=Path(static_source).resolve();c=api(full);start=time.monotonic();code=identity()
    for parent in (full,frame):
        m=c.read(parent/'manifest.json')
        if c.digest({k:v for k,v in m.items() if k!='bundle_id'})[:24]!=m['bundle_id']:raise ValueError('parent manifest differs')
        c.verify_inventory(parent,m['files'])
    for folder in ('runtime','firmware'):
        if c.inventory(full/folder)!=c.inventory(frame/folder):raise ValueError('parents have different '+folder)
    provenance=c.read(static/'source_provenance.json')
    if provenance['source_bundle']!=str(full) or provenance['source_manifest_sha256']!=c.sha256(full/'manifest.json'):
        raise ValueError('static source belongs to different programs')
    ranked=c.read(static/'static_rankings.json');pool=pool_from_rankings(ranked)
    if pool!=c.read(static/'static_pool.json'):raise ValueError('static input pool differs')
    folder=ROOT/'artifacts/frame_stream_top20_builds';folder.mkdir(parents=True,exist_ok=True)
    work=Path(tempfile.mkdtemp(prefix='prepare-',dir=folder))
    for directory in ('runtime','firmware','inputs','references'):
        for name,item in c.inventory(full/directory).items():_copy(full/directory/name,work/directory/name,item['sha256'])
    for directory in ('base_controller','cpu_strategy','native_source'):
        shutil.copytree(full/'sources'/directory,work/'sources'/directory,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
    shutil.copytree(frame/'sources/reference',work/'sources/reference')
    # SDK recipes reference a small immutable source include tree.
    shutil.copytree(frame/'sources/vta',work/'sources/vta',ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('inputs.json','runtime_build.json','compiler_identity.json','sdk_runtime.json'):
        _copy(full/'plans'/name,work/'plans'/name)
    all_programs=c.read(full/'plans/programs.json')
    names={s['program'] for cfg in pool.values() for s in cfg['stages']}
    programs={name:all_programs[name] for name in sorted(names)}
    for name,p in programs.items():
        for leaf,meta in p['files'].items():_copy(full/'programs'/name/leaf,work/'programs'/name/leaf,meta['sha256'])
    c.write(work/'plans/programs.json',programs);c.write(work/'plans/full_programs.json',all_programs)
    for name in ('static_rankings.json','static_pool.json','source_provenance.json','isolated_costs.json','ddr.json'):
        _copy(static/name,work/'plans'/name)
    # Bind DP/oracle checkpoints as evidence, without importing old throughput files.
    for path in (static/'search').glob('*.json'):_copy(path,work/'provenance/static_search'/path.name)
    c.write(work/'plans/top20_protocol.json',protocol())
    c.write(work/'plans/board_execution.json',{'revision':'cpu-frame-stream-top20-1','udmabuf_bytes':201326592,
            'storage_margin_bytes':256*1024**2,'boot_policy':protocol()['boot_policy']})
    native=compile_runner(frame,work)
    policy=c.read(full/'plans/executors.json')['policy'];_copy(full/policy['path'],work/policy['path'],policy['sha256'])
    executors={'policy':policy,'frame_stream':{'path':native['runner_path'],'sha256':native['runner_sha256']}}
    c.write(work/'plans/executors.json',executors)
    c.write(work/'plans/cpu_groups.json',groups(pool,programs,c.read(work/'plans/inputs.json'),executors['frame_stream']))
    from cpu_vta_partitioning.prepare.package import elf
    closure={}
    for p in [*list((work/'runner').iterdir()),*list((work/'runtime').iterdir()),*list((work/'programs').glob('*/module.so'))]:
        if not p.is_file():continue
        info=elf(p)
        if any(not (work/'runtime'/name).is_file() for name in info['needed']):raise ValueError('unresolved ELF dependency')
        closure[str(p.relative_to(work))]=info
    c.write(work/'plans/elf_dependencies.json',closure)
    emulator=shutil.which('qemu-aarch64-static')
    if not emulator:raise RuntimeError('QEMU missing')
    check=subprocess.run([emulator,str(work/'runtime/ld-linux-aarch64.so.1'),'--library-path',str(work/'runtime'),
                          str(work/native['runner_path']),'--help'],capture_output=True,text=True,timeout=45)
    if check.returncode or '--cpu-frame-stream-spec' not in check.stdout+check.stderr:raise ValueError('QEMU loader/mode check failed')
    tests=subprocess.run([sys.executable,'-m','pytest','-c','pyproject.toml','-q','tests/test_frame_stream_top20.py'],
                         cwd=ROOT,env=dict(os.environ,PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'),capture_output=True,text=True,timeout=180)
    c.write(work/'plans/offline_tests.json',{'passed':tests.returncode==0,'stdout':tests.stdout,'stderr':tests.stderr})
    if tests.returncode:raise RuntimeError(tests.stdout+tests.stderr)
    if code!=identity():raise ValueError('controller changed during preparation')
    snapshot=work/'sources/top20_controller'
    for name in ('run_frame_stream_top20.py','pyproject.toml'):_copy(ROOT/name,snapshot/name)
    for name in ('frame_stream_top20','frame_stream','cpu_policies','configs'):
        shutil.copytree(ROOT/name,snapshot/name,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
    _copy(ROOT/'tests/test_frame_stream_top20.py',snapshot/'tests/test_frame_stream_top20.py')
    from cpu_vta_partitioning.prepare.payload import execution_files
    payload=execution_files(work);c.write(work/'plans/board_payload.json',{'files':payload,'unpacked_bytes':sum(v['bytes'] for v in payload.values())})
    c.write(work/'plans/preparation_cost.json',{'elapsed_seconds':time.monotonic()-start,'CPU_groups':len(c.read(work/'plans/cpu_groups.json')['groups']),
        'candidate_count':len(pool),'compiled_runners':1,'stage_modules_compiled':0,'runtime_rebuilt':False,
        'execution_bytes':sum(v['bytes'] for v in payload.values())})
    c.write(work/'OFFLINE_READINESS.json',{'status':'PREPARED_OFFLINE','tests':{'passed':True},'candidate_count':len(pool),
        'domain_contexts':9,'full_domain_counts':ranked['counts'],'pipeline_executor_unchanged':True,
        'board_validation_pending':['CPU frame conservation','affinity','complete output correctness']})
    manifest={'experiment':protocol()['experiment'],'controller_identity':code,'files':c.inventory(work)}
    manifest['bundle_id']=c.digest(manifest)[:24];c.write(work/'manifest.json',manifest)
    target=ROOT/'artifacts/bundles'/manifest['bundle_id'];work.rename(target);validate_bundle(target)
    return target
