from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import time
from .common import ROOT,FULL_PARENT,STATIC_PARENT,api,identity,protocol
from cpu_policies.diagnosis.prepare import _copy

def validate_bundle(bundle):
    from cpu_policies.open_resources.domain import validate_candidate,configuration,contexts
    from .audit import verify_upstream
    bundle=Path(bundle).resolve();c=api(bundle);m=c.read(bundle/'manifest.json')
    from cpu_vta_partitioning.prepare.payload import execution_files
    if c.digest({k:v for k,v in m.items() if k!='bundle_id'})[:24]!=m['bundle_id']:raise ValueError('manifest identity differs')
    if m['controller_identity']!=identity():raise ValueError('controller changed; use frozen selection_controller')
    c.verify_inventory(bundle,m['files']);verify_upstream()
    if c.read(bundle/'plans/selection_protocol.json')!=protocol():raise ValueError('protocol differs')
    ranked=c.read(bundle/'plans/selection_rankings.json');pool=c.read(bundle/'plans/static_pool.json')
    all_rows=[]
    for name in ('pure_static20','diverse20','coverage20'):
        rows=ranked[name]
        if len(rows)!=20 or len({r['id'] for r in rows})!=20:raise ValueError('twenty distinct entries required')
        all_rows+=rows
    if [r['id'] for r in ranked['coverage20'][:5]]!=[r['id'] for r in ranked['pure_static20'][:5]]:
        raise ValueError('static first five changed')
    if {r['context_id'] for r in ranked['coverage20']}!={v['id'] for v in contexts()}:
        raise ValueError('resource context coverage differs')
    expected={r['id']:{**configuration(r),'id':r['id'],'score_ns':r['score_ns'],'score_ms':r['score_ms']} for r in all_rows}
    if pool!=dict(sorted(expected.items())) or len(pool)>43:raise ValueError('measurement union differs')
    programs=c.read(bundle/'plans/programs.json')
    for cfg in pool.values():validate_candidate(cfg,programs)
    if set(programs)!={s['program'] for v in pool.values() for s in v['stages']}:raise ValueError('program closure differs')
    source=c.read(bundle/'plans/source_provenance.json')
    if (not source['component_evidence_replayed'] or not source['same_original_lists']
            or source['isolated_costs']!=775 or source['traffic_records']!=87 or source['baseline_union']!=28):
        raise ValueError('component/search proof incomplete')
    audit=c.read(bundle/'plans/runtime_audit/runtime_audit.json')
    if audit['runtime_sha256']!=c.sha256(bundle/'runtime/libtvm_runtime.so') or audit['hotspot_or_causation_proven']:
        raise ValueError('runtime audit identity/conclusion differs')
    executors=c.read(bundle/'plans/executors.json')
    if set(executors)!={'policy'} or c.sha256(bundle/executors['policy']['path'])!=executors['policy']['sha256']:
        raise ValueError('pipeline executor differs')
    if execution_files(bundle)!=c.read(bundle/'plans/board_payload.json')['files']:raise ValueError('execution payload differs')
    if c.read(bundle/'OFFLINE_READINESS.json')['status']!='PREPARED_OFFLINE':raise ValueError('offline preparation incomplete')
    return m

def prepare(base_bundle=None,static_source=None,selection_source=None):
    from .static import build
    from .audit import audit,verify_upstream
    full=Path(base_bundle or FULL_PARENT).resolve();c=api(full);start=time.monotonic();code=identity()
    full_manifest=c.read(full/'manifest.json')
    if c.digest({k:v for k,v in full_manifest.items() if k!='bundle_id'})[:24]!=full_manifest['bundle_id']:
        raise ValueError('parent manifest differs')
    c.verify_inventory(full,full_manifest['files'])
    if selection_source is None:selection_source=build(full,static_source)
    selected=Path(selection_source).resolve();source=c.read(selected/'source_provenance.json')
    if not source['component_evidence_replayed'] or source['source']['source_manifest_sha256']!=c.sha256(full/'manifest.json'):
        raise ValueError('selection lacks matching raw component replay')
    ranked=c.read(selected/'rankings.json');pool=c.read(selected/'pool.json')
    if selection_source is not None:
        # Reproduce the selected lists from authenticated compact checkpoints;
        # a caller-supplied JSON cannot become a frozen ranking on assertion alone.
        check=build(full,static_source,output=ROOT/'artifacts/selection_top20_prepare_replay',replay_components=False)
        if c.read(check/'rankings.json')!=ranked or c.read(check/'pool.json')!=pool:
            raise ValueError('selection does not reproduce from the component-only checkpoints')
    folder=ROOT/'artifacts/selection_top20_builds';folder.mkdir(parents=True,exist_ok=True)
    work=Path(tempfile.mkdtemp(prefix='prepare-',dir=folder))
    for directory in ('runtime','firmware','inputs','references'):
        for name,item in c.inventory(full/directory).items():_copy(full/directory/name,work/directory/name,item['sha256'])
    for name in ('base_controller','cpu_strategy','native_source'):
        shutil.copytree(full/'sources'/name,work/'sources'/name,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
    reference=STATIC_PARENT/'sources/reference';reference_manifest=c.read(STATIC_PARENT/'manifest.json')
    for path in reference.rglob('*'):
        if path.is_file():
            relative=str(path.relative_to(STATIC_PARENT));_copy(path,work/relative,reference_manifest['files'][relative]['sha256'])
    for name in ('inputs.json','runtime_build.json','compiler_identity.json','sdk_runtime.json'):
        _copy(full/'plans'/name,work/'plans'/name)
    all_programs=c.read(full/'plans/programs.json');names={s['program'] for v in pool.values() for s in v['stages']}
    programs={n:all_programs[n] for n in sorted(names)}
    for name,p in programs.items():
        for leaf,meta in p['files'].items():_copy(full/'programs'/name/leaf,work/'programs'/name/leaf,meta['sha256'])
    c.write(work/'plans/programs.json',programs);c.write(work/'plans/full_programs.json',all_programs)
    for old,new in (('rankings.json','selection_rankings.json'),('pool.json','static_pool.json'),('source_provenance.json','source_provenance.json')):
        _copy(selected/old,work/'plans'/new)
    from .common import STATIC_SOURCE
    static=Path(static_source or STATIC_SOURCE)
    for name in ('isolated_costs.json','ddr.json'):_copy(static/name,work/'plans'/name)
    for context_id in source['compact_checkpoint_hashes']:
        path=static/'search'/(context_id+'.json')
        _copy(path,work/'provenance/compact_search'/path.name)
    policy=c.read(full/'plans/executors.json')['policy'];_copy(full/policy['path'],work/policy['path'],policy['sha256'])
    c.write(work/'plans/executors.json',{'policy':policy});c.write(work/'plans/selection_protocol.json',protocol())
    c.write(work/'plans/upstream_selector.json',verify_upstream())
    c.write(work/'plans/board_execution.json',{'revision':'autotvm-coverage-top20-1','udmabuf_bytes':201326592,
        'storage_margin_bytes':256*1024**2,'boot_policy':protocol()['boot_policy']})
    audit(work,work/'plans/runtime_audit')
    from cpu_vta_partitioning.prepare.package import elf
    closure={}
    for path in [*list((work/'runner').iterdir()),*list((work/'runtime').iterdir()),*list((work/'programs').glob('*/module.so'))]:
        if not path.is_file():continue
        info=elf(path)
        if any(not (work/'runtime'/n).is_file() for n in info['needed']):raise ValueError('unresolved ELF dependency')
        closure[str(path.relative_to(work))]=info
    c.write(work/'plans/elf_dependencies.json',closure)
    emulator=shutil.which('qemu-aarch64-static')
    if not emulator:raise RuntimeError('QEMU missing')
    result=subprocess.run([emulator,str(work/'runtime/ld-linux-aarch64.so.1'),'--library-path',str(work/'runtime'),
        str(work/policy['path']),'--help'],capture_output=True,text=True,timeout=45)
    if result.returncode or '--cpu-worker-pinning' not in result.stdout+result.stderr:raise ValueError('QEMU loader/parameter check failed')
    c.write(work/'plans/qemu_loader.json',{'passed':True,'runner_sha256':policy['sha256'],'stdout':result.stdout,'stderr':result.stderr})
    tests=subprocess.run([sys.executable,'-m','pytest','-c','pyproject.toml','-q','tests/test_selection_top20.py',
        'tests/test_frame_stream_top20.py'],cwd=ROOT,env=dict(os.environ,PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'),
        capture_output=True,text=True,timeout=240)
    c.write(work/'plans/offline_tests.json',{'passed':tests.returncode==0,'stdout':tests.stdout,'stderr':tests.stderr})
    if tests.returncode:raise RuntimeError(tests.stdout+tests.stderr)
    if identity()!=code:raise ValueError('controller changed during preparation')
    snapshot=work/'sources/selection_controller'
    for name in ('run_diverse_top20.py','pyproject.toml'):_copy(ROOT/name,snapshot/name)
    for name in ('selection_top20','cpu_policies','configs'):
        shutil.copytree(ROOT/name,snapshot/name,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
    _copy(ROOT/'tests/test_selection_top20.py',snapshot/'tests/test_selection_top20.py')
    from cpu_vta_partitioning.prepare.payload import execution_files
    payload=execution_files(work);size=sum(v['bytes'] for v in payload.values())
    c.write(work/'plans/board_payload.json',{'files':payload,'unpacked_bytes':size,'host_only':['sources','references','provenance']})
    c.write(work/'plans/preparation_cost.json',{'elapsed_seconds':time.monotonic()-start,'candidate_count':len(pool),
        'static_selection_seconds':source['elapsed_seconds'],'compiled_runners':0,'stage_modules_compiled':0,
        'runtime_rebuilt':False,'execution_bytes':size})
    c.write(work/'OFFLINE_READINESS.json',{'status':'PREPARED_OFFLINE','tests':{'passed':True},'candidate_count':len(pool),
        'context_count':9,'pipeline_executor_unchanged':True,'new_component_profiles':0,
        'board_validation_pending':['actual affinity','complete output correctness','one-boot performance']})
    manifest={'experiment':protocol()['experiment'],'controller_identity':code,'files':c.inventory(work)}
    manifest['bundle_id']=c.digest(manifest)[:24];c.write(work/'manifest.json',manifest)
    target=ROOT/'artifacts/bundles'/manifest['bundle_id'];work.rename(target);validate_bundle(target)
    return target
