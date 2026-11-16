"""Local-only capsule assembly; reuse authenticated graph/runtime assets."""
from copy import deepcopy
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
from .common import ROOT, REPO, DEFAULT_PARENT, FULL_PARENT, api, identity, protocol
from .domain import configurations, components, groups, diagnostics, program_names
from ..diagnosis.prepare import _copy
from ..run_scope.prepare import _sdk_checked

def compile_runner(parent,work,reuse_executor=None):
    from .patch_runner import prepare_source
    c=api(parent); native=prepare_source(parent,work/'sources/contention_native')
    record,env=_sdk_checked(parent,work)
    output=work/'runner/vta_stage_pipeline_runner_contention'
    if reuse_executor is not None:
        source=Path(reuse_executor).resolve(); manifest=c.read(source/'manifest.json')
        if c.digest({k:v for k,v in manifest.items() if k!='bundle_id'})[:24]!=manifest['bundle_id']: raise ValueError('executor source manifest differs')
        c.verify_inventory(source,manifest['files'])
        actual=c.inventory(work/'sources/contention_native')
        expected=c.inventory(source/'sources/contention_native')
        if actual!=expected or c.read(source/'plans/runtime_build.json')!=record:
            raise ValueError('executor reuse requires identical native source/header and toolchain identities')
        runner=source/'runner/vta_stage_pipeline_runner_contention'
        _copy(runner,output,manifest['files']['runner/vta_stage_pipeline_runner_contention']['sha256'])
        native.update(reused_executor_bundle_id=manifest['bundle_id'],executor_recompiled=False)
    else:
        command=['bash','-c','source "$1" || exit $?; shift; exec "$@"','contention-sdk',
                 record['toolchain']['sdk_setup'],'bash','-c',record['runner_compiler_recipe'],
                 'contention-runner',native['source_path'],str(output),str(REPO),str(work/'runtime')]
        result=subprocess.run(command,capture_output=True,text=True,env=env,timeout=300)
        (work/'build_logs/contention_compile.log').write_text(result.stdout+result.stderr)
        if result.returncode: raise RuntimeError('contention runner compile failed: '+str(work/'build_logs/contention_compile.log'))
        native['executor_recompiled']=True
    native.update(runner_path=str(output.relative_to(work)),runner_sha256=c.sha256(output),
                  source_path_in_bundle=str(Path(native.pop('source_path')).relative_to(work)),
                  runtime_rebuilt=False,stage_modules_recompiled=False)
    c.write(work/'plans/contention_build.json',native)
    return native

def validate_bundle(bundle):
    bundle=Path(bundle).resolve(); c=api(bundle); m=c.read(bundle/'manifest.json')
    if c.digest({k:v for k,v in m.items() if k!='bundle_id'})[:24]!=m['bundle_id']: raise ValueError('manifest identity differs')
    if m['controller_identity']!=identity(): raise ValueError('controller changed; use frozen sources/contention_controller')
    c.verify_inventory(bundle,m['files'])
    if c.read(bundle/'plans/contention_protocol.json')!=protocol(): raise ValueError('protocol differs')
    cases={'pipelines':configurations(),'components':components(),'groups':groups(),'diagnostics':diagnostics()}
    if c.read(bundle/'plans/contention_cases.json')!=cases: raise ValueError('cases differ')
    if set(c.read(bundle/'plans/programs.json'))!=set(program_names()): raise ValueError('program closure differs')
    if c.read(bundle/'OFFLINE_READINESS.json')['status']!='PREPARED_OFFLINE': raise ValueError('not ready')
    from cpu_vta_partitioning.prepare.payload import execution_files
    if execution_files(bundle)!=c.read(bundle/'plans/board_payload.json')['files']: raise ValueError('payload closure differs')
    return m

def prepare(parent=None,full_parent=None,reuse_executor=None):
    parent=Path(parent or DEFAULT_PARENT).resolve(); full=Path(full_parent or FULL_PARENT).resolve()
    c=api(parent); code=identity()
    manifests=[]
    for source in (parent,full):
        m=c.read(source/'manifest.json')
        if c.digest({k:v for k,v in m.items() if k!='bundle_id'})[:24]!=m['bundle_id']: raise ValueError('parent identity differs')
        c.verify_inventory(source,m['files']); manifests.append(m)
    for folder in ('runtime','firmware'):
        if c.inventory(parent/folder)!=c.inventory(full/folder): raise ValueError('parents have different '+folder)
    buildroot=ROOT/'artifacts/contention_builds'; buildroot.mkdir(parents=True,exist_ok=True)
    work=Path(tempfile.mkdtemp(prefix='prepare-',dir=buildroot))
    for folder in ('runtime','runner','firmware','inputs','references'):
        for name,item in c.inventory(parent/folder).items(): _copy(parent/folder/name,work/folder/name,item['sha256'])
    for folder in ('base_controller','cpu_strategy','native_source','vta'):
        shutil.copytree(parent/'sources'/folder,work/'sources'/folder,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
    for name in ('inputs.json','executors.json','compiler_identity.json','runtime_build.json','sdk_runtime.json',
                 'policy_native_acceptance.json','policy_runner_patch.json','baseline_costs.json','baseline_ddr.json','baseline_predictions.json'):
        _copy(parent/'plans'/name,work/'plans'/name)
    programs=c.read(full/'plans/programs.json')
    from ..open_resources.search import _topologies
    from itertools import combinations
    paths=_topologies(programs,21,3)
    cpu_pairs=set(); contextual_pairs=set()
    for path in paths:
        cpus=[n for n in path if programs[n]['device']=='cpu']
        vtas=tuple(n for n in path if programs[n]['device']=='vta')
        for pair in combinations(cpus,2):
            cpu_pairs.add(pair); contextual_pairs.add((*pair,vtas))
    c.write(work/'plans/full_domain_profile_inventory.json',{
        'source_manifest_sha256':c.sha256(full/'manifest.json'),'topologies':len(paths),
        'distinct_ordered_CPU_interval_pairs':len(cpu_pairs),
        'CPU_pair_physical_signatures_four_core':len(cpu_pairs)*15*15,
        'distinct_CPU_pair_and_VTA_sequence_contexts':len(contextual_pairs),
        'CPU_VTA_signatures_shared4':len(contextual_pairs)*15*15,
        'CPU_VTA_signatures_if_extended_to_nine_contexts':len(contextual_pairs)*(5*15*15+4*7*7),
        'measured_this_round':20,'automatically_scheduled':False,
        'limitations':['pairwise profiles do not cover three/four-CPU interactions',
                      'counts assume caller at minimum core; no worker permutation search',
                      'numbers are preparation estimates, not a recommendation to measure exhaustively']})
    programs={name:programs[name] for name in program_names()}
    for name,p in programs.items():
        for leaf,item in p['files'].items(): _copy(full/'programs'/name/leaf,work/'programs'/name/leaf,item['sha256'])
    # Reference proof resources are authenticated from the existing complete capsule.
    shutil.copytree(parent/'sources/reference',work/'sources/reference')
    c.write(work/'plans/programs.json',programs)
    c.write(work/'plans/contention_cases.json',{'pipelines':configurations(),'components':components(),'groups':groups(),'diagnostics':diagnostics()})
    c.write(work/'plans/contention_protocol.json',protocol())
    c.write(work/'plans/pilot_protocol.json',protocol())
    c.write(work/'plans/board_execution.json',{'revision':'contention-1','udmabuf_bytes':201326592,
             'storage_margin_bytes':256*1024**2,'boot_policy':protocol()['boot_policy']})
    native=compile_runner(parent,work,reuse_executor)
    specs=c.read(work/'plans/executors.json'); specs['contention']={'path':native['runner_path'],'sha256':native['runner_sha256']}
    c.write(work/'plans/executors.json',specs,replace=True)
    sources={}
    for i,source in enumerate((parent,full)):
        m=manifests[i]; sources[str(m['bundle_id'])]={'manifest_sha256':c.sha256(source/'manifest.json')}
        c.write(work/f'provenance/parent_{i}_manifest.json',m)
    c.write(work/'plans/source_provenance.json',{'parents':sources,'new_stage_compilations':0,'new_runtime_compilations':0})
    snapshot=work/'sources/contention_controller'
    _copy(ROOT/'run_contention.py',snapshot/'run_contention.py')
    _copy(ROOT/'run_contention_compat.py',snapshot/'run_contention_compat.py')
    _copy(ROOT/'pyproject.toml',snapshot/'pyproject.toml')
    shutil.copytree(ROOT/'cpu_policies',snapshot/'cpu_policies',ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
    shutil.copytree(ROOT/'configs',snapshot/'configs')
    for p in (ROOT/'tests').glob('test_contention*.py'): _copy(p,snapshot/'tests'/p.name)
    from cpu_vta_partitioning.prepare.package import elf
    files=[*list((work/'runner').glob('*')),*list((work/'runtime').glob('*')),*list((work/'programs').glob('*/module.so'))]
    closure={}
    for p in files:
        if p.is_file():
            info=elf(p)
            if any(not (work/'runtime'/n).is_file() for n in info['needed']): raise ValueError('unresolved runtime dependency')
            closure[str(p.relative_to(work))]=info
    c.write(work/'plans/elf_dependencies.json',closure)
    emulator=shutil.which('qemu-aarch64-static')
    if not emulator: raise RuntimeError('QEMU missing')
    check=subprocess.run([emulator,str(work/'runtime/ld-linux-aarch64.so.1'),'--library-path',str(work/'runtime'),
                          str(work/native['runner_path']),'--help'],capture_output=True,text=True,timeout=45)
    if check.returncode or '--contention-spec' not in check.stdout+check.stderr: raise ValueError('loader/argument acceptance failed')
    tests=subprocess.run([sys.executable,'-m','pytest','-c','pyproject.toml','-q','tests/test_contention.py'],
                         cwd=ROOT,env=dict(os.environ,PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'),capture_output=True,text=True,timeout=180)
    c.write(work/'plans/offline_tests.json',{'passed':tests.returncode==0,'stdout':tests.stdout,'stderr':tests.stderr})
    if tests.returncode: raise RuntimeError(tests.stdout+tests.stderr)
    if code!=identity(): raise ValueError('controller changed during preparation')
    from cpu_vta_partitioning.prepare.payload import execution_files
    payload=execution_files(work)
    c.write(work/'plans/board_payload.json',{'revision':1,'files':payload,'unpacked_bytes':sum(v['bytes'] for v in payload.values())})
    c.write(work/'OFFLINE_READINESS.json',{'status':'PREPARED_OFFLINE','sessions':48,'component_signatures':15,'pipelines':8,
             'tests':{'passed':True},'new_stage_compilations':0,'new_runtime_compilations':0,
             'board_validation_pending':['affinity','outputs','background coverage','cross-topology validation']})
    m={'experiment':protocol()['experiment'],'controller_identity':code,'files':c.inventory(work)}
    m['bundle_id']=c.digest(m)[:24]; c.write(work/'manifest.json',m)
    target=ROOT/'artifacts/bundles'/m['bundle_id']; work.rename(target)
    validate_bundle(target)
    return target
