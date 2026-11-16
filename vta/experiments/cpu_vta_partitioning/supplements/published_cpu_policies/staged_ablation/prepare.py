"""Self-contained capsule using the previously qualified native executor."""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from .common import ROOT,PARENT,SOURCE,MODELS,settings,protocol,api,identity
from .model import score
from .search import applicable_score, method

def _controller(work):
    from run_resource_behavior import SELECTION
    from resource_envelope.search_adapter import _replace
    base=SELECTION/'sources/selection_controller/selection_top20'
    snapshot=work/'sources/staged_ablation_controller';package=snapshot/'staged_ablation'
    package.mkdir(parents=True)
    for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,package/p.name)
    campaign=(base/'campaign.py').read_text()
    campaign=_replace(campaign,'from .common import api,identity,protocol','from .common import api,identity,protocol,MODELS')
    campaign=_replace(campaign,"('pure_static20','diverse20','coverage20')","tuple(row[0] for row in MODELS)")
    campaign=campaign.replace('[AutoTVM Top20]','[分阶段消融]').replace('frozen selection_controller','frozen staged_ablation_controller')
    (package/'campaign.py').write_text(campaign)
    shutil.copy2(ROOT/'run_staged_ablation.py',snapshot/'run_staged_ablation.py')
    shutil.copy2(ROOT/'pyproject.toml',snapshot/'pyproject.toml')
    for folder in ('cpu_policies','configs','resource_envelope','model_ablation','host_work_ablation'):
        shutil.copytree(ROOT/folder,snapshot/folder,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
    # The frozen CLI does not dispatch local search/prepare into live sources.
    (snapshot/'run_staged_ablation.py').write_text('import sys\nsys.dont_write_bytecode=True\nfrom staged_ablation.cli import main\nif __name__ == "__main__": main()\n')
    return snapshot

def _module(folder):
    c=api();name='ablation_proof_'+c.digest(str(folder))[:20]
    spec=importlib.util.spec_from_file_location(name,folder/'__init__.py',submodule_search_locations=[str(folder)])
    package=importlib.util.module_from_spec(spec);sys.modules[name]=package;spec.loader.exec_module(package)
    spec=importlib.util.spec_from_file_location(name+'.search',folder/'search.py')
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    return mod

def validate_proofs(bundle):
    from cpu_policies.open_resources.costs import Costs
    from cpu_policies.open_resources.search import _digest
    c=api(bundle);b=Path(bundle);proofs=c.read(b/'plans/search_proofs.json')
    lists=c.read(b/'plans/selection_rankings.json');programs=c.read(b/'plans/full_programs.json')
    records=c.read(b/'plans/isolated_costs.json')['costs'];ddr=c.read(b/'plans/ddr.json')
    for key,*_ in MODELS:
        proof=proofs[key];source=b/'provenance/search'/key/'source';module=_module(source)
        impl={n:c.sha256(source/n) for n in ('search.py','oracle.cc','picojson.h')}
        if proof['implementation']!=impl or proof['model']!=settings(key):raise ValueError('proof model/source differs')
        if proof['component_hashes']!={n:c.sha256(b/'plans'/n) for n in ('isolated_costs.json','ddr.json')}:
            raise ValueError('proof component identity differs')
        costs=Costs(records,programs=programs,ddr=ddr,distribution='session_weights' if settings(key)['thread_weights'] else 'balanced')
        graphs=module._topologies(programs,21,3);contexts={v['id']:v for v in module.domain.contexts()}
        expanded=[];total=0
        for name,ctx in contexts.items():
            cp=c.read(b/'provenance/search'/key/'checkpoints'/(name+'.json'))
            payload=dict(programs=module._prepare(programs,costs,ctx,ddr),topologies=graphs,
                context=ctx,k=20,score_scale=module.SCORE_SCALE,implementation=impl)
            fields=('counts','gate_counts','rankings','topology_strata','resource_strata');v=cp['verification']
            if (cp['checkpoint_sha256']!=_digest({k:x for k,x in cp.items() if k!='checkpoint_sha256'})
                or cp['input_sha256']!=_digest(payload) or not v['passed']
                or v['source_sha256']!=impl['oracle.cc']
                or v['result_sha256']!=_digest({f:cp[f] for f in fields})):
                raise ValueError('complete-domain proof checkpoint differs')
            total+=cp['counts'][method(key)]
            for row in cp['rankings'][method(key)]:
                terminal=module.Terminal.from_json(row)
                cfg=module._configuration(terminal,graphs,programs,contexts)
                expanded.append({**cfg,'score_ns':terminal.score,'score_ms':terminal.score/1e6})
        if total!=proof['selected_count']:raise ValueError('complete-domain count differs')
        expected=sorted(expanded,key=module._row_key)[:20]
        for pos,row in enumerate(expected,1):row['static_rank']=pos
        if expected!=c.read(b/'plans/full_rankings.json')[key] or expected[:settings(key)['measurement_budget']]!=lists[key]:raise ValueError('Top-20 differs from complete-domain proof')
    return True

def validate_bundle(bundle):
    from cpu_policies.open_resources.domain import configuration,validate_candidate
    from cpu_policies.open_resources.costs import Costs
    b=Path(bundle).resolve();c=api(b);m=c.read(b/'manifest.json')
    from cpu_vta_partitioning.prepare.payload import execution_files
    if c.digest({k:v for k,v in m.items() if k!='bundle_id'})[:24]!=m['bundle_id']:raise ValueError('manifest differs')
    if m['controller_identity']!=identity():raise ValueError('controller identity differs; use frozen capsule')
    c.verify_inventory(b,m['files'])
    if c.read(b/'plans/selection_protocol.json')!=protocol():raise ValueError('protocol differs')
    frozen=c.read(b/'plans/ranking_freeze.json')
    if frozen['pipeline_FPS_used']:raise ValueError('selection used measured FPS')
    for name,digest in frozen['files'].items():
        if c.sha256(b/'plans'/name)!=digest:raise ValueError('ranking freeze changed')
    ranked=c.read(b/'plans/selection_rankings.json');pool=c.read(b/'plans/static_pool.json')
    if set(ranked)!={r[0] for r in MODELS}:raise ValueError('all staged model lists required')
    wanted={};programs=c.read(b/'plans/programs.json')
    for key,rows in ranked.items():
        if len(rows)!=settings(key)['measurement_budget'] or len({r['id'] for r in rows})!=len(rows):raise ValueError('declared unique selection budget required')
        for row in rows:
            validate_candidate(row,programs)
            cfg={**configuration(row),'id':row['id']}
            if row['id'] in wanted and wanted[row['id']]!=cfg:raise ValueError('execution identity collision')
            wanted[row['id']]=cfg
    for row in c.read(b/'plans/mechanism_audit.json')['required_new_configurations']:
        cfg=row['configuration'];validate_candidate(cfg,programs);wanted[cfg['id']]=cfg
    if pool!=dict(sorted(wanted.items())) or len(pool)>121:raise ValueError('measurement union differs')
    if set(programs)!={s['program'] for cfg in pool.values() for s in cfg['stages']}:raise ValueError('program closure differs')
    import random
    order=sorted(pool);random.Random(protocol()['seed']).shuffle(order)
    if order!=frozen['configuration_order'] or frozen['candidate_count']!=len(pool):raise ValueError('execution order differs')
    if frozen['native_sessions']!=len(pool)*2 or frozen['complete_inferences']!=len(pool)*29:raise ValueError('task budget differs')
    scores=c.read(b/'plans/model_scores.json');records=c.read(b/'plans/isolated_costs.json')['costs'];ddr=c.read(b/'plans/ddr.json')
    for key,*_ in MODELS:
        costs=Costs(records,programs=programs,ddr=ddr,distribution='session_weights' if settings(key)['thread_weights'] else 'balanced')
        if scores[key]!={i:applicable_score(cfg,costs,key) for i,cfg in pool.items()}:raise ValueError('static score accounting differs')
    validate_proofs(b)
    policy=c.read(b/'plans/executors.json')['policy'];loader=c.read(b/'plans/qemu_loader.json')
    if c.sha256(b/policy['path'])!=policy['sha256'] or not loader['passed'] or loader['runner_sha256']!=policy['sha256']:
        raise ValueError('executor/loader differs')
    if execution_files(b)!=c.read(b/'plans/board_payload.json')['files']:raise ValueError('board payload differs')
    return m

def prepare(output):
    from .search import freeze
    from cpu_policies.diagnosis.prepare import _copy
    from run_resource_behavior import SELECTION
    c=api();started=time.monotonic();output=Path(output).resolve();selected=freeze(output)
    pool=c.read(selected/'static_pool.json');proofs=c.read(selected/'search_proofs.json')
    parent=c.read(PARENT/'manifest.json');c.verify_inventory(PARENT,parent['files'])
    build=ROOT/'artifacts/staged_ablation_builds';build.mkdir(exist_ok=True)
    work=Path(tempfile.mkdtemp(prefix='prepare-',dir=build))
    for folder in ('runtime','firmware','inputs','references'):
        for name,info in c.inventory(PARENT/folder).items():_copy(PARENT/folder/name,work/folder/name,info['sha256'])
    for name in ('base_controller','cpu_strategy','native_source'):
        shutil.copytree(PARENT/'sources'/name,work/'sources'/name,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
    for p in (SELECTION/'sources/reference').rglob('*'):
        if p.is_file():_copy(p,work/p.relative_to(SELECTION))
    all_programs=c.read(PARENT/'plans/programs.json');names={s['program'] for cfg in pool.values() for s in cfg['stages']}
    programs={n:all_programs[n] for n in sorted(names)}
    for name,program in programs.items():
        for leaf,info in program['files'].items():_copy(PARENT/'programs'/name/leaf,work/'programs'/name/leaf,info['sha256'])
    for name in ('inputs.json','runtime_build.json','compiler_identity.json','sdk_runtime.json'):_copy(PARENT/'plans'/name,work/'plans'/name)
    for name in ('isolated_costs.json','ddr.json'):_copy(SOURCE/name,work/'plans'/name)
    _copy(SOURCE/'source_provenance.json',work/'plans/component_source_provenance.json')
    if (output/'RANK_COST.json').is_file():_copy(output/'RANK_COST.json',work/'plans/rank_cost.json')
    for p in selected.iterdir():
        if p.is_file():_copy(p,work/'plans'/('ranking_freeze.json' if p.name=='FREEZE.json' else p.name))
    c.write(work/'plans/programs.json',programs);c.write(work/'plans/full_programs.json',all_programs)
    policy=c.read(PARENT/'plans/executors.json')['policy'];_copy(PARENT/policy['path'],work/policy['path'],policy['sha256'])
    c.write(work/'plans/executors.json',{'policy':policy})
    c.write(work/'plans/board_execution.json',dict(revision='staged-ablation-1',udmabuf_bytes=201326592,
        storage_margin_bytes=256*1024**2,boot_policy=protocol()['boot_policy']))
    for key,proof in proofs.items():
        source=Path(proof['source_directory']);target=work/'provenance/search'/key
        for name in ('search.py','oracle.cc','picojson.h','domain.py','costs.py','__init__.py'):
            if (source/name).is_file():_copy(source/name,target/'source'/name)
            elif name=='__init__.py':(target/'source'/name).write_text('')
            else:raise ValueError('search source closure missing '+name)
        for p in (output/key/'search').glob('*.json'):
            if p.stem in [ctx['id'] for ctx in __import__('cpu_policies.open_resources.domain',fromlist=['contexts']).contexts()]:
                _copy(p,target/'checkpoints'/p.name)
    shutil.copytree(output/'evidence',work/'provenance/evidence')
    evidence=c.read(output/'evidence/EVIDENCE_CHAIN.json');copied={}
    for i,(name,info) in enumerate(evidence['source_files'].items()):
        dest=work/'provenance/historical_sources'/(str(i)+'_'+Path(name).name)
        _copy(Path(name),dest,info['sha256']);copied[name]=str(dest.relative_to(work))
    c.write(work/'provenance/historical_sources.json',copied)
    snapshot=_controller(work)
    tests=subprocess.run([sys.executable,'-m','pytest','-c','pyproject.toml','-q','tests/test_staged_ablation.py'],
        cwd=ROOT,env=dict(os.environ,PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'),capture_output=True,text=True,timeout=240)
    c.write(work/'plans/offline_tests.json',dict(passed=tests.returncode==0,stdout=tests.stdout,stderr=tests.stderr))
    if tests.returncode:raise RuntimeError(tests.stdout+tests.stderr)
    emulator=shutil.which('qemu-aarch64-static')
    if not emulator:raise RuntimeError('QEMU required')
    loaded=subprocess.run([emulator,str(work/'runtime/ld-linux-aarch64.so.1'),'--library-path',str(work/'runtime'),
        str(work/policy['path']),'--help'],capture_output=True,text=True,timeout=45)
    if loaded.returncode or '--cpu-worker-pinning' not in loaded.stdout+loaded.stderr:raise ValueError('QEMU runner load failed')
    c.write(work/'plans/qemu_loader.json',dict(passed=True,runner_sha256=policy['sha256'],stdout=loaded.stdout,stderr=loaded.stderr))
    from cpu_vta_partitioning.prepare.payload import execution_files
    payload=execution_files(work);size=sum(v['bytes'] for v in payload.values())
    c.write(work/'plans/board_payload.json',dict(files=payload,unpacked_bytes=size))
    c.write(work/'plans/preparation_cost.json',dict(elapsed_seconds=time.monotonic()-started,candidate_count=len(pool),
        execution_bytes=size,compiled_runners=0,stage_modules_compiled=0,runtime_rebuilt=False))
    c.write(work/'OFFLINE_READINESS.json',dict(status='PREPARED_OFFLINE',candidate_count=len(pool),formal_sessions=2*len(pool),
        complete_inferences=29*len(pool),tests_passed=True,new_component_profiles=0,same_pipeline_executor=True))
    child=subprocess.run([sys.executable,'-c','from staged_ablation.common import identity; print(identity())'],
        cwd=snapshot,capture_output=True,text=True,check=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
    manifest=dict(experiment=protocol()['experiment'],controller_identity=child.stdout.strip(),files=c.inventory(work))
    manifest['bundle_id']=c.digest(manifest)[:24];c.write(work/'manifest.json',manifest)
    target=ROOT/'artifacts/bundles'/manifest['bundle_id'];work.rename(target)
    checked=subprocess.run([sys.executable,str(target/'sources/staged_ablation_controller/run_staged_ablation.py'),'validate','--bundle',str(target)],
        cwd=target/'sources/staged_ablation_controller',capture_output=True,text=True,timeout=240)
    if checked.returncode:raise RuntimeError(checked.stdout+checked.stderr)
    return target
