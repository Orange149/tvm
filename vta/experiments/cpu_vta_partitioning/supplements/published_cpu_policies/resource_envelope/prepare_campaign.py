"""Package frozen lists and adapt the already tested transport controller."""
from pathlib import Path
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

from run_resource_behavior import ROOT,PARENT,SELECTION,SOURCE,api
from .model import score
from .selection import profile_scales,select
from .search_adapter import _replace

PROTOCOL={'experiment':'resnet18_resource_envelope_top20','seed':20261109,'input_index':0,'queue_depth':2,
    'performance':{'warmup':8,'scored':16,'tail':4},
    'environment':{'cores':4,'frequency_khz':1066666,'poll_sleep_ns':1000,'post_start_sleep_ns':1000,
        'spin_count':0,'exclude_worker0':1,'bind_threads':1},
    'top_k':20,'max_pool':60,'boot_policy':'one_boot_no_automatic_reboot','anchors':False,
    'automatic_remeasurement':False,'performance_rejection_threshold':None,'single_boot_observation_only':True,
    'new_component_profiles':0,'runtime_changes':False,'board_reboot_forbidden':True}

def model_settings(search_cost):
    distribution=search_cost.get('cpu_distribution','balanced')
    variant=search_cost.get('variant','combined')
    if distribution not in ('balanced','session_weights') or variant not in ('original','combined'):
        raise ValueError('unsupported static model settings')
    return {'cpu_distribution':distribution,'score_variant':variant}

def campaign_protocol(settings):
    weighted=settings['cpu_distribution']=='session_weights'
    result={**PROTOCOL,**settings}
    result['method_labels']={'pure_static20':'原均分核时模型严格前20项',
        'diverse20':('逐线程核时模型' if settings['score_variant']=='original' else '逐线程核时＋资源包络')+'严格前20项'
            if weighted else '资源包络严格前20项',
        'coverage20':('逐线程核时模型' if settings['score_variant']=='original' else '逐线程核时＋资源包络')+'按资源行为筛选20项'
            if weighted else '资源包络按资源行为筛选20项'}
    if weighted:result['experiment']='resnet18_thread_work_'+settings['score_variant']+'_top20'
    return result

def rank(search_root):
    from selection_top20.static import compact_records
    from cpu_policies.open_resources.costs import Costs
    from cpu_policies.open_resources.domain import contexts,configuration
    c=api();search_root=Path(search_root).resolve();ranked=c.read(search_root/'rankings.json')
    source_manifest=c.read(SELECTION/'manifest.json')
    if c.digest({k:v for k,v in source_manifest.items() if k!='bundle_id'})[:24]!=source_manifest['bundle_id']:
        raise ValueError('profile source manifest differs')
    for name in ('isolated_costs.json','ddr.json'):
        if c.sha256(SOURCE/name)!=source_manifest['files']['plans/'+name]['sha256']:
            raise ValueError('profile source differs')
    search_cost=c.read(search_root/'cost.json')
    settings=model_settings(search_cost)
    if (search_cost['costs_sha256']!=c.sha256(SOURCE/'isolated_costs.json')
            or search_cost['ddr_sha256']!=c.sha256(SOURCE/'ddr.json')):
        raise ValueError('search and ranking costs differ')
    if not ranked['verification']['passed'] or ranked['counts']['enhanced']!=966240306:
        raise ValueError('complete-domain search must finish before ranking')
    programs=c.read(PARENT/'plans/programs.json')
    checkpoints=[c.read(search_root/'search'/(ctx['id']+'.json')) for ctx in contexts()]
    for cp in checkpoints:
        from cpu_policies.open_resources.search import _digest
        if cp['checkpoint_sha256']!=_digest({k:v for k,v in cp.items() if k!='checkpoint_sha256'}):raise ValueError('checkpoint differs')
    records,expand=compact_records(programs,checkpoints)
    source_costs=c.read(SOURCE/'isolated_costs.json')['costs'];ddr=c.read(SOURCE/'ddr.json')
    costs=Costs(source_costs,programs=programs,ddr=ddr,distribution=settings['cpu_distribution'])
    baseline=Costs(source_costs,programs=programs,ddr=ddr)
    behavior,selection=select(records,expand,profile_scales(programs,costs),
        preserve_worker_roles=settings['cpu_distribution']=='session_weights')
    original=c.read(SOURCE/'static_rankings.json')['pure_static20']
    result={'pure_static20':original,'diverse20':ranked['pure_static20'],'coverage20':behavior,
        'selection_protocol':selection,'counts':ranked['counts'],'model_settings':settings}
    pool={}
    for method in ('pure_static20','diverse20','coverage20'):
        for row in result[method]:
            cfg={**configuration(row),'id':row['id']};prediction=score(cfg,costs)
            value=prediction[settings['score_variant']]
            if method!='pure_static20' and row['score_ns']!=value:raise ValueError('search and scalar model disagree')
            cfg.update(score_ns=value,score_ms=value/1e6,
                original_score_ns=score(cfg,baseline)['original'],host_guard_ns=prediction['host_concentration_guard_ns'],
                team_guard_ns=prediction['CPU_team_guard_ns'])
            if row['id'] in pool and pool[row['id']]!=cfg:raise ValueError('pool identity differs')
            pool[row['id']]=cfg
    if len(pool)>60:raise ValueError('candidate budget exceeded')
    target=search_root/'frozen_selection';target.mkdir(exist_ok=True)
    c.write(target/'rankings.json',result,replace=True);c.write(target/'pool.json',dict(sorted(pool.items())),replace=True)
    c.write(target/'FREEZE.json',{'rankings_sha256':c.sha256(target/'rankings.json'),'pool_sha256':c.sha256(target/'pool.json'),
        'frozen_at':c.now(),'pipeline_FPS_used':False,'scope':'component-only static ranking; all later evaluation is conditional on this frozen union'},replace=True)
    return target

def _controller(work,protocol_value=None):
    protocol_value=protocol_value or campaign_protocol({'cpu_distribution':'balanced','score_variant':'combined'})
    c=api();base=SELECTION/'sources/selection_controller';snapshot=work/'sources/envelope_controller'
    target=snapshot/'envelope_campaign';target.mkdir(parents=True,exist_ok=True)
    for name in ('campaign.py','runtime.py'):
        text=(base/'selection_top20'/name).read_text().replace('[AutoTVM Top20]','[资源包络 Top20]')
        (target/name).write_text(text)
    common=(base/'selection_top20/common.py').read_text()
    common=common.replace("'run_diverse_top20.py'","'run_envelope_top20.py'")
    common=common.replace("'selection_top20'","'envelope_campaign'")
    common=common.replace('test_selection_top20.py','test_envelope_top20.py')
    common=common.replace("('envelope_campaign', 'cpu_policies')","('envelope_campaign', 'resource_envelope', 'cpu_policies')")
    common=common.replace("'resnet18_autotvm_diverse_top20_01'","'resnet18_resource_envelope_top20_01'")
    if protocol_value['cpu_distribution']=='session_weights':
        common=common.replace('resnet18_resource_envelope_top20_01',protocol_value['experiment']+'_01')
    common+="\nRUN_ROOT=PROJECT/'supplements/published_cpu_policies/runs'\n"
    (target/'common.py').write_text(common)
    (target/'__init__.py').write_text('"""Immutable component-only envelope campaign."""\n')
    (target/'protocol.json').write_text(json.dumps(protocol_value,ensure_ascii=False,indent=2)+'\n')
    _copy=lambda src,dst: (Path(dst).parent.mkdir(parents=True,exist_ok=True),shutil.copy2(src,dst))
    _copy(Path(__file__).with_name('controller_validation.py'),target/'prepare.py')
    cli=(base/'selection_top20/cli.py').read_text()
    cli=cli.replace('ROOT,DEFAULT_RUN_ID,DEFAULT_LOCAL_CONFIG','ROOT,DEFAULT_RUN_ID,DEFAULT_LOCAL_CONFIG,RUN_ROOT')
    cli=cli.replace("p.add_argument('--run-id',default=DEFAULT_RUN_ID)",
        "p.add_argument('--run-id',default=DEFAULT_RUN_ID);p.add_argument('--run-root',type=Path,default=RUN_ROOT)")
    cli=cli.replace("root=ROOT/'runs'/args.run_id","root=args.run_root.resolve()/args.run_id")
    cli=cli.replace('AutoTVM覆盖筛选与冻结Top-20单启动验证','资源包络与行为家族Top-20验证')
    # Runtime audit belongs to the earlier capsule; this campaign keeps the
    # same binary and carries that audit as provenance rather than rerunning it.
    cli=cli.replace("p=sub.add_parser('audit-runtime');p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)","")
    (target/'cli.py').write_text(cli)
    _copy(Path(__file__).with_name('analysis_template.py'),target/'analysis.py')
    (snapshot/'run_envelope_top20.py').write_text('from envelope_campaign.cli import main\nif __name__ == "__main__": main()\n')
    for name in ('pyproject.toml',):_copy(ROOT/name,snapshot/name)
    _copy(ROOT/'tests/test_envelope_top20.py',snapshot/'tests/test_envelope_top20.py')
    for name in ('cpu_policies','configs'):
        shutil.copytree(ROOT/name,snapshot/name,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
    # Only the pure score implementation is needed for validation at runtime.
    for leaf in ('__init__.py','model.py'):_copy(Path(__file__).parent/leaf,snapshot/'resource_envelope'/leaf)
    c.write(work/'plans/controller_origin.json',{'parent_manifest_sha256':c.sha256(SELECTION/'manifest.json'),
        'parent_files':{name:c.sha256(base/'selection_top20'/name) for name in ('campaign.py','runtime.py','common.py','cli.py','analysis.py')},
        'adaptation':'unchanged detached transport/raw validation; separate protocol, validation and analysis labels; first-five preservation removed'})
    return snapshot

def prepare(search_root):
    from cpu_policies.diagnosis.prepare import _copy
    c=api();started=time.monotonic();search_root=Path(search_root).resolve();selected=rank(search_root)
    pool=c.read(selected/'pool.json');ranked=c.read(selected/'rankings.json')
    protocol_value=campaign_protocol(ranked['model_settings'])
    source_manifest=c.read(PARENT/'manifest.json');c.verify_inventory(PARENT,source_manifest['files'])
    folder=ROOT/'artifacts/envelope_top20_builds';folder.mkdir(parents=True,exist_ok=True)
    work=Path(tempfile.mkdtemp(prefix='prepare-',dir=folder))
    for directory in ('runtime','firmware','inputs','references'):
        for name,info in c.inventory(PARENT/directory).items():_copy(PARENT/directory/name,work/directory/name,info['sha256'])
    for name in ('base_controller','cpu_strategy','native_source'):
        shutil.copytree(PARENT/'sources'/name,work/'sources'/name,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
    for p in (SELECTION/'sources/reference').rglob('*'):
        if p.is_file():_copy(p,work/p.relative_to(SELECTION))
    all_programs=c.read(PARENT/'plans/programs.json');names={s['program'] for r in pool.values() for s in r['stages']}
    programs={name:all_programs[name] for name in sorted(names)}
    for name,p in programs.items():
        for leaf,info in p['files'].items():_copy(PARENT/'programs'/name/leaf,work/'programs'/name/leaf,info['sha256'])
    for name in ('inputs.json','runtime_build.json','compiler_identity.json','sdk_runtime.json'):_copy(PARENT/'plans'/name,work/'plans'/name)
    for name in ('isolated_costs.json','ddr.json'):_copy(SOURCE/name,work/'plans'/name)
    for a,b in [('rankings.json','selection_rankings.json'),('pool.json','static_pool.json')]:_copy(selected/a,work/'plans'/b)
    _copy(selected/'FREEZE.json',work/'plans/ranking_freeze.json')
    c.write(work/'plans/programs.json',programs);c.write(work/'plans/full_programs.json',all_programs)
    policy=c.read(PARENT/'plans/executors.json')['policy'];_copy(PARENT/policy['path'],work/policy['path'],policy['sha256'])
    c.write(work/'plans/executors.json',{'policy':policy});c.write(work/'plans/selection_protocol.json',protocol_value)
    c.write(work/'plans/board_execution.json',{'revision':'resource-envelope-top20-1','udmabuf_bytes':201326592,
        'storage_margin_bytes':256*1024**2,'boot_policy':protocol_value['boot_policy']})
    source_ranked=c.read(search_root/'rankings.json')
    c.write(work/'plans/source_provenance.json',{'domain_counts':ranked['counts'],
        'oracle_verification':source_ranked['verification'],'static_search_cost':c.read(search_root/'cost.json'),
        'source_model_identity':c.read(search_root/'source_identity.json'),'new_component_profiles':0,
        'selection':ranked['selection_protocol'],'model_settings':ranked['model_settings'],'pipeline_FPS_used':False})
    # History annotation happens only after the lists have been frozen. It is
    # evaluation metadata, never an input to either score or selection.
    from .history_labels import annotate
    c.write(work/'plans/history_labels.json',annotate(pool))
    shutil.copytree(search_root/'sources',work/'provenance/generated_dp')
    shutil.copytree(Path(__file__).parent,work/'provenance/resource_envelope',
        ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
    _copy(ROOT/'run_envelope_top20.py',work/'provenance/run_envelope_top20.py')
    for p in (search_root/'search').glob('*.json'):_copy(p,work/'provenance/search'/p.name)
    _copy(Path(__file__).parent/'DESIGN_zh.md',work/'provenance/DESIGN_zh.md')
    snapshot=_controller(work,protocol_value)
    tests=subprocess.run([sys.executable,'-m','pytest','-c','pyproject.toml','-q','tests/test_resource_envelope.py','tests/test_envelope_top20.py','tests/test_weighted_full_domain.py'],
        cwd=ROOT,env=dict(os.environ,PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'),capture_output=True,text=True,timeout=180)
    c.write(work/'plans/offline_tests.json',{'passed':tests.returncode==0,'stdout':tests.stdout,'stderr':tests.stderr})
    if tests.returncode:raise RuntimeError(tests.stdout+tests.stderr)
    emulator=shutil.which('qemu-aarch64-static')
    if not emulator:raise RuntimeError('QEMU missing')
    result=subprocess.run([emulator,str(work/'runtime/ld-linux-aarch64.so.1'),'--library-path',str(work/'runtime'),
        str(work/policy['path']),'--help'],capture_output=True,text=True,timeout=45)
    if result.returncode or '--cpu-worker-pinning' not in result.stdout+result.stderr:
        raise ValueError('QEMU loader failed')
    c.write(work/'plans/qemu_loader.json',{'passed':True,'runner_sha256':policy['sha256'],
        'stdout':result.stdout,'stderr':result.stderr})
    from cpu_vta_partitioning.prepare.payload import execution_files
    payload=execution_files(work);size=sum(v['bytes'] for v in payload.values())
    c.write(work/'plans/board_payload.json',{'files':payload,'unpacked_bytes':size})
    c.write(work/'plans/preparation_cost.json',{'elapsed_seconds':time.monotonic()-started,
        'candidate_count':len(pool),'execution_bytes':size,'compiled_runners':0,'stage_modules_compiled':0,'runtime_rebuilt':False})
    c.write(work/'OFFLINE_READINESS.json',{'status':'PREPARED_OFFLINE','candidate_count':len(pool),
        'formal_sessions':len(pool)*2,'complete_inferences':len(pool)*29,'tests_passed':True,
        'new_component_profiles':0,'same_pipeline_executor':True})
    child=subprocess.run([sys.executable,'-c','from envelope_campaign.common import identity; print(identity())'],
        cwd=snapshot,capture_output=True,text=True,check=True)
    manifest={'experiment':protocol_value['experiment'],'controller_identity':child.stdout.strip(),'files':c.inventory(work)}
    manifest['bundle_id']=c.digest(manifest)[:24];c.write(work/'manifest.json',manifest)
    target=ROOT/'artifacts/bundles'/manifest['bundle_id'];work.rename(target)
    verify=subprocess.run([sys.executable,str(target/'sources/envelope_controller/run_envelope_top20.py'),'validate','--bundle',str(target)],
        cwd=target/'sources/envelope_controller',capture_output=True,text=True,timeout=120)
    if verify.returncode:raise RuntimeError(verify.stdout+verify.stderr)
    return target
