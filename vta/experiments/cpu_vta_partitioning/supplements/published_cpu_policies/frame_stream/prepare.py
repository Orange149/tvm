"""Authenticate old evidence, compile one runner, seal a CPU-only capsule."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
from .common import ROOT, REPO, DEFAULT_PARENT, DEFAULT_SOURCE_RUN, api, identity, protocol
from .domain import cases, program_names
from cpu_policies.diagnosis.prepare import _copy
from cpu_policies.run_scope.prepare import _sdk_checked

def import_source(source_run,parent):
    c=api(parent); source_run=Path(source_run)
    state=c.read(source_run/'state.json')
    if state['status']!='COMPLETE' or state['completed_tasks']!=48:raise ValueError('source experiment is incomplete')
    c.verify_inventory(source_run,state['frozen_files'])
    source_bundle=Path(state['bundle']); manifest=c.read(source_bundle/'manifest.json')
    if c.digest({k:v for k,v in manifest.items() if k!='bundle_id'})[:24]!=state['bundle_id']:raise ValueError('source manifest differs')
    c.verify_inventory(source_bundle,manifest['files'])
    for directory in ('runtime','inputs','references'):
        if c.inventory(parent/directory)!=c.inventory(source_bundle/directory):raise ValueError('source '+directory+' identity differs')
    programs=c.read(source_bundle/'plans/programs.json')
    parent_programs=c.read(parent/'plans/programs.json')
    for name in program_names():
        if programs[name]!=parent_programs[name]:raise ValueError('CPU program identity differs')
    costs=c.read(source_run/'costs.json')
    if costs['pipeline_feedback_used'] is not False or len(costs['costs'])!=15:raise ValueError('invalid source cost table')
    tasks=c.read(source_run/'profile_tasks.json')
    from cpu_policies.contention.runtime import validate_summary as old_validate
    for key,record in costs['costs'].items():
        doc=next(t for t in tasks if t['task_id']==record['task_id'])
        receipt=c.read(source_run/'accepted'/record['boot_id']/(record['task_id']+'.json'))
        if receipt['task']!=doc or receipt['boot_id']!=record['boot_id']:raise ValueError('source receipt identity differs')
        c.verify_inventory(source_run,receipt['files'])
        summary=old_validate(source_run/receipt['raw'],doc,source_bundle)
        if summary!=receipt['summary']:raise ValueError('isolated source replay differs')
        if any(summary['actors']['0'][field]!=value for field,value in record['cost'].items()):raise ValueError('isolated cost differs from raw')
    old=c.read(source_run/'predictions.json')
    if old.get('pipeline_feedback_used') is not False or set(old['predictions'])!=set(cases()):raise ValueError('invalid source predictions')
    freeze=c.read(source_run/'prediction_freeze.json')
    if freeze['predictions_sha256']!=c.sha256(source_run/'predictions.json'):raise ValueError('source prediction freeze differs')
    model={'costs':costs,'original_predictions':{},'reference_rankings':{}}
    for name,row in old['predictions'].items():
        if row['configuration']!=cases()[name]['configuration']:raise ValueError('source configuration differs')
        model['original_predictions'][name]={'original_score_ns':row['baseline_ledger']['score_ns'],
            'saturated_score_ms':row['estimates_ms']['cpu_corun'],'baseline_ledger':row['baseline_ledger']}
    for topology,value in old['rankings'].items():
        model['reference_rankings'][topology]={'original':value['original'],'saturated':value['cpu_corun']}
    reference={}; evidence={}
    final_tasks=c.read(source_run/'final_tasks.json')
    for doc in final_tasks:
        if doc['kind']!='pipeline':continue
        receipt=c.read(source_run/'accepted'/state['boot_id']/(doc['task_id']+'.json'))
        if receipt['task']!=doc or receipt['boot_id']!=state['boot_id']:raise ValueError('source performance receipt differs')
        c.verify_inventory(source_run,receipt['files'])
        if old_validate(source_run/receipt['raw'],doc,source_bundle)!=receipt['summary']:raise ValueError('source performance replay differs')
        name=doc['name'].split('/',1)[1]
        reference[name]={'ii_ms':receipt['summary']['ii_ms'],'fps':receipt['summary']['fps'],
                         'source_task_id':doc['task_id'],'source_boot_id':state['boot_id']}
        evidence[name]={'receipt_sha256':c.sha256(source_run/'accepted'/state['boot_id']/(doc['task_id']+'.json')),
                        'raw_files':receipt['files']}
    if set(reference)!=set(cases()):raise ValueError('source eight-pipeline coverage differs')
    return model,reference,{'source_run':str(source_run),'source_state_sha256':c.sha256(source_run/'state.json'),
        'source_bundle_id':state['bundle_id'],'source_manifest_sha256':c.sha256(source_bundle/'manifest.json'),
        'source_costs_sha256':c.sha256(source_run/'costs.json'),'source_predictions_sha256':c.sha256(source_run/'predictions.json'),
        'pipeline_feedback_used_for_scores':False,'isolated_profiles_replayed':15,'pipeline_results_replayed':8,
        'source_runner_sha256':c.sha256(source_bundle/'runner/vta_stage_pipeline_runner_contention'),
        'source_performance_evidence':evidence,'interpretation':'known-configuration development replay, not independent confirmation'}

def compile_runner(parent,work):
    from .patch_runner import prepare_source
    c=api(parent); native=prepare_source(parent,work/'sources/frame_stream_native')
    record,env=_sdk_checked(parent,work)
    output=work/'runner/vta_stage_pipeline_runner_frame_stream'
    output.parent.mkdir(parents=True,exist_ok=True)
    result=subprocess.run(['bash','-c','source "$1" || exit $?; shift; exec "$@"','frame-stream-sdk',
        record['toolchain']['sdk_setup'],'bash','-c',record['runner_compiler_recipe'],'frame-stream-runner',
        native['source_path'],str(output),str(REPO),str(work/'runtime')],capture_output=True,text=True,env=env,timeout=300)
    (work/'build_logs/frame_stream_compile.log').write_text(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError('frame-stream compiler failed: '+str(work/'build_logs/frame_stream_compile.log'))
    native.update(runner_path=str(output.relative_to(work)),runner_sha256=c.sha256(output),
                  source_path_in_bundle=str(Path(native.pop('source_path')).relative_to(work)),
                  runtime_rebuilt=False,stage_modules_recompiled=False)
    c.write(work/'plans/frame_stream_build.json',native)
    return native

def validate_bundle(bundle):
    bundle=Path(bundle).resolve();c=api(bundle);m=c.read(bundle/'manifest.json')
    if c.digest({k:v for k,v in m.items() if k!='bundle_id'})[:24]!=m['bundle_id']:raise ValueError('manifest identity differs')
    if m['controller_identity']!=identity():raise ValueError('controller changed; use frozen frame_stream_controller')
    c.verify_inventory(bundle,m['files'])
    if c.read(bundle/'plans/frame_stream_protocol.json')!=protocol() or c.read(bundle/'plans/cases.json')!=cases():raise ValueError('protocol or cases differ')
    if set(c.read(bundle/'plans/programs.json'))!=set(program_names()):raise ValueError('CPU program closure differs')
    from cpu_vta_partitioning.prepare.payload import execution_files
    if execution_files(bundle)!=c.read(bundle/'plans/board_payload.json')['files']:raise ValueError('execution closure differs')
    if c.read(bundle/'OFFLINE_READINESS.json')['status']!='PREPARED_OFFLINE':raise ValueError('not prepared')
    return m

def prepare(parent=None,source_run=None):
    parent=Path(parent or DEFAULT_PARENT).resolve();source_run=Path(source_run or DEFAULT_SOURCE_RUN).resolve()
    c=api(parent);code=identity();m=c.read(parent/'manifest.json')
    if c.digest({k:v for k,v in m.items() if k!='bundle_id'})[:24]!=m['bundle_id']:raise ValueError('parent manifest differs')
    c.verify_inventory(parent,m['files'])
    model,reference,provenance=import_source(source_run,parent)
    folder=ROOT/'artifacts/frame_stream_builds';folder.mkdir(parents=True,exist_ok=True)
    work=Path(tempfile.mkdtemp(prefix='prepare-',dir=folder))
    for directory in ('runtime','firmware','inputs','references'):
        for name,item in c.inventory(parent/directory).items():_copy(parent/directory/name,work/directory/name,item['sha256'])
    for directory in ('base_controller','cpu_strategy','native_source','vta','reference'):
        shutil.copytree(parent/'sources'/directory,work/'sources'/directory,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
    for name in ('inputs.json','runtime_build.json','compiler_identity.json','sdk_runtime.json'):
        _copy(parent/'plans'/name,work/'plans'/name)
    programs=c.read(parent/'plans/programs.json');programs={name:programs[name] for name in program_names()}
    for name,p in programs.items():
        for leaf,item in p['files'].items():_copy(parent/'programs'/name/leaf,work/'programs'/name/leaf,item['sha256'])
    c.write(work/'plans/programs.json',programs);c.write(work/'plans/cases.json',cases())
    c.write(work/'plans/model_inputs.json',model)
    c.write(work/'plans/analysis_reference.json',reference)
    c.write(work/'plans/source_provenance.json',provenance)
    c.write(work/'provenance/source_parent_manifest.json',m)
    c.write(work/'plans/frame_stream_protocol.json',protocol())
    c.write(work/'plans/board_execution.json',{'revision':'cpu-frame-stream-1','udmabuf_bytes':201326592,
            'storage_margin_bytes':256*1024**2,'boot_policy':protocol()['boot_policy']})
    native=compile_runner(parent,work)
    c.write(work/'plans/executors.json',{'frame_stream':{'path':native['runner_path'],'sha256':native['runner_sha256']}})
    snapshot=work/'sources/frame_stream_controller'
    for name in ('run_frame_stream.py','pyproject.toml'):_copy(ROOT/name,snapshot/name)
    for name in ('frame_stream','cpu_policies','configs'):
        shutil.copytree(ROOT/name,snapshot/name,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
    _copy(ROOT/'tests/test_frame_stream.py',snapshot/'tests/test_frame_stream.py')
    from cpu_vta_partitioning.prepare.package import elf
    closure={}
    for p in [work/native['runner_path'],*list((work/'runtime').iterdir()),*list((work/'programs').glob('*/module.so'))]:
        if not p.is_file():continue
        info=elf(p)
        if any(not (work/'runtime'/name).is_file() for name in info['needed']):raise ValueError('unresolved shared library')
        closure[str(p.relative_to(work))]=info
    c.write(work/'plans/elf_dependencies.json',closure)
    emulator=shutil.which('qemu-aarch64-static')
    if not emulator:raise RuntimeError('QEMU missing')
    check=subprocess.run([emulator,str(work/'runtime/ld-linux-aarch64.so.1'),'--library-path',str(work/'runtime'),
                         str(work/native['runner_path']),'--help'],capture_output=True,text=True,timeout=45)
    if check.returncode or '--cpu-frame-stream-spec' not in check.stdout+check.stderr:raise ValueError('loader/explicit mode check failed')
    tests=subprocess.run([sys.executable,'-m','pytest','-c','pyproject.toml','-q','tests/test_frame_stream.py'],
        cwd=ROOT,env=dict(os.environ,PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'),capture_output=True,text=True,timeout=180)
    c.write(work/'plans/offline_tests.json',{'passed':tests.returncode==0,'stdout':tests.stdout,'stderr':tests.stderr})
    if tests.returncode:raise RuntimeError(tests.stdout+tests.stderr)
    if code!=identity():raise ValueError('controller changed during preparation')
    from cpu_vta_partitioning.prepare.payload import execution_files
    payload=execution_files(work);c.write(work/'plans/board_payload.json',{'files':payload,'unpacked_bytes':sum(v['bytes'] for v in payload.values())})
    c.write(work/'OFFLINE_READINESS.json',{'status':'PREPARED_OFFLINE','sessions':8,'stage_calls':300,
        'tests':{'passed':True},'CPU_programs':4,'VTA_programs':0,'new_stage_compilations':0,'new_runtime_compilations':0,
        'source_state_sha256':provenance['source_state_sha256'],'board_validation_pending':['frame conservation','affinity','output correctness']})
    manifest={'experiment':protocol()['experiment'],'controller_identity':code,'files':c.inventory(work)}
    manifest['bundle_id']=c.digest(manifest)[:24];c.write(work/'manifest.json',manifest)
    target=ROOT/'artifacts/bundles'/manifest['bundle_id'];work.rename(target);validate_bundle(target)
    return target
