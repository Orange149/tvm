"""Offline assembly of both executors, all 172 modules and audited cost imports."""
from pathlib import Path
from importlib.metadata import version
import json
import os
import shutil
import subprocess
import sys
import tempfile

from cpu_vta_partitioning.common import REPO,digest,inventory,locked,read,sha256,verify_inventory,write
from cpu_vta_partitioning.prepare.payload import execution_files
from cpu_vta_partitioning.prepare.package import elf
from cpu_vta_partitioning.prepare.build import sdk_command
from cpu_vta_partitioning.prepare.toolchain import inspect_toolchain
from .. import prepare as legacy
from ..evidence import Evidence as LegacyEvidence
from .domain import ROOT,PROJECT,protocol,component,component_key,policy_components,complete_components
from .native import patch


DEFAULT_PARENT=PROJECT/'artifacts/bundles/62a5cff7a2439cf3a7828476'
DEFAULT_GROUP=ROOT/'artifacts/bundles/c471d4e0d4ab822f052faa76'
DEFAULT_SOURCE=ROOT/'runs/resnet18_cpu_strategy_01'


def identity():
    return digest({'sources':{str(p.relative_to(ROOT)):sha256(p) for p in [ROOT/'run.py',*sorted((ROOT/'cpu_strategy').rglob('*.py'))]},
                   'protocol':protocol(),'dependencies':{n:version(n) for n in ('numpy','scipy','matplotlib')},
                   'native':{p.name:sha256(p) for p in legacy._native_source().iterdir() if p.suffix in ('.cc','.h')},
                   'base_controller':legacy.base_identity(),'python':sys.version})


def payload(parent,workspace):
    manifest=read(parent/'manifest.json');verify_inventory(parent,manifest['files'])
    programs=read(parent/'plans/programs.json')
    if len(programs)!=172:raise ValueError('complete frozen 172-module domain is required')
    for name,p in programs.items():
        for filename,item in p['files'].items():legacy._copy(parent/'programs'/name/filename,workspace/'programs'/name/filename,item['sha256'])
    inputs=read(parent/'plans/inputs.json');wanted=set();derived=[]
    for index in legacy.AVAILABLE_INDICES:wanted.update(legacy._record_files(inputs['records'][index]))
    for name in sorted(wanted):legacy._copy(parent/name,workspace/name,manifest['files'][name]['sha256'])
    for index in legacy.AVAILABLE_INDICES:
        record=inputs['records'][index]
        for p in programs.values():
            if p['begin']==0:continue
            for node in p['input_node_ids']:
                f=record['frontiers'][str(node)]
                if 'raw' in f:continue
                import numpy as np
                a=np.load(workspace/f['npy'],allow_pickle=False)
                if list(a.shape)!=f['shape'] or str(a.dtype)!=f['dtype']:raise ValueError('frontier NPY metadata differs')
                name=f'inputs/derived_frontiers/{index:03}/{node}.bin';target=workspace/name;target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(a.tobytes(order='C'));f['raw']=name;wanted.add(name)
                derived.append({'path':name,'source':f['npy'],'sha256':sha256(target)})
    inputs['available_indices']=list(legacy.AVAILABLE_INDICES)
    inputs['files']={n:{'sha256':sha256(workspace/n),'bytes':(workspace/n).stat().st_size} for n in sorted(wanted)}
    write(workspace/'plans/inputs.json',inputs,replace=True);write(workspace/'plans/programs.json',programs,replace=True)
    write(workspace/'plans/derived_inputs.json',derived,replace=True)
    for folder in ('runtime','firmware'):
        for name,item in inventory(parent/folder).items():legacy._copy(parent/folder/name,workspace/folder/name,item['sha256'])
    for name in ('compiler_identity.json','runtime_build.json','sdk_runtime.json'):legacy._copy(parent/'plans'/name,workspace/'plans'/name)
    return programs


def import_profiles(source,group_bundle,workspace):
    if source is None:return {'records':{},'source':None}
    source=Path(source).resolve();group_bundle=Path(group_bundle).resolve()
    check=subprocess.run([sys.executable,str(group_bundle/'sources/cpu_strategy/run.py'),'validate','--bundle',str(group_bundle)],capture_output=True,text=True)
    if check.returncode:raise ValueError('legacy source bundle does not validate: '+check.stderr[-1500:])
    old=read(group_bundle/'manifest.json');table=read(source/'costs.json')
    if table['source']!='fresh_isolated_board_calibration' or table['bundle_id']!=old['bundle_id']:raise ValueError('source costs have wrong identity')
    boot=table['boot_id'];env=read(source/('environment-'+boot+'.json'))
    if env['identity']['bundle_id']!=old['bundle_id'] or env['identity']['boot_id']!=boot:raise ValueError('source environment boot differs')
    freq=[line.split()[-1] for line in env['raw'].splitlines() if 'scaling_cur_freq ' in line]
    if freq!=['1066666']*4:raise ValueError('source frequency differs from the frozen environment')
    ev=LegacyEvidence(source,old['bundle_id'],group_bundle)
    target=workspace/'sources/profile_imports/source';target.mkdir(parents=True,exist_ok=True)
    clone=workspace/'sources/profile_imports/bundle'
    if clone.exists():shutil.rmtree(clone)
    shutil.copytree(group_bundle,clone)
    legacy._copy(source/'costs.json',target/'costs.json');legacy._copy(source/('environment-'+boot+'.json'),target/('environment-'+boot+'.json'))
    result={}
    for signature,blocks in sorted(table['blocks'].items()):
        qualification=read(source/'records'/boot/'components/qualify'/(signature+'.json'))
        for entry in [qualification,*blocks]:
            receipt=ev.summary(entry['summary'],entry['document'],entry['boot_id'])
            for name,item in receipt['files'].items():legacy._copy(source/name,target/name,item['sha256'])
            ref=entry['summary']['_receipt'];legacy._copy(source/ref['path'],target/ref['path'],ref['sha256'])
        c=component(blocks[0]['document']['stages'][0]);key=component_key(c);program=c['stages'][0]['program']
        required=[n for n in old['files'] if n.startswith(('runtime/','firmware/','programs/'+program+'/'))]
        required+=list(read(group_bundle/'plans/inputs.json')['files'])
        result[key]={'configuration':c,'cost':{f:table['costs'][signature][f] for f in (
            'service_ms','run_core_ms','boundary_core_ms','set_ms','run_ms','get_ms','set_core_ms','get_core_ms')},
            'source_root':'sources/profile_imports/source','source_bundle':'sources/profile_imports/bundle',
            'source_bundle_id':old['bundle_id'],'original_signature':signature,'original_boot_id':boot,
            'blocks':blocks,'qualification':qualification,'required_assets':sorted(set(required)),
            'provenance':'byte_matched_legacy_component_no_new_boot_claim'}
    return {'source':str(source),'source_costs_sha256':sha256(source/'costs.json'),'records':result,
            'original_blocks':sum(len(v) for v in table['blocks'].values())}


def build_native(workspace,local,group_bundle):
    group=legacy.build_runner(workspace,local)
    expected=sha256(Path(group_bundle)/'runner/vta_stage_pipeline_runner')
    if sha256(workspace/'runner/vta_stage_pipeline_runner')!=expected:raise ValueError('legacy group executor is not byte-identical; reuse is refused')
    legacy._copy(workspace/'sources/vta/apps/native_deploy/vta_stage_pipeline_runner.cc',workspace/'sources/group_runner.cc')
    legacy.native_acceptance(workspace)
    source=(legacy._native_source()/'vta_stage_pipeline_runner.cc').read_text();patched,meta=patch(source)
    snapshot=workspace/'sources/vta/apps/native_deploy';(snapshot/'vta_stage_pipeline_runner.cc').write_text(patched)
    (workspace/'sources/policy_runner.patch').write_text(meta.pop('diff'))
    recipe=group['recipe'];output=workspace/'runner/vta_stage_pipeline_runner_pinned'
    sdk_command(local,['bash','-c',recipe,'policy-runner',snapshot/'vta_stage_pipeline_runner.cc',output,REPO,workspace/'runtime',snapshot],workspace/'build_logs/policy_runner.log')
    write(workspace/'plans/policy_runner_patch.json',meta,replace=True)
    write(workspace/'plans/executors.json',{'legacy':{'path':'runner/vta_stage_pipeline_runner','sha256':expected},
        'policy':{'path':'runner/vta_stage_pipeline_runner_pinned','sha256':sha256(output)}},replace=True)
    return meta


def native_acceptance(workspace):
    qemu=shutil.which('qemu-aarch64-static')
    if not qemu:raise RuntimeError('QEMU required for offline native acceptance')
    runner=workspace/'runner/vta_stage_pipeline_runner_pinned';loader=workspace/'runtime/ld-linux-aarch64.so.1'
    closure={}
    for path in [*list((workspace/'runner').iterdir()),*[workspace/'programs'/n/'module.so' for n in read(workspace/'plans/programs.json')],*list((workspace/'runtime').iterdir())]:
        if not path.is_file():continue
        info=elf(path)
        if any(not (workspace/'runtime'/n).is_file() for n in info['needed']):raise ValueError('unresolved native dependency')
        closure[str(path.relative_to(workspace))]=info
    write(workspace/'plans/elf_dependencies.json',closure,replace=True)
    result=subprocess.run([qemu,str(loader),'--library-path',str(workspace/'runtime'),str(runner),'--help'],capture_output=True,text=True,timeout=45)
    if result.returncode or '--cpu-worker-pinning' not in result.stdout+result.stderr:raise ValueError('native pinned parser/loader failed')
    # Host CPU ids are used only under QEMU; board qualification always uses 0-3.
    cpus=sorted(os.sched_getaffinity(0))[:2]
    from cpu_vta_partitioning.measurement.tasks import task,native_command
    from cpu_vta_partitioning.domain import Stage
    from cpu_vta_partitioning.measurement import frames,correctness,input_sequence
    output=workspace/'build_logs/qemu_pinned';output.mkdir(parents=True,exist_ok=True)
    doc=task('qualify_component',[Stage('cpu',2,3,2,tuple(cpus))],{'cores':4,'poll_sleep_ns':1000,'post_start_sleep_ns':1000},warmup=0,scored=4,tail=0,indices=list(legacy.AVAILABLE_INDICES))
    args,lines=native_command(doc,workspace,str(workspace),str(output));(output/'inputs.txt').write_text(lines)
    position=args.index(str(workspace/'runner/vta_stage_pipeline_runner'));args[position]=str(runner)
    args+=['--allow-cpu-affinity-superset','--cpu-worker-pinning','--resource-affinity-probe']
    args.insert(args.index(str(loader)),qemu)
    test=subprocess.run(args,capture_output=True,text=True,timeout=60);(output/'native.log').write_text(test.stdout+test.stderr)
    if test.returncode:raise RuntimeError('QEMU actual pinned component failed: '+test.stderr[-2500:])
    rows=frames(output/'result.jsonl',4);input_sequence(rows,doc);correctness(rows,doc,workspace,output)
    affinity=read(output/'affinity.json')['stages'][0];slots=affinity['worker_slots']
    if len(slots)!=2 or sorted((s['slot'],s['cpus']) for s in slots)!=[(i,[c]) for i,c in enumerate(cpus)]:raise ValueError('QEMU slot pinning is not bijective')
    if next(s for s in slots if s['slot']==0)['tid']!=affinity['caller_tid']:raise ValueError('QEMU caller not worker0')
    record={'passed':True,'scope':'172-module aarch64 ELF closure, pinned CPU numerics and logical-worker mapping under QEMU; physical board qualification pending',
            'qemu_cores':cpus,'worker_slots':slots}
    write(workspace/'plans/policy_native_acceptance.json',record,replace=True)
    return record


def extra_native_acceptance(workspace,local):
    from cpu_vta_partitioning.measurement.tasks import task,native_command
    from cpu_vta_partitioning.domain import Stage
    from cpu_vta_partitioning.measurement import frames,correctness,input_sequence
    cpus=sorted(os.sched_getaffinity(0))[:4]
    if len(cpus)<4:raise RuntimeError('offline pinning acceptance needs four allowed host CPUs')
    qemu=shutil.which('qemu-aarch64-static');loader=workspace/'runtime/ld-linux-aarch64.so.1'
    runner=workspace/'runner/vta_stage_pipeline_runner_pinned';cases={}
    for n in (1,3,4):
        out=workspace/'build_logs'/('qemu_pinned_'+str(n));out.mkdir(parents=True,exist_ok=True)
        d=task('qualify_component',[Stage('cpu',2,3,n,tuple(cpus[:n]))],{'cores':4,'poll_sleep_ns':1000,'post_start_sleep_ns':1000},
               warmup=0,scored=4,tail=0,indices=list(legacy.AVAILABLE_INDICES))
        args,lines=native_command(d,workspace,str(workspace),str(out));(out/'inputs.txt').write_text(lines)
        args[args.index(str(workspace/'runner/vta_stage_pipeline_runner'))]=str(runner)
        args+=['--allow-cpu-affinity-superset','--cpu-worker-pinning','--resource-affinity-probe']
        args.insert(args.index(str(loader)),qemu)
        result=subprocess.run(args,capture_output=True,text=True,timeout=60);(out/'native.log').write_text(result.stdout+result.stderr)
        if result.returncode:raise RuntimeError('QEMU pinned width '+str(n)+' failed: '+result.stderr[-1800:])
        rows=frames(out/'result.jsonl',4);input_sequence(rows,d);correctness(rows,d,workspace,out)
        report=read(out/'affinity.json')['stages'][0];slots=report['worker_slots']
        if sorted((r['slot'],r['cpus']) for r in slots)!=[(i,[cpu]) for i,cpu in enumerate(cpus[:n])]:raise ValueError('pinned width mapping failed')
        if next(r for r in slots if r['slot']==0)['tid']!=report['caller_tid']:raise ValueError('caller mapping failed')
        cases[str(n)]=slots
    snapshot=workspace/'sources/vta/apps/native_deploy';source=workspace/'build_logs/policy_control_selftest.cc'
    source.write_text(r'''#define VTA_STAGE_PIPELINE_NO_MAIN
#include "vta_stage_pipeline_runner.cc"
int main(int argc,char** argv) {
  auto cpus=ParseCPUList(argv[1]);
  CapturePolicyControl(0,{cpus.at(0)});
  std::thread producer([&]{CapturePolicyControl(1,{cpus.at(1)});});producer.join();
  if (ParseCPUList(CurrentThreadAffinity())!=std::vector<unsigned int>{cpus.at(0)}) return 10;
  if (policy_control_slots.at(1).cpus!=std::vector<unsigned int>{cpus.at(1)}) return 11;
  Args args;StageArgs s;s.device="cpu";s.runtime_num_threads=2;s.cpu_affinity={0,1,2,3};args.stages={s};
  args.allow_cpu_affinity_superset=true;args.cpu_worker_pinning=true;
  bool rejected=false;try {ValidateStageAffinities(args);}catch(const std::runtime_error&){rejected=true;}
  if(!rejected)return 12;
  args.stages[0].cpu_affinity={0,1};args.process_cpu_affinity={0,1,2,3};rejected=false;
  try {ValidateStageAffinities(args);}catch(const std::runtime_error&){rejected=true;}
  if(!rejected)return 13;
  return 0;
}
''')
    recipe=read(workspace/'plans/parent_identity.json')['native_build']['recipe'] if (workspace/'plans/parent_identity.json').exists() else (
        'exec $CXX -std=c++17 -O2 "$1" -o "$2" -I"$3/include" -I"$3/3rdparty/dlpack/include" -I"$3/3rdparty/dmlc-core/include" '
        '-I"$3/3rdparty/vta-hw/include" -I"$5" -L"$4" -Wl,-rpath,\'$ORIGIN/../runtime\' -ltvm_runtime -lvta -ldl -pthread')
    binary=workspace/'build_logs/policy_control_selftest'
    sdk_command(local,['bash','-c',recipe,'policy-control-selftest',source,binary,REPO,workspace/'runtime',snapshot],workspace/'build_logs/policy_control_build.log')
    result=subprocess.run([qemu,str(loader),'--library-path',str(workspace/'runtime'),str(binary),','.join(map(str,cpus[:2]))],capture_output=True,text=True,timeout=45)
    if result.returncode:raise RuntimeError('control-role native selftest failed: '+result.stderr[-1500:])
    value=read(workspace/'plans/policy_native_acceptance.json');value['extra_pinned_widths']=cases;value['control_role_selftest']=True
    write(workspace/'plans/policy_native_acceptance.json',value,replace=True)


def validate(bundle):
    bundle=Path(bundle).resolve();m=read(bundle/'manifest.json')
    if digest({k:v for k,v in m.items() if k!='bundle_id'})[:24]!=m['bundle_id']:raise ValueError('bundle manifest identity differs')
    if m['controller_identity']!=identity() or m['experiment']!=protocol()['experiment']:raise ValueError('controller changed; use frozen snapshot')
    verify_inventory(bundle,m['files'])
    if read(bundle/'plans/resource_protocol.json')!=protocol():raise ValueError('frozen resource protocol differs')
    if len(read(bundle/'plans/programs.json'))!=172:raise ValueError('full program coverage missing')
    for e in read(bundle/'plans/executors.json').values():
        if sha256(bundle/e['path'])!=e['sha256']:raise ValueError('executor identity differs')
    from .evidence import validate_import
    validate_import(bundle)
    payload=read(bundle/'plans/board_payload.json')
    if payload['files']!=execution_files(bundle):raise ValueError('execution closure differs')
    ready=read(bundle/'OFFLINE_READINESS.json')
    if ready['status']!='PREPARED_OFFLINE' or not ready['tests']['passed']:raise ValueError('offline acceptance incomplete')
    return m


def prepare(parent=DEFAULT_PARENT,group_bundle=DEFAULT_GROUP,source=DEFAULT_SOURCE,local_config=None,config=None):
    if config is not None and read(config)!=protocol():raise ValueError('configuration differs from implemented resource protocol')
    parent=Path(parent).resolve();group_bundle=Path(group_bundle).resolve();local=read(local_config or PROJECT/'configs/local.json')
    code=identity();workspace=ROOT/'artifacts/resource_builds'/digest({'code':code,'parent':read(parent/'manifest.json')['bundle_id']})[:24]
    with locked(workspace):
        programs=payload(parent,workspace)
        tc=inspect_toolchain(workspace,local)
        if tc!=read(parent/'plans/compiler_identity.json')['toolchain']:raise ValueError('SDK toolchain changed')
        build_native(workspace,local,group_bundle);native_acceptance(workspace);extra_native_acceptance(workspace,local)
        imported=import_profiles(source,group_bundle,workspace)
        write(workspace/'plans/imported_profiles.json',imported,replace=True)
        assets={}
        for folder in ('programs','runtime','firmware','inputs','references','runner'):
            assets.update({folder+'/'+k:v for k,v in inventory(workspace/folder).items()})
        write(workspace/'plans/execution_assets.json',assets,replace=True)
        write(workspace/'plans/resource_protocol.json',protocol(),replace=True)
        write(workspace/'plans/policy_components.json',policy_components(),replace=True)
        needed=policy_components();old=set(imported['records']);full={}
        for mode in ('group','pinned'):
            for reserved in (False,True):
                keys=set(complete_components(programs,mode,reserved,True))
                full[mode+('_reserved' if reserved else '_shared')]={'required_representative_signatures':len(keys),
                    'reusable_from_existing_source':len(keys&old),
                    'missing_after_initial_policy_profiles':len(keys-(old|set(needed)))}
        write(workspace/'plans/profile_budget.json',{'policy':{'required':len(needed),'exact_reusable':len(set(needed)&old),
            'missing':len(set(needed)-old)},'full_domain':full,
            'note':'component sessions, not complete pipeline candidates; nonhomogeneous fallback may require more signatures',
            'max_policy_pipeline_configurations':26,'max_top20_pipeline_configurations':30},replace=True)
        # Explicit offline algorithm fixture; never imported as campaign costs.
        fixture=PROJECT/'runs/resnet18_v1_main_01/env0/costs.json'
        if not fixture.is_file():raise ValueError('full-domain algorithm acceptance fixture is missing')
        write(workspace/'plans/search_test_fixture.json',{'scope':'offline_search_test_only_not_campaign_calibration',
            'source_sha256':sha256(fixture),'costs':read(fixture)['costs']},replace=True)
        write(workspace/'plans/board_execution.json',{'revision':'resource-policy-1','udmabuf_bytes':201326592,'storage_margin_bytes':256*1024**2,'boot_policy':'manual_reboot_only'},replace=True)
        files=execution_files(workspace);write(workspace/'plans/board_payload.json',{'revision':1,'files':files,'unpacked_bytes':sum(v['bytes'] for v in files.values())},replace=True)
        from .evidence import validate_import
        validate_import(workspace)
        env=dict(os.environ,PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR='/tmp/cpu-resource-mpl',
                 CPUVTA_CPU_STRATEGY_WORKSPACE=str(workspace),CPUVTA_RESOURCE_WORKSPACE=str(workspace))
        log=workspace/'build_logs/resource_tests.log'
        with log.open('w') as stream:
            t=subprocess.run([sys.executable,'-m','pytest','-c',str(ROOT/'pyproject.toml'),'--confcutdir='+str(ROOT),str(ROOT/'tests'),'-q'],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        if t.returncode:raise RuntimeError('resource acceptance tests failed: '+str(log))
        if identity()!=code:raise ValueError('source changed during preparation')
        legacy._snapshot_controller(workspace)
        legacy._copy(log,workspace/'plans/resource_tests.log')
        write(workspace/'OFFLINE_READINESS.json',{'status':'PREPARED_OFFLINE','controller_identity':code,'programs':len(programs),'new_model_compilations':0,
            'tests':{'passed':True,'result':log.read_text()[-6000:],'sha256':sha256(log)},'board_validation_pending':['actual pinning','frontier correctness','six independent boots','measured conclusions']},replace=True)
        dest=ROOT/'artifacts/bundles';dest.mkdir(parents=True,exist_ok=True)
        staging=Path(tempfile.mkdtemp(prefix='.resource-',dir=dest))
        try:
            for folder in ('programs','runtime','runner','firmware','inputs','references','plans','sources'):shutil.copytree(workspace/folder,staging/folder)
            legacy._copy(workspace/'OFFLINE_READINESS.json',staging/'OFFLINE_READINESS.json')
            m={'schema_version':1,'experiment':protocol()['experiment'],'controller_identity':code,'parent_bundle_id':read(parent/'manifest.json')['bundle_id'],'files':inventory(staging)}
            m['bundle_id']=digest(m)[:24];write(staging/'manifest.json',m);validate(staging)
            with tempfile.TemporaryDirectory(prefix='resource-relocation-') as temp:
                relocated=Path(temp)/'bundle';shutil.copytree(staging,relocated)
                check=subprocess.run([sys.executable,str(relocated/'sources/cpu_strategy/run.py'),'validate','--bundle',str(relocated)],capture_output=True,text=True,timeout=240)
                if check.returncode:raise RuntimeError('relocated resource package failed: '+check.stderr[-2500:])
            result=dest/m['bundle_id']
            if result.exists():validate(result);shutil.rmtree(staging)
            else:staging.rename(result)
            write(ROOT/'artifacts/LATEST_RESOURCE_BUNDLE.json',{'bundle_id':m['bundle_id'],'path':str(result)},replace=True)
            return result
        finally:
            if staging.exists():shutil.rmtree(staging)
