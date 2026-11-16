"""Common, single-boot validation of the frozen run-scope comparison.

Reuse the original executable and existing durable task/measurement helpers.
This entry never reboots, reprograms the FPGA, or retunes the environment.
The larger native campaign is run only after its concrete cost is approved.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import json
import shlex
import shutil
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,write,sha256,digest,inventory,verify_inventory,locked,now,sync_tree
from cpu_vta_partitioning.measurement.tasks import native_command as base_command
from cpu_vta_partitioning.measurement import frames,correctness
from cpu_policies.open_resources.domain import component,configuration
from phase_profile.prepare import COMPONENT_PARENTS,link_checked
from common_top20.prepare import runtime_program
from common_top20.runtime import task,command,summary
from common_top20.campaign import accept,qemu_cpu_summary
from board_transport_without_reprogram import transport
from run_scope_rank import ROOT,freeze

OUT=ROOT/'artifacts/run_scope_common_validation_01'
RUN=ROOT/'runs/run_scope_common_validation_01'
CONFIG=ROOT/'artifacts/cross_network_model_01/board.local.json'
RUNNER_SHA='25833e7f8b47fc8824ff9460afc0b968f24934dcee41c3b1f28619a99b6b6fe6'


def sources():
    project=ROOT.parents[1]/'src/cpu_vta_partitioning'
    paths=[Path(__file__),ROOT/'board_transport_without_reprogram.py',
        ROOT/'common_top20/runtime.py',ROOT/'common_top20/campaign.py',ROOT/'common_top20/prepare.py',
        ROOT/'phase_profile/prepare.py',ROOT/'cpu_policies/open_resources/runtime.py',
        ROOT/'cpu_policies/open_resources/domain.py',ROOT/'yolo_validation/components.py',
        *[project/p for p in ('common.py','transport/__init__.py','transport/scripts.py',
                             'measurement/__init__.py','measurement/tasks.py','prepare/payload.py')]]
    return {str(p):sha256(p) for p in paths}


def selection():
    seal=read(OUT/'FREEZE.json')
    for leaf,wanted in seal['files'].items():
        if sha256(OUT/leaf)!=wanted:
            raise ValueError('frozen validation selection changed')
    plan=read(OUT/'PLAN.json');pool=read(OUT/'POOL.json');order=read(OUT/'ORDER.json')
    if (len(order)!=plan['configurations'] or len({(r['network'],r['configuration']) for r in order})!=len(order)
            or {(r['network'],r['configuration']) for r in order}!={(n,i) for n,p in pool.items() for i in p}):
        raise ValueError('frozen execution order incomplete or duplicated')
    return plan,pool,order


def assert_sources():
    prepared=read(OUT/'PREPARATION.json')
    if prepared['controllers']!=sources() or prepared['selection_sha256']!=sha256(OUT/'FREEZE.json'):
        raise ValueError('prepared controller or selection identity changed')
    return prepared


def validate(bundle):
    bundle=Path(bundle);m=read(bundle/'manifest.json')
    if m['bundle_id']!=digest({k:v for k,v in m.items() if k!='bundle_id'})[:24]:
        raise ValueError('capsule identity differs')
    verify_inventory(bundle,m['files'])
    prep=read(bundle/'plans/preparation.json')
    if prep['controllers']!=sources() or prep['selection_sha256']!=sha256(OUT/'FREEZE.json'):
        raise ValueError('capsule source or selection changed')
    if m['runner_sha256']!=RUNNER_SHA or sha256(bundle/'runner/vta_stage_pipeline_runner')!=RUNNER_SHA:
        raise ValueError('original comparison executable required')
    if read(bundle/'plans/inputs.json')['available_indices']!=[0]:
        raise ValueError('only input zero is frozen')
    rows=read(bundle/'plans/rankings.json');network=m['network']
    if rows['pool']!=read(OUT/'POOL.json')[network]:
        raise ValueError('capsule execution configurations differ')
    for cfg in rows['pool'].values():
        for phase,mode in [('qualify','raw_all_stages'),('short','raw')]:
            doc=task(bundle,cfg,phase);argv,_=command(doc,bundle,str(bundle),'/tmp/scope-command-validation')
            if argv[argv.index('--output-mode')+1]!=mode:
                raise ValueError('qualification or performance output mode differs')
            if argv[argv.index('--cpu-stage-gate')+1]!=cfg['execution_mode']:
                raise ValueError('gate configuration was not transmitted')
    return m


def prepare():
    plan,pool,order=selection();controllers=sources();protocol=read(OUT/'PROTOCOL.json')
    lists=read(OUT/'LISTS.json');rank_plan=read(ROOT/'artifacts/run_scope_rank_01/PLAN.json');bundles={}
    for network,parent in COMPONENT_PARENTS.items():
        bundle=OUT/network/'capsule'
        if (bundle/'manifest.json').exists():
            bundles[network]=validate(bundle)['bundle_id'];continue
        old=read(parent/'manifest.json')
        if old['bundle_id']!=digest({k:v for k,v in old.items() if k!='bundle_id'})[:24]:
            raise ValueError('source capsule identity differs')
        catalog=read(parent/'plans/programs.json')
        frozen_programs=read(rank_plan['networks'][network]['programs_path'])
        names={s['program'] for cfg in pool[network].values() for s in cfg['stages']}
        for name in names:
            for leaf in ('graph.json','module.so','params.bin'):
                if catalog[name]['files'][leaf]['sha256']!=frozen_programs[name]['files'][leaf]['sha256']:
                    raise ValueError('selected program differs from isolated search input')
            for key in ('input_node_ids','output_node_ids','input_names'):
                if catalog[name][key]!=frozen_programs[name][key]:
                    raise ValueError('selected frontier contract differs')
        programs={n:runtime_program(catalog[n]) for n in sorted(names)}
        record=deepcopy(next(r for r in read(parent/'plans/inputs.json')['records'] if r['index']==0))
        nodes={str(node) for p in programs.values() for node in p['input_node_ids']+p['output_node_ids']}
        if not nodes.issubset(record['frontiers']):
            raise ValueError('selected frontier input or reference missing')
        record['frontiers']={n:v for n,v in record['frontiers'].items() if n in nodes}
        paths={record['input']}
        for row in record['frontiers'].values():
            paths.update(row[k] for k in ('raw','npy'))
        for path in paths:
            link_checked(parent/path,bundle/path,old['files'][path]['sha256'])
        for name,p in programs.items():
            for leaf,entry in p['files'].items():
                path='programs/'+name+'/'+leaf
                link_checked(parent/path,bundle/path,entry['sha256'])
        for path,entry in old['files'].items():
            if path.startswith(('runtime/','firmware/')):
                link_checked(parent/path,bundle/path,entry['sha256'])
        runner=ROOT/'artifacts/cross_network_model_01/phase_profile'/network/'capsule/runner/vta_stage_pipeline_runner'
        link_checked(runner,bundle/'runner/vta_stage_pipeline_runner',RUNNER_SHA)
        for leaf,value in {
            'programs.json':programs,'inputs.json':dict(records=[record],available_indices=[0]),
            'rankings.json':dict(lists=lists[network],pool=pool[network],
                execution_order=[r['configuration'] for r in order if r['network']==network]),
            'protocol.json':protocol,'selection_seal.json':read(OUT/'FREEZE.json'),
            'board_execution.json':dict(udmabuf_bytes=201326592,storage_margin_bytes=268435456,
                boot_policy='same_boot_no_reboot_or_reprogram'),
            'preparation.json':dict(controllers=controllers,selection_sha256=sha256(OUT/'FREEZE.json'),
                source_manifest=str(parent/'manifest.json'),source_manifest_sha256=sha256(parent/'manifest.json'),
                new_modules=0,new_runtimes=0,new_executors=0,
                reused_runtime_assets_only=True)}.items():
            freeze(bundle/'plans'/leaf,value)
        for i,(path,wanted) in enumerate(sorted(controllers.items())):
            leaf=f'{i:02}_'+Path(path).name;target=bundle/'sources'/leaf
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():shutil.copy2(path,target)
            if sha256(target)!=wanted:raise ValueError('controller snapshot differs')
        assets=inventory(bundle)
        payload={n:v for n,v in assets.items() if n.startswith(('programs/','runtime/','runner/','firmware/','inputs/','references/'))}
        payload['plans/board_execution.json']=assets['plans/board_execution.json']
        freeze(bundle/'plans/board_payload.json',dict(files=payload,unpacked_bytes=sum(v['bytes'] for v in payload.values())))
        m=dict(network=network,experiment=protocol['experiment'],runner_sha256=RUNNER_SHA,
            runtime_sha256=sha256(bundle/'runtime/libtvm_runtime.so'),files=inventory(bundle))
        m['bundle_id']=digest(m)[:24];freeze(bundle/'manifest.json',m)
        bundles[network]=validate(bundle)['bundle_id']
        print(json.dumps(dict(network=network,bundle_id=m['bundle_id'],programs=len(programs),
            payload_MiB=round(sum(v['bytes'] for v in payload.values())/2**20,2))),flush=True)
    value=dict(bundles=bundles,controllers=controllers,selection_sha256=sha256(OUT/'FREEZE.json'),
        native_sessions=plan['native_sessions'],DNN_inferences=plan['DNN_inferences'],
        no_board_connection=True,board_execution_approved=False)
    freeze(OUT/'PREPARATION.json',value)
    return value


def qemu_cases(bundle):
    programs=read(Path(bundle)/'plans/programs.json');pool=read(Path(bundle)/'plans/rankings.json')['pool']
    cpu={n:p for n,p in programs.items() if p['device']=='cpu'}
    names={min(cpu,key=lambda n:(cpu[n]['begin'],cpu[n]['end'],n)),
           max(cpu,key=lambda n:(len(cpu[n]['input_node_ids']),cpu[n]['end'],n)),
           max(cpu,key=lambda n:(cpu[n]['end'],cpu[n]['begin'],n))}
    for name in sorted(names):
        cfg=next(c for c in pool.values() if any(s['program']==name for s in c['stages']))
        choices=[s for c in pool.values() for s in c['stages'] if s['program']==name]
        row=max(choices,key=lambda s:(s['threads'],s['mask']))
        doc=task(bundle,cfg,'qualify');single=component(row)
        doc.update(configuration=single,stages=single['stages'],kind='qualify_component')
        doc.pop('task_id');doc['task_id']=digest(doc)[:24]
        yield name,doc


def qemu():
    prepared=assert_sources();binary=shutil.which('qemu-aarch64-static')
    if not binary:raise ValueError('QEMU is required for ARM loading/output checks')
    if (OUT/'QEMU.json').exists():
        old=read(OUT/'QEMU.json')
        if old['bundles']!=prepared['bundles'] or not old['passed']:
            raise ValueError('QEMU receipt does not match the prepared capsules')
        return old
    results=[]
    for network in prepared['bundles']:
        bundle=OUT/network/'capsule';validate(bundle)
        for name,doc in qemu_cases(bundle):
            raw=OUT/'qemu'/(network+'_'+name);raw.mkdir(parents=True,exist_ok=True)
            argv,inputs=base_command(doc,bundle,str(bundle),str(raw));(raw/'inputs.txt').write_text(inputs)
            at=argv.index(str(bundle)+'/runtime/ld-linux-aarch64.so.1')
            argv[at:at]=['TVM_BIND_THREADS=1','TVM_EXCLUDE_WORKER0=1']
            argv+=['--cpu-worker-pinning','--allow-cpu-affinity-superset','--resource-affinity-probe']
            at=argv.index(str(bundle)+'/runtime/ld-linux-aarch64.so.1')
            call=[binary,*[x for item in argv[2:at] for x in ('-E',item)],*argv[at:]]
            result=subprocess.run(call,capture_output=True,text=True,timeout=180)
            (raw/'native.log').write_text(result.stdout+result.stderr)
            if result.returncode:raise ValueError('QEMU CPU qualification failed: '+str(raw))
            checked=qemu_cpu_summary(raw,doc,bundle)
            results.append(dict(network=network,program=name,threads=doc['stages'][0]['threads'],
                input_tensors=len(read(bundle/'plans/programs.json')[name]['input_node_ids']),summary=checked))
            print('QEMU output verified',network,name,flush=True)
    value=dict(passed=True,bundles=prepared['bundles'],cases=results,
        new_board_sessions=0,hardware_performance_not_verified_by_QEMU=True)
    freeze(OUT/'QEMU.json',value);return value


def bind_run(run_dir,boot,bundles):
    identity=dict(boot_id=boot,bundles=bundles,selection_sha256=sha256(OUT/'FREEZE.json'),
                  preparation_sha256=sha256(OUT/'PREPARATION.json'))
    freeze(Path(run_dir)/'run.json',identity)
    return identity


def board(run_dir=RUN,config=CONFIG,*,execute_board=False):
    if not execute_board:raise ValueError('--execute-board is required')
    prepared=assert_sources();plan,pool,order=selection();run_dir=Path(run_dir)
    proof=read(OUT/'QEMU.json')
    if not proof['passed'] or proof['bundles']!=prepared['bundles']:
        raise ValueError('matching QEMU qualification required')
    bundles={n:OUT/n/'capsule' for n in prepared['bundles']}
    for b in bundles.values():validate(b)
    ssh=transport(read(config)['board'],command,execute_board=True);boot=ssh.boot_id()
    with locked(run_dir):
        bind_run(run_dir,boot,prepared['bundles']);started=time.monotonic();remotes={};completed=0
        def state(status,**extra):
            write(run_dir/'state.json',dict(status=status,boot_id=boot,updated_at=now(),
                completed_configurations=completed,total_configurations=len(order),**extra),replace=True)
        try:
            for n,b in bundles.items():
                (run_dir/n).mkdir(parents=True,exist_ok=True)
                remotes[n]=ssh.deploy(b,run_dir/n)
                ssh.qualify_environment(remotes[n],boot,run_dir/n,b)
            for row in order:
                n,identity=row['network'],row['configuration'];b=bundles[n];cfg=pool[n][identity]
                state('RUNNING',current_network=n,current_configuration=identity)
                q=accept(ssh,task(b,cfg,'qualify'),b,remotes[n],run_dir,boot)
                accept(ssh,task(b,cfg,'short'),b,remotes[n],run_dir,boot,q['summary']['output_fingerprints'])
                completed+=1
                print('Common validation',completed,'/',len(order),n,identity,flush=True)
            for remote in remotes.values():
                ssh.idle_command('cd '+shlex.quote(remote)+' && sha256sum -c board.sha256 > scope_post_verify.log && sync',
                    boot=boot,timeout=600)
            if ssh.boot_id()!=boot:raise ValueError('boot changed after common validation')
            state('COMPLETE',native_sessions=plan['native_sessions'],DNN_inferences=plan['DNN_inferences'],
                  seconds=time.monotonic()-started,no_reboot=True,no_reprogram=True,no_extra_measurement=True)
            sync_tree(run_dir)
        except Exception as error:
            state('STOPPED_ON_ERROR',error_type=type(error).__name__,error=str(error));raise
    return read(run_dir/'state.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('prepare','validate','qemu','board','status'))
    parser.add_argument('--execute-board',action='store_true');parser.add_argument('--run-dir',type=Path,default=RUN)
    parser.add_argument('--config',type=Path,default=CONFIG);args=parser.parse_args()
    if args.command=='prepare':value=prepare()
    elif args.command=='validate':value={n:validate(OUT/n/'capsule')['bundle_id'] for n in assert_sources()['bundles']}
    elif args.command=='qemu':value=qemu()
    elif args.command=='board':value=board(args.run_dir,args.config,execute_board=args.execute_board)
    else:value=read(args.run_dir/'state.json') if (args.run_dir/'state.json').exists() else dict(status='NOT_STARTED')
    print(json.dumps(value,ensure_ascii=False,indent=2))
