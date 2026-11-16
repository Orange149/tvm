"""Reuse the formal candidate closure; no modules or native programs rebuilt."""
import json
from pathlib import Path
import shutil

from cpu_vta_partitioning.common import digest,inventory,read,sha256,verify_inventory,write
from cpu_policies.open_resources.domain import stage,component
from phase_profile.prepare import link_checked
from workload_model.common import OUTPUT,ROOT,freeze,require_space


def sources():
    paths=[*sorted(Path(__file__).parent.glob('*.py')),ROOT/'run_pool_temporal.py',
           *[ROOT/p for p in ('phase_profile/runtime.py','phase_profile/worker.py','phase_model/extract.py',
             'device_phase/collect.py','device_phase/extract.py','wait_diagnosis/runtime.py')]]
    paths += [ROOT/p for p in ('common_top20/campaign.py','cpu_policies/open_resources/runtime.py',
                              'cpu_policies/open_resources/domain.py')]
    return {str(p):sha256(p) for p in paths}


def validate(bundle):
    bundle=Path(bundle);m=read(bundle/'manifest.json')
    if m['bundle_id']!=digest({k:v for k,v in m.items() if k!='bundle_id'})[:24]:raise ValueError('intrinsic capsule seal differs')
    verify_inventory(bundle,m['files'])
    for record in read(bundle/'plans/inputs.json')['records']:
        paths={record['input']}
        for frontier in record['frontiers'].values():
            paths.update(frontier[k] for k in ('raw','npy'))
        if any(path not in m['files'] for path in paths):
            raise ValueError('isolated capsule input/reference closure incomplete')
    for path,expected in read(bundle/'plans/preparation.json')['controllers'].items():
        if sha256(path)!=expected:raise ValueError('intrinsic controller changed; use new identity')
    return m


def prepare(output):
    output=Path(output).resolve();require_space();plan=read(output/'PLAN.json');tasks=[];ids={}
    for network,required in plan['missing_profiles'].items():
        parent=OUTPUT/'common_top20'/network/'capsule';old=read(parent/'manifest.json')
        if sha256(OUTPUT/'common_top20'/network/'RANKINGS.json')!=plan['frozen_pool_hashes'][network]:
            raise ValueError('intrinsic candidate pool changed')
        bundle=output/network/'capsule';components={}
        for kind in ('CPU','VTA'):
            for spec in required[kind]:
                begin,end=map(int,spec['program'].split('_')[1:]);device=kind.lower()
                row=stage(device,begin,end+1,spec['threads'],spec['mask'])
                ctx='shared4' if device=='cpu' or len(spec['mask'])==4 else 'host_shared_'+str(spec['mask'][0])
                cfg=component(row,ctx);case=spec['program']+'_t'+str(spec['threads'])+'_a'+'-'.join(map(str,spec['mask']))
                components[case]=cfg
                for mode in ('qualify','observed'):tasks.append(dict(network=network,case=case,mode=mode))
        if not (bundle/'manifest.json').exists():
            for leaf,entry in old['files'].items():
                if leaf.startswith(('runtime/','runner/','firmware/','inputs/','references/','programs/')):
                    link_checked(parent/leaf,bundle/leaf,entry['sha256'])
            phase=OUTPUT/'phase_profile'/network/'capsule'
            for leaf in ('vta_stage_pipeline_runner_phase','libwait_adapter.so'):
                link_checked(phase/'runner'/leaf,bundle/'runner'/leaf,sha256(phase/'runner'/leaf))
            device=OUTPUT/'device_phase_02'/network/'capsule/runner/libdevice_phase.so'
            link_checked(device,bundle/'runner/libdevice_phase.so',sha256(device))
            for leaf in ('programs.json','inputs.json','board_execution.json'):
                dest=bundle/'plans'/leaf;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(parent/'plans'/leaf,dest)
            write(bundle/'plans/components.json',components)
            write(bundle/'plans/preparation.json',dict(controllers=sources(),
                parent_manifest_sha256=sha256(parent/'manifest.json'),plan_sha256=sha256(output/'PLAN.json'),
                no_runtime_build=True,no_executor_build=True,no_module_build=True))
            snapshot=bundle/'sources/pool_temporal';snapshot.mkdir(parents=True,exist_ok=True)
            for path in sources():shutil.copy2(path,snapshot/(Path(path).parent.name+'_'+Path(path).name))
            assets=inventory(bundle);payload={k:v for k,v in assets.items()
                if k.startswith(('runtime/','runner/','firmware/','inputs/','programs/'))}
            payload['plans/board_execution.json']=assets['plans/board_execution.json']
            write(bundle/'plans/board_payload.json',dict(files=payload,unpacked_bytes=sum(v['bytes'] for v in payload.values())))
            m=dict(network=network,experiment=plan['experiment'],profile_runner_sha256=sha256(bundle/'runner/vta_stage_pipeline_runner_phase'),
                wait_adapter_sha256=sha256(bundle/'runner/libwait_adapter.so'),device_adapter_sha256=sha256(bundle/'runner/libdevice_phase.so'),
                runtime_sha256=sha256(bundle/'runtime/libtvm_runtime.so'),files=inventory(bundle))
            m['bundle_id']=digest(m)[:24];write(bundle/'manifest.json',m)
        ids[network]=validate(bundle)['bundle_id']
    if len(tasks)!=168:raise ValueError('frozen missing signature count differs')
    freeze(output/'TASKS.json',tasks)
    freeze(output/'PREPARATION_FREEZE.json',dict(bundles=ids,controllers=sources(),
        plan_sha256=sha256(output/'PLAN.json'),tasks_sha256=sha256(output/'TASKS.json')))
    return dict(new_signatures=len(tasks)//2,native_sessions=len(tasks),bundles=ids)
