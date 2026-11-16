"""Small per-network capsules, hard-linking immutable local assets."""
from copy import deepcopy
import os
from pathlib import Path
import random
import shutil
import subprocess
import time

from cpu_vta_partitioning.common import digest,inventory,now,read,sha256,verify_inventory,write
from cpu_policies.open_resources import domain
from workload_model.common import ROOT,require_space,freeze

COMPONENT_PARENTS=dict(resnet18=ROOT/'artifacts/bundles/8a942b2047c031339168f9d2',
    yolov3_tiny416=ROOT/'artifacts/yolo_transfer_01/profile_bundle_bridge02')
PIPELINE_PARENTS=dict(resnet18=ROOT/'artifacts/bundles/e542003fc5a64fb17ee1e751',
    yolov3_tiny416=ROOT/'artifacts/yolo_cpu_policy_check_01/pipeline_bundle')
ADAPTER=ROOT/'artifacts/yolo_wait_diagnosis_01/capsule/adapter/libwait_adapter.so'


def protocol():
    return dict(experiment='cross_network_phase_profile_01',revision=1,input_index=0,seed=20261110,
        component=dict(warmup=2,scored=8,tail=0),qualification=dict(warmup=0,scored=1,tail=0),
        pipeline=dict(warmup=8,scored=16,tail=4),
        queue_depth=2,CPU_gate='off',host_and_control_mask=[0,1,2,3],worker_pinning=True,
        startup_affinity_probe=True,original_wait=True,no_reboot=True,drift_rejection=False,
        automatic_remeasurement=False,phase_clock='CLOCK_MONOTONIC and caller CLOCK_THREAD_CPUTIME_ID',
        task_counter_scope='stage-local RunStage calls; startup empty callbacks separately identified',
        interpretation='diagnostic component inputs and later frozen predictive validation; no full-domain profiling')


def configurations(network):
    if network=='resnet18':
        ranges=[('cpu',0,3),('vta',3,5),('cpu',5,8),('vta',8,20),('cpu',20,21)]
        cases=dict(overlap=[[0,1,2,3],[0,1,2,3],[0]],split=[[0,1,2],[3],[0]])
        required=[(0,3,3),(0,3,4),(5,8,1),(5,8,4),(20,21,1)]
    elif network=='yolov3_tiny416':
        ranges=[('cpu',0,3),('vta',3,7),('cpu',7,9),('vta',9,13),('cpu',13,14)]
        cases=dict(overlap=[[0,1,2,3],[1,2,3],[0]],split=[[0,1,2],[3],[3]])
        required=[(0,3,3),(0,3,4),(7,9,1),(7,9,3),(13,14,1)]
    else:raise ValueError('unknown network')
    pipes={}
    for name,masks in cases.items():
        masks=iter(masks);stages=[]
        for device,begin,end in ranges:
            mask=next(masks) if device=='cpu' else [0,1,2,3]
            stages.append(domain.stage(device,begin,end,len(mask) if device=='cpu' else 1,mask))
        pipes[name]=domain.candidate(stages,'shared4','off')
    components={f'cpu_{a:02}_{b-1:02}_t{n}':domain.component(domain.stage('cpu',a,b,n,list(range(n))))
                for a,b,n in required}
    return dict(components=components,pipelines=pipes)


def link_checked(source,target,expected):
    source,target=Path(source),Path(target)
    if not source.is_file() or source.is_symlink() or sha256(source)!=expected:
        raise ValueError('immutable source missing or changed: '+str(source))
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if sha256(target)!=expected:raise ValueError('asset changed in new capsule')
        return
    try:os.link(source,target)
    except OSError:shutil.copy2(source,target)


def task_specs():
    specs=[]
    for network in COMPONENT_PARENTS:
        cases=list(configurations(network)['components'])
        random.Random(protocol()['seed']).shuffle(cases)
        for case in cases:
            for phase in ('qualify','clean','observed'):
                specs.append(dict(network=network,scope='component',case=case,phase=phase,
                                  probe=phase!='clean',runner='phase'))
    # Pair clean/observed after qualification; do not execute pipeline timings
    # until component-derived predictions have their own freeze.
    return specs


def prepare(output):
    output=Path(output).resolve();require_space();build=read(output/'build/BUILD.json')
    if sha256(output/'build/vta_stage_pipeline_runner_phase')!=build['runner_sha256']:
        raise ValueError('optional runner binary changed')
    frozen_specs=task_specs();freeze(output/'COMPONENT_TASKS.json',frozen_specs)
    result={}
    for network,source in COMPONENT_PARENTS.items():
        target=output/network/'capsule'
        if (target/'manifest.json').exists():result[network]=validate(target);continue
        started=time.monotonic();manifest=read(source/'manifest.json')
        if digest({k:v for k,v in manifest.items() if k!='bundle_id'})[:24]!=manifest['bundle_id']:
            raise ValueError('source component capsule seal changed')
        catalog=read(source/'plans/programs.json');cfg=configurations(network)
        programs={s['program']:catalog[s['program']] for section in cfg.values() for c in section.values() for s in c['stages']}
        source_inputs=read(source/'plans/inputs.json');record=deepcopy(source_inputs['records'][0])
        needed={str(n) for p in programs.values() for n in p['input_node_ids']+p['output_node_ids']}
        record['frontiers']={n:v for n,v in record['frontiers'].items() if n in needed}
        paths={record['input']}
        for v in record['frontiers'].values():paths.update(v[k] for k in ('raw','npy'))
        for path in paths:link_checked(source/path,target/path,manifest['files'][path]['sha256'])
        for name,p in programs.items():
            for leaf in ('graph.json','module.so','params.bin'):
                path='programs/'+name+'/'+leaf
                link_checked(source/path,target/path,p['files'][leaf]['sha256'])
        for path,value in manifest['files'].items():
            if path.startswith(('runtime/','firmware/')):
                link_checked(source/path,target/path,value['sha256'])
        pipe=PIPELINE_PARENTS[network];pm=read(pipe/'manifest.json')
        if (pipe/'plans/executors.json').exists():old=read(pipe/'plans/executors.json')['policy']
        else:
            old=dict(path=next(n for n,v in pm['files'].items() if n.startswith('runner/') and v['sha256']==pm['runner_sha256']),sha256=pm['runner_sha256'])
        link_checked(pipe/old['path'],target/'runner/vta_stage_pipeline_runner',old['sha256'])
        link_checked(output/'build/vta_stage_pipeline_runner_phase',target/'runner/vta_stage_pipeline_runner_phase',build['runner_sha256'])
        link_checked(ADAPTER,target/'runner/libwait_adapter.so',sha256(ADAPTER))
        write(target/'plans/programs.json',programs)
        write(target/'plans/inputs.json',dict(records=[record],available_indices=[0]))
        write(target/'plans/configurations.json',cfg);write(target/'plans/protocol.json',protocol())
        write(target/'plans/board_execution.json',dict(boot_policy='one_boot_no_automatic_reboot',
            udmabuf_bytes=201326592,storage_margin_bytes=268435456))
        # The only newly compiled native file is this optional runner.
        assets=inventory(target)
        payload={n:v for n,v in assets.items() if n.startswith(('runtime/','runner/','firmware/','inputs/','programs/'))}
        payload['plans/board_execution.json']=assets['plans/board_execution.json']
        write(target/'plans/board_payload.json',dict(files=payload,unpacked_bytes=sum(v['bytes'] for v in payload.values())))
        source_files={str(p):sha256(p) for p in [source/'manifest.json',pipe/'manifest.json',output/'build/BUILD.json',ADAPTER]}
        sources={str(p.relative_to(ROOT)):sha256(p) for p in Path(__file__).parent.iterdir() if p.suffix in ('.py','.h')}
        sources['run_phase_profile.py']=sha256(ROOT/'run_phase_profile.py')
        (target/'sources').mkdir(exist_ok=True)
        for p in Path(__file__).parent.iterdir():
            if p.suffix in ('.py','.h'):shutil.copy2(p,target/'sources'/p.name)
        shutil.copy2(ROOT/'run_phase_profile.py',target/'sources/run_phase_profile.py')
        write(target/'plans/preparation.json',dict(source_files=source_files,sources=sources,
            new_runner_sha256=build['runner_sha256'],source_runner_sha256=old['sha256'],
            new_runtime_builds=0,new_module_builds=0,local_asset_reuse='hardlink, immutable bytes',seconds=time.monotonic()-started))
        m=dict(experiment=protocol()['experiment'],network=network,
            old_runner_sha256=old['sha256'],phase_runner_sha256=build['runner_sha256'],
            adapter_sha256=sha256(ADAPTER),runtime_sha256=sha256(target/'runtime/libtvm_runtime.so'),files=inventory(target))
        m['bundle_id']=digest(m)[:24];write(target/'manifest.json',m)
        result[network]=validate(target)
    return result


def validate(bundle):
    bundle=Path(bundle);m=read(bundle/'manifest.json')
    if digest({k:v for k,v in m.items() if k!='bundle_id'})[:24]!=m['bundle_id']:
        raise ValueError('phase capsule changed')
    verify_inventory(bundle,m['files'])
    if read(bundle/'plans/protocol.json')!=protocol():raise ValueError('phase protocol changed')
    prep=read(bundle/'plans/preparation.json')
    for name,expected in prep['sources'].items():
        if sha256(ROOT/name)!=expected:raise ValueError('phase controller source changed; use frozen source')
    for name,expected in prep['source_files'].items():
        if sha256(name)!=expected:raise ValueError('phase input source changed')
    return dict(bundle_id=m['bundle_id'],network=m['network'],validated=True,
                payload_bytes=read(bundle/'plans/board_payload.json')['unpacked_bytes'])
