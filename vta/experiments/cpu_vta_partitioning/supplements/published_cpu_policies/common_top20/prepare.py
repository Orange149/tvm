"""Compact, input-zero capsules using immutable hardlinked original assets."""
from copy import deepcopy
import os
from pathlib import Path
import shutil

from cpu_vta_partitioning.common import digest,inventory,read,sha256,verify_inventory,write
from phase_profile.prepare import COMPONENT_PARENTS,link_checked
from workload_model.common import OUTPUT,ROOT,freeze,require_space


def sources():
    paths=[*sorted(Path(__file__).parent.glob('*.py')),ROOT/'run_common_top20.py',
           *[ROOT/p for p in ('cpu_policies/open_resources/runtime.py','cpu_policies/open_resources/domain.py',
                              'cpu_policies/open_resources/costs.py','staged_ablation/model.py','resource_envelope/model.py')]]
    project=ROOT.parents[1]/'src/cpu_vta_partitioning'
    paths += [project/p for p in ('common.py','transport/__init__.py','measurement/__init__.py','measurement/tasks.py')]
    return {str(p):sha256(p) for p in paths}


def runtime_program(program):
    value=deepcopy(program);required=('graph.json','module.so','params.bin')
    if any(name not in value['files'] for name in required):raise ValueError('required compiled runtime file missing')
    value['excluded_nonexecution_proof_files']={n:v for n,v in value['files'].items() if n not in required}
    value['files']={n:value['files'][n] for n in required}
    return value


def validate(bundle):
    bundle=Path(bundle);manifest=read(bundle/'manifest.json')
    if manifest['bundle_id']!=digest({k:v for k,v in manifest.items() if k!='bundle_id'})[:24]:
        raise ValueError('common capsule seal differs')
    verify_inventory(bundle,manifest['files'])
    for path,expected in read(bundle/'plans/preparation.json')['controllers'].items():
        if sha256(path)!=expected:raise ValueError('common controller source differs')
    rows=read(bundle/'plans/rankings.json')
    if set(rows['pool'])!=set(rows['execution_order']):raise ValueError('execution order coverage differs')
    if len(rows['pool'])>80 or any(len(r)!=20 or len({c['id'] for c in r})!=20 for r in rows['lists'].values()):
        raise ValueError('method budget or candidate uniqueness differs')
    if manifest['runner_sha256']!='25833e7f8b47fc8824ff9460afc0b968f24934dcee41c3b1f28619a99b6b6fe6':
        raise ValueError('formal comparison must reuse the original pinned executor')
    if read(bundle/'plans/inputs.json')['available_indices']!=[0]:raise ValueError('wrong formal input')
    return manifest


def prepare(output):
    output=Path(output).resolve();require_space();seal=read(output/'RANK_FREEZE.json')
    if seal['pipeline_FPS_read'] or sha256(output/'PLAN.json')!=seal['plan_sha256']:
        raise ValueError('rank plan identity differs or feedback occurred')
    if sha256(Path(__file__).with_name('ranking.py'))!=seal['ranking_source_sha256']:
        raise ValueError('ranking source changed')
    result={}
    for network,parent in COMPONENT_PARENTS.items():
        if sha256(output/network/'RANKINGS.json')!=seal['files'][network+'/RANKINGS.json']:
            raise ValueError('frozen method lists changed')
        bundle=output/network/'capsule'
        if (bundle/'manifest.json').exists():result[network]=validate(bundle)['bundle_id'];continue
        rows=read(output/network/'RANKINGS.json');old=read(parent/'manifest.json')
        catalog=read(parent/'plans/programs.json');names={s['program'] for cfg in rows['pool'].values() for s in cfg['stages']}
        programs={name:runtime_program(catalog[name]) for name in sorted(names)}
        inp=read(parent/'plans/inputs.json');record=deepcopy(inp['records'][0])
        nodes={str(n) for p in programs.values() for n in p['input_node_ids']+p['output_node_ids']}
        record['frontiers']={n:v for n,v in record['frontiers'].items() if n in nodes}
        needed={record['input']}
        for row in record['frontiers'].values():needed.update(row[k] for k in ('raw','npy'))
        for path in needed:link_checked(parent/path,bundle/path,old['files'][path]['sha256'])
        for name,p in programs.items():
            for leaf,value in p['files'].items():
                path='programs/'+name+'/'+leaf
                link_checked(parent/path,bundle/path,value['sha256'])
        for path,entry in old['files'].items():
            if path.startswith(('runtime/','firmware/')):link_checked(parent/path,bundle/path,entry['sha256'])
        runner=OUTPUT/'phase_profile'/network/'capsule/runner/vta_stage_pipeline_runner'
        runner_hash=sha256(runner)
        link_checked(runner,bundle/'runner/vta_stage_pipeline_runner',runner_hash)
        write(bundle/'plans/programs.json',programs)
        write(bundle/'plans/inputs.json',dict(records=[record],available_indices=[0]))
        write(bundle/'plans/rankings.json',rows);write(bundle/'plans/protocol.json',read(output/'PLAN.json'))
        write(bundle/'plans/ranking_seal.json',seal)
        write(bundle/'plans/board_execution.json',dict(udmabuf_bytes=201326592,
            storage_margin_bytes=268435456,boot_policy='same_boot_no_reboot'))
        controller=sources();write(bundle/'plans/preparation.json',dict(
            parent_manifest_sha256=sha256(parent/'manifest.json'),controllers=controller,
            new_runtime_builds=0,new_module_builds=0,new_executor_builds=0,
            asset_policy='immutable hardlinks; only input zero and selected modules'))
        snapshot=bundle/'sources/common_top20';snapshot.mkdir(parents=True,exist_ok=True)
        for path in controller:shutil.copy2(path,snapshot/(Path(path).parent.name+'_'+Path(path).name))
        assets=inventory(bundle)
        payload={n:v for n,v in assets.items() if n.startswith(('programs/','runner/','runtime/','firmware/','inputs/'))}
        payload['plans/board_execution.json']=assets['plans/board_execution.json']
        write(bundle/'plans/board_payload.json',dict(files=payload,unpacked_bytes=sum(v['bytes'] for v in payload.values())))
        manifest=dict(network=network,experiment='cross_network_constraint_semantics_01',
            runner_sha256=runner_hash,runtime_sha256=sha256(bundle/'runtime/libtvm_runtime.so'),
            firmware_sha256={p:v['sha256'] for p,v in payload.items() if p.startswith('firmware/')},
            files=inventory(bundle))
        manifest['bundle_id']=digest(manifest)[:24];write(bundle/'manifest.json',manifest)
        validate(bundle);result[network]=manifest['bundle_id']
    freeze(output/'PREPARATION_FREEZE.json',dict(bundles=result,controllers=sources(),
        rank_freeze_sha256=sha256(output/'RANK_FREEZE.json')))
    return result
