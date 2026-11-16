"""Reuse frozen modules and executors; freeze six holdouts without FPS input."""
from copy import deepcopy
import os
from pathlib import Path
import random
import shutil

from cpu_vta_partitioning.common import digest,inventory,read,sha256,verify_inventory,write
from cpu_policies.open_resources.domain import component,stage
from phase_profile.prepare import validate as parent_validate
from workload_model.common import OUTPUT,ROOT,freeze,require_space


def source_files():
    return [*sorted(Path(__file__).parent.glob('*.py')),ROOT/'run_phase_validation.py']


def validate(bundle):
    bundle=Path(bundle);result=parent_validate(bundle)
    proof=read(bundle/'plans/validation_preparation.json')
    for name,value in proof['controllers'].items():
        if sha256(name)!=value:raise ValueError('frozen validation controller changed')
    return result


def prepare(output):
    output=Path(output).resolve();require_space();plan=read(output/'PLAN.json')
    for path,expected in plan['model_sources_frozen'].items():
        if sha256(path)!=expected:raise ValueError('frozen predictive mechanism changed')
    components=[];pipelines=[]
    for network,cases in plan['networks'].items():
        parent=OUTPUT/'phase_profile'/network/'capsule';parent_validate(parent)
        old=read(parent/'manifest.json');bundle=output/network/'capsule'
        cpu=next(iter(cases.values()))['stages']
        required=[component(stage('cpu',s['begin'],s['end'],2,[0,1]))
                  for s in cpu if s['device']=='cpu' and s['end']-s['begin']>1]
        if len(required)!=2:raise ValueError('holdout requires exactly two new t2 components per network')
        cfgs={'components':{r['stages'][0]['program']+'_t2':r for r in required},'pipelines':cases}
        if not (bundle/'manifest.json').exists():
            for leaf,entry in old['files'].items():
                src=parent/leaf;dest=bundle/leaf;dest.parent.mkdir(parents=True,exist_ok=True)
                if sha256(src)!=entry['sha256']:raise ValueError('parent immutable asset changed')
                if leaf.startswith(('programs/','runtime/','runner/','inputs/','firmware/')):os.link(src,dest)
                else:shutil.copy2(src,dest)
            write(bundle/'plans/configurations.json',cfgs,replace=True)
            write(bundle/'plans/validation_preparation.json',dict(
                plan_sha256=sha256(output/'PLAN.json'),parent_manifest_sha256=sha256(parent/'manifest.json'),
                controllers={str(p):sha256(p) for p in source_files()},
                executor_and_runtime_reused=True,stage_modules_recompiled=False))
            snapshots=bundle/'sources/phase_validation';snapshots.mkdir(parents=True,exist_ok=True)
            for path in source_files():shutil.copy2(path,snapshots/path.name)
            manifest={k:v for k,v in old.items() if k not in ('files','bundle_id')}
            manifest.update(experiment='cross_network_phase_validation_01',files=inventory(bundle))
            manifest['bundle_id']=digest(manifest)[:24];write(bundle/'manifest.json',manifest)
        validate(bundle)
        for case in cfgs['components']:
            for mode in ('qualify','clean','observed'):
                components.append(dict(network=network,scope='component',case=case,phase=mode,
                                       probe=mode!='clean',runner='phase'))
        for case in cases:
            pipelines.append(dict(network=network,case=case))
    random.Random(20261110).shuffle(pipelines)
    tasks=[]
    for item in pipelines:
        for mode in ('qualify','clean'):
            tasks.append(dict(**item,scope='pipeline',phase=mode,probe=mode!='clean',runner='phase'))
    freeze(output/'COMPONENT_TASKS.json',components);freeze(output/'PIPELINE_TASKS.json',tasks)
    freeze(output/'PREPARATION_FREEZE.json',dict(
        plan_sha256=sha256(output/'PLAN.json'),controllers={str(p):sha256(p) for p in source_files()},
        capsule_ids={n:read(output/n/'capsule/manifest.json')['bundle_id'] for n in plan['networks']},
        components_sha256=sha256(output/'COMPONENT_TASKS.json'),
        pipelines_sha256=sha256(output/'PIPELINE_TASKS.json')))
    return dict(new_components=len(components)//3,pipeline_configurations=len(tasks)//2,
                new_compilations=0)
