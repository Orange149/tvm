"""Three full-domain searches with independent C++ enumeration."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .common import ROOT, PARENT, SOURCE, MODELS, LEGACY_SEARCH, LEGACY_MODELS, settings, protocol, api
from .model import score

DOMAIN_COUNT = 966240306
CACHES = {key: LEGACY_SEARCH/legacy for key,legacy in LEGACY_MODELS.items()}

def engine(output, key):
    """Copy hash-verified previous engine; only ignored mode changes accounting."""
    from resource_envelope.search_adapter import _replace
    c=api();settings(key)
    legacy=LEGACY_MODELS.get(key,'weights_team')
    proof=c.read(LEGACY_SEARCH/legacy/'proof.json')
    if proof['model']['key']!=legacy:
        raise ValueError('legacy model provenance differs')
    source=Path(proof['source_directory'])
    for leaf,digest in proof['implementation'].items():
        if c.sha256(source/leaf)!=digest:raise ValueError('legacy source hash differs')
    contents={leaf:(source/leaf).read_text() for leaf in
        ('search.py','oracle.cc','picojson.h','__init__.py','costs.py','domain.py')}
    if key=='ignored':
        contents['search.py']=_replace(contents['search.py'],
            '    service = np.maximum(np.maximum(cpu_max, vta), ddr)\n',
            '    host = 0  # Host CPU demand excluded from scoring; measured costs remain intact.\n'
            '    service = np.maximum(np.maximum(cpu_max, vta), ddr)\n')
        contents['oracle.cc']=_replace(contents['oracle.cc'],
            '  std::int64_t sum=0,capacity=0;\n',
            '  host_work=0;  // Scoring ablation only; VTA wall and DDR costs unchanged.\n'
            '  std::int64_t sum=0,capacity=0;\n')
    name='host_work_'+c.digest(contents)[:20]
    target=Path(output)/'sources'/name;target.mkdir(parents=True,exist_ok=True)
    for leaf,value in contents.items():
        p=target/leaf
        if p.exists() and p.read_text()!=value:raise ValueError('generated host-work source changed')
        if not p.exists():p.write_text(value)
    spec=importlib.util.spec_from_file_location(name,target/'__init__.py',submodule_search_locations=[str(target)])
    package=importlib.util.module_from_spec(spec);sys.modules[name]=package;spec.loader.exec_module(package)
    spec=importlib.util.spec_from_file_location(name+'.search',target/'search.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module

def inputs():
    from run_resource_behavior import SELECTION
    c = api(); manifest = c.read(SELECTION/'manifest.json')
    if (c.digest({k:v for k,v in manifest.items() if k!='bundle_id'})[:24] != manifest['bundle_id']
        or manifest['bundle_id']!=SELECTION.name):
        raise ValueError('component manifest identity differs')
    hashes = {name:c.sha256(SOURCE/name) for name in ('isolated_costs.json','ddr.json')}
    if any(value != manifest['files']['plans/'+name]['sha256'] for name,value in hashes.items()):
        raise ValueError('frozen component costs differ')
    records = c.read(SOURCE/'isolated_costs.json')['costs']
    ddr = c.read(SOURCE/'ddr.json'); programs = c.read(PARENT/'plans/programs.json')
    if len(records)!=775 or len(programs)!=172: raise ValueError('component/program coverage differs')
    provenance=c.read(SOURCE/'source_provenance.json')
    if provenance['isolated_costs_replayed']!=775 or provenance['traffic_records_replayed']!=87:
        raise ValueError('component/DDR source evidence coverage differs')
    parent=c.read(PARENT/'manifest.json')
    if (c.digest({k:v for k,v in parent.items() if k!='bundle_id'})[:24]!=parent['bundle_id']
        or parent['bundle_id']!=PARENT.name
        or c.sha256(PARENT/'plans/programs.json')!=parent['files']['plans/programs.json']['sha256']):
        raise ValueError('program source manifest differs')
    return programs, records, ddr, hashes

def run_one(output, key):
    from cpu_policies.open_resources.costs import Costs
    c = api(); started = time.monotonic(); output = Path(output).resolve(); config = settings(key)
    programs, records, ddr, hashes = inputs()
    costs = Costs(records, programs=programs, ddr=ddr,
                  distribution='session_weights' if config['thread_weights'] else 'balanced')
    module = engine(output, key)
    cached = CACHES.get(key); cache_used = bool(cached and (cached/'search').is_dir())
    target = output/'search'; target.mkdir(parents=True,exist_ok=True)
    if cache_used:
        for file in (cached/'search').glob('*.json'):
            if file.stem not in [ctx['id'] for ctx in module.domain.contexts()]: continue
            dst = target/file.name
            if not dst.exists(): shutil.copy2(file,dst)
    # rank_all reconstructs the complete input fingerprint (including generated
    # Python/C++ sources), checks every checkpoint and its oracle result digest.
    ranked = module.rank_all(programs,costs,ddr=ddr,output_dir=target,verify=True)
    if ranked['counts']['enhanced']!=DOMAIN_COUNT or not ranked['verification']['passed']:
        raise ValueError('full-domain enumeration incomplete')
    for row in ranked['pure_static20']:
        if score(row,costs,key)['score_ns'] != row['score_ns']: raise ValueError('scalar score differs from DP')
    if cache_used:
        filename = 'rankings.json'
        if ranked['pure_static20'] != c.read(cached/filename)['pure_static20']:
            raise ValueError('cached Top-20 replay differs')
    c.write(output/'rankings.json',ranked,replace=True)
    implementation = {name:c.sha256(Path(module.__file__).with_name(name)) for name in ('search.py','oracle.cc','picojson.h')}
    c.write(output/'proof.json',dict(model=config, component_hashes=hashes,
        implementation=implementation, source_directory=str(Path(module.__file__).parent),
        counts=ranked['counts'], verification=ranked['verification'], cache_replayed=cache_used,
        cache_source=str(cached) if cache_used else None,
        cache_identity_verified=True, pipeline_FPS_used=False,
        elapsed_seconds=time.monotonic()-started),replace=True)
    return {'model':key,'first':ranked['pure_static20'][0]['id'],'cache_replayed':cache_used}

def verify_one(output, key):
    """Replay all nine checkpoint fingerprints without enumerating new states."""
    c=api(); output=Path(output); proof=c.read(output/'proof.json')
    if proof['model']!=settings(key):raise ValueError('model proof switches changed')
    programs,records,ddr,hashes=inputs()
    if proof['component_hashes']!=hashes:raise ValueError('proof costs changed')
    from cpu_policies.open_resources.costs import Costs
    module=engine(output,key)
    implementation={n:c.sha256(Path(module.__file__).with_name(n)) for n in ('search.py','oracle.cc','picojson.h')}
    if implementation!=proof['implementation']:raise ValueError('search implementation differs')
    costs=Costs(records,programs=programs,ddr=ddr,distribution='session_weights' if settings(key)['thread_weights'] else 'balanced')
    graphs=module._topologies(programs,21,3)
    from cpu_policies.open_resources.search import _digest
    total=0
    for ctx in module.domain.contexts():
        cp=c.read(output/'search'/(ctx['id']+'.json'))
        payload=dict(programs=module._prepare(programs,costs,ctx,ddr),topologies=graphs,context=ctx,
                     k=20,score_scale=module.SCORE_SCALE,implementation=implementation)
        fields=('counts','gate_counts','rankings','topology_strata','resource_strata')
        v=cp['verification']
        if (cp['checkpoint_sha256']!=_digest({k:x for k,x in cp.items() if k!='checkpoint_sha256'})
            or cp['input_sha256']!=_digest(payload) or not v['passed']
            or v['source_sha256']!=implementation['oracle.cc']
            or v['result_sha256']!=_digest({k:cp[k] for k in fields})):
            raise ValueError('checkpoint/oracle fingerprint differs')
        total+=cp['counts']['enhanced']
    if total!=DOMAIN_COUNT:raise ValueError('complete-domain count differs')
    ranked=c.read(output/'rankings.json')
    # Reconstruct global Top-K from proved per-context terminal rankings.
    context_map={ctx['id']:ctx for ctx in module.domain.contexts()}; expanded=[]
    for ctx in context_map:
        cp=c.read(output/'search'/(ctx+'.json'))
        for terminal in cp['rankings']['enhanced']:
            t=module.Terminal.from_json(terminal)
            cfg=module._configuration(t,graphs,programs,context_map)
            expanded.append({**cfg,'score_ns':t.score,'score_ms':t.score/1e6})
    expected=sorted(expanded,key=module._row_key)[:20]
    for i,row in enumerate(expected,1):row['static_rank']=i
    if expected!=ranked['pure_static20']:raise ValueError('global Top-20 not backed by checkpoints')
    return ranked,proof

def freeze(output):
    from cpu_policies.open_resources.costs import Costs
    from cpu_policies.open_resources.domain import configuration
    c=api(); output=Path(output).resolve(); programs,records,ddr,hashes=inputs()
    lists={}; pool={}; proofs={}
    for key,*_ in MODELS:
        ranked,proof=verify_one(output/key,key); rows=ranked['pure_static20']
        if len(rows)!=20 or len({r['id'] for r in rows})!=20:raise ValueError('twenty unique configurations required')
        lists[key]=rows;proofs[key]=proof
        for row in rows:
            cfg={**configuration(row),'id':row['id']}
            if row['id'] in pool and pool[row['id']]!=cfg:raise ValueError('execution identity collision')
            pool[row['id']]=cfg
    if len(pool)>60:raise ValueError('candidate union exceeds budget')
    all_scores={}
    for key,*_ in MODELS:
        costs=Costs(records,programs=programs,ddr=ddr,distribution='session_weights' if settings(key)['thread_weights'] else 'balanced')
        all_scores[key]={name:score(cfg,costs,key) for name,cfg in sorted(pool.items())}
        for row in lists[key]:
            if row['score_ns']!=all_scores[key][row['id']]['score_ns']:raise ValueError('frozen score differs')
    selected=output/'frozen';selected.mkdir(exist_ok=True)
    docs={'selection_rankings.json':lists,'static_pool.json':dict(sorted(pool.items())),
          'model_scores.json':all_scores,'search_proofs.json':proofs,'selection_protocol.json':protocol()}
    for leaf,value in docs.items():
        p=selected/leaf
        if p.exists() and c.read(p)!=value:raise ValueError('refusing to replace frozen selection')
        if not p.exists():c.write(p,value)
    import random
    order=sorted(pool);random.Random(protocol()['seed']).shuffle(order)
    frozen=dict(files={n:c.sha256(selected/n) for n in docs},component_hashes=hashes,
        configuration_order=order,pipeline_FPS_used=False,candidate_count=len(pool),
        native_sessions=2*len(pool),complete_inferences=29*len(pool))
    p=selected/'FREEZE.json'
    if p.exists():
        old=c.read(p)
        if any(old[k]!=v for k,v in frozen.items()):raise ValueError('freeze identity differs')
    else:c.write(p,{**frozen,'frozen_at':c.now()})
    return selected

def run_all(output, workers=2):
    c=api(); output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    def worker(key):
        directory=output/key
        if (directory/'proof.json').exists():
            verify_one(directory,key);return {'model':key,'resumed_verified':True}
        directory.mkdir(exist_ok=True)
        print('[VTA主机需求消融] search '+key,flush=True)
        with (directory/'search.log').open('a') as log:
            p=subprocess.run([sys.executable,str(ROOT/'run_host_work_ablation.py'),'rank-one','--output',str(directory),'--model',key],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
        if p.returncode:raise RuntimeError('search failed; see '+str(directory/'search.log'))
        print('[VTA主机需求消融] verified '+key,flush=True)
        return {'model':key}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results=list(executor.map(worker,[r[0] for r in MODELS]))
    target=freeze(output)
    return {'frozen':str(target),'models':results,'candidate_count':c.read(target/'FREEZE.json')['candidate_count']}
