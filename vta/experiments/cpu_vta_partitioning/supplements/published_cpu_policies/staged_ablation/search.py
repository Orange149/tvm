"""Full resource DP and independent C++ oracle, with immutable source replay."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from .common import ROOT,LEGACY_SEARCH,HOST_SEARCH,MODELS,settings,protocol,api
from .model import score
from host_work_ablation.search import inputs

DOMAIN_COUNT=966240306
CACHES={
    'coarse_host':LEGACY_SEARCH/'uniform',
    'fine_no_host':HOST_SEARCH/'ignored',
    'fine_host':LEGACY_SEARCH/'weights_team',
    'weights_only':LEGACY_SEARCH/'weights',
    'team_only':LEGACY_SEARCH/'team',
    'concentrated':LEGACY_SEARCH/'complete',
}

def method(key):return 'single_core' if key=='tarnawski' else 'enhanced'

def engine(output,key):
    from resource_envelope.search_adapter import _replace
    c=api();settings(key);cached=CACHES.get(key,LEGACY_SEARCH/'uniform')
    proof=c.read(cached/'proof.json');source=Path(proof['source_directory'])
    for leaf,digest in proof['implementation'].items():
        if c.sha256(source/leaf)!=digest:raise ValueError('legacy search source differs')
    contents={leaf:(source/leaf).read_text() for leaf in
        ('search.py','oracle.cc','picojson.h','__init__.py','costs.py','domain.py')}
    if key in ('coarse_no_host','tarnawski'):
        contents['search.py']=_replace(contents['search.py'],
            '    service = np.maximum(np.maximum(cpu_max, vta), ddr)\n',
            '    host = 0  # Exclude host CPU capacity; wall service and DDR remain.\n'
            '    service = np.maximum(np.maximum(cpu_max, vta), ddr)\n')
        contents['oracle.cc']=_replace(contents['oracle.cc'],
            '  std::int64_t sum=0,capacity=0;\n',
            '  host_work=0;  // CPU-resource ablation only.\n'
            '  std::int64_t sum=0,capacity=0;\n')
    if key=='tarnawski':
        contents['search.py']=_replace(contents['search.py'],
            '    return tuple(mask for n in range(1, len(pool) + 1) for mask in combinations(pool, n))',
            '    return tuple((core,) for core in pool)  # Physical single-core devices only.')
        contents['oracle.cc']=_replace(contents['oracle.cc'],
            'for(int n=1;n<=static_cast<int>(pool.size());++n)make_options(0,n,scratch);',
            'for(int n=1;n<=1;++n)make_options(0,n,scratch);')
        contents['search.py']=_replace(contents['search.py'],
            '    serial = np.maximum(enhanced, groups.max(axis=1))',
            '    serial = np.maximum(service, groups.max(axis=1))  # Exact adapted device-load objective.')
        contents['oracle.cc']=_replace(contents['oracle.cc'],
            '  auto serial=enhanced;',
            '  auto serial=service;  // Device wall service, not CPU-work capacity.')
    name='staged_engine_'+c.digest(contents)[:20]
    target=Path(output)/'sources'/name;target.mkdir(parents=True,exist_ok=True)
    for leaf,value in contents.items():
        p=target/leaf
        if p.exists() and p.read_text()!=value:raise ValueError('frozen search source changed')
        if not p.exists():p.write_text(value)
    spec=importlib.util.spec_from_file_location(name,target/'__init__.py',submodule_search_locations=[str(target)])
    package=importlib.util.module_from_spec(spec);sys.modules[name]=package;spec.loader.exec_module(package)
    spec=importlib.util.spec_from_file_location(name+'.search',target/'search.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module

def _costs(key,programs,records,ddr):
    from cpu_policies.open_resources.costs import Costs
    return Costs(records,programs=programs,ddr=ddr,
        distribution='session_weights' if settings(key)['thread_weights'] else 'balanced')

def run_one(output,key):
    c=api();started=time.monotonic();output=Path(output).resolve();config=settings(key)
    programs,records,ddr,hashes=inputs();costs=_costs(key,programs,records,ddr)
    module=engine(output,key);cached=CACHES.get(key);target=output/'search';target.mkdir(parents=True,exist_ok=True)
    cache_used=bool(cached and (cached/'search').is_dir())
    if cache_used:
        for file in (cached/'search').glob('*.json'):
            if file.stem not in [ctx['id'] for ctx in module.domain.contexts()]:continue
            if not (target/file.name).exists():shutil.copy2(file,target/file.name)
    ranked=module.rank_all(programs,costs,ddr=ddr,output_dir=target,verify=True)
    if not ranked['verification']['passed']:raise ValueError('independent enumeration failed')
    if key!='tarnawski' and ranked['counts']['enhanced']!=DOMAIN_COUNT:raise ValueError('full candidate domain differs')
    selected=ranked['methods'][method(key)]['rankings']
    for row in selected:
        if score(row,costs,key)['score_ns']!=row['score_ns']:raise ValueError('scalar score differs from search')
    ranked['staged_model']=config
    if key=='tarnawski':
        ranked['model_formulas']['single_core']='max(singleton physical CPU cumulative isolated wall service, all VTA wall service, DDR); host CPU demand excluded'
    c.write(output/'rankings.json',ranked,replace=True)
    implementation={n:c.sha256(Path(module.__file__).with_name(n)) for n in ('search.py','oracle.cc','picojson.h')}
    c.write(output/'proof.json',dict(model=config,component_hashes=hashes,implementation=implementation,
        source_directory=str(Path(module.__file__).parent),counts=ranked['counts'],selected_method=method(key),
        selected_count=ranked['counts'][method(key)],verification=ranked['verification'],
        cache_replayed=cache_used,cache_source=str(cached) if cache_used else None,
        pipeline_FPS_used=False,elapsed_seconds=time.monotonic()-started),replace=True)
    return dict(model=key,first=selected[0]['id'],candidate_count=ranked['counts'][method(key)],cache_replayed=cache_used)

def verify_one(output,key):
    from cpu_policies.open_resources.search import _digest
    c=api();output=Path(output);proof=c.read(output/'proof.json')
    if proof['model']!=settings(key) or proof['selected_method']!=method(key):raise ValueError('model proof differs')
    programs,records,ddr,hashes=inputs();module=engine(output,key);costs=_costs(key,programs,records,ddr)
    impl={n:c.sha256(Path(module.__file__).with_name(n)) for n in ('search.py','oracle.cc','picojson.h')}
    if proof['component_hashes']!=hashes or proof['implementation']!=impl:raise ValueError('source/cost fingerprint differs')
    graphs=module._topologies(programs,21,3);contexts={ctx['id']:ctx for ctx in module.domain.contexts()}
    total=0;expanded=[]
    for ctx in contexts.values():
        cp=c.read(output/'search'/(ctx['id']+'.json'))
        payload=dict(programs=module._prepare(programs,costs,ctx,ddr),topologies=graphs,context=ctx,
            k=20,score_scale=module.SCORE_SCALE,implementation=impl)
        fields=('counts','gate_counts','rankings','topology_strata','resource_strata');v=cp['verification']
        if (cp['checkpoint_sha256']!=_digest({k:x for k,x in cp.items() if k!='checkpoint_sha256'})
            or cp['input_sha256']!=_digest(payload) or not v['passed']
            or v['source_sha256']!=impl['oracle.cc']
            or v['result_sha256']!=_digest({f:cp[f] for f in fields})):
            raise ValueError('checkpoint/oracle proof differs')
        total+=cp['counts'][method(key)]
        for raw in cp['rankings'][method(key)]:
            t=module.Terminal.from_json(raw);cfg=module._configuration(t,graphs,programs,contexts)
            expanded.append({**cfg,'score_ns':t.score,'score_ms':t.score/1e6})
    if total!=proof['selected_count'] or (key!='tarnawski' and total!=DOMAIN_COUNT):raise ValueError('domain coverage differs')
    ranked=c.read(output/'rankings.json');expected=sorted(expanded,key=module._row_key)[:20]
    for i,row in enumerate(expected,1):row['static_rank']=i
    if ranked['methods'][method(key)]['rankings']!=expected:raise ValueError('global Top20 differs from proof')
    return ranked,proof

def applicable_score(cfg,costs,key):
    if key=='tarnawski' and (cfg['execution_mode']!='resource_group' or any(
            s['device']=='cpu' and s['threads']!=1 for s in cfg['stages'])):
        return dict(applicable=False,reason='outside the physical-single-core execution domain')
    return dict(applicable=True,**score(cfg,costs,key))

def freeze(output):
    from cpu_policies.open_resources.domain import configuration
    c=api();output=Path(output).resolve()
    evidence=output/'evidence'
    if not (evidence/'BASELINE_CONTRACT.json').is_file():raise ValueError('complete evidence audit before ranking')
    programs,records,ddr,hashes=inputs();lists={};full_lists={};pool={};proofs={}
    for key,*_ in MODELS:
        ranked,proof=verify_one(output/key,key);full=ranked['methods'][method(key)]['rankings']
        if len(full)!=20 or len({r['id'] for r in full})!=20:raise ValueError('full unique Top20 required')
        full_lists[key]=full;lists[key]=full[:settings(key)['measurement_budget']];proofs[key]=proof
        for row in lists[key]:
            cfg={**configuration(row),'id':row['id']}
            if row['id'] in pool and pool[row['id']]!=cfg:raise ValueError('configuration identity collision')
            pool[row['id']]=cfg
    selection_count=len(pool)
    if selection_count>115:raise ValueError('selection pool exceeds declared budget')
    mechanism=c.read(evidence/'MECHANISM_AUDIT.json')
    for row in mechanism['required_new_configurations']:
        cfg=row['configuration'];pool[cfg['id']]=cfg
    if len(mechanism['required_new_configurations'])>6 or len(pool)>121:raise ValueError('mechanism budget exceeded')
    scores={key:{i:applicable_score(cfg,_costs(key,programs,records,ddr),key) for i,cfg in sorted(pool.items())}
            for key,*_ in MODELS}
    for key,rows in lists.items():
        if any(scores[key][r['id']]['score_ns']!=r['score_ns'] for r in rows):raise ValueError('frozen score differs')
    selected=output/'frozen';selected.mkdir(exist_ok=True)
    docs={'selection_rankings.json':lists,'full_rankings.json':full_lists,'static_pool.json':dict(sorted(pool.items())),
        'model_scores.json':scores,'search_proofs.json':proofs,'selection_protocol.json':protocol(),
        'mechanism_audit.json':mechanism,'baseline_contract.json':c.read(evidence/'BASELINE_CONTRACT.json')}
    for leaf,value in docs.items():
        p=selected/leaf
        if p.exists() and c.read(p)!=value:raise ValueError('refusing to replace frozen selection')
        if not p.exists():c.write(p,value)
    import random
    order=sorted(pool);random.Random(protocol()['seed']).shuffle(order)
    frozen=dict(files={n:c.sha256(selected/n) for n in docs},component_hashes=hashes,
        configuration_order=order,pipeline_FPS_used=False,candidate_count=len(pool),selection_count=selection_count,
        native_sessions=2*len(pool),complete_inferences=29*len(pool))
    p=selected/'FREEZE.json'
    if p.exists():
        old=c.read(p)
        if any(old[k]!=v for k,v in frozen.items()):raise ValueError('freeze identity differs')
    else:c.write(p,{**frozen,'frozen_at':c.now()})
    return selected

def run_all(output,workers=2):
    c=api();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    if not (output/'evidence/BASELINE_CONTRACT.json').is_file():raise ValueError('run audit first')
    started=time.monotonic()
    def worker(key):
        directory=output/key
        if (directory/'proof.json').exists():verify_one(directory,key);return {'model':key,'resumed_verified':True}
        directory.mkdir(exist_ok=True);print('[分阶段消融] search '+key,flush=True)
        with (directory/'search.log').open('a') as log:
            p=subprocess.run([sys.executable,str(ROOT/'run_staged_ablation.py'),'rank-one','--output',str(directory),'--model',key],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
        if p.returncode:raise RuntimeError('search failed: '+str(directory/'search.log'))
        print('[分阶段消融] verified '+key,flush=True);return {'model':key}
    with ThreadPoolExecutor(max_workers=workers) as executor:results=list(executor.map(worker,[r[0] for r in MODELS]))
    target=freeze(output)
    c.write(output/'RANK_COST.json',dict(elapsed_seconds=time.monotonic()-started),replace=True)
    return dict(frozen=str(target),models=results,candidate_count=c.read(target/'FREEZE.json')['candidate_count'])
