"""Full-domain DP and independent C++ verification, with isolated inputs only."""
from pathlib import Path
import os
import random
import subprocess
import sys
import time
from cpu_vta_partitioning.common import digest, now, read, sha256, write
from cpu_vta_partitioning.prepare.build import copy_checked
from cpu_policies.open_resources.costs import Costs
from cpu_policies.open_resources.domain import configuration
from staged_ablation.model import score
from staged_ablation.search import engine, method
from yolo_transfer.domain_counts import allocation_counts
from yolo_transfer.ranking import ddr_table, replay_traffic
from .common import METHODS,HARDWARE,ROOT,assert_sources,frozen,protocol
from .components import replay_complete

def inputs(output):
    """Only frozen programs, isolated costs and DDR are admissible ranking inputs."""
    root=Path(output)/'ranking/inputs'
    programs={n:p for n,p in read(root/'programs.json').items() if p['eligible_search']}
    records=read(root/'isolated_costs.json')['costs']
    ddr=read(Path(output)/'ranking/ddr.json')
    hashes={n:sha256(root/n) for n in ('programs.json','isolated_costs.json','traffic.json','hardware.json')}
    hashes['ddr.json']=sha256(Path(output)/'ranking/ddr.json')
    return programs,records,ddr,hashes

def prepare_inputs(output,traffic_run):
    output=Path(output);prep=assert_sources(output);bundle=Path(prep['profile_bundle'])
    replay_complete(output)
    traffic=replay_traffic(bundle,traffic_run)
    sources={'programs.json':bundle/'plans/programs.json',
        'isolated_costs.json':output/'isolated_costs.json',
        'traffic.json':Path(traffic_run)/'traffic.json','hardware.json':HARDWARE}
    target=output/'ranking/inputs'
    for leaf,path in sources.items():
        dest=target/leaf
        if dest.exists() and sha256(dest)!=sha256(path):raise ValueError('static input changed')
        if not dest.exists():copy_checked(path,dest)
    programs=read(target/'programs.json')
    frozen(output/'ranking/ddr.json',ddr_table(programs,traffic,read(HARDWARE)['provenance']))
    frozen(output/'ranking/INPUTS.json',{leaf:dict(source=str(path.resolve()),sha256=sha256(path)) for leaf,path in sources.items()})

def costs_for(key,programs,records,ddr):
    return Costs(records,programs=programs,ddr=ddr,
        distribution='session_weights' if key=='fine_host' else 'balanced')

def rank_one(output,key):
    output=Path(output).resolve();assert_sources(output)
    if key not in METHODS:raise ValueError('unknown validation model')
    started=time.monotonic();programs,records,ddr,hashes=inputs(output)
    target=output/'ranking'/key;module=engine(target,key)
    costs=costs_for(key,programs,records,ddr)
    ranked=module.rank_all(programs,costs,ddr=ddr,output_dir=target/'search',unit_count=14,max_vta_islands=3,verify=True)
    counts=allocation_counts();selected=method(key)
    expected=3067532 if key=='tarnawski' else 472266548
    if (ranked['counts']['topologies']!=2509 or ranked['counts'][selected]!=expected
            or not ranked['verification']['passed']):
        raise ValueError('full-domain coverage/oracle proof differs')
    rows=ranked['methods'][selected]['rankings']
    for row in rows:
        if score(row,costs,key)['score_ns']!=row['score_ns']:
            raise ValueError('Python scalar score differs from full search')
    frozen(target/'rankings.json',ranked)
    implementation={n:sha256(Path(module.__file__).with_name(n)) for n in ('search.py','oracle.cc','picojson.h')}
    proof=dict(model=key,component_hashes=hashes,implementation=implementation,
        source_directory=str(Path(module.__file__).parent),counts=ranked['counts'],
        selected_method=selected,selected_count=expected,verification=ranked['verification'],
        pipeline_FPS_used=False)
    frozen(target/'proof.json',proof)
    write(target/'COST.json',dict(seconds=time.monotonic()-started),replace=True)
    return dict(model=key,first=rows[0]['id'],count=expected)

def verify_one(output,key):
    output=Path(output);target=output/'ranking'/key;programs,records,ddr,hashes=inputs(output)
    proof=read(target/'proof.json');module=engine(target,key)
    impl={n:sha256(Path(module.__file__).with_name(n)) for n in ('search.py','oracle.cc','picojson.h')}
    if proof['component_hashes']!=hashes or proof['implementation']!=impl or proof['model']!=key:
        raise ValueError('search source/input proof changed')
    costs=costs_for(key,programs,records,ddr)
    graphs=module._topologies(programs,14,3)
    expanded=[];total=0
    contexts={ctx['id']:ctx for ctx in module.domain.contexts()}
    for ctx in contexts.values():
        cp=read(target/'search'/(ctx['id']+'.json'))
        payload=dict(programs=module._prepare(programs,costs,ctx,ddr),topologies=graphs,context=ctx,
            k=20,score_scale=module.SCORE_SCALE,implementation=impl)
        fields=('counts','gate_counts','rankings','topology_strata','resource_strata');v=cp['verification']
        if (cp['checkpoint_sha256']!=module._digest({k:x for k,x in cp.items() if k!='checkpoint_sha256'})
                or cp['input_sha256']!=module._digest(payload) or not v['passed']
                or v['source_sha256']!=impl['oracle.cc']
                or v['result_sha256']!=module._digest({f:cp[f] for f in fields})):
            raise ValueError('complete-domain checkpoint/oracle proof differs')
        total+=cp['counts'][method(key)]
        for raw in cp['rankings'][method(key)]:
            terminal=module.Terminal.from_json(raw)
            cfg=module._configuration(terminal,graphs,programs,contexts)
            expanded.append({**cfg,'score_ns':terminal.score,'score_ms':terminal.score/1e6})
    expected=sorted(expanded,key=module._row_key)[:20]
    for i,row in enumerate(expected,1):row['static_rank']=i
    rows=read(target/'rankings.json')['methods'][method(key)]['rankings']
    if expected!=rows or total!=proof['selected_count']:
        raise ValueError('global Top20 differs from independent complete search')
    for row in rows:
        if score(row,costs,key)['score_ns']!=row['score_ns']:raise ValueError('scalar ranking differs')
    return rows,proof

def freeze(output):
    output=Path(output);lists={};pool={};proofs={};scores={}
    programs,records,ddr,hashes=inputs(output)
    for key in METHODS:
        rows,proofs[key]=verify_one(output,key)
        if len(rows)!=20 or len({r['id'] for r in rows})!=20:raise ValueError('twenty unique candidates required')
        lists[key]=rows
        for row in rows:
            cfg={**configuration(row),'id':row['id']}
            if row['id'] in pool and pool[row['id']]!=cfg:raise ValueError('execution identity collision')
            pool[row['id']]=cfg
    if len(pool)>60:raise ValueError('candidate budget exceeded')
    for key in METHODS:
        costs=costs_for(key,programs,records,ddr)
        scores[key]={i:(dict(applicable=False) if key=='tarnawski' and (
            cfg['execution_mode']!='resource_group' or any(s['device']=='cpu' and s['threads']!=1 for s in cfg['stages']))
            else dict(applicable=True,**score(cfg,costs,key))) for i,cfg in sorted(pool.items())}
    root=output/'ranking';docs={'rankings.json':dict(lists=lists), 'pool.json':dict(sorted(pool.items())),
        'proofs.json':proofs,'scores.json':scores,'protocol.json':protocol()}
    for leaf,doc in docs.items():frozen(root/leaf,doc)
    order=sorted(pool);random.Random(protocol()['seed']).shuffle(order)
    frozen(root/'execution_order.json',order)
    seal=dict(files={n:sha256(root/n) for n in (*docs,'execution_order.json')},
        component_hashes=hashes,candidate_count=len(pool),pipeline_FPS_used=False)
    if (root/'FREEZE.json').exists():
        old=read(root/'FREEZE.json')
        if any(old[k]!=v for k,v in seal.items()):raise ValueError('ranking freeze changed')
    else:write(root/'FREEZE.json',{**seal,'frozen_at':now()})
    return dict(candidate_count=len(pool),frozen=str(root/'FREEZE.json'))

def rank(output):
    output=Path(output).resolve();assert_sources(output)
    started=time.monotonic()
    for key in METHODS:
        if (output/'ranking'/key/'proof.json').exists():verify_one(output,key);continue
        target=output/'ranking'/key;target.mkdir(parents=True,exist_ok=True)
        print('YOLO full-domain search',key,flush=True)
        with (target/'search.log').open('a') as log:
            p=subprocess.run([sys.executable,str(ROOT/'run_yolo_validation.py'),'rank-one','--output',str(output),'--model',key],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
        if p.returncode:raise RuntimeError('search failed: '+str(target/'search.log'))
        print('YOLO search independently verified',key,flush=True)
    result=freeze(output)
    write(output/'ranking/COST.json',dict(seconds=time.monotonic()-started),replace=True)
    return result
