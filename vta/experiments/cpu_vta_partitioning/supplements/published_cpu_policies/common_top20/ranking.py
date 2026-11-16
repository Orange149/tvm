"""Whole-domain score replay, semantic revision, and existing policy controls."""
from pathlib import Path
import random
import time

from cpu_vta_partitioning.common import digest,read,sha256,write
from cpu_policies.open_resources.costs import Costs
from cpu_policies.open_resources import domain
from policy_branch_eval.common import isolated_inputs_only,require_ranked
from policy_branch_eval.ranking import inputs,engine_from
from policy_branch_eval.selection import fixed_family,mixed
from staged_ablation.model import score
from staged_ablation.search import engine as new_engine
from workload_model.common import ROOT,freeze,require_space


def verify_rows(root,network,key,programs,records,ddr,hashes,units):
    root=Path(root);proof=read(root/'proof.json')
    if proof['component_hashes']!=hashes or proof['pipeline_FPS_used'] is not False:
        raise ValueError('whole-domain cost identity or feedback flag differs')
    module=engine_from(proof,network+'_'+key)
    costs=Costs(records,programs=programs,ddr=ddr,distribution='balanced' if key=='tarnawski' else 'session_weights')
    graphs=module._topologies(programs,units,3);contexts={c['id']:c for c in domain.contexts()}
    method='single_core' if key=='tarnawski' else 'enhanced';total=0;expanded=[];files={}
    for context in contexts.values():
        path=root/'search'/(context['id']+'.json');cp=read(path)
        payload=dict(programs=module._prepare(programs,costs,context,ddr),topologies=graphs,
                     context=context,k=20,score_scale=module.SCORE_SCALE,implementation=proof['implementation'])
        fields=('counts','gate_counts','rankings','topology_strata','resource_strata');v=cp['verification']
        if (cp['checkpoint_sha256']!=module._digest({k:x for k,x in cp.items() if k!='checkpoint_sha256'})
            or cp['input_sha256']!=module._digest(payload) or not v['passed']
            or v['source_sha256']!=proof['implementation']['oracle.cc']
            or v['result_sha256']!=module._digest({f:cp[f] for f in fields})):
            raise ValueError('complete-domain independent enumeration proof changed')
        total+=cp['counts'][method];files[str(path)]=sha256(path)
        for raw in cp['rankings'][method]:
            t=module.Terminal.from_json(raw);cfg=module._configuration(t,graphs,programs,contexts)
            expanded.append({**cfg,'score_ns':t.score,'score_ms':t.score/1e6})
    rows=sorted(expanded,key=module._row_key)[:20]
    for i,row in enumerate(rows,1):row['static_rank']=i
    if rows!=read(root/'rankings.json')['methods'][method]['rankings'] or total!=proof['selected_count']:
        raise ValueError('global strict Top20 or complete coverage changed')
    expected=sum(len(c['compute_pool'])**sum(programs[n]['device']=='cpu' for n in graph)
                 for c in contexts.values() for graph in graphs) if key=='tarnawski' else (966240306 if units==21 else 472266548)
    if total!=expected:raise ValueError('full legal domain count differs')
    for row in rows:
        if score(row,costs,key)['score_ns']!=row['score_ns']:raise ValueError('scalar formula differs from enumerated result')
    files.update({str(root/n):sha256(root/n) for n in ('proof.json','rankings.json')})
    return rows,dict(domain_count=total,full_domain_verified=True,files=files),module,graphs,costs


def fixed_rows(network,programs,costs,ddr,module,graphs):
    parent=ROOT/'artifacts/policy_branch_eval_01';seal=require_ranked(parent)
    root=parent/network/'fixed_search';expanded=[];total=0;files={}
    proof=read(root/'oracle/FILTER_PROOF.json')
    if (proof['original_sha256']!=sha256(Path(module.__file__).with_name('oracle.cc'))
            or proof['score_expressions_changed'] is not False):
        raise ValueError('fixed family changed score expressions')
    if sha256(root/'oracle/oracle.cc')!=proof['filtered_sha256']:
        raise ValueError('fixed-family C++ source changed')
    for context in domain.contexts():
        path=root/(context['id']+'.json');cp=read(path)
        payload=dict(programs=module._prepare(programs,costs,context,ddr),topologies=graphs,
                     context=context,k=20,score_scale=module.SCORE_SCALE,policy_family='fixed')
        binaries=[p for p in (root/'oracle').glob('scalar_oracle_*') if p.is_file()]
        if (cp['content_sha256']!=digest({n:v for n,v in cp.items() if n!='content_sha256'})
            or cp['input_sha256']!=digest(payload) or not cp['verification_passed']
            or not any(sha256(p)==cp['oracle_sha256'] for p in binaries)):
            raise ValueError('fixed-family complete-domain proof differs')
        total+=cp['count'];expanded.extend(cp['rankings']);files[str(path)]=sha256(path)
    rows=sorted(expanded,key=module._row_key)[:20]
    prior=read(parent/network/'rankings.json')
    if rows!=prior['lists']['fixed'] or total!=prior['fixed_proof']['count']:
        raise ValueError('fixed-family global Top20 differs')
    if any(not fixed_family(r) or score(r,costs,'fine_host')['score_ns']!=r['score_ns'] for r in rows):
        raise ValueError('fixed-family identity or scalar score differs')
    return rows,dict(domain_count=total,restricted_family_explicit=True,files=files,
                     source_seal_sha256=sha256(parent/'RANK_FREEZE.json'))


def new_revision(output,network,programs,records,ddr,hashes,units):
    target=Path(output)/network/'semantics_revision';target.mkdir(parents=True,exist_ok=True)
    if not (target/'proof.json').exists():
        module=new_engine(target,'weights_only')
        costs=Costs(records,programs=programs,ddr=ddr,distribution='session_weights')
        ranked=module.rank_all(programs,costs,ddr=ddr,output_dir=target/'search',
                               unit_count=units,max_vta_islands=3,verify=True)
        write(target/'rankings.json',ranked)
        implementation={n:sha256(Path(module.__file__).with_name(n)) for n in ('search.py','oracle.cc','picojson.h')}
        write(target/'proof.json',dict(component_hashes=hashes,implementation=implementation,
            source_directory=str(Path(module.__file__).parent),selected_count=ranked['counts']['enhanced'],
            pipeline_FPS_used=False,selected_method='enhanced'))
    return verify_rows(target,network,'weights_only',programs,records,ddr,hashes,units)[:2]


def rank(output):
    output=Path(output).resolve();require_space();started=time.monotonic();plan=read(output/'PLAN.json')
    result={};counts={};orders={}
    with isolated_inputs_only():
        for network in plan['networks']:
            programs,records,ddr,hashes,current,units=inputs(network)
            main,main_proof,module,graphs,costs=verify_rows(current,network,'fine_host',programs,records,ddr,hashes,units)
            baseline=ROOT/('artifacts/staged_ablation_01/tarnawski' if network=='resnet18' else 'artifacts/yolo_validation_01/ranking/tarnawski')
            tarnawski,tproof,*_=verify_rows(baseline,network,'tarnawski',programs,records,ddr,hashes,units)
            fixed,fproof=fixed_rows(network,programs,costs,ddr,module,graphs)
            blended,reasons=mixed(main,fixed)
            if network=='resnet18':
                revised,rproof,*_=verify_rows(ROOT/'artifacts/staged_ablation_01/weights_only',network,'weights_only',programs,records,ddr,hashes,units)
            else:revised,rproof=new_revision(output,network,programs,records,ddr,hashes,units)
            lists=dict(current=main,semantics_revision=revised,fixed_family=fixed,mixed=blended,tarnawski=tarnawski)
            pool={}
            for rows in lists.values():
                if len(rows)!=20 or len({r['id'] for r in rows})!=20:raise ValueError('every method requires twenty unique configurations')
                for r in rows:
                    cfg={**domain.configuration(r),'id':r['id']}
                    domain.validate_candidate(cfg,programs,unit_count=units)
                    if r['id'] in pool and cfg!=pool[r['id']]:raise ValueError('execution identity collision')
                    pool[r['id']]=cfg
            if len(pool)>80:raise ValueError('common pool exceeds predeclared bound')
            order=sorted(pool);random.Random(plan['seed']).shuffle(order)
            freeze(output/network/'RANKINGS.json',dict(lists=lists,pool=pool,execution_order=order,
                mixed_reasons=reasons,component_hashes=hashes,proofs=dict(current=main_proof,
                    semantics_revision=rproof,fixed_family=fproof,tarnawski=tproof),pipeline_FPS_read=False))
            result[network]=dict(pool_count=len(pool),method_first={k:v[0]['id'] for k,v in lists.items()})
            counts[network]=len(pool);orders[network]=order
            print('共同Top20名单已核验',network,len(pool),flush=True)
        freeze(output/'RANK_FREEZE.json',dict(files={n+'/RANKINGS.json':sha256(output/n/'RANKINGS.json') for n in result},
            plan_sha256=sha256(output/'PLAN.json'),ranking_source_sha256=sha256(Path(__file__)),
            candidate_counts=counts,pipeline_FPS_read=False,selection_order=orders))
    write(output/'RANK_COST.json',dict(seconds=time.monotonic()-started),replace=True)
    return result
