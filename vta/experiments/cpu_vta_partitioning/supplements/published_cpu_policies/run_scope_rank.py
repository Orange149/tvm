"""Full-domain, unchanged-formula check of the isolated run-share input repair.

Both arms use the historical strongest complete formula and the identical
Python DP / independent C++ enumerator. No pipeline performance is read here.
"""
from collections import Counter
from pathlib import Path
import argparse
import json
import shutil
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,write,sha256,digest,now
from cpu_policies.open_resources.costs import Costs
from model_ablation.model import score
from policy_branch_eval.common import isolated_inputs_only,offline_guard
from policy_branch_eval.ranking import engine_from
from run_scope_inputs import RunScopeCosts

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/run_scope_rank_01'
VARIANTS=('original','run_scope')
NETWORKS={
    'resnet18':dict(units=21,topologies=4623,allocations=966240306,
        ddr=ROOT/'artifacts/frame_stream_top20_static_validation/ddr.json',
        original_cache=ROOT/'artifacts/weighted_envelope_full_domain_01/search'),
    'yolov3_tiny416':dict(units=14,topologies=2509,allocations=472266548,
        ddr=ROOT/'artifacts/yolo_validation_01/ranking/ddr.json',original_cache=None)}


def freeze(path,value):
    path=Path(path)
    if path.exists():
        if read(path)!=value:
            raise ValueError('frozen input or result differs: '+str(path))
    else:
        write(path,value)
    return value


def prepare():
    input_path=ROOT/'artifacts/run_scope_weights_01/INPUTS.json'
    inputs=read(input_path)
    if not inputs['no_pipeline_performance_input'] or inputs['new_measured_costs']:
        raise ValueError('isolated-only, unchanged timing inputs required')
    for path,expected in inputs['source_hashes'].items():
        if sha256(path)!=expected:
            raise ValueError('isolated scope evidence changed')
    source_proof=read(ROOT/'artifacts/model_ablation_01/complete/proof.json')
    if source_proof['model']!={
            'key':'complete','label':'完整模型','thread_weights':True,'host_guard':True,'team_guard':True}:
        raise ValueError('strong complete formula contract differs')
    source_dir=Path(source_proof['source_directory'])
    destination=OUT/'sources/complete'
    destination.mkdir(parents=True,exist_ok=True)
    copied={}
    for leaf in ('__init__.py','search.py','oracle.cc','costs.py','domain.py','picojson.h'):
        source=source_dir/leaf;expected=sha256(source)
        if leaf in source_proof['implementation'] and expected!=source_proof['implementation'][leaf]:
            raise ValueError('historical complete engine source differs')
        target=destination/leaf
        if target.exists() and sha256(target)!=expected:
            raise ValueError('copied engine source differs')
        if not target.exists():
            shutil.copy2(source,target)
        copied[leaf]=expected
    source_files=[Path(__file__),ROOT/'run_scope_inputs.py',ROOT/'run_budget_reconstruction.py',
        ROOT/'cpu_policies/open_resources/costs.py',ROOT/'cpu_policies/open_resources/domain.py',
        ROOT/'resource_envelope/model.py',ROOT/'model_ablation/model.py',ROOT/'model_ablation/common.py',
        ROOT/'policy_branch_eval/common.py',ROOT/'policy_branch_eval/ranking.py']
    source_hashes={str(p):sha256(p) for p in source_files}
    networks={}
    for network,spec in NETWORKS.items():
        source=inputs['networks'][network]
        for leaf in ('cost','programs'):
            if sha256(source[leaf+'_path'])!=source[leaf+'_sha256']:
                raise ValueError('source cost or program identity differs')
        programs=read(source['programs_path'])
        if network=='yolov3_tiny416':
            excluded={n:p['purpose'] for n,p in programs.items() if not p['eligible_search']}
            if excluded!={'cpu_01_01':'atomic_thread_curve_only','cpu_12_12':'atomic_thread_curve_only'}:
                raise ValueError('YOLO curve-only inventory differs')
        else:
            excluded={}
        networks[network]={**source,**{k:v for k,v in spec.items() if k not in ('ddr','original_cache')},
            'ddr_path':str(spec['ddr']),'ddr_sha256':sha256(spec['ddr']),
            'original_cache':str(spec['original_cache']) if spec['original_cache'] else None,
            'excluded_curve_only_modules':excluded}
    return freeze(OUT/'PLAN.json',dict(networks=networks,variants=list(VARIANTS),
        input_variant_path=str(input_path),input_variant_sha256=sha256(input_path),
        copied_engine_files=copied,implementation=source_proof['implementation'],source_hashes=source_hashes,
        model_switches=dict(thread_weights=True,host_CPU_demand=True,team_guard=True,host_concentration=True),
        max_VTA_islands=3,contexts=9,top_k=20,full_configuration_domain_preserved=True,
        pipeline_performance_read=False,new_board_sessions=0,new_profile_measurements=0,
        execution_mechanism_changed=False,shared_timing_gap_claimed_solved=False))


def assert_plan():
    plan=read(OUT/'PLAN.json')
    for path,expected in plan['source_hashes'].items():
        if sha256(path)!=expected:
            raise ValueError('rank implementation changed')
    if sha256(plan['input_variant_path'])!=plan['input_variant_sha256']:
        raise ValueError('scope variant identity changed')
    for leaf,expected in plan['copied_engine_files'].items():
        if sha256(OUT/'sources/complete'/leaf)!=expected:
            raise ValueError('copied complete engine changed')
    return plan


def inputs_for(plan,network,variant):
    if variant not in VARIANTS or network not in NETWORKS:
        raise ValueError('unsupported network or input variant')
    spec=plan['networks'][network]
    for leaf in ('cost','programs','ddr'):
        if sha256(spec[leaf+'_path'])!=spec[leaf+'_sha256']:
            raise ValueError('frozen ranking input changed')
    records=read(spec['cost_path'])['costs']
    programs=read(spec['programs_path'])
    if network=='yolov3_tiny416':
        programs={n:p for n,p in programs.items() if p['eligible_search']}
    ddr=read(spec['ddr_path'])
    costs=(Costs(records,programs=programs,ddr=ddr,distribution='session_weights') if variant=='original'
           else RunScopeCosts(records,spec['patches'],programs=programs,ddr=ddr))
    return programs,costs,ddr


def load_engine(plan):
    return engine_from(dict(source_directory=str(OUT/'sources/complete'),
        implementation=plan['implementation']),'run_scope_complete_'+plan['implementation']['search.py'][:12])


def payload_for(module,programs,costs,ddr,graphs,ctx):
    implementation={n:sha256(Path(module.__file__).with_name(n)) for n in ('search.py','oracle.cc','picojson.h')}
    return dict(programs=module._prepare(programs,costs,ctx,ddr),topologies=graphs,
        context=ctx,k=20,score_scale=module.SCORE_SCALE,implementation=implementation)


def checked_checkpoint(module,path,payload):
    cp=read(path);fields=('counts','gate_counts','rankings','topology_strata','resource_strata')
    if cp['checkpoint_sha256']!=module._digest({k:v for k,v in cp.items() if k!='checkpoint_sha256'}):
        raise ValueError('checkpoint content hash differs')
    if cp['input_sha256']!=module._digest(payload):
        return None
    proof=cp.get('verification',{})
    if (not proof.get('passed') or proof['source_sha256']!=payload['implementation']['oracle.cc'] or
            proof['result_sha256']!=module._digest({k:cp[k] for k in fields})):
        raise ValueError('checkpoint independent oracle proof differs')
    return cp


def copy_matching_contexts(module,programs,costs,ddr,graphs,source,target):
    source,target=Path(source),Path(target);matched={}
    target.mkdir(parents=True,exist_ok=True)
    for ctx in module.domain.contexts():
        path=source/(ctx['id']+'.json')
        if not path.exists():
            continue
        payload=payload_for(module,programs,costs,ddr,graphs,ctx)
        if checked_checkpoint(module,path,payload) is None:
            continue
        dest=target/path.name
        if dest.exists() and sha256(dest)!=sha256(path):
            raise ValueError('cache destination differs')
        if not dest.exists():
            shutil.copy2(path,dest)
        matched[ctx['id']]=dict(source=str(path),sha256=sha256(path))
    return matched


def rank_one(network,variant):
    plan=assert_plan();spec=plan['networks'][network]
    target=OUT/network/variant
    if (target/'PROOF.json').exists():
        rows,_=verify_one(network,variant)
        print(json.dumps(dict(network=network,variant=variant,state='COMPLETE_CACHED',first=rows[0]['id'])),flush=True)
        return read(target/'rankings.json')
    programs,costs,ddr=inputs_for(plan,network,variant);module=load_engine(plan)
    target.mkdir(parents=True,exist_ok=True)
    graphs=module._topologies(programs,spec['units'],3)
    cache={}
    if variant=='original' and spec['original_cache']:
        cache=copy_matching_contexts(module,programs,costs,ddr,graphs,spec['original_cache'],target/'search')
    freeze(target/'INPUTS.json',dict(plan_sha256=sha256(OUT/'PLAN.json'),network=network,variant=variant,
        cost_sha256=spec['cost_sha256'],programs_sha256=spec['programs_sha256'],ddr_sha256=spec['ddr_sha256'],
        patch_digest=digest(spec['patches']) if variant=='run_scope' else None,
        same_complete_formula=True,cache=cache,no_pipeline_performance_input=True))
    started=time.monotonic();done=[]
    with offline_guard(OUT) as guard,isolated_inputs_only():
        for ctx in module.domain.contexts():
            print(json.dumps(dict(network=network,variant=variant,context=ctx['id'],state='SEARCHING')),flush=True)
            module.rank_all(programs,costs,ddr=ddr,output_dir=target/'search',unit_count=spec['units'],
                            max_vta_islands=3,verify=True,contexts=[ctx],k=20)
            done.append(ctx['id'])
            write(target/'PROGRESS.json',dict(completed_contexts=done,total_contexts=9,
                elapsed_seconds=time.monotonic()-started,state='SEARCHING'),replace=True)
            print(json.dumps(dict(network=network,variant=variant,context=ctx['id'],state='VERIFIED')),flush=True)
        ranked=module.rank_all(programs,costs,ddr=ddr,output_dir=target/'search',unit_count=spec['units'],
                               max_vta_islands=3,verify=True,k=20)
        for row in ranked['pure_static20']:
            if score(row,costs,'complete')['score_ns']!=row['score_ns']:
                raise ValueError('unchanged complete scalar formula differs from search')
    if (ranked['counts']['enhanced']!=spec['allocations'] or
            ranked['counts']['topologies']!=spec['topologies'] or ranked['counts']['contexts']!=9 or
            not ranked['verification']['passed']):
        raise ValueError('full domain or oracle verification incomplete')
    # Correct provenance labels without changing the archived implementation.
    ranked['experiment']='isolated_run_scope_complete_domain'
    ranked['network']=network;ranked['input_variant']=variant
    ranked['assumptions']['cpu_work_split']=(
        'original isolated whole-session shares, retaining explicit uniform fallbacks' if variant=='original' else
        'run-budget reconstruction where observable; original shares or explicit fallbacks otherwise')
    ranked['model_switches']=plan['model_switches']
    ranked['model_formulas']['enhanced']='max(original capacity, CPU team service, CPU per-core work plus all allowed host work)'
    freeze(target/'rankings.json',ranked)
    proof=dict(plan_sha256=sha256(OUT/'PLAN.json'),rankings_sha256=sha256(target/'rankings.json'),
        input_sha256=sha256(target/'INPUTS.json'),implementation=plan['implementation'],
        counts=ranked['counts'],verification=ranked['verification'],
        checkpoints={ctx['id']:sha256(target/'search'/(ctx['id']+'.json')) for ctx in module.domain.contexts()},
        guard=guard,pipeline_FPS_read=False)
    freeze(target/'PROOF.json',proof)
    write(target/'PROGRESS.json',dict(completed_contexts=done,total_contexts=9,state='COMPLETE',
        elapsed_seconds=time.monotonic()-started),replace=True)
    print(json.dumps(dict(network=network,variant=variant,state='COMPLETE',
        first=ranked['pure_static20'][0]['id'],allocations=spec['allocations'])),flush=True)
    return ranked


def verify_one(network,variant):
    plan=assert_plan();spec=plan['networks'][network];target=OUT/network/variant
    proof=read(target/'PROOF.json');ranked=read(target/'rankings.json')
    if (proof['plan_sha256']!=sha256(OUT/'PLAN.json') or
            proof['rankings_sha256']!=sha256(target/'rankings.json') or
            proof['input_sha256']!=sha256(target/'INPUTS.json')):
        raise ValueError('rank freeze identity differs')
    programs,costs,ddr=inputs_for(plan,network,variant);module=load_engine(plan)
    graphs=module._topologies(programs,spec['units'],3)
    contexts={ctx['id']:ctx for ctx in module.domain.contexts()};expanded=[];total=0
    for ctx in contexts.values():
        path=target/'search'/(ctx['id']+'.json')
        if sha256(path)!=proof['checkpoints'][ctx['id']]:
            raise ValueError('proved context changed')
        cp=checked_checkpoint(module,path,payload_for(module,programs,costs,ddr,graphs,ctx))
        if cp is None:
            raise ValueError('context input no longer matches')
        total+=cp['counts']['enhanced']
        for raw in cp['rankings']['enhanced']:
            terminal=module.Terminal.from_json(raw)
            expanded.append({**module._configuration(terminal,graphs,programs,contexts),
                'score_ns':terminal.score,'score_ms':terminal.score/1e6})
    expected=sorted(expanded,key=module._row_key)[:20]
    for i,row in enumerate(expected,1):row['static_rank']=i
    if total!=spec['allocations'] or expected!=ranked['pure_static20']:
        raise ValueError('global Top20 not supported by independent complete enumeration')
    for row in expected:
        if score(row,costs,'complete')['score_ns']!=row['score_ns']:
            raise ValueError('scalar score differs')
    return expected,costs


def compare():
    plan=assert_plan();lists={};summaries=[];scores={}
    with isolated_inputs_only():
        for network in NETWORKS:
            rows={};costs={}
            for variant in VARIANTS:
                rows[variant],costs[variant]=verify_one(network,variant)
                lists[network+'/'+variant]=rows[variant]
            a,b=rows['original'],rows['run_scope'];ia=[r['id'] for r in a];ib=[r['id'] for r in b]
            pool={r['id']:r for r in a+b}
            scores[network]={variant:{key:score(row,costs[variant],'complete') for key,row in pool.items()}
                             for variant in VARIANTS}
            summaries.append(dict(network=network,original_first=ia[0],run_scope_first=ib[0],
                same_first=ia[0]==ib[0],same_ordered20=ia==ib,overlap20=len(set(ia)&set(ib)),
                union=len(pool),top5_overlap=len(set(ia[:5])&set(ib[:5])),
                original_contexts=dict(Counter(r['context_id'] for r in a)),
                run_scope_contexts=dict(Counter(r['context_id'] for r in b)),
                inference_or_FPS_gain_not_claimed=True))
    result=dict(summaries=summaries,plan_sha256=sha256(OUT/'PLAN.json'),
        same_formula_and_full_domain=True,only_CPU_thread_share_input_changed=True,
        no_new_profile_or_board_sessions=True,no_pipeline_performance_read=True,
        shared_timing_gap_not_claimed_solved=True)
    freeze(OUT/'COMPARE.json',result);freeze(OUT/'LISTS.json',lists);freeze(OUT/'SCORES.json',scores)
    freeze(OUT/'SELECTION_FREEZE.json',dict(files={n:sha256(OUT/n) for n in ('COMPARE.json','LISTS.json','SCORES.json')},
        plan_sha256=sha256(OUT/'PLAN.json'),new_board_sessions=0,no_pipeline_FPS_input=True))
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('prepare');one=sub.add_parser('rank-one')
    one.add_argument('--network',choices=NETWORKS,required=True)
    one.add_argument('--variant',choices=VARIANTS,required=True)
    sub.add_parser('compare')
    args=parser.parse_args()
    if args.command=='prepare':
        prepare();print('Prepared isolated-only full-domain scope comparison.',flush=True)
    elif args.command=='rank-one':
        rank_one(args.network,args.variant)
    else:
        compare()
