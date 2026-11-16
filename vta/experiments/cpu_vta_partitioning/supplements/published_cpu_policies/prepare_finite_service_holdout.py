#!/usr/bin/env python3
"""FPS-free small prospective selection check; original full domain is untouched."""
from copy import deepcopy
from itertools import product
from pathlib import Path
import argparse,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,write,sha256,digest
from cpu_policies.open_resources import domain
from cpu_policies.open_resources.costs import Costs
from staged_ablation.model import score
from policy_branch_eval.ranking import inputs
from policy_branch_eval.common import isolated_inputs_only
from pool_temporal.ranking import model_stages
from model_semantics_audit.common import metadata_projection
from model_semantics_audit.replay import validate_isolated
from thread_history_audit.simulator import HistoryReplay
from finite_service_replay import FiniteReplay
from finite_service.common import ROOT,OUTPUT,freeze,sources as platform_sources

DEST=OUTPUT/'prospective'
WINDOW=dict(frames=28,warmup=8,scored=16,queue_depth=2)


def fingerprint(value):
    """Conservatively exclude old resource placements even under other runners."""
    rows=value.get('stages')
    if not isinstance(rows,list) or not rows or any(not isinstance(r,dict) for r in rows):return None
    if rows[0].get('begin')!=0 or rows[-1].get('end') not in (14,21):return None
    if not any(r.get('device')=='vta' for r in rows):return None
    if any(not all(k in r for k in ('device','begin','end','threads','mask')) for r in rows):return None
    if not all(isinstance(r['mask'],list) for r in rows):return None
    stages=[dict(device=r['device'],begin=r['begin'],end=r['end'],threads=r['threads'],mask=r['mask'],
        worker_core_map=r.get('worker_core_map',r['mask'] if r['device']=='cpu' else [])) for r in rows]
    return digest(dict(stages=stages,host_mask=value.get('host_mask',next(r['mask'] for r in rows if r['device']=='vta')),
        control_mask=value.get('control_mask',[0,1,2,3]),execution_mode=value.get('execution_mode','off'),
        affinity_mode=value.get('affinity_mode','pinned')))


def collect_known(output):
    ids=set();files={};parsed=unparsed=0
    def visit(x):
        nonlocal parsed,unparsed
        if isinstance(x,dict):
            if 'stages' in x:
                f=fingerprint(x)
                if f:ids.add(f);parsed+=1
                else:unparsed+=1
            for v in x.values():visit(v)
        elif isinstance(x,list):
            for v in x:visit(v)
    projects=ROOT.parents[1]
    roots=[projects/'runs',*sorted(ROOT.parent.glob('*/runs'))]
    receipt_count=0;missing_task=0
    for base in roots:
        if not base.exists():continue
        for path in sorted(base.glob('**/accepted/**/*.json')):
            if not path.is_file():continue
            try:meta=metadata_projection(path,{'task'})
            except ValueError:missing_task+=1;continue
            visit(meta['task'].get('configuration',{}));files[str(path)]=sha256(path);receipt_count+=1
    # Excluding prepared-but-unmeasured configs is conservative. These files
    # contain configuration metadata, not performance outcomes.
    for base in [ROOT/'artifacts',*(p for p in ROOT.parent.glob('*/artifacts') if p!=ROOT/'artifacts')]:
        if not base.exists():continue
        for path in sorted(base.glob('**/plans/configurations.json')):
            if OUTPUT in path.parents:continue
            visit(read(path));files[str(path)]=sha256(path)
    evidence=dict(fingerprints=sorted(ids),files=files,accepted_receipts=receipt_count,
        valid_configuration_occurrences=parsed,non_full_or_noncanonical_stage_containers=unparsed,
        receipt_without_top_level_task=missing_task,
        summary_FPS_subtrees_decoded=False,prepared_but_unmeasured_also_excluded=True,
        scope='canonical full configurations in current project accepted receipts and prepared configuration manifests; pre-project scripts not exhaustively inventoried')
    freeze(output/'KNOWN_EXECUTIONS.json',evidence);return ids


def prepare(output=DEST):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'PLAN.json').exists():
        p=read(output/'PLAN.json')
        if p['source_sha256']!=sha256(__file__):raise ValueError('prospective pool source changed')
        return dict(cached=True,counts=p['counts'])
    if not read(OUTPUT/'MIXED_VALIDATION.json')['finite_service_platform_check_passed']:
        raise ValueError('platform model failed')
    known=collect_known(output)
    iso=validate_isolated(read(ROOT/'artifacts/model_semantics_audit_01/ISOLATED_INPUTS.json'))
    intr_path=ROOT/'artifacts/cross_network_model_01/pool_temporal/INTRINSICS.json';intr=read(intr_path)
    cal=read(OUTPUT/'CALIBRATION.json');counts={};pools={};projections={};cost_hashes={}
    with isolated_inputs_only():
        for network,key in [('resnet18','resnet18/static_first'),('yolov3_tiny416','yolov3_tiny416/overlap')]:
            programs,records,ddr,hashes,_,units=inputs(network)
            cost_hashes[network]=hashes
            costs=Costs(records,programs=programs,ddr=ddr,distribution='session_weights')
            base=iso['cases'][key]['configuration'];cpu_indices=[i for i,s in enumerate(base['stages']) if s['device']=='cpu']
            first,second=cpu_indices[:2];available=[]
            for index in (first,second):
                name=base['stages'][index]['program']
                ns={v['identity']['threads'] for v in intr['CPU'].values() if v['identity']['network']==network and v['identity']['program']==name}
                available.append([m for m in domain.nonempty_masks() if len(m) in ns])
            representatives={};eligible=0;excluded=0
            for a,b in product(*available):
                rows=deepcopy(base['stages'])
                for index,mask in [(first,a),(second,b)]:
                    r=rows[index];rows[index]=domain.stage('cpu',r['begin'],r['end'],len(mask),mask)
                cfg=domain.candidate(rows,'shared4','off');domain.validate_candidate(cfg,programs,unit_count=units)
                if fingerprint(cfg) in known:excluded+=1;continue
                eligible+=1
                relation='same' if a==b else 'partial' if set(a)&set(b) else 'disjoint'
                group=(len(a),len(b),relation);value=score(cfg,costs,'fine_host')['score_ns']
                order=(value,domain.canonical_tuple(cfg))
                if group not in representatives or order<representatives[group][0]:representatives[group]=(order,cfg)
            configs=sorted({c['id']:c for _,c in representatives.values()}.values(),key=domain.canonical_tuple)
            if not configs:raise ValueError('no canonical unseen configurations in prospective pool')
            pools[network]=configs;projections[network]={}
            for cfg in configs:
                stages,assumptions=model_stages(network,cfg,programs,costs,intr)
                projections[network][cfg['id']]=dict(stages=stages,configuration=cfg,
                    static_score_ns=score(cfg,costs,'fine_host')['score_ns'],assumptions=assumptions)
            counts[network]=dict(eligible_resources=eligible,known_excluded=excluded,representatives=len(configs))
    freeze(output/'ISOLATED_POOL.json',dict(networks=projections,window=WINDOW,quanta_ns=cal['quanta_ns'],
        contains_pipeline_performance=False,original_full_domain_pruned=False,diagnostic_local_pool=True))
    plan=dict(source_sha256=sha256(__file__),predictor_sha256=sha256(ROOT/'finite_service_replay.py'),
        platform_sources=platform_sources(),calibration_sha256=sha256(OUTPUT/'CALIBRATION.json'),
        isolated_pool_sha256=sha256(output/'ISOLATED_POOL.json'),known_inventory_sha256=sha256(output/'KNOWN_EXECUTIONS.json'),
        intrinsics_sha256=sha256(intr_path),counts=counts,methods=['static_fine_host','fluid_instant','fluid_wake_boundary','finite_wake_boundary'],
        component_cost_hashes=cost_hashes,
        score_source_hashes={str(ROOT/p):sha256(ROOT/p) for p in ('staged_ablation/model.py','resource_envelope/model.py','cpu_policies/open_resources/costs.py','pool_temporal/ranking.py')},
        primary_finite_placement='wake_boundary',no_retain_variant_selection=True,window=WINDOW,
        selection='union of four predicted first choices; if fewer than two distinct choices, add maximum rank disagreement, at most two more',
        max_native_configurations_per_network=4,max_native_sessions=16,max_DNN_inferences=232,
        execution_seed=20261111,
        original_search_domain_unchanged=True,no_runtime_or_executor_change=True,
        local_selection_check_not_complete_DP_Top20=True,
        success='report frozen-model top-1 selection and ordering on the jointly measured pool; phase error alone is insufficient',
        stop='no output/hash/affinity/frame/boot failures accepted; no automatic extra window or repetition')
    freeze(output/'PLAN.json',plan);return dict(counts=counts,known_fingerprints=len(known))


def rank(output=DEST):
    output=Path(output);plan=read(output/'PLAN.json')
    if plan['source_sha256']!=sha256(__file__) or plan['predictor_sha256']!=sha256(ROOT/'finite_service_replay.py') or plan['platform_sources']!=platform_sources():
        raise ValueError('prospective predictor changed')
    for p,h in plan['score_source_hashes'].items():
        if sha256(p)!=h:raise ValueError('prospective score source changed')
    if plan['isolated_pool_sha256']!=sha256(output/'ISOLATED_POOL.json'):raise ValueError('prospective inputs changed')
    doc=read(output/'ISOLATED_POOL.json');result={};selected={};started=time.monotonic()
    for network,configs in doc['networks'].items():
        scores={};files={}
        with isolated_inputs_only():
            for count,(cid,x) in enumerate(configs.items(),1):
                path=output/'predictions'/network/(cid+'.json')
                if path.exists():
                    r=read(path)
                    if r['input_identity']!=digest(x):raise ValueError('prospective checkpoint changed')
                else:
                    common=dict(runnable_completion_wait=True,controls=None,gate='off',host_order='before',**WINDOW)
                    r=dict(input_identity=digest(x),configuration=x['configuration'],static_fine_host=x['static_score_ns'])
                    for placement in ('instant','wake_boundary'):
                        v=HistoryReplay(x['stages'],placement=placement,**common).run()
                        r['fluid_'+placement]=v['replay']['interval_ns']
                    v=FiniteReplay(x['stages'],quanta=doc['quanta_ns'],placement='wake_boundary',**common).run()
                    r['finite_wake_boundary']=v['replay']['interval_ns']
                    r['finite_stage_ns']=v['replay']['stage_service_ns'];r['native_FPS_read']=False
                    freeze(path,r)
                scores[cid]=r;files[str(path.relative_to(output))]=sha256(path)
                print('未见配置离线排名',network,count,'/',len(configs),flush=True)
        rankings={m:sorted(scores,key=lambda cid:(scores[cid][m],domain.canonical_tuple(configs[cid]['configuration']))) for m in plan['methods']}
        chosen=list(dict.fromkeys(rankings[m][0] for m in plan['methods']))
        if len(chosen)<2:
            candidates=sorted(scores,key=lambda cid:(-max(abs(rankings[a].index(cid)-rankings[b].index(cid)) for a,b in product(rankings,repeat=2)),domain.canonical_tuple(configs[cid]['configuration'])))
            for cid in candidates:
                if cid not in chosen:chosen.append(cid)
                if len(chosen)==min(3,len(configs)):break
        if len(chosen)>4:raise ValueError('prospective native budget exceeded')
        selected[network]=[configs[cid]['configuration'] for cid in chosen]
        result[network]=dict(rankings=rankings,scores=scores,selected_ids=chosen,prediction_files=files)
    freeze(output/'SELECTION_FREEZE.json',dict(networks=result,selected_configurations=selected,
        plan_sha256=sha256(output/'PLAN.json'),selection_before_native_measurement=True,
        performance_files_read=False,DNN_budget=sum(len(v) for v in selected.values())*29))
    write(output/'RANK_COST.json',dict(seconds=time.monotonic()-started),replace=True)
    return dict(selected_counts={k:len(v) for k,v in selected.items()},DNN_budget=sum(len(v) for v in selected.values())*29)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','rank'))
    p.add_argument('--output',type=Path,default=DEST);a=p.parse_args()
    print(json.dumps(globals()[a.command](a.output),ensure_ascii=False))
