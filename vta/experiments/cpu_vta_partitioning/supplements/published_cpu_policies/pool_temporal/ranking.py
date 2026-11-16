"""Strict pool-local Top20; preselection/domain limits remain explicit."""
from copy import deepcopy
from pathlib import Path
import time

from cpu_vta_partitioning.common import read,sha256,write
from cpu_policies.open_resources.costs import Costs,quantize_ns
from cpu_policies.open_resources.domain import canonical_tuple
from policy_branch_eval.common import isolated_inputs_only
from policy_branch_eval.ranking import inputs
from staged_ablation.model import score
from workload_model.common import OUTPUT,ROOT,freeze
from .simulator import PoolReplay


def model_stages(network,cfg,programs,costs,intrinsics):
    result=[];assumptions=[]
    for row in cfg['stages']:
        p=programs[row['program']];cost=costs.get(row,cfg)
        value=dict(device=row['device'],mask=row['mask'],worker_core_map=row['worker_core_map'])
        if row['device']=='cpu':
            found=[v for v in intrinsics['CPU'].values() if v['identity']['network']==network
                   and v['identity']['program']==row['program'] and v['identity']['threads']==row['threads']]
            if len(found)!=1:raise ValueError('missing or duplicate real CPU work projection')
            profile=found[0]
            if profile['identity']['module_sha256']!=p['files']['module.so']['sha256'] or not profile['no_pipeline_performance']:
                raise ValueError('CPU projection belongs to a different module or scope')
            value['profile']=profile['average']
            if profile['identity']['mask']!=row['mask']:
                assumptions.append(dict(program=row['program'],assumption='homogeneous_core_permutation',
                    source=profile['identity']['mask'],requested=row['mask']))
        else:
            key=network+'/'+row['program']+'/a'+'-'.join(map(str,row['mask']))
            if key not in intrinsics['VTA']:raise ValueError('no exact VTA host-mask phase projection')
            profile=intrinsics['VTA'][key]
            if profile['module_sha256']!=p['files']['module.so']['sha256'] or not profile['no_pipeline_performance']:
                raise ValueError('VTA projection belongs to a different module or scope')
            value['device_profile']=profile['stages']
        for op in ('set','get'):
            cpu=quantize_ns(cost[op+'_core_ms']);wall=quantize_ns(cost[op+'_ms'])
            value[op+'_cpu_ns']=cpu;value[op+'_nonrunning_ns']=max(0,wall-cpu)
        result.append(value)
    return result,assumptions


def validate_rank(output):
    output=Path(output);seal=read(output/'RANK_FREEZE.json')
    for path,expected in seal['files'].items():
        if sha256(output/path)!=expected:raise ValueError('candidate temporal ranking artifact changed')
    for path,expected in seal['sources'].items():
        if sha256(path)!=expected:raise ValueError('candidate temporal predictor changed')
    if seal['common_rank_freeze_sha256']!=sha256(OUTPUT/'common_top20/RANK_FREEZE.json'):
        raise ValueError('candidate temporal pool differs from formal common pool')
    return seal


def rank(output):
    output=Path(output).resolve()
    if (output/'RANK_FREEZE.json').exists():return dict(resumed_verified=True,**validate_rank(output)['counts'])
    accepted=ROOT/'runs/cross_network_common_top20_01/accepted'
    if accepted.exists() and any(accepted.rglob('*.json')):raise ValueError('new temporal selection cannot occur after formal native timing')
    started=time.monotonic();result={};counts={}
    with isolated_inputs_only():
        plan=read(output/'PLAN.json');intrinsics=read(output/'INTRINSICS.json')
        if intrinsics['pipeline_FPS_read'] or intrinsics['macro_costs_modified']:raise ValueError('inadmissible intrinsic projection')
        for network in ('resnet18','yolov3_tiny416'):
            parent=OUTPUT/'common_top20'/network/'RANKINGS.json'
            if sha256(parent)!=plan['frozen_pool_hashes'][network]:raise ValueError('frozen pool changed')
            pool=read(parent)['pool'];programs,records,ddr,hashes,_,_=inputs(network)
            costs=Costs(records,programs=programs,ddr=ddr,distribution='session_weights');rows=[]
            for number,(identity,cfg) in enumerate(sorted(pool.items()),1):
                stages,assumptions=model_stages(network,cfg,programs,costs,intrinsics)
                current=score(cfg,costs,'fine_host');capacity=score(cfg,costs,'weights_only')
                models={}
                for order in ('before','after'):
                    run=PoolReplay(stages,gate=cfg['execution_mode'],host_order=order).run()
                    models[order]=dict(replay=run,replace_score_ns=max(capacity['score_ns'],run['interval_ns']),
                                      append_score_ns=max(current['score_ns'],run['interval_ns']))
                rows.append(dict(configuration=cfg,models=models,current_score_ns=current['score_ns'],
                    capacity_score_ns=capacity['score_ns'],assumptions=assumptions))
                if number%10==0:print('候选时序评分',network,number,'/',len(pool),flush=True)
            lists={}
            for label in ('replace','append'):
                ordered=sorted(rows,key=lambda r:(r['models']['before'][label+'_score_ns'],canonical_tuple(r['configuration'])))
                lists[label]=[dict(**r['configuration'],score_ns=r['models']['before'][label+'_score_ns'],rank=i)
                              for i,r in enumerate(ordered[:20],1)]
            result[network]=dict(lists=lists,all_scores=rows,component_hashes=hashes,
                pool_preselection_explicit=True,global_temporal_Top20_claimed=False,pipeline_FPS_read=False)
            freeze(output/network/'RANKINGS.json',result[network]);counts[network]=len(rows)
        sources={str(p):sha256(p) for p in [Path(__file__),Path(__file__).with_name('simulator.py'),
            ROOT/'phase_model/simulator.py',ROOT/'staged_ablation/model.py',ROOT/'run_common_top20_guarded.py']}
        freeze(output/'RANK_FREEZE.json',dict(files={str(p.relative_to(output)):sha256(p) for p in
            [output/'PLAN.json',output/'INTRINSICS.json',*[output/n/'RANKINGS.json' for n in result]]},
            sources=sources,counts=counts,common_rank_freeze_sha256=sha256(OUTPUT/'common_top20/RANK_FREEZE.json'),
            pipeline_FPS_read=False,formal_native_tasks_before_freeze=0,
            objective='max(physical capacity proxy, isolated phase replay); append guard is an ablation',
            scope='Top20 within frozen full-DP candidate union, no claim of full-domain temporal optimum'))
    write(output/'RANK_COST.json',dict(seconds=time.monotonic()-started),replace=True)
    return {n:{k:v[0]['id'] for k,v in data['lists'].items()} for n,data in result.items()}
