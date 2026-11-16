"""Explain the existing per-subgraph domain, without pruning or FPS inputs."""
import csv
import json
from pathlib import Path
from collections import Counter, defaultdict
from cpu_vta_partitioning.common import read, sha256, write
from policy_branch_eval.ranking import inputs
from policy_branch_eval.common import isolated_inputs_only
from .common import PARENT, freeze


def summarize_group(rows):
    rows=sorted(rows,key=lambda r:r['threads'])
    if [r['threads'] for r in rows]!=[1,2,3,4]:raise ValueError('thread curve incomplete or duplicated')
    first=rows[0];result=[]
    for i,r in enumerate(rows):
        if r['service_ms']<=0 or r['run_core_ms']<0:raise ValueError('invalid isolated curve')
        result.append({**r,'speedup_vs_one':first['service_ms']/r['service_ms'],
                       'core_growth_vs_one':r['run_core_ms']/first['run_core_ms'] if first['run_core_ms'] else None,
                       'marginal_service_reduction':None if not i else 1-r['service_ms']/rows[i-1]['service_ms'],
                       'nonmonotonic_vs_previous':i>0 and r['service_ms']>rows[i-1]['service_ms'],
                       'retained_in_local_table':True})
    fastest=min(rows,key=lambda r:(r['service_ms'],r['threads']))
    gain=first['service_ms']/fastest['service_ms']
    # Labels are descriptive bins for this single historical measurement,
    # never significance/equivalence claims or a pruning rule.
    label='时间变化不超过10%的观察' if gain<=1.1 else '超过1.5倍加速的观察' if gain>=1.5 else '中等加速的观察'
    near=[r['threads'] for r in rows if r['service_ms']<=fastest['service_ms']*1.1]
    return result,dict(program=first['program'],best_observed_threads=fastest['threads'],
        best_speedup=gain,first_threads_within_10pct_of_observed_best=min(near),
        descriptive_label=label,label_thresholds_are_not_pruning=True,
        nonmonotonic_pairs=sum(r['nonmonotonic_vs_previous'] for r in result))


def csv_write(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:raise ValueError('cannot silently write empty evidence table')
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else v for k,v in r.items()})


def curves(output):
    output=Path(output);seal=read(PARENT/'FEATURE_FREEZE.json');intrinsics=read(PARENT/'pool_temporal/INTRINSICS.json')
    trace_index={}
    for value in intrinsics['CPU'].values():
        identity=value['identity'];a=value['average'];counts=sorted(set(k['tasks'] for k in a['kernels']))
        trace_index[(identity['network'],identity['program'],identity['threads'])]=dict(
            actual_participant_counts=counts,scope='isolated per-kernel callbacks, not session TID count',
            provenance=value['provenance'])
    all_rows=[];groups=[];sources={};counts={}
    with isolated_inputs_only():
        for network in ('resnet18','yolov3_tiny416'):
            f=PARENT/network/'isolated_features.json'
            if sha256(f)!=seal['inputs'][network]:raise ValueError('isolated feature freeze changed')
            feature=read(f);programs,records,ddr,hashes,_,_=inputs(network)
            if hashes!=feature['component_hashes']:raise ValueError('macro costs no longer match feature sources')
            sources[str(f)]=sha256(f);grouped=defaultdict(list)
            for raw in feature['features']:
                r={**raw};record=records[r['source_key']];cfg=record['configuration']
                if any(abs(r[k]-record['cost'][k])>1e-9 for k in ('service_ms','run_ms','run_core_ms','boundary_core_ms')):
                    raise ValueError('derived curve differs from unchanged source cost')
                evidence=trace_index.get((network,r['program'],r['threads']))
                r.update(mask=cfg['stages'][0]['mask'],worker_core_map=cfg['stages'][0]['worker_core_map'],
                    input_indices=[0],source_record=record,
                    actual_participation=evidence if evidence else dict(status='UNKNOWN_NO_ISOLATED_CALLBACK_TRACE'),
                    eligible_for_current_search=r['program'] in programs,
                    core_cost_scope='measured prefix mask; other same-size masks use declared homogeneous-core approximation',
                    run_core_scope='isolated process CPU during run; includes runtime work, not pure arithmetic')
                grouped[r['program']].append(r)
            counts[network]=len(grouped)
            for rows in grouped.values():
                expanded,summary=summarize_group(rows);all_rows+=expanded;groups.append(dict(network=network,**summary))
    if len(all_rows)!=664 or counts!={'resnet18':85,'yolov3_tiny416':81}:raise ValueError('expected full historical thread coverage')
    result=dict(schema='subgraph-thread-curves-v1',rows=all_rows,summaries=groups,sources=sources,
        intrinsic_source_sha256=sha256(PARENT/'pool_temporal/INTRINSICS.json'),
        no_pipeline_FPS_read=True,no_candidates_pruned=True,new_native_sessions=0,
        threads_equal_core_set_size_in_current_domain=True,per_operator_thread_assignment=False)
    freeze(output/'curves/THREAD_CURVES.json',result)
    slim=[{k:v for k,v in r.items() if k not in ('source_record','recognized_dense_convolutions','graph_functions')} for r in all_rows]
    csv_write(output/'curves/thread_options.csv',slim);csv_write(output/'curves/subgraph_summary.csv',groups)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for network in counts:
        rows=[r for r in all_rows if r['network']==network];names=sorted({r['program'] for r in rows})
        matrix=[[next(r['speedup_vs_one'] for r in rows if r['program']==p and r['threads']==n) for n in range(1,5)] for p in names]
        fig,ax=plt.subplots(figsize=(5,max(8,len(names)*0.17)))
        im=ax.imshow(matrix,aspect='auto',vmin=1,vmax=4,cmap='viridis')
        ax.set(xticks=range(4),xticklabels=[1,2,3,4],yticks=range(len(names)),yticklabels=names,
               xlabel='Configured threads',title=network+' isolated service speedup')
        ax.tick_params(axis='y',labelsize=6);fig.colorbar(im,ax=ax,label='T(1)/T(n)');fig.tight_layout()
        fig.savefig(output/'curves'/(network+'_speedup.png'),dpi=160);plt.close(fig)
    write(output/'curves/SUMMARY.json',dict(subgraphs=counts,signatures=len(all_rows),
        actual_trace_signatures=sum(r['actual_participation'].get('actual_participant_counts') is not None for r in all_rows),
        labels=dict(Counter(g['descriptive_label'] for g in groups)),pruned=0),replace=True)
    return read(output/'curves/SUMMARY.json')
