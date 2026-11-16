"""Read-only preparation of the requested TopHub/TPS full-network comparison.

Uses sealed graph/config identities, never history latency as current timing.
No build, target invocation, board contact, parameter change or schedule hook.
The existing analytical 3x3 model is screened over its full thread domain;
1x1 is explicitly a pending model/qualification task, not a silent fallback.
"""
import argparse
from collections import Counter
from pathlib import Path
from c3_tps_aggregation_prepare import HERE,REPO,HOLD,read,write,sha,require,verify_bindings
from c3_node34_tps_applicability import enumerate_shape

OFFICIAL=HOLD/'20261014_c3_official_valid_window_run01'
BASE=OFFICIAL/'baseline/yolov3_tiny_416_final01'
LOG=Path('/home/orange/.tvm/tophub/vta_v0.10.log')


def group_calls(calls,graph):
    require(len(calls)==12,'EXPECTED_12_VTA_CALLS')
    grouped={}
    for row in calls:
        require(row['network']=='yolov3_tiny_416' and row['official_baseline_bound'] and row['roundtrip_passed'],
                'OFFICIAL_IDENTITY_NOT_ESTABLISHED')
        record=row['source_record']
        require(row['workload']==record['workload'],'EXACT_WORKLOAD_CHANGED')
        require(row['effective_config_entity']==record['complete_config_entity'],'EFFECTIVE_CONFIG_CHANGED')
        require(record['raw_record']['result'][1]==0 and record['source_model'].startswith('pynq_'),
                'UNSUCCESSFUL_OR_DIFFERENT_SOURCE_RECORD')
        node=row['earliest_graph_node']
        require(graph['nodes'][node].get('attrs',{}).get('func_name')==row['global_symbol'],'GRAPH_FUNCTION_CHANGED')
        entries=grouped.setdefault(row['global_symbol'],[])
        if entries:
            require(all(entries[0][k]==row[k] for k in ('abi','workload','semantic_hash','effective_config_entity')),
                    'SHARED_SYMBOL_HAS_DIFFERENT_SEMANTICS')
        entries.append(row)
    require(len(grouped)==11 and sum(len(v) for v in grouped.values())==12,'UNIQUE_FUNCTION_COVERAGE_CHANGED')
    return sorted(grouped.values(),key=lambda rows:rows[0]['earliest_graph_node'])


def screen(row,capacity):
    workload=row['workload'];x,w=workload[1][1],workload[2][1]
    require(workload[0]=='conv2d_packed.vta' and workload[3]==[1,1] and workload[5]==[1,1],
            'UNEXPECTED_CONVOLUTION_FAMILY')
    require(len(x)==len(w)==6 and x[0]==x[4]==1 and x[5]==w[4]==w[5]==16,'PACKED_UNIT_CHANGED')
    kernel=w[2:4]
    if kernel==[1,1]:
        require(workload[4]==[0,0,0,0],'UNEXPECTED_POINTWISE_PADDING')
        return {'status':'POINTWISE_MODEL_AND_COMPILER_QUALIFICATION_PENDING',
                'existing_3x3_cost_applied':False,'compiled_TPS':False}
    require(kernel==[3,3] and workload[4]==[1,1,1,1],'UNEXPECTED_KERNEL_OR_PADDING')
    dims=(1,x[2],x[3],x[1],w[0])
    candidates=enumerate_shape(dims,capacity)
    ranked=sorted((r for r in candidates if r['necessary_conditions_passed']),
                  key=lambda r:(r['paper_score_bytes'],r['id']))
    require(ranked,'NO_ANALYTICALLY_LEGAL_3X3_TILE')
    oc2=[r for r in ranked if r['tile']['oc_nthread']==2]
    return {'status':'EXISTING_3X3_ANALYTICAL_RULE_SCREENED_NOT_COMPILED','dimensions':list(dims),
            'raw_candidates':len(candidates),'necessary_domain':len(ranked),
            'all_supported_threads_first':ranked[0], 'oc2_family_first':oc2[0] if oc2 else None,
            'forced_oc2_has_higher_score':bool(oc2 and oc2[0]['paper_score_bytes']>ranked[0]['paper_score_bytes']),
            'selection_read_performance':False,'compiled_TPS':False}


def run(out):
    out.mkdir(exist_ok=False)
    seal=read(OFFICIAL/'completion_manifest.json')
    bindings={str(OFFICIAL/p):v for p,v in seal['artifact_sha256'].items()}
    bindings[str(OFFICIAL/'completion_manifest.json')]=sha(OFFICIAL/'completion_manifest.json')
    for p in (Path(__file__).resolve(),HERE/'c3_node34_tps_applicability.py',HERE/'c3_banerjee_baseline.py',
              REPO/'3rdparty/vta-hw/config/vta_config.json',REPO/'build/libtvm.so',LOG):
        bindings[str(p)]=sha(p)
    require(sha(LOG)==read(BASE/'compiler_identity.json')['official_log_sha256'],'TOPHUB_LOG_CHANGED')
    verify_bindings(bindings)
    identity=read(BASE/'baseline_identity.json');graph=read(BASE/'B_raw/graph.json')
    require(identity['status']=='OFFICIAL_QUERY_AND_ROUNDTRIP_PASSED','BASELINE_QUALIFICATION_MISSING')
    groups=group_calls(identity['callsites'],graph)
    cfg=read(REPO/'3rdparty/vta-hw/config/vta_config.json')
    capacity={k:1<<cfg['LOG_'+k+'_BUFF_SIZE'] for k in ('INP','WGT','ACC')}
    rows=[]
    for calls in groups:
        row=calls[0]
        for key in ('parent_module','final_module'):
            require(sha(row[key])==row[key+'_sha256'],'ARCHIVED_IR_CHANGED')
        rows.append({'object_id':row['object_id'],'nodes':[r['earliest_graph_node'] for r in calls],
                     'symbol':row['global_symbol'],'abi':row['abi'],'workload':row['workload'],
                     'TopHub_source':row['source_record'],'effective_config_entity':row['effective_config_entity'],
                     'TPS_preparation':screen(row,capacity)})
    kinds=Counter(tuple(r['workload'][2][1][2:4]) for r in rows)
    require(kinds=={(3,3):7,(1,1):4},'EXPECTED_KERNEL_COVERAGE_CHANGED')
    write(out/'freeze.json',{'bindings':bindings,'network':'yolov3_tiny_416','new_target_calls':0})
    write(out/'inventory.json',rows)
    summary={'status':'YOLO416_COMPARISON_SCOPE_AUDITED_NOT_BUILT','network':'yolov3_tiny_416',
             'graph_VTA_calls':12,'unique_complete_VTA_functions':11,'kernel_3x3_functions':7,'kernel_1x1_functions':4,
             'raw_graph_outputs':len(graph['heads']),'official_exact_workload_coverage':'12/12 calls,11/11 workloads',
             'source_platform':'PYNQ','source_search_budget':'unknown','TopHub_record_cost_used_as_current_timing':False,
             'TPS_graph_available':False,'TPS_graph_qualified':False,
             '3x3_forced_oc2_higher_analytical_score_objects':[r['object_id'] for r in rows
                   if r['TPS_preparation'].get('forced_oc2_has_higher_score')],
             'new_compilations':0,'new_FSim_calls':0,'FPGA_calls':0,'board_contacted':False,
             'next_required':['general TPS model for 1x1','complete 11-function schedule adaptation and certificates',
                              'same-graph same-parameter roundtrip and memory/call contracts','local qualification before FPGA']}
    write(out/'summary.json',summary)
    verify_bindings(bindings)
    print(__import__('json').dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.output.resolve())
