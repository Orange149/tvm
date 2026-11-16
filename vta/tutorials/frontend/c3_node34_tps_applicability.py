"""Paper-formula applicability check, not a compiler or performance search.

Restricted to batch 1, packed 16x16, 3x3 stride-1 same-padding convolution.
Historical node24 code remains unchanged. No TVM import, target or board call.
"""
from dataclasses import asdict
from itertools import product
from pathlib import Path
import argparse
import hashlib
import json

from c3_banerjee_baseline import Tile, divisors, paper_cost

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HOLD = HERE/'report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout'


def cost(tile, dimensions, capacity):
    inner = (tile.b, tile.h, tile.w, tile.ci, tile.co)
    if dimensions[0] != 1 or any(type(v) is not int or v <= 0 or n % v for n, v in zip(dimensions, inner)):
        raise ValueError('unsupported shape or non-integral tile')
    if tile.oc_nthread not in (1,2) or tile.h_nthread not in (1,2):
        raise ValueError('thread count')
    bo, ho, wo, cio, coo = (n//v for n,v in zip(dimensions,inner))
    threads = tile.oc_nthread*tile.h_nthread
    usage = {'INP': tile.b*tile.ci*16*(tile.h+2)*(tile.w+2)*threads,
             'WGT': tile.co*tile.ci*256*9*threads,
             'ACC': (tile.b*tile.co*16*tile.h*tile.w+tile.co*16)*4*threads}
    reasons = ['paper_both_threads_two'] if threads == 4 else []
    exact = coo % tile.oc_nthread == 0 and ho % tile.h_nthread == 0
    if not exact: reasons.append('local_exact_outer_thread_divisibility')
    reasons += [k+'_paper_capacity' for k,v in usage.items() if v>capacity[k]]
    mul = bo*(ho//tile.h_nthread)*(coo//tile.oc_nthread)*wo*cio if exact else None
    loads = None if mul is None else {'INP':mul*usage['INP'], 'WGT':mul*usage['WGT'],
                                     'ACC_bias_proxy':bo*ho*wo*(dimensions[4]*16)*4}
    return {'id':tile.identity,'tile':asdict(tile),'paper_allocation_bytes':usage,
            'paper_payload_bytes':loads,'paper_score_bytes':None if loads is None else sum(loads.values()),
            'necessary_conditions_passed':not reasons,'rejections':reasons,'full_fusion_qualified':False}


def enumerate_shape(dimensions, capacity):
    return [cost(Tile(*values),dimensions,capacity) for values in product(
        *(divisors(n) for n in dimensions),(1,2),(1,2))]


def main(output):
    output.mkdir(exist_ok=False)
    bindings = {}
    def bind(p):
        bindings[str(p.resolve())] = hashlib.sha256(p.read_bytes()).hexdigest()
        return p
    cfg=json.loads(bind(ROOT/'3rdparty/vta-hw/config/vta_config.json').read_text())
    for k,v in {'LOG_BATCH':0,'LOG_BLOCK':4,'LOG_INP_WIDTH':3,'LOG_WGT_WIDTH':3,'LOG_ACC_WIDTH':5}.items():
        if cfg[k]!=v:raise ValueError('hardware unit contract')
    capacity={k:1<<cfg['LOG_'+k+'_BUFF_SIZE'] for k in ('INP','WGT','ACC')}
    reference=enumerate_shape((1,14,14,8,16),capacity)
    for row in reference:
        if row!=paper_cost(Tile(**row['tile']),capacity):raise ValueError('node24 regression mismatch')
    rows=enumerate_shape((1,7,7,32,64),capacity)
    ranked=sorted((r for r in rows if r['necessary_conditions_passed']),key=lambda r:(r['paper_score_bytes'],r['id']))
    oc=[r for r in ranked if r['tile']['oc_nthread']==2]
    best=ranked[0]['paper_score_bytes'];best_oc=oc[0]['paper_score_bytes']
    tied=[r for r in ranked if r['paper_score_bytes']==best]
    tied_oc=[r for r in oc if r['paper_score_bytes']==best_oc]
    a=tied_oc[0]
    if (a['tile']['h'],a['tile']['w'])!=(7,7):raise ValueError('unexpected spatial conclusion')
    if a['paper_payload_bytes']['WGT']!=4718592:raise ValueError('weight-once model')
    prospective={'not_a_compiled_ConfigEntity':True,'origin':'Appendix A reconstruction, not author TPS executable',
        'paper_minimum_all_supported_threads':tied,'paper_minimum_oc2_family':tied_oc,
        'prospective_oc2_representative':a,'tie_rule':'paper score then existing Tile.identity lexicographic order; no timing',
        'oc2_family_is_local_support_choice_not_paper_global_optimum':True,
        'predicted_actual_tensor_payload_if_lowering_preserves_model':{
            'WGT_bytes':4718592,'WGT_address_multiplicity':1,
            'INP_before_shared_bank_reuse_bytes':25088*(64//a['tile']['co']),
            'INP_after_shared_bank_reuse_bytes':25088*(64//a['tile']['co']//2),
            'ADD_bytes':4096,'OUT_bytes':50176},
        'prediction_limits':['logical allocation not final effective SRAM addresses','paper input estimate includes padding',
            'actual loads and reduction order require compiled full-fusion audit','no latency prediction']}
    author=HOLD/'20261012_c3_banerjee_double_buffer_local_run01/author_sources/vta_conv2d.py'
    if hashlib.sha256(bind(author).read_bytes()).hexdigest()!='c8428db2c98a82e55cf38e737b0d5ce1c29265e0a18991d7b51ba182e56361eb':raise ValueError('author source')
    decision=HOLD/'20261017_c3_candidate_decision_readonly01'
    bind(decision/'completion_manifest.json');bind(decision/'node_34_descriptors.json')
    graph=HOLD/'20261008_c3_yolo224_incremental_prepare_run03/G2'
    symbol='tvmgen_default_fused_nn_conv2d_add_right_shift_clip_cast_5'
    tir=bind(graph/(symbol+'.tir')).read_text()
    if '0, 3, 1, 6)' not in tir or '0, 0, 1, 127)' not in tir or '0, 1, 1, -127)' not in tir:raise ValueError('fusion constants')
    for path in [HERE/'c3_banerjee_baseline.py',HERE/'c3_banerjee_local.py',HERE/'c3_banerjee_completion.py',
                 HERE/'c3_tps_sync_release_prepare.py',Path(__file__).resolve(),graph/'graph.json',graph/'relay_000.txt']:
        bind(path)
    result={'status':'ANALYTICAL_APPLICABILITY_ONLY_NOT_COMPILED','node':34,'dimensions':[1,7,7,32,64],
        'capacity_bytes':capacity,'raw_configurations':len(rows),'necessary_domain':len(ranked),
        'node24_exact_regression_rows':len(reference),'ranked_necessary_domain':ranked,
        'rejected':{reason:sum(reason in r['rejections'] for r in rows) for reason in sorted({x for r in rows for x in r['rejections']})},
        'prospective':prospective,'new_target_calls':0,'new_compilations':0,'board_contacted':False,'source_sha256':bindings}
    (output/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','raw_configurations','necessary_domain','node24_exact_regression_rows']},indent=2))
    print(json.dumps(prospective,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);main(p.parse_args().output)
