"""Freeze one node46 TPS representative before compilation or performance.

Same analytical score/tie rule and oc2 support restriction as node34.
The broader thread-domain minimum is retained explicitly, not hidden.
No timing, AutoTVM search, candidate transform or board access.
"""
import argparse
from pathlib import Path
from c3_node34_tps_applicability import enumerate_shape
from c3_node46_tps_local import HERE,REPO,HOLD,GRAPH,AUTHOR,SHAPES,read,write,sha,verify,require


def main(root):
    root.mkdir(exist_ok=False)
    review=HOLD/'20261017_c3_uop_identity_applicability_readonly03'
    verify({str(review/p):v for p,v in read(review/'completion_manifest.json')['artifact_sha256'].items()})
    cfg=read(REPO/'3rdparty/vta-hw/config/vta_config.json')
    require(all(cfg[k]==v for k,v in {'LOG_BATCH':0,'LOG_BLOCK':4,'LOG_INP_WIDTH':3,
                                    'LOG_WGT_WIDTH':3,'LOG_ACC_WIDTH':5}.items()),'HARDWARE_UNIT_CHANGED')
    capacity={k:1<<cfg['LOG_'+k+'_BUFF_SIZE'] for k in ('INP','WGT','ACC')}
    rows=enumerate_shape((1,14,14,24,16),capacity)
    ranked=sorted((r for r in rows if r['necessary_conditions_passed']),
                  key=lambda r:(r['paper_score_bytes'],r['id']))
    oc=[r for r in ranked if r['tile']['oc_nthread']==2]
    tied=[r for r in ranked if r['paper_score_bytes']==ranked[0]['paper_score_bytes']]
    tied_oc=[r for r in oc if r['paper_score_bytes']==oc[0]['paper_score_bytes']]
    chosen=tied_oc[0]
    require(chosen['id']=='b1h14w14ci1co4oc2hth1','INHERITED_RULE_SELECTION_CHANGED')
    prospective={'origin':'paper Appendix A reconstruction, not author search executable',
        'not_a_compiled_ConfigEntity':True,'paper_minimum_all_supported_threads':tied,
        'paper_minimum_oc2_family':tied_oc,'prospective_oc2_representative':chosen,
        'tie_rule':'paper score then existing Tile.identity lexicographic order; no timing or redundancy used',
        'oc2_family_is_local_support_choice_not_paper_global_optimum':True,
        'broader_thread_domain_has_lower_model_score':ranked[0]['paper_score_bytes']<chosen['paper_score_bytes'],
        'predicted_actual_payload_if_compilation_preserves_model':{
            'WGT':884736,'WGT_address_multiplicity':1,'INP':150528,'ADD':1024,'OUT':50176},
        'prediction_is_not_final_address_or_performance_certification':True}
    from c3_node46_tps_local import extract_relay
    relay=extract_relay((GRAPH/'relay_000.txt').read_text())
    require('right_shift(%24, 8' in relay,'SHIFT_EIGHT_REQUIRED')
    paths=[Path(__file__),HERE/'c3_node46_tps_local.py',HERE/'c3_node34_tps_applicability.py',
           HERE/'c3_banerjee_baseline.py',HERE/'c3_banerjee_local.py',HERE/'c3_banerjee_completion.py',
           REPO/'3rdparty/vta-hw/config/vta_config.json',GRAPH/'relay_000.txt',GRAPH/'graph.json',
           AUTHOR/'vta_conv2d.py',review/'completion_manifest.json']
    result={'status':'ANALYTICAL_APPLICABILITY_ONLY_NOT_COMPILED','node':46,'dimensions':[1,14,14,24,16],
            'abi_shapes':SHAPES,'arithmetic_shift':8,'capacity_bytes':capacity,
            'raw_configurations':len(rows),'necessary_domain':len(ranked),'ranked_necessary_domain':ranked,
            'prospective':prospective,'new_target_calls':0,'new_compilations':0,'board_contacted':False,
            'source_sha256':{str(p.resolve()):sha(p) for p in paths}}
    write(root/'audit.json',result)
    write(root/'completion_manifest.json',{'status':result['status'],'target_calls':0,
          'artifact_sha256':{'audit.json':sha(root/'audit.json')}})
    print(chosen['id'], 'score',chosen['paper_score_bytes'], 'broader_minimum',ranked[0]['paper_score_bytes'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    main(p.parse_args().output.resolve())
