"""Read-only residual-cost audit after the single node34 TPS is qualified.

No compilation, target execution, board access or executable variant output.
An optional cache identity quotient is ONLY replayed through the frozen queue
model, with exact value comparison for every reference; it is not a candidate
qualification or a latency estimate.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path

from c3_node34_tps_local import HERE,REPO,HOLD,AUTHOR,read,write,sha,verify,require


def classify_kernels(events):
    """Exact complete initializer equivalence within each static MAC handle."""
    by_scope=defaultdict(list)
    for e in events:
        if e['kind']=='kernel' and e['mode_reset']==[0,0]:by_scope[e['scope']].append(e)
    rows=[];aliases={}
    for scope,sequence in sorted(by_scope.items()):
        signatures={};classes=[]
        for e in sequence:
            sig=tuple(e['signature'])
            if sig in signatures:
                require(signatures[sig]['uops']==e['uops'],'SIGNATURE_CONTENT_MUTATION')
            signatures[sig]=e
            index=next((i for i,cls in enumerate(classes) if cls['uops']==e['uops']),None)
            if index is None:
                index=len(classes);classes.append({'uops':e['uops'],'signatures':set()})
            classes[index]['signatures'].add(sig)
            aliases[(scope,sig)]=f'readonly-equivalence-scope{scope}-class{index}'
        mapping=[{'signatures':[list(v) for v in sorted(c['signatures'])],
                  'base_uops':len(c['uops'])} for c in classes]
        parity=all(len(sig)==1 and e['uops']==signatures[(sig[0]%2,)]['uops']
                   for sig,e in signatures.items()) if (0,) in signatures and (1,) in signatures else False
        rows.append({'scope':scope,'dynamic_calls':len(sequence),'signature_count':len(signatures),
                     'complete_initializer_classes':len(classes),'parity_equals_full_content_on_observed_domain':parity,
                     'classes':mapping})
    return rows,aliases


def model_summary(instructions,packed,wraps,guarded):
    loads=[r for r in instructions if r['kind']=='load']
    return {'serialized_uop_bytes':len(packed)*4,'UOP_LOADs':len(loads),
            'UOP_LOAD_bytes':sum(r['size']*4 for r in loads),'wraps':wraps,
            'guarded_greedy_conflicts':guarded,
            'compute_references':sum(r['kind']=='compute' for r in instructions)}


def audit(baseline,output):
    from c3_valid_window_uop_values import _normalize
    from c3_tps_window_native import _simulate
    from c3_five_tradeoff_commands import decode_submissions
    from c3_tps_device_cost_readonly import decode
    output.mkdir(exist_ok=False)
    require(read(baseline/'status.json')['status']=='NODE34_LOCAL_QUALIFIED','BASELINE_NOT_QUALIFIED')
    verify({str(baseline/p):v for p,v in read(baseline/'completion_manifest.json')['artifact_sha256'].items()})
    verify(read(baseline/'freeze.json')['bindings'])
    parent=baseline/'prepare/R'
    events=read(parent/'events.json');classes,aliases=classify_kernels(events)
    ks=[_normalize(e) for e in events if e['kind']=='kernel']
    defs={k['identity']:k['values'] for k in ks}
    first=read(baseline/'R_detailed/native_certificate_01.json')
    barriers={int(k):v for k,v in first['submissions'][0]['compute_nop_greedy_barriers'].items()}
    simulated,packed,wraps,guarded=_simulate(ks,defs,barriers)
    before=model_summary(simulated,packed,wraps,guarded)
    require(before['serialized_uop_bytes']==first['uop_values']['serialized_uop_bytes'] and
            before['wraps']==first['uop_values']['uop_sram_wraps'] and
            before['UOP_LOADs']==first['uop_values']['uop_loads'],'MODEL_NOT_MATCHING_CERTIFIED_BASELINE')

    quotient=[]
    for event,k in zip((e for e in events if e['kind']=='kernel'),ks):
        new=deepcopy(k)
        if event['mode_reset']==[0,0]:new['identity']=aliases[event['scope'],tuple(event['signature'])]
        quotient.append(new)
    compact={}
    for k in quotient:
        require(k['identity'] not in compact or compact[k['identity']]==k['values'],'BAD_EQUIVALENCE_CLASS')
        compact[k['identity']]=k['values']
    predicted,qpacked,qwraps,qguarded=_simulate(quotient,compact,barriers)
    # Exact per-reference values in a model, not hash-only equivalence.
    live=[None]*8192;ordinal=0
    for item in predicted:
        if item['kind']=='load':
            live[item['begin']:item['begin']+item['size']]=qpacked[item['source']:item['source']+item['size']]
        else:
            require(live[item['begin']:item['end']]==ks[ordinal]['values'],'MODEL_REFERENCE_VALUES_DIFFER')
            require(quotient[ordinal]['args']==ks[ordinal]['args'] and quotient[ordinal]['loops']==ks[ordinal]['loops'],
                    'MODEL_CHANGED_COMPUTE_CONTRACT')
            ordinal+=1
    require(ordinal==len(ks),'MODEL_MISSING_REFERENCE')
    after=model_summary(predicted,qpacked,qwraps,qguarded)

    actual=[];move_maps=[]
    for mode in ('detailed','minimal'):
        for index in range(1,8):
            unit=baseline/f'R_{mode}';text=(unit/f'native_call_{index:02d}.txt').read_text()
            cert=read(unit/f'native_certificate_{index:02d}.json')
            records=decode_submissions(text)[0]
            loads=[r for r in records if r['description']=='LOAD UOP']
            physical=read(unit/f'physical_certificate_{index:02d}.json')
            moves=[r for r in physical['moves'] if not r['reset_producer'] and r['category']=='POSSIBLE_EARLIER_RELEASE']
            move_maps.append(moves)
            work=Counter()
            for r in decode(text):work[r['kind']]+=r['steps']
            uop_bytes=sum(int(__import__('re').search(r'x: size=(\d+)',r['detail'])[1])*4 for r in loads)
            actual.append({'mode':mode,'call':index,'instructions':len(records),'UOP_LOADs':len(loads),
                'UOP_LOAD_bytes':uop_bytes,'serialized_uop_bytes':cert['uop_values']['serialized_uop_bytes'],
                'wraps':cert['uop_values']['uop_sram_wraps'],'work':dict(work),'MAC_wait_move_preliminary_sites':len(moves)})
    require(all(r['work']==actual[0]['work'] and r['UOP_LOAD_bytes']==actual[0]['UOP_LOAD_bytes'] and
                r['wraps']==actual[0]['wraps'] for r in actual),'QUALIFICATION_PATH_COST_DIFFERS')
    require(all(m==move_maps[0] for m in move_maps),'WAIT_SITE_MAPPING_DIFFERS')
    write(output/'actual_calls.json',actual)
    write(output/'initializer_classes.json',classes)
    write(output/'cache_model_counterfactual.json',{'status':'READONLY_QUEUE_MODEL_ONLY',
        'before':before,'quotient':after,'exact_reference_checks':ordinal,'compiled_variant':False,
        'joint_notification_graph_certified':False,'full_call_latency_prediction':None,
        'domain':'only observed 0..31 signatures, separate static handles; no raw-address-only aliasing'})
    write(output/'wait_sites.json',{'preliminary_independent_sites':move_maps[0],
        'combined_transform_certified':False,'input_payload_per_site_bytes':784,
        'input_sram_write_per_site_bytes':1296,'latency_saving_measured':False})
    source_paths=[Path(__file__),parent/'events.json',parent/'tensor_requests.json',parent/'module.ll',
        baseline/'completion_manifest.json',AUTHOR/'runtime.cc',REPO/'src/target/llvm/codegen_cpu.cc',
        HERE/'c3_official_window_runtime_uop_queue.h',HERE/'c3_tps_window_native.py']
    work=actual[0]['work'];slots=work['MAC']+work['RESET']+2*work['ALU']
    result={'status':'QUALIFIED_BASELINE_RESIDUAL_AUDIT','baseline':str(baseline),
        'source_bindings':{str(p):sha(p) for p in source_paths},'actual_cost':actual[0],
        'static_MAC_signature_identities':sum(r['signature_count'] for r in classes),
        'exact_MAC_initializer_classes':sum(r['complete_initializer_classes'] for r in classes),
        'cache_candidate':'NONADJACENT_EQUIVALENT_INITIALIZERS_UNDER_DIFFERENT_SIGNATURES',
        'scope':'repeated within this qualified operator; no cross-network prevalence claim',
        'actual_latency_cost_isolated':False,'novelty_established':False,
        'ideal_issue_slots':slots,'nominal_100MHz_issue_ms':slots/100000,
        'ideal_issue_slots_are_not_measured_cycles':True,
        'new_target_calls':0,'new_compilations':0,'board_contacted':False}
    write(output/'audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='source_bindings'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('baseline',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args();audit(args.baseline.resolve(),args.output.resolve())
