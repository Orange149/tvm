"""Offline exact equivalence and malformed multi-submit gates; no targets."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time

from c3_valid_window_native import native_binding_details as previous
from c3_official_window_native_binding import native_binding_details as current
from c3_official_window_runtime_native import certify_native

HERE=Path(__file__).resolve().parent
GROUPED=HERE/'report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout'
OLD=GROUPED/'20261014_c3_valid_window_local_run01'


def load(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,data):
    with path.open('x') as stream:json.dump(data,stream,indent=2,sort_keys=True);stream.write('\n')
def rejection(function):
    try:function()
    except ValueError as error:return str(error)
    raise AssertionError('expected strict rejection')


def run(root,out):
    out.mkdir(parents=True,exist_ok=False)
    local=root/'runtime_local_final'
    comparisons=[]
    for program in ('Q1','Q8','TPS'):
        path=OLD/'local_final02'/program/'expected_trace.json' if program!='TPS' else OLD/'sealed_tps_expected_trace.json'
        events=load(path);events=events['events'] if isinstance(events,dict) else events
        dump=local/'mode_equivalence02'/(program+'_detailed')/'native_call_01.txt'
        text=dump.read_text()
        begin=time.perf_counter();a=previous(events,text);old_ms=(time.perf_counter()-begin)*1000
        begin=time.perf_counter();b=current(events,text);new_ms=(time.perf_counter()-begin)*1000
        assert a==b
        comparisons.append({'program':program,'events':len(events),'instructions':b['instruction_count'],
            'full_evidence_exact_equal':True,'old_algorithm_ms':old_ms,'new_algorithm_ms':new_ms,
            'events_path':str(path),'events_sha256':sha(path),'dump':str(dump),'dump_sha256':sha(dump)})
        if program=='Q1':q1_events=events
    dual=[]
    for mode in ('detailed','minimal'):
        directory=local/('two_explicit_syncs_'+mode+'01')
        result=load(directory/'result.json');cache=None
        for index in (1,2):
            text=(directory/('native_%d.txt'%index)).read_text()
            ledger=result['cases'][index-1]['ledger']
            begin=time.perf_counter();new=certify_native(q1_events+q1_events,text,ledger,cache)
            elapsed=(time.perf_counter()-begin)*1000
            cache=new['cache_definitions']
            old=load(directory/('proof_%d.json'%index))
            for key,value in old.items():assert new[key]==value,key
            assert len(new['tensor_domain_dram_bases'])==4
            write(out/(mode+'_two_submit_%d.json'%index),new)
            dual.append({'mode':mode,'call':index,'all_previous_fields_exact_equal':True,
                         'new_full_certificate_ms':elapsed,'cross_submit_dram_bases_stable':True})
            if mode=='detailed' and index==1:fixture=(text,ledger)
    text,ledger=fixture
    blocks=list(re.finditer(r'There are \d+ instructions\n.*?(?=There are \d+ instructions|\Z)',text,re.S))
    assert len(blocks)==2
    second=blocks[1].group()
    pattern=r'(INSTRUCTION \d+: LOAD INP\n.*?DRAM: 0x)([0-9a-fA-F]+)'
    second,count=re.subn(pattern,lambda match:match[1]+format(int(match[2],16)+1,'x'),second,flags=re.S)
    assert count>0
    # Each isolated old fragment accepts a constant shifted tensor base. The
    # new whole-call gate must refuse that change across successful Syncs.
    previous(q1_events,second)
    bad=text[:blocks[1].start()]+second
    errors={'second_submit_tensor_base_change':rejection(lambda:certify_native(q1_events+q1_events,bad,ledger))}
    assert errors['second_submit_tensor_base_change']=='CROSS_SUBMIT_TENSOR_DOMAIN_DRAM_BASE_CHANGED'
    write(out/'result.json',{'status':'EXACT_NATIVE_DAG_PAYLOAD_OPTIMIZATION_VERIFIED','real_short_dumps':comparisons,
        'real_two_submit_cold_warm':dual,'negative_gates':errors,
        'algorithm':'Same FIFO/token edges; topological recurrence A[target] |= A[source] | mapped_TIR_bit(source)',
        'strictness':['all material/dependency/loop/address checks retained','token census and cycle rejection retained',
                     'instruction IDs must be contiguous','fragment event IDs must be contiguous before mask shift',
                     'same tensor/domain physical DDR base must stay fixed across explicit Syncs'],
        'sources':{str(HERE/name):sha(HERE/name) for name in (
            'c3_official_window_native_binding.py','c3_official_window_runtime_native.py',
            'c3_official_window_native_binding_test.py','c3_valid_window_native.py')},
        'runtime_contract_unmodified':str(root/'runtime_contract.json'),
        'runtime_contract_sha256':sha(root/'runtime_contract.json'),
        'target_calls':0,'board_contacted':False,
        'timing_scope':'offline certificate algorithm wall-clock only, never target/operator performance'})
    print(json.dumps({'status':'EXACT_NATIVE_DAG_PAYLOAD_OPTIMIZATION_VERIFIED','real_short_dumps':3,
                      'real_two_submit_cold_warm':4,'target_calls':0}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();run(a.root.resolve(),a.output_dir.resolve())
