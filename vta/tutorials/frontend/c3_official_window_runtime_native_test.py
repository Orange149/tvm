"""Offline native gates over real saved dumps, plus actual-ABI queue fixture."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

import c3_official_window_runtime as build
from c3_official_window_runtime_native import certify_native

HERE = Path(__file__).resolve().parent
GROUPED = HERE/'report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout'
OLD = GROUPED/'20261014_c3_valid_window_local_run01'


def write(path,data):
    with path.open('x') as stream:json.dump(data,stream,indent=2,sort_keys=True);stream.write('\n')


def expect_reject(function,contains=None):
    try:function()
    except ValueError as error:
        if contains:assert contains in str(error),(contains,str(error))
        return str(error)
    raise AssertionError('malformed certificate unexpectedly accepted')


def run(out,comparison):
    out.mkdir(parents=True,exist_ok=False)
    cases=[]
    for program in ('Q1','Q8','TPS'):
        source = OLD/'local_final02'/program/'expected_trace.json' if program != 'TPS' else OLD/'sealed_tps_expected_trace.json'
        events=json.loads(source.read_text())
        if isinstance(events,dict):events=events['events']
        for mode in ('detailed','minimal'):
            directory=comparison/(program+'_'+mode)
            result=json.loads((directory/'result.json').read_text())
            cache=None
            for index,call in enumerate(result['calls'],1):
                text=Path(call['dump_file']).read_text()
                proof=certify_native(events,text,call['ledger'],cache)
                cache=proof['cache_definitions']
                destination=out/(program+'_'+mode+'_%02d.json'%index)
                write(destination,proof)
                cases.append({'program':program,'mode':mode,'call':index,'certificate':str(destination),
                    'certificate_sha256':build.digest(destination),'submission_count':proof['submission_count'],
                    'uop_values':{k:v for k,v in proof['uop_values'].items() if k!='checks'}})
                if program=='Q1' and mode=='detailed' and index==1:
                    fixture=(events,text,call['ledger'],cache)
    events,text,ledger,cache=fixture
    negative={}
    tuples=copy.deepcopy(events)
    for event in tuples:
        if event['kind']=='kernel' and 'signature' in event:
            event['signature']=tuple(event['signature'])
    assert certify_native(tuples,text,ledger)['status']=='ACTUAL_BOUNDED_NATIVE_FIFO_TOKEN_UOP_CERTIFIED'
    negative['empty_tuple_signature_accepted']=True
    bad=copy.deepcopy(ledger);bad['overflow']=True
    negative['overflow']=expect_reject(lambda:certify_native(events,text,bad))
    bad=copy.deepcopy(ledger);bad['submits'][0]['completion']=False
    negative['incomplete']=expect_reject(lambda:certify_native(events,text,bad))
    bad=copy.deepcopy(ledger);bad['submits'][0]['serialized_uop_bytes']+=4
    negative['serialized_bytes']=expect_reject(lambda:certify_native(events,text,bad))
    bad=copy.deepcopy(ledger);bad['submits'][0]['reason']='threshold_auto_sync'
    negative['unknown_autosync']=expect_reject(lambda:certify_native(events,text,bad))
    bad=copy.deepcopy(ledger);bad['submits'][0]['ordinal']=2
    negative['ordinal']=expect_reject(lambda:certify_native(events,text,bad))
    bad=copy.deepcopy(ledger);bad['attempted_submits']=17
    negative['excess_submit_census']=expect_reject(lambda:certify_native(events,text,bad))
    negative['truncated']=expect_reject(lambda:certify_native(events,text[:text.rfind('INSTRUCTION')],ledger))
    bad=copy.deepcopy(events)
    for event in bad:
        if event['kind']=='kernel':
            event['signature']=[99];break
    negative['signature99']=expect_reject(lambda:certify_native(bad,text,ledger))
    warm=json.loads((comparison/'Q1_detailed/result.json').read_text())['calls'][1]
    warm_text=Path(warm['dump_file']).read_text()
    bad=copy.deepcopy(events)
    for event in bad:
        if event['kind']=='kernel' and event['uops'][0]['args'][0]==1:
            for uop in event['uops']:uop['args'][7]+=1
    negative['same_identity_warm_imm_mutation']=expect_reject(lambda:certify_native(bad,warm_text,warm['ledger'],cache))
    defs=shlex.split(subprocess.check_output([str(build.PYTHON),str(build.REPO/'3rdparty/vta-hw/config/vta_config.py'),'--defs'],text=True))
    queue=[]
    for original in (False,True):
        binary=out/('queue_original' if original else 'queue_safe')
        command=['g++','-std=c++17','-O2',str(HERE/'c3_official_window_runtime_queue_test.cc'),
                 '-I'+str(HERE),'-I'+str(build.REPO/'3rdparty/vta-hw/include'),*defs]
        if original:command+=['-DORIGINAL_QUEUE']
        command+=['-o',str(binary)]
        proc=subprocess.run(command,capture_output=True,text=True)
        (out/(binary.name+'_build.log')).write_text(proc.stdout+proc.stderr)
        assert proc.returncode==0,proc.stderr
        for mode in ('detailed','minimal'):
            env={**os.environ,'C3_OFFICIAL_WINDOW_COLLECTION':mode}
            command=[str(binary)]
            result=subprocess.run(command,capture_output=True,text=True,env=env)
            assert result.returncode==0,result.stderr
            queue.append({'original':original,'mode':mode,'returncode':result.returncode,'evidence':json.loads(result.stdout)})
        for operation in ('endpoint','oversize'):
            result=subprocess.run([str(binary),operation],capture_output=True,text=True)
            queue.append({'original':original,'operation':operation,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
            assert result.returncode==0 if operation=='endpoint' else result.returncode<0
    write(out/'result.json',{'status':'SAVED_REAL_NATIVE_AND_ACTUAL_ABI_QUEUE_GATES_PASSED','real_dump_calls':len(cases),
        'cases':cases,'negative_gates':negative,'actual_8192_queue':queue,
        'old_source_hashes':{str(HERE/'c3_banerjee_uop_queue.h'):build.digest(HERE/'c3_banerjee_uop_queue.h')},
        'new_VTA_targets':0,'board_contacted':False})
    print(json.dumps({'status':'SAVED_REAL_NATIVE_AND_ACTUAL_ABI_QUEUE_GATES_PASSED','real_dump_calls':len(cases),'new_targets':0}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--comparison-dir',type=Path,required=True)
    args=parser.parse_args();run(args.output_dir,args.comparison_dir)
