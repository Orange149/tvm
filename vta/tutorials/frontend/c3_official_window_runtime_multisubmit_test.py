"""Real FSim two-explicit-Sync fixture; never a performance candidate."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys

from c3_official_window_runtime_native import certify_native

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
GROUPED=HERE/'report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout'
OLD=GROUPED/'20261014_c3_valid_window_local_run01'
DATA=GROUPED/'20261009_c3_node24_partial_run01/board_qualify03'
SYMBOL='tvmgen_default_fused_nn_conv2d_add_right_shift_clip_cast'


def run(out,runtime,helper,mode):
    out.mkdir(parents=True,exist_ok=False)
    os.environ['TVM_LIBRARY_PATH']=str(runtime)
    os.environ['C3_OFFICIAL_WINDOW_COLLECTION']=mode
    os.environ['VTA_REPLAY_POLICY']='disabled'
    os.environ.pop('VTA_COMMAND_MANIFEST_ID',None)
    source=out/'two_complete_sync_fixture.cc'
    source.write_text('''#include <tvm/runtime/c_backend_api.h>
extern "C" TVM_DLL void* __tvm_module_ctx = nullptr;
extern "C" TVM_DLL int c3_two_complete_syncs(TVMValue* a,int* t,int n,TVMValue* r,int* rt) {
 if(n!=4)return -1;
 TVMFunctionHandle f=nullptr;
 if(TVMBackendGetFuncFromEnv(__tvm_module_ctx,"'''+SYMBOL+'''",&f))return -2;
 for(int i=0;i<2;++i)if(TVMFuncCall(f,a,t,n,r,rt))return -3;
 return 0; }
''')
    binary=out/'two_complete_sync_fixture.so'
    command=['g++','-O2','-shared','-fPIC',str(source),'-I'+str(REPO/'include'),
             '-I'+str(REPO/'3rdparty/dlpack/include'),'-I'+str(REPO/'3rdparty/dmlc-core/include'),'-o',str(binary)]
    proc=subprocess.run(command,capture_output=True,text=True)
    (out/'build.log').write_text(proc.stdout+proc.stderr)
    assert proc.returncode==0
    sys.path[:0]=[str(REPO/'python'),str(REPO/'vta/python')]
    import numpy as np
    import tvm,vta
    with vta.Environment(dict(vta.get_env().cfg_dict,TARGET='sim')):
        from vta.testing import simulator
        assert Path(simulator.LIBS[0]._name).resolve()==(runtime/'libvta_fsim.so').resolve()
        sealed=tvm.runtime.load_module(str(OLD/'local_final02/Q1/fsim_module.so'))
        module=tvm.runtime.load_module(str(binary));module.import_module(sealed)
        helper_module=tvm.runtime.load_module(str(helper))
        events=json.loads((OLD/'local_final02/Q1/expected_trace.json').read_text())
        if isinstance(events,dict):events=events['events']
        # This fixture deliberately calls one unchanged sealed operator twice;
        # its full final trace is the exact concatenation. No candidate/timing.
        events=events+events
        ref=np.load(DATA/'Y24_0.npy')
        arrays=[np.load(DATA/name) for name in ('X_0.npy','W_A.npy','ADD.npy')]+[np.bitwise_xor(ref,np.int8(1))]
        buffers=[tvm.nd.array(value,tvm.device('ext_dev',0)) for value in arrays]
        once=tvm.get_global_func('c3.official_window.fused_diagnostic_once')
        reset=tvm.get_global_func('c3.official_window.reset_counts');reset()
        ledger_clear=tvm.get_global_func('vta.runtime.official_submit_ledger_clear')
        ledger_read=tvm.get_global_func('vta.runtime.official_submit_ledger')
        timer=tvm.get_global_func('c3.official_window.fused_timed_once')
        cases=[];cache=None
        for index in (1,2):
            ledger_clear()
            text=str(once(module,'c3_two_complete_syncs',*buffers))
            (out/('native_%d.txt'%index)).write_text(text)
            ledger=json.loads(str(ledger_read()))
            proof=certify_native(events,text,ledger,cache)
            cache=proof['cache_definitions']
            assert proof['submission_count']==2
            assert np.array_equal(buffers[-1].numpy(),ref)
            assert all(np.array_equal(b.numpy(),value) for b,value in zip(buffers[:3],arrays[:3]))
            assert int(tvm.get_global_func('c3.official_window.counts')())==index
            first_material=[e for e in proof['material_map'] if e['submission']==1]
            first_mask=sum(1<<entry['tir_event'] for entry in first_material)
            assert all(proof['ancestors'][entry['tir_event']]&first_mask==first_mask
                       for entry in proof['material_map'] if entry['submission']==2)
            (out/('proof_%d.json'%index)).write_text(json.dumps(proof,indent=2,sort_keys=True))
            cases.append({'index':index,'kind':'diagnostic','ledger':ledger,'submission_count':2,'output_correct':True})
        ledger_clear();latency=float(timer(module,'c3_two_complete_syncs',*buffers));ledger=json.loads(str(ledger_read()))
        status=json.loads(str(tvm.get_global_func('vta.runtime.official_collection_status')()))
        assert latency>0 and len(ledger['submits'])==2 and all(row['completion'] for row in ledger['submits'])
        assert int(tvm.get_global_func('c3.official_window.counts')())==3
        assert status['debug_flag']==0 and np.array_equal(buffers[-1].numpy(),ref)
        if mode=='minimal':assert status['detailed_clock_reads']==0
        cases.append({'index':3,'kind':'timer','ledger':ledger,'call_ms':latency,'submission_count':2,'output_correct':True,'collection_status':status})
        result={'status':'REAL_FSIM_TWO_EXPLICIT_SYNCS_CERTIFIED','mode':mode,'fixture_helper_targets':3,
                'unchanged_sealed_operator_invocations':6,'cases':cases,'timer_adds_sync':False,
                'timer_target_calls_exactly_once':True,'role':'runtime-only correctness fixture; no candidate or performance inference',
                'board_contacted':False}
        with (out/'result.json').open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True)
        print(json.dumps({'status':result['status'],'mode':mode,'fixture_targets':3,'sealed_operator_invocations':6}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--runtime-dir',type=Path,required=True)
    parser.add_argument('--helper',type=Path,required=True)
    parser.add_argument('--mode',choices=('detailed','minimal'),required=True)
    a=parser.parse_args();run(a.output_dir,a.runtime_dir,a.helper,a.mode)
