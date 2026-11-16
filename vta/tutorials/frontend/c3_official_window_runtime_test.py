#!/usr/bin/env python3
"""Real fresh-process FSim collection gates and generic CPU helper regression."""
import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
GROUPED = HERE/'report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout'
OLD = GROUPED/'20261014_c3_valid_window_local_run01'
DATA = GROUPED/'20261009_c3_node24_partial_run01/board_qualify03'
SEEDS = (0,20250901,20260910)
SYMBOL = 'tvmgen_default_fused_nn_conv2d_add_right_shift_clip_cast'


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


def fsim_worker(out,runtime_dir,helper,program,mode):
    os.environ['TVM_LIBRARY_PATH'] = str(runtime_dir)
    os.environ['C3_OFFICIAL_WINDOW_COLLECTION'] = mode
    os.environ['VTA_REPLAY_POLICY'] = 'disabled'
    os.environ.pop('VTA_COMMAND_MANIFEST_ID',None)
    sys.path[:0] = [str(REPO/'python'),str(REPO/'vta/python'),str(HERE)]
    import numpy as np
    import tvm,vta
    with vta.Environment(dict(vta.get_env().cfg_dict,TARGET='sim')):
        from vta.testing import simulator
        expected_runtime = runtime_dir/'libvta_fsim.so'
        assert Path(simulator.LIBS[0]._name).resolve() == expected_runtime.resolve()
        module_path = OLD/'local_final02'/program/'fsim_module.so' if program != 'TPS' else GROUPED/'20261013_c3_banerjee_baseline_completion_run02/local_final06/TPS-oc-reuse-ci4/fsim_module.so'
        module_sha,rt_sha,helper_sha = digest(module_path),digest(expected_runtime),digest(helper)
        module = tvm.runtime.load_module(str(module_path))
        helper_module = tvm.runtime.load_module(str(helper))
        funcs = {name:tvm.get_global_func('vta.runtime.'+name) for name in ('official_collection_status','official_submit_ledger_clear','official_submit_ledger','profiler_clear','profiler_status')}
        diagnostic = tvm.get_global_func('c3.official_window.fused_diagnostic_once')
        reset,counts,metadata = [tvm.get_global_func('c3.official_window.'+name) for name in ('reset_counts','counts','metadata')]
        reset()
        x={s:np.load(DATA/('X_%d.npy' % s)) for s in SEEDS}
        y={s:np.load(DATA/('Y24_%d.npy' % s)) for s in SEEDS}
        wa,wb,add,yb=[np.load(DATA/name) for name in ('W_A.npy','W_B.npy','ADD.npy','Y_B.npy')]
        dev=tvm.device('ext_dev',0)
        buffers=tuple(tvm.nd.array(v,dev) for v in (x[0],wa,add,np.bitwise_xor(y[0],np.int8(1))))
        handles=[int(ctypes.cast(b.handle,ctypes.c_void_p).value) for b in buffers]
        initial=json.loads(str(funcs['official_collection_status']()))
        assert initial['mode']==mode and initial['debug_flag']==0 and initial['replay']['mode']=='disabled'
        result=[]
        cases=[(s,wa,y[s],'seed') for s in SEEDS]+[(0,w,ref,label) for w,ref,label in ((wa,y[0],'A'),(wb,yb,'B'),(wa,y[0],'A_return'))]
        for index,(seed,weight,reference,label) in enumerate(cases,1):
            values=(x[seed],weight,add,np.bitwise_xor(reference,np.int8(1)))
            for b,value in zip(buffers,values): b.copyfrom(value)
            funcs['profiler_clear']();funcs['official_submit_ledger_clear']()
            dump=str(diagnostic(module,SYMBOL,*buffers))
            ledger=json.loads(str(funcs['official_submit_ledger']()))
            status=json.loads(str(funcs['official_collection_status']()))
            profile=json.loads(str(funcs['profiler_status']()))
            helper_metadata=json.loads(str(metadata()))
            assert int(counts())==index and helper_metadata['started_calls']==helper_metadata['completed_calls']==index
            assert helper_metadata['parameter_count']==4 and not helper_metadata['adds_synchronize']
            assert status['mode']==mode and status['debug_flag']==0 and status['replay']['mode']=='disabled'
            assert ledger['recorded_submits']==ledger['attempted_submits']==1 and not ledger['overflow']
            assert ledger['submits'][0]['completion'] and ledger['submits'][0]['timeout']==0
            if mode=='minimal':
                assert status['detailed_clock_reads']==0 and profile['event_count']==0
                assert all(profile[key]==0 for key in ('push_gemm_op_calls','push_alu_op_calls','load_buffer_2d_calls','store_buffer_2d_calls','synchronize_calls'))
            else:
                assert status['detailed_clock_reads']>0 and profile['event_count']>0 and profile['synchronize_calls']==1
            actual=buffers[-1].numpy()
            assert np.array_equal(actual,reference)
            assert all(np.array_equal(b.numpy(),value) for b,value in zip(buffers[:3],values[:3]))
            assert handles==[int(ctypes.cast(b.handle,ctypes.c_void_p).value) for b in buffers]
            dump_path=out/('native_call_%02d.txt' % index);dump_path.write_text(dump)
            result.append({'index':index,'seed':seed,'case':label,'dump_file':str(dump_path),'dump_sha256':digest(dump_path),
                'output_sha256':hashlib.sha256(actual.tobytes()).hexdigest(),'ledger':ledger,'collection_status':status,
                'profile':profile,'helper_metadata':helper_metadata,'correct':True,'arguments_unchanged':True,'same_handles':True})
        assert digest(module_path)==module_sha and digest(expected_runtime)==rt_sha and digest(helper)==helper_sha
        write(out/'result.json',{'status':'REAL_FSIM_COLLECTION_AND_ONCE_HELPER_PASSED','program':program,'mode':mode,'target_calls':6,
            'module':str(module_path),'module_sha256':module_sha,'runtime_sha256':rt_sha,'helper_sha256':helper_sha,'calls':result,'board_contacted':False})


def compare(out,runtime_dir,helper):
    out.mkdir(parents=True,exist_ok=False)
    from c3_valid_window_board import command_fingerprint
    all_results={}
    for program in ('Q1','Q8','TPS'):
        all_results[program]={}
        for mode in ('detailed','minimal'):
            dest=out/(program+'_'+mode);dest.mkdir()
            command=[sys.executable,__file__,'worker','--output-dir',str(dest),'--runtime-dir',str(runtime_dir),'--helper',str(helper),'--program',program,'--mode',mode]
            with (dest/'worker.log').open('x') as f:
                proc=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=300)
            write(dest/'exit.json',{'returncode':proc.returncode})
            if proc.returncode:raise RuntimeError('real collection worker failed: '+str(dest))
            all_results[program][mode]=json.loads((dest/'result.json').read_text())
        detailed,minimal=[all_results[program][m] for m in ('detailed','minimal')]
        assert detailed['runtime_sha256']==minimal['runtime_sha256'] and detailed['module_sha256']==minimal['module_sha256']
        for a,b in zip(detailed['calls'],minimal['calls']):
            assert a['output_sha256']==b['output_sha256']
            assert a['ledger']['submits']==b['ledger']['submits']
            assert command_fingerprint(Path(a['dump_file']).read_text())==command_fingerprint(Path(b['dump_file']).read_text())
            # Exact initializer/UOP lines are independent of physical DDR relocation.
            uop=lambda text:[line for line in text.splitlines() if line.startswith('There are ') and line.endswith('uops') or 'acc=' in line and 'inp=' in line and 'wgt=' in line]
            assert uop(Path(a['dump_file']).read_text())==uop(Path(b['dump_file']).read_text())
    write(out/'comparison.json',{'status':'DETAILED_MINIMAL_REAL_FSIM_EQUAL','programs':['Q1','Q8','TPS'],'targets_per_mode_per_program':6,
          'total_target_calls':36,'same_binary_same_runtime':True,'same_normalized_all_commands':True,'same_uop_initializers':True,
          'same_submit_ledger':True,'same_seed_ABA_outputs':True,'minimal_runtime_detail_clocks_events_disabled':True,'board_contacted':False})
    print(json.dumps({'status':'DETAILED_MINIMAL_REAL_FSIM_EQUAL','target_calls':36,'board_contacted':False}))


def helper_cpu(out,helper):
    out.mkdir(parents=True,exist_ok=False)
    source=out/'counter.cc';binary=out/'counter.so'
    source.write_text('''#include <tvm/runtime/c_backend_api.h>
extern "C" TVM_DLL int c3_generic_counter(TVMValue* a,int* t,int n,TVMValue* r,int* rt) {
 if(n<1)return -1; for(int i=0;i<n;++i)if(t[i]!=kTVMDLTensorHandle&&t[i]!=kTVMNDArrayHandle)return -2;
 auto tensor=static_cast<DLTensor*>(a[n-1].v_handle);if(tensor->device.device_type!=kDLCPU||tensor->dtype.bits!=32)return -3;
 auto p=reinterpret_cast<int32_t*>(static_cast<char*>(tensor->data)+tensor->byte_offset);*p+=1;
 r->v_int64=0;*rt=kTVMNullptr;return 0; }
''')
    command=['g++','-O2','-shared','-fPIC',str(source),'-I'+str(REPO/'include'),'-I'+str(REPO/'3rdparty/dlpack/include'),'-I'+str(REPO/'3rdparty/dmlc-core/include'),'-o',str(binary)]
    proc=subprocess.run(command,capture_output=True,text=True);(out/'build.log').write_text(proc.stdout+proc.stderr)
    assert proc.returncode==0
    sys.path[:0]=[str(REPO/'python')]
    import numpy as np,tvm
    loaded=tvm.runtime.load_module(str(helper));module=tvm.runtime.load_module(str(binary))
    timer,reset,counts,metadata=[tvm.get_global_func('c3.official_window.'+n) for n in ('fused_timed_once','reset_counts','counts','metadata')]
    for arity in (1,3,4,6):
        buffers=[tvm.nd.array(np.array([0],dtype='int32')) for _ in range(arity)];reset()
        for i in range(1,4):
            latency=float(timer(module,'c3_generic_counter',*buffers));meta=json.loads(str(metadata()))
            assert math.isfinite(latency) and latency>0 and int(counts())==i and int(buffers[-1].numpy()[0])==i
            assert meta['parameter_count']==arity and meta['started_calls']==meta['completed_calls']==i
            assert all(int(b.numpy()[0])==0 for b in buffers[:-1])
    write(out/'result.json',{'status':'GENERIC_HELPER_REAL_CPU_ARITIES_PASSED','arities':[1,3,4,6],'invocations_each':3,'total_target_calls':12,
            'one_call_output_increment_exact':True,'helper_sha256':digest(helper),'helper_metadata_outside_timer':True,'VTA_target_calls':0,'board_contacted':False})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('worker','compare','helper-cpu'))
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--runtime-dir',type=Path)
    parser.add_argument('--helper',type=Path,required=True)
    parser.add_argument('--program',choices=('Q1','Q8','TPS'))
    parser.add_argument('--mode',choices=('detailed','minimal'))
    args=parser.parse_args()
    if args.phase=='worker':fsim_worker(args.output_dir,args.runtime_dir,args.helper,args.program,args.mode)
    elif args.phase=='compare':compare(args.output_dir,args.runtime_dir,args.helper)
    else:helper_cpu(args.output_dir,args.helper)


if __name__=='__main__':main()
