#!/usr/bin/env python3
"""V4 first TPS window gates. This command cannot contact an FPGA.

The complete function changes only its first dynamic bank pair. Until the
hardware feedback proof is closed, these are local diagnostic proposals, not
qualified full N/G programs or latency samples. Sealed inputs are read-only.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
HOLD=HERE/'report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout'
TPS=HOLD/'20261013_c3_banerjee_baseline_completion_run02/local_final06/TPS-oc-reuse-ci4'
OFFICIAL=HOLD/'20261014_c3_official_valid_window_run01'
PYTHON=Path('/home/orange/miniconda3/envs/vta-resnet/bin/python')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n')


def worker(out,variant):
    runtime=OFFICIAL/'runtime_local_final'
    helper=OFFICIAL/'helper/local_final02/c3_official_window_runtime_helper.so'
    os.environ['TVM_LIBRARY_PATH']=str(runtime)
    os.environ['C3_OFFICIAL_WINDOW_COLLECTION']='detailed'
    os.environ['VTA_REPLAY_POLICY']='disabled'
    os.environ.pop('VTA_COMMAND_MANIFEST_ID',None)
    import numpy as np
    import tvm
    import vta
    from vta.testing import simulator
    from c3_tps_window_method import migrate,acc_spacing
    from c3_tps_window_progress import certify_progress,LANES
    from c3_five_tradeoff_commands import decode_submissions
    from c3_valid_window import certify_trace
    from c3_official_window_baseline import buffer_abi,normalize_llvm
    from c3_official_window_method import audit_function
    from c3_official_window_reference import (array_sha,from_workload,
        postprocess_from_output_terms,semantic_gates)
    from c3_tps_window_native import certify_native,HEADER
    from c3_official_window_workflow import llvm_alu_contract,llvm_cache_contract,native_certificate_for_storage
    out.mkdir()
    start_identity=json.loads((out.parent/'identity.json').read_text())
    for path,digest in start_identity['source_bindings'].items():
        if sha(path)!=digest:raise ValueError('FROZEN_SOURCE_CHANGED:'+path)
    loaded_runtime=Path(simulator.LIBS[0]._name).resolve()
    contract=json.loads((OFFICIAL/'runtime_contract.json').read_text())
    if loaded_runtime!=(runtime/'libvta_fsim.so').resolve() or sha(loaded_runtime)!=contract['runtime'][0]['binary_sha256']:
        raise ValueError('ACTUAL_RUNTIME_IDENTITY_CHANGED')
    if sha(helper)!=contract['helper']['local']['binary_sha256']:
        raise ValueError('ACTUAL_HELPER_IDENTITY_CHANGED')
    build=json.loads((runtime/'build.json').read_text())
    if build['source_bindings'].get(str(HEADER))!=sha(HEADER):
        raise ValueError('NATIVE_MODEL_NOT_BOUND_TO_RUNTIME')
    parent=tvm.ir.load_json((TPS/'013_after_tir_vta_FoldUopLoop.json').read_text())
    restored=tvm.ir.load_json(tvm.ir.save_json(parent))
    if not tvm.ir.structural_equal(parent,restored,map_free_vars=False):
        raise ValueError('PARENT_ROUNDTRIP_CHANGED')
    raw_final=tvm.ir.load_json((TPS/'015_after_tir_vta_CPUAccessRewrite.json').read_text())
    rt_final=vta.transform.CPUAccessRewrite()(restored)
    if not tvm.ir.structural_equal(raw_final,rt_final,map_free_vars=False):
        raise ValueError('REMAINING_LOWERING_ROUNDTRIP_CHANGED')
    gv=next(iter(parent.functions));original=parent[gv]
    hardware=json.loads((out.parent/'hardware_archive.json').read_text())
    minimum_gap=hardware.get('verified_acc_interval')
    proposed,decisions=migrate(original,variant,first_only=True)
    baseline=audit_function(original)
    audit=audit_function(proposed)
    baseline_sem=certify_trace(baseline['events'])
    sem=certify_trace(audit['events'])
    if sem['output_terms']!=baseline_sem['output_terms']:
        raise ValueError('OUTPUT_PRODUCTS_OR_TAIL_CHANGED')
    tensor=lambda es:[e for e in es if e['kind'] in ('load','store')]
    if tensor(audit['events'])!=tensor(baseline['events']):
        raise ValueError('ORDERED_TENSOR_REQUESTS_CHANGED')
    old_kernels=[e for e in baseline['events'] if e['kind']=='kernel']
    new_kernels=[e for e in audit['events'] if e['kind']=='kernel']
    if variant=='N' and old_kernels!=new_kernels:
        raise ValueError('N_CHANGED_KERNEL_CONTENT_OR_CACHE_IDENTITY')
    abi=buffer_abi(original)
    workload=['conv2d_packed.vta',['TENSOR',abi[0]['shape'],'int8'],['TENSOR',abi[1]['shape'],'int8'],
              [1,1],[1,1,1,1],[1,1],'NCHW1n16c','int32']
    tail=postprocess_from_output_terms(sem['output_terms'],abi)
    spec=from_workload(workload,abi,tail)
    independent=semantic_gates(spec,sem['output_terms'])
    write(out/'reference_contract.json',spec.contract())
    write(out/'independent_semantics.json',independent)
    write(out/'logical_certificate.json',{
        'status':'LOCAL_LOGICAL_GATES_PASSED','variant':variant,'first_window_only':True,
        'baseline_counts':baseline['counts'],'proposal_counts':audit['counts'],
        'decisions':decisions,'output_products_and_tail_exactly_equal':True,
        'tensor_requests_exactly_equal':True,'N_kernel_and_cache_identity_equal':variant=='N',
        'acc_spacing':acc_spacing(audit['events'],minimum_gap),
        'hardware_pipeline_qualified':minimum_gap is not None,'board_qualified':False})
    module=tvm.IRModule(parent.functions,parent.type_definitions,parent.attrs)
    module.update_func(gv,proposed)
    final=vta.transform.CPUAccessRewrite()(module)
    (out/'parent.json').write_text(tvm.ir.save_json(module))
    (out/'parent.tir').write_text(module.script())
    (out/'final.json').write_text(tvm.ir.save_json(final))
    symbol=str(original.attrs['global_symbol'])
    with vta.Environment(dict(vta.get_env().cfg_dict,TARGET='sim')):
        target=tvm.target.Target(vta.get_env().target,host=vta.get_env().target_host)
        with tvm.transform.PassContext(opt_level=3,disabled_pass={'AlterOpLayout','tir.CommonSubexprElimTIR'}):
            lib=tvm.build(final,target=target)
            if variant=='B':
                direct=tvm.build(raw_final,target=target)
                if normalize_llvm(lib.get_source('ll'))!=normalize_llvm(direct.get_source('ll')):
                    raise ValueError('ROUNDTRIP_LLVM_OR_CACHE_SHARING_CHANGED')
        lib.export_library(str(out/'fsim_module.so'))
        llvm=lib.get_source('ll');(out/'module.ll').write_text(llvm)
        write(out/'alu_initializer_contract.json',llvm_alu_contract(final[gv],llvm))
        write(out/'compiled_cache_contract.json',llvm_cache_contract(llvm))
        lib=tvm.runtime.load_module(str(out/'fsim_module.so'))
        helper_module=tvm.runtime.load_module(str(helper))
        diagnostic=tvm.get_global_func('c3.official_window.fused_diagnostic_once')
        reset=tvm.get_global_func('c3.official_window.reset_counts')
        counts=tvm.get_global_func('c3.official_window.counts')
        metadata=tvm.get_global_func('c3.official_window.metadata')
        clear=tvm.get_global_func('vta.runtime.official_submit_ledger_clear')
        ledger_func=tvm.get_global_func('vta.runtime.official_submit_ledger')
        cases=spec.qualification_cases()
        buffers=tuple(tvm.nd.array(value,tvm.device('ext_dev',0)) for value in cases[0]['arrays'])
        handles=[int(ctypes.cast(b.handle,ctypes.c_void_p).value) for b in buffers]
        storage=[int(b.handle.contents.data) for b in buffers]
        depths=hardware['fifo_depths']
        command_depths={1:depths['load_queue'],2:depths['gemm_queue'],3:depths['store_queue']}
        token_depths=dict(zip(LANES,[depths[k] for k in ('l2g_queue','g2l_queue','g2s_queue','s2g_queue')]))
        definitions,rows=None,[]
        reset()
        for index,case in enumerate(cases,1):
            for buf,value in zip(buffers,case['arrays']):buf.copyfrom(value)
            clear()
            write(out/f'call_{index:02d}_begin.json',{'call_index':index,'variant':variant,
                'kind':case['kind'],'seed':case['seed'],'target_calls_planned':1,
                'performance_sample':False,'board_contacted':False})
            dump=diagnostic(lib,symbol,*buffers)
            ledger=json.loads(ledger_func())
            actual=buffers[-1].numpy()
            if not np.array_equal(actual,case['expected']):raise ValueError('FULL_FUSION_MISMATCH')
            if any(not np.array_equal(b.numpy(),a) for b,a in zip(buffers[:-1],case['arrays'][:-1])):
                raise ValueError('READONLY_ARGUMENT_CHANGED')
            if handles!=[int(ctypes.cast(b.handle,ctypes.c_void_p).value) for b in buffers] or \
               storage!=[int(b.handle.contents.data) for b in buffers]:raise ValueError('PARAMETER_STORAGE_CHANGED')
            meta=json.loads(metadata())
            if int(counts())!=index or meta['started_calls']!=index or meta['completed_calls']!=index:
                raise ValueError('HIDDEN_OR_MISSING_TARGET_CALL')
            (out/f'native_call_{index:02d}.txt').write_text(dump)
            native=certify_native(audit['events'],dump,ledger,definitions)
            definitions=native['cache_definitions']
            native_sem=certify_trace(audit['events'],ancestors=native['ancestors'])
            if native_sem['output_terms']!=baseline_sem['output_terms']:
                raise ValueError('NATIVE_DEPENDENCIES_CHANGED_OUTPUT_SEMANTICS')
            progress=[]
            for records in decode_submissions(dump):
                certificate=certify_progress(records,command_depths,token_depths)
                if certificate['status']!='FINITE_FIFO_PROGRESS_PROVED':
                    raise ValueError('FINITE_FIFO_PROGRESS_NOT_PROVED:'+certificate['status'])
                progress.append(certificate)
            write(out/f'native_certificate_{index:02d}.json',native_certificate_for_storage(native))
            write(out/f'progress_{index:02d}.json',progress)
            call={'call_index':index,'variant':variant,'kind':case['kind'],'seed':case['seed'],
                'correct':True,'target_calls':1,'helper_metadata':meta,'submit_ledger':ledger,
                'buffer_handles':handles,'actual_storage':storage,'same_weight_storage':storage[1],
                'native_dump_sha256':hashlib.sha256(dump.encode()).hexdigest(),
                'output_sha256':array_sha(actual),'reference_sha256':array_sha(case['expected']),
                'input_sha256':[array_sha(a) for a in case['arrays'][:-1]],
                'performance_sample':False,'board_contacted':False}
            if case['kind']=='directed':
                call['output_distribution']={
                    'zero':int(np.count_nonzero(actual==0)),
                    'clip_endpoints':int(np.count_nonzero((actual==127)|(actual==-127))),
                    'nonzero_unsaturated':int(np.count_nonzero((actual!=0)&(actual!=127)&(actual!=-127))),
                    'total':int(actual.size)}
            rows.append(call);write(out/f'call_{index:02d}_end.json',call)
            print(json.dumps({'variant':variant,'local_full_call':index,'status':'PASS'}),flush=True)
        write(out/'calls.json',rows)
    if helper_module is None:raise ValueError('HELPER_LIFETIME')
    write(out/'summary.json',{'status':'FIRST_WINDOW_LOCAL_LOGICAL_NATIVE_AND_FSIM_PASSED',
        'variant':variant,'first_window_only':True,'local_complete_calls':len(rows),
        'hardware_pipeline_qualified':minimum_gap is not None,'board_qualified':False,
        'runtime_sha256':sha(loaded_runtime),'helper_sha256':sha(helper),
        'module_sha256':sha(out/'fsim_module.so'),'source_bindings':start_identity['source_bindings'],
        'counts':audit['counts'],'statistics':audit['statistics'],
        'native_submissions':native['submissions'],'native_uop_values':native['uop_values']})


def run(output):
    from c3_tps_window_hardware import audit_archive
    from c3_official_window_workflow import _tree_rss
    output.mkdir(exist_ok=False)
    inputs=[HERE/'c3_tps_window_local.py',HERE/'c3_tps_window_method.py',HERE/'c3_tps_window_progress.py',HERE/'c3_tps_window_native.py',
            HERE/'c3_tps_window_hardware.py',TPS/'013_after_tir_vta_FoldUopLoop.json',
            TPS/'015_after_tir_vta_CPUAccessRewrite.json',OFFICIAL/'runtime_contract.json',
            OFFICIAL/'runtime_local_final/libvta_fsim.so',
            OFFICIAL/'helper/local_final02/c3_official_window_runtime_helper.so',
            HOLD.parent/'00_governance/C3_TPS_VALID_WINDOW_FOLLOWUP_PLAN_V4.md']
    original=json.loads((OFFICIAL/'runtime_contract.json').read_text())['source_bindings']
    final_seal=json.loads((OFFICIAL/'completion_manifest.json').read_text())
    verifier_revisions=[]
    bindings={str(p):sha(p) for p in inputs}
    for name in ('c3_valid_window.py','c3_official_window_method.py',
                 'c3_official_window_reference.py','c3_official_window_llvm.py',
                 'c3_official_window_workflow.py','c3_official_window_baseline.py',
                 'c3_official_window_native_binding.py','c3_five_tradeoff_commands.py'):
        path=HERE/name
        bindings[str(path)]=sha(path)
    for path,digest in original.items():
        actual=sha(path)
        if actual!=digest:
            # The final seal already includes this verifier-only strengthening
            # after runtime_contract was written. Do not silently allow an
            # arbitrary changed runtime, generator, helper or analyzer.
            if path!=str(HERE/'c3_official_window_runtime_native.py') or \
               final_seal['source_sha256'].get(path)!=actual:
                raise ValueError('SEALED_COMMON_SOURCE_CHANGED:'+path)
            verifier_revisions.append({'path':path,'runtime_contract_sha256':digest,
                'authoritative_final_seal_sha256':actual,
                'change':'sealed verifier adds cross-submit tensor-domain DRAM-base consistency; no DSO change'})
        bindings[path]=actual
    bindings[str(OFFICIAL/'completion_manifest.json')]=sha(OFFICIAL/'completion_manifest.json')
    write(output/'identity.json',{'schema':'c3_tps_window_first_attempt_v1',
        'scope':'first TPS window; gate stop before complete B/N/G until ACC hardware proof',
        'source_bindings':bindings,'verifier_contract_revisions':verifier_revisions,
        'board_contacted':False,'performance_samples':0})
    write(output/'hardware_archive.json',audit_archive(REPO))
    results=[]
    for variant in ('B','N','G'):
        command=[str(PYTHON),str(Path(__file__).resolve()),'--worker',variant,'--output',str(output/variant)]
        environment=dict(os.environ,PYTHONPATH=':'.join(str(REPO/p) for p in ('python','vta/python','vta/tutorials/frontend')),
                         OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        start,peak,limit=time.monotonic(),0,None
        with (output/f'{variant}.log').open('x') as stream:
            child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=environment,start_new_session=True)
            while child.poll() is None:
                peak=max(peak,_tree_rss(child.pid))
                if peak>8*1024**3:limit='AUDIT_MEMORY_LIMIT'
                elif time.monotonic()-start>600:limit='AUDIT_TIME_LIMIT'
                if limit:
                    os.killpg(child.pid,signal.SIGKILL);break
                time.sleep(0.2)
            code=child.wait()
        result={'variant':variant,'returncode':code,'status':limit or ('PASS' if code==0 else 'LOCAL_GATE_FAILED'),
                'wall_seconds':time.monotonic()-start,'process_tree_peak_rss':peak}
        results.append(result);write(output/f'{variant}_worker.json',result)
        print(json.dumps(result),flush=True)
        if code!=0:break
    write(output/'worker_results.json',results)
    for path,digest in bindings.items():
        if sha(path)!=digest:raise ValueError('FROZEN_INPUT_CHANGED_DURING_ATTEMPT:'+path)
    write(output/'stage_result.json',{'status':'FIRST_WINDOW_GATES_PASSED' if len(results)==3 and all(r['status']=='PASS' for r in results)
         else 'STOP_LOCAL_GATE_FAILED','hardware_pipeline_qualified':True,'board_qualified':False,
         'full_BNG_generated':False,'official_transfer_started':False,'board_contacted':False,
         'worker_results':results,'sealed_inputs_unchanged':True})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--worker',choices=('B','N','G'))
    args=parser.parse_args()
    if args.worker:worker(Path(args.output).resolve(),args.worker)
    else:run(Path(args.output).resolve())
