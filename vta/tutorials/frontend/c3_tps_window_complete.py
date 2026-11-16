#!/usr/bin/env python3
"""V4 complete fixed-object gates. This command cannot contact an FPGA.

The caller must provide first-window gates and source-bound bank provenance.
Only the fixed TPS/official objects are accepted; no performance sample is
produced. Sealed inputs are read-only.
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


def worker(out,variant,object_id,mode):
    runtime=OFFICIAL/'runtime_local_final'
    helper=OFFICIAL/'helper/local_final02/c3_official_window_runtime_helper.so'
    os.environ['TVM_LIBRARY_PATH']=str(runtime)
    os.environ['C3_OFFICIAL_WINDOW_COLLECTION']=mode
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
    descriptor=json.loads((out.parent/'objects.json').read_text())[object_id]
    parent=tvm.ir.load_json(Path(descriptor['parent_module']).read_text())
    restored=tvm.ir.load_json(tvm.ir.save_json(parent))
    if not tvm.ir.structural_equal(parent,restored,map_free_vars=False):
        raise ValueError('PARENT_ROUNDTRIP_CHANGED')
    raw_final=tvm.ir.load_json(Path(descriptor['final_module']).read_text())
    rt_final=vta.transform.CPUAccessRewrite()(restored)
    if not tvm.ir.structural_equal(raw_final,rt_final,map_free_vars=False):
        raise ValueError('REMAINING_LOWERING_ROUNDTRIP_CHANGED')
    gv=parent.get_global_var(descriptor['global_var']);original=parent[gv]
    hardware=json.loads((out.parent/'hardware_archive.json').read_text())
    minimum_gap=hardware.get('verified_acc_interval')
    proposed,decisions=migrate(original,variant,first_only=False,bank_loop_source_verified=descriptor['bank_loop_source_verified'])
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
        'status':'LOCAL_LOGICAL_GATES_PASSED','variant':variant,'first_window_only':False,'object_id':object_id,
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
        collection=tvm.get_global_func('vta.runtime.official_collection_status')
        if json.loads(collection())['mode']!=mode:raise ValueError('COLLECTION_MODE_NOT_EFFECTIVE')
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
            collected=json.loads(collection())
            if collected['mode']!=mode or collected['debug_flag']!=0:
                raise ValueError('COLLECTION_MODE_OR_DEBUG_CHANGED')
            if mode=='minimal' and collected['detailed_clock_reads']!=0:
                raise ValueError('MINIMAL_RETAINED_DETAIL_CLOCK_READS')
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
    write(out/'summary.json',{'status':'COMPLETE_WINDOW_LOCAL_LOGICAL_NATIVE_AND_FSIM_PASSED',
        'variant':variant,'collection':mode,'first_window_only':False,'object_id':object_id,'local_complete_calls':len(rows),
        'hardware_pipeline_qualified':minimum_gap is not None,'board_qualified':False,
        'runtime_sha256':sha(loaded_runtime),'helper_sha256':sha(helper),
        'module_sha256':sha(out/'fsim_module.so'),'source_bindings':start_identity['source_bindings'],
        'counts':audit['counts'],'statistics':audit['statistics'],
        'native_submissions':native['submissions'],'native_uop_values':native['uop_values']})


def objects():
    """Fixed source-derived objects; no capacity or performance selection."""
    from c3_official_window_baseline import compiler_sources,assert_effective_config
    from tvm.autotvm.task.space import ConfigEntity
    descriptor_path=OFFICIAL/'local_final01/yolov3_tiny_416_node024_descriptor.json'
    official=json.loads(descriptor_path.read_text())
    compiler_path=OFFICIAL/'baseline/yolov3_tiny_416_final01/compiler_identity.json'
    compiler=json.loads(compiler_path.read_text())
    current_compiler=compiler_sources()
    if current_compiler.keys()!=compiler['source_sha256'].keys():
        raise ValueError('OFFICIAL_COMPILER_SOURCE_SET_CHANGED')
    capture_revisions=[]
    final_seal=json.loads((OFFICIAL/'completion_manifest.json').read_text())
    for path,digest in compiler['source_sha256'].items():
        if current_compiler[path]!=digest:
            if path!=str(HERE/'c3_official_window_baseline.py') or \
               final_seal['source_sha256'].get(path)!=current_compiler[path]:
                raise ValueError('OFFICIAL_COMPILER_OR_TEMPLATE_CHANGED:'+path)
            capture_revisions.append({'path':path,'at_capture_sha256':digest,
                'execution_utility_sha256':current_compiler[path],
                'scope':'ABI/config decoding utility only; archived parent is used, schedule is not regenerated'})
    if sha(compiler['official_log'])!=compiler['official_log_sha256']:
        raise ValueError('OFFICIAL_LOG_CHANGED')
    record=official['source_record']
    assert_effective_config(record,ConfigEntity.from_json_dict(official['effective_config_entity']))
    line=json.loads(Path(compiler['official_log']).read_text().splitlines()[record['line']-1])
    if line!=record['raw_record'] or line['result'][1]!=0:
        raise ValueError('OFFICIAL_SELECTED_LOG_LINE_CHANGED')
    entities={key:value for key,kind,value in record['complete_config_entity']['entity']}
    if entities.get('oc_nthread')!=2 or entities.get('h_nthread')!=1:
        raise ValueError('SOURCE_NOT_TWO_OUTPUT_VIRTUAL_THREADS')
    passes=json.loads((OFFICIAL/'baseline/yolov3_tiny_416_final01/B_raw/pass_sequence.json').read_text())
    if 'tir.InjectVirtualThread' not in passes['actual_passes']:
        raise ValueError('VIRTUAL_THREAD_LOWERING_PROVENANCE_MISSING')
    result={
        'TPS':{'parent_module':str(TPS/'013_after_tir_vta_FoldUopLoop.json'),
               'final_module':str(TPS/'015_after_tir_vta_CPUAccessRewrite.json'),
               'global_var':'tvmgen_default_fused_nn_conv2d_add_right_shift_clip_cast',
               'bank_loop_source_verified':False,'source_kind':'already adjacent explicit bank scopes'},
        'official':{**official,'parent_module':str(REPO/official['parent_module']),
                    'final_module':str(REPO/official['final_module']),
                    'bank_loop_source_verified':True,
                    'bank_provenance':{'origin':'source-bound original VTA output-vthread schedule',
                        'compiler_identity_sha256':sha(compiler_path),
                        'capture_utility_revisions':capture_revisions,
                        'descriptor_sha256':sha(descriptor_path),
                        'oc_nthread':2,'h_nthread':1,'InjectVirtualThread_executed':True,
                        'support':'only the unique direct two-iteration COMPUTE/MAC axis; no other host loops'}},
    }
    return result,current_compiler


def run(output,first_gate,selected):
    from c3_tps_window_hardware import audit_archive
    from c3_official_window_workflow import _tree_rss
    gate=json.loads((first_gate/'stage_result.json').read_text())
    if gate['status']!='FIRST_WINDOW_GATES_PASSED':
        raise ValueError('FIRST_WINDOW_NOT_QUALIFIED')
    output.mkdir(exist_ok=False)
    inventory,compiler_bindings=objects()
    write(output/'objects.json',inventory)
    first_identity=json.loads((first_gate/'identity.json').read_text())
    bindings=dict(first_identity['source_bindings'])
    for path,digest in bindings.items():
        if sha(path)!=digest:raise ValueError('FIRST_WINDOW_SOURCE_CHANGED:'+path)
    for name in ('c3_tps_window_complete.py','c3_tps_window_native.py','test_c3_tps_window.py'):
        bindings[str(HERE/name)]=sha(HERE/name)
    bindings.update(compiler_bindings)
    for row in inventory.values():
        for key in ('parent_module','final_module'):
            bindings[row[key]]=sha(row[key])
    bindings[str(first_gate/'stage_result.json')]=sha(first_gate/'stage_result.json')
    write(output/'identity.json',{'scope':'complete fixed programs; no FPGA or latency samples',
        'first_gate':str(first_gate),'source_bindings':bindings})
    write(output/'hardware_archive.json',audit_archive(REPO))
    results=[]
    for object_id in selected:
        for variant in ('B','N','G'):
            for mode in ('detailed','minimal'):
                name=f'{object_id}_{variant}_{mode}'
                command=[str(PYTHON),str(Path(__file__).resolve()),'--worker',variant,
                         '--object',object_id,'--mode',mode,'--output',str(output/name)]
                environment=dict(os.environ,PYTHONPATH=':'.join(str(REPO/p) for p in ('python','vta/python','vta/tutorials/frontend')),
                                 OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
                start,peak,limit=time.monotonic(),0,None
                with (output/f'{name}.log').open('x') as stream:
                    child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=environment,start_new_session=True)
                    while child.poll() is None:
                        peak=max(peak,_tree_rss(child.pid))
                        if peak>8*1024**3:limit='AUDIT_MEMORY_LIMIT'
                        elif time.monotonic()-start>600:limit='AUDIT_TIME_LIMIT'
                        if limit:
                            os.killpg(child.pid,signal.SIGKILL);break
                        time.sleep(0.2)
                    code=child.wait()
                result={'object_id':object_id,'variant':variant,'mode':mode,'returncode':code,
                    'status':limit or ('PASS' if code==0 else 'LOCAL_GATE_FAILED'),
                    'wall_seconds':time.monotonic()-start,'process_tree_peak_rss':peak}
                results.append(result);write(output/f'{name}_worker.json',result)
                print(json.dumps(result),flush=True)
                if code!=0:
                    write(output/'stage_result.json',{'status':'STOP_LOCAL_GATE_FAILED',
                         'worker_results':results,'board_contacted':False})
                    return
            detailed=output/f'{object_id}_{variant}_detailed'
            minimal=output/f'{object_id}_{variant}_minimal'
            from c3_official_window_baseline import normalize_llvm
            if normalize_llvm((detailed/'module.ll').read_text())!=normalize_llvm((minimal/'module.ll').read_text()):
                raise ValueError('COLLECTION_MODE_CHANGED_MODULE')
            d=json.loads((detailed/'summary.json').read_text())
            m=json.loads((minimal/'summary.json').read_text())
            if d['native_submissions']!=m['native_submissions'] or d['native_uop_values']!=m['native_uop_values']:
                raise ValueError('COLLECTION_MODE_CHANGED_NATIVE_EXECUTION')
    for path,digest in bindings.items():
        if sha(path)!=digest:raise ValueError('FROZEN_SOURCE_CHANGED_DURING_RUN:'+path)
    write(output/'stage_result.json',{'status':'COMPLETE_LOCAL_GATES_PASSED',
        'objects':selected,'worker_results':results,'board_contacted':False,
        'local_complete_calls':sum(json.loads((output/f"{r['object_id']}_{r['variant']}_{r['mode']}"/'summary.json').read_text())['local_complete_calls'] for r in results)})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--worker',choices=('B','N','G'))
    parser.add_argument('--object',choices=('TPS','official'))
    parser.add_argument('--mode',choices=('detailed','minimal'),default='detailed')
    parser.add_argument('--first-gate')
    parser.add_argument('--objects',nargs='+',choices=('TPS','official'),default=['TPS','official'])
    args=parser.parse_args()
    if args.worker:worker(Path(args.output).resolve(),args.worker,args.object,args.mode)
    else:run(Path(args.output).resolve(),Path(args.first_gate).resolve(),args.objects)

