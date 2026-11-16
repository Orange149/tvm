"""YOLO416-only qualification worker, with explicit finite FIFO proof adoption.

The worker body equals the archived common worker except the progress proof
import. No compiler transform, instruction, runtime or target call changes.
The AST equivalence is checked before every new attempt.
"""
import ctypes
import json
import os
from pathlib import Path
from c3_tps_sync_release_local import (HELPER,RUNTIME,capture_helper_call,descriptor,
    initialize_helper,read,require,sha,validate_calls,validate_helper_completion,verify,write)


def qualification_worker(root, program, mode):
    os.environ['TVM_LIBRARY_PATH']=str(RUNTIME)
    os.environ['C3_OFFICIAL_WINDOW_COLLECTION']=mode
    os.environ['VTA_REPLAY_POLICY']='disabled'
    os.environ.pop('VTA_COMMAND_MANIFEST_ID',None)
    import numpy as np
    import tvm
    from vta.testing import simulator
    from c3_official_window_method import audit_function
    from c3_valid_window import certify_trace
    from c3_official_window_reference import from_workload,array_sha,semantic_gates
    from c3_official_window_workflow import native_certificate_for_storage
    from c3_tps_window_native import certify_native
    from c3_yolo416_fifo_certificate import certify_action_graph as certify_progress
    from c3_tps_window_progress import LANES
    from c3_five_tradeoff_commands import decode_submissions
    from c3_sync_release_audit import analyze
    from c3_tps_sync_release_method import certify_joint
    out=root/f'{program}_{mode}';out.mkdir()
    verify(read(root/'freeze.json')['bindings'])
    d=descriptor(root,program)
    verify(d['artifact_bindings'])
    actual_runtime=Path(simulator.LIBS[0]._name).resolve()
    require(actual_runtime==(RUNTIME/'libvta_fsim.so').resolve(),'ACTUAL_RUNTIME_PATH_CHANGED')
    require(sha(actual_runtime)==read(root/'freeze.json')['bindings'][str(actual_runtime)],'ACTUAL_RUNTIME_SHA_CHANGED')
    parent=tvm.ir.load_json(Path(d['parent_module']).read_text());function=parent[parent.get_global_var(d['global_var'])]
    audit=audit_function(function);events=audit['events'];sem=certify_trace(events)
    contract=read(d['reference_contract'])
    workload=['conv2d_packed.vta',['TENSOR',contract['input_shape'],'int8'],['TENSOR',contract['weight_shape'],'int8'],
              contract['strides'],contract['padding'],contract['dilation'],'NCHW1n16c','int32']
    spec=from_workload(workload,contract['abi'],contract['postprocess'])
    require(spec.contract()==contract,'REFERENCE_CONTRACT_CHANGED')
    write(out/'independent_semantics.json',semantic_gates(spec,sem['output_terms']))
    lib=tvm.runtime.load_module(d['module_path']); helper=tvm.runtime.load_module(str(HELPER))
    diagnostic=tvm.get_global_func('c3.official_window.fused_diagnostic_once')
    reset=tvm.get_global_func('c3.official_window.reset_counts');counts=tvm.get_global_func('c3.official_window.counts')
    metadata=tvm.get_global_func('c3.official_window.metadata')
    clear=tvm.get_global_func('vta.runtime.official_submit_ledger_clear')
    ledger_func=tvm.get_global_func('vta.runtime.official_submit_ledger')
    collection=tvm.get_global_func('vta.runtime.official_collection_status')
    require(json.loads(collection())['mode']==mode,'COLLECTION_MODE_CHANGED')
    replay=tvm.get_global_func('vta.runtime.replay_status')
    capacity=tvm.get_global_func('vta.runtime.queue_capacity_status')
    inventory={'replay':json.loads(replay()),'queue':json.loads(capacity())}
    require(inventory['replay']['mode']=='disabled' and inventory['replay']['template_count']==0 and
            inventory['queue']['replay_policy']=='disabled','REPLAY_NOT_DISABLED')
    require(all(inventory['queue'][k]==33554432 for k in
                ('insn_capacity_bytes','uop_capacity_bytes','insn_submit_threshold_bytes')) and
            inventory['queue']['submissions']==0,'ACTUAL_QUEUE_CONFIGURATION_CHANGED')
    write(out/'runtime_inventory.json',inventory)
    cases=spec.qualification_cases()
    planned=[p for p in read(root/'call_ledger.json') if p['program']==program and p['mode']==mode]
    require([(p['kind'],p['seed']) for p in planned]==[(c['kind'],c['seed']) for c in cases], 'INPUT_CASE_BUDGET_CHANGED')
    buffers=tuple(tvm.nd.array(a,tvm.device('ext_dev',0)) for a in cases[0]['arrays'])
    handles=[int(ctypes.cast(b.handle,ctypes.c_void_p).value) for b in buffers]
    storage=[int(b.handle.contents.data) for b in buffers]
    h=read(root/'hardware_archive.json')['fifo_depths']
    command_depths={i:h[k] for i,k in enumerate(('load_queue','gemm_queue','store_queue'),1)}
    token_depths=dict(zip(LANES,[h[k] for k in ('l2g_queue','g2l_queue','g2s_queue','s2g_queue')]))
    definitions,rows,backend_bases=None,[],None
    initial=initialize_helper(reset,counts,metadata,
                              lambda value:write(out/'helper_initial_state.json',value))
    write(out/'helper_initial_metadata.json',initial['metadata'])
    for index,(case,expected_row) in enumerate(zip(cases,planned),1):
        require(np.all(case['arrays'][-1]!=case['expected']),'OUTPUT_SENTINEL_MISSING')
        for b,a in zip(buffers,case['arrays']):b.copyfrom(a)
        clear()
        write(out/f'call_{index:02d}_begin.json',expected_row)
        dump,execution=capture_helper_call(index,
            lambda:diagnostic(lib,str(function.attrs['global_symbol']),*buffers),
            counts,metadata,ledger_func,
            lambda value:write(out/f'call_{index:02d}_execution.json',value))
        (out/f'native_call_{index:02d}.txt').write_text(dump)
        validate_helper_completion(execution,index)
        submits,meta=execution['submits'],execution['metadata']
        require(submits['recorded_submits']==submits['attempted_submits']==1,'UNEXPECTED_MULTI_SUBMISSION')
        actual=buffers[-1].numpy()
        require(np.array_equal(actual,case['expected']),'FULL_FUSION_MISMATCH')
        require(all(np.array_equal(b.numpy(),a) for b,a in zip(buffers[:-1],case['arrays'][:-1])),'READONLY_ARGUMENT_CHANGED')
        require(handles==[int(ctypes.cast(b.handle,ctypes.c_void_p).value) for b in buffers] and
                storage==[int(b.handle.contents.data) for b in buffers],'PARAMETER_STORAGE_CHANGED')
        collected=json.loads(collection())
        require(collected['mode']==mode and collected['debug_flag']==0,'MODE_OR_DEBUG_CHANGED')
        require(mode!='minimal' or collected['detailed_clock_reads']==0,'MINIMAL_DETAIL_CLOCK_READS')
        native=certify_native(events,dump,submits,definitions);definitions=native['cache_definitions']
        if backend_bases is None:backend_bases=native['tensor_domain_dram_bases']
        require(native['tensor_domain_dram_bases']==backend_bases,'ACTUAL_BACKEND_STORAGE_CHANGED')
        require(json.loads(replay())['mode']=='disabled' and json.loads(capacity())['replay_policy']=='disabled',
                'REPLAY_CHANGED_DURING_QUALIFICATION')
        require(certify_trace(events,ancestors=native['ancestors'])['output_terms']==sem['output_terms'],'ACTUAL_DEPENDENCY_SEMANTICS_CHANGED')
        blocks=decode_submissions(dump);require(len(blocks)==1,'DUMP_SUBMISSION_CHANGED');records=blocks[0]
        progress=certify_progress(records,command_depths,token_depths)
        require(progress['status']=='FINITE_FIFO_PROGRESS_PROVED','FINITE_FIFO_UNRESOLVED:'+progress['status'])
        physical=analyze(records,native)
        if program=='S':
            before=root/f'R_{mode}'
            original_records=decode_submissions((before/f'native_call_{index:02d}.txt').read_text())[0]
            original_native=read(before/f'native_certificate_{index:02d}.json')
            joint=certify_joint(original_records,original_native,records,native,command_depths,token_depths)
            write(out/f'joint_certificate_{index:02d}.json',joint)
        write(out/f'native_certificate_{index:02d}.json',native_certificate_for_storage(native))
        write(out/f'physical_certificate_{index:02d}.json',physical)
        write(out/f'progress_{index:02d}.json',progress)
        row={**expected_row,'status':'PASS','correct':True,'helper_metadata':meta,
             'submit_ledger':submits,'buffer_handles':handles,'actual_storage':storage,
             'storage_pointer_meaning':'DLTensor.data DataBuffer handle; backend addresses separately bound by decoded requests',
             'backend_tensor_domain_dram_bases':backend_bases,
             'input_sha256':[array_sha(a) for a in case['arrays'][:-1]],
             'output_sha256':array_sha(actual),'reference_sha256':array_sha(case['expected']),
             'native_dump_sha256':sha(out/f'native_call_{index:02d}.txt')}
        if case['kind']=='directed':
            row['output_distribution']={'zero':int(np.count_nonzero(actual==0)),
                'clip_endpoints':int(np.count_nonzero((actual==127)|(actual==-127))),
                'nonzero_unsaturated':int(np.count_nonzero((actual!=0)&(actual!=127)&(actual!=-127))),
                'total':int(actual.size),'directed_scale':case['directed_scale']}
        write(out/f'call_{index:02d}_end.json',row);rows.append(row)
        print(json.dumps({'program':program,'mode':mode,'complete_call':index,'status':'PASS'}),flush=True)
    validate_calls(planned,rows);verify(read(root/'freeze.json')['bindings']);verify(d['artifact_bindings'])
    require(helper is not None,'HELPER_LIFETIME')
    write(out/'summary.json',{'status':'FULL_LOCAL_QUALIFICATION_PASSED','program':program,'mode':mode,
        'calls':rows,'target_calls':7,'module_sha256':sha(d['module_path']),
        'runtime_sha256':sha(actual_runtime),'helper_sha256':sha(HELPER),'performance_conclusion':None})
