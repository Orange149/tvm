"""Whole YOLO416 FSim qualification with finite action graph progress proof, with no board/performance entry point.

Two variants, two modes, six complete calls each. Reference CPU functions
execute individually in a separate byte-identical library; VTA convolutions
use independent NumPy mathematics. They never invoke GraphExecutor.run().
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
from c3_tps_sync_release_local import (HERE,REPO,PYTHON,RUNTIME,read,write,sha,verify,require,
                                     env_for_worker,capture_helper_call)


SEEDS=(0,20250901,20260910)


def ledger():
    cases=[('three_seed',s) for s in SEEDS]+[(s,0) for s in ('W0','W1','W0_return')]
    return [{'variant':v,'mode':m,'call_index':i,'kind':k,'seed':s,'GraphExecutor_calls':1,
             'performance_sample':False} for v in ('TH','P') for m in ('detailed','minimal')
            for i,(k,s) in enumerate(cases,1)]


def helper_gate(execution,index):
    meta=execution.get('metadata',{})
    require(all(type(v) is int for v in (execution.get('counts'),meta.get('started_calls'),
                meta.get('completed_calls'),meta.get('parameter_count'),meta.get('packed_calls_per_invocation'))),
            'GRAPH_HELPER_COUNTER_TYPES')
    require(execution.get('helper_returned') is True and execution.get('counts')==index and
            meta.get('started_calls')==meta.get('completed_calls')==index,'GRAPH_HIDDEN_OR_INCOMPLETE_CALL')
    require(meta.get('schema')=='c3_yolo416_graph_once_v1' and meta.get('last_kind')=='diagnostic'
            and meta.get('parameter_count')==0 and meta.get('packed_calls_per_invocation')==1
            and meta.get('adds_synchronize') is False and not execution.get('observation_errors'),
            'GRAPH_HELPER_CONTRACT_CHANGED')


def recording_specs(specs,records):
    class Recording:
        def __init__(self,symbol,spec):self.symbol,self.spec,self.abi=symbol,spec,spec.abi
        def reference(self,inputs):
            result=self.spec.reference(inputs)
            records.append({'symbol':self.symbol,'inputs':[x.copy() for x in inputs],'output':result.copy()})
            return result
    return {s:Recording(s,p) for s,p in specs.items()}


def make_cases(graph,params,specs,cpu):
    import numpy as np
    from c3_yolo416_graph_reference import evaluate
    rowptr=graph['node_row_ptr'];shape=graph['attrs']['shape'][1]
    input_nodes=[i for i,n in enumerate(graph['nodes']) if n['op']=='null' and n['name']=='data']
    require(len(input_nodes)==1 and shape[rowptr[input_nodes[0]]]==[1,3,416,416],'GRAPH_DATA_INPUT_CHANGED')
    first=next(n for n in graph['nodes'] if n.get('attrs',{}).get('func_name') in specs)
    parent,output,version=first['inputs'][1]
    require(version==output==0 and graph['nodes'][parent]['op']=='null','GRAPH_WEIGHT_NOT_LONG_PARAMETER')
    weight_name=graph['nodes'][parent]['name'];w0=params[weight_name]

    def golden(seed,weight):
        data=np.random.default_rng(seed).uniform(0,1,size=(1,3,416,416)).astype('float32')
        p=dict(params);p[weight_name]=weight;records=[]
        heads,audit=evaluate(graph,p,{'data':data},recording_specs(specs,records),cpu)
        require(len(heads)==8 and len(records)==12 and all(np.isfinite(x).all() for x in heads),
                'GRAPH_REFERENCE_COVERAGE_OR_FINITE_VALUES')
        return {'seed':seed,'data':data,'weight':weight,'heads':heads,'convolutions':records,
                'reference_audit':audit}

    seeded=[golden(s,w0) for s in SEEDS];base=seeded[0];changed=None;choice=None
    candidates=[np.negative(w0.astype(np.int16)).clip(-128,127).astype('int8'),
                np.full_like(w0,127),np.full_like(w0,-128)]
    for i,weight in enumerate(candidates):
        if np.array_equal(weight,w0):continue
        proposal=golden(0,weight)
        if any(not np.array_equal(a,b) for a,b in zip(proposal['heads'],base['heads'])):
            changed,choice=proposal,i;break
    require(changed is not None,'GRAPH_W0_W1_FINAL_OUTPUTS_NOT_OBSERVABLY_DISTINCT')
    return [*seeded,base,changed,base],weight_name,{'candidate_index':choice,
              'selection_uses_target_or_timing':False,'comparison':'at least one raw final output differs'}


def identity(array):
    return {'handle':int(ctypes.cast(array.handle,ctypes.c_void_p).value),
            'data_handle':int(array.handle.contents.data),'byte_offset':int(array.handle.contents.byte_offset)}


def qualify_worker(root,variant,mode):
    os.environ['C3_OFFICIAL_WINDOW_COLLECTION']=mode
    import numpy as np
    import tvm,vta
    from tvm import relay
    from tvm.contrib import graph_executor
    from vta.testing import simulator
    from c3_official_window_reference import from_workload,array_sha
    from c3_yolo416_graph_cpu_reference import CPUCallback
    from c3_yolo416_graph_trace import storage_bindings
    from c3_tps_window_native import certify_native
    from c3_yolo416_fifo_certificate import certify_action_graph as certify_progress
    from c3_tps_window_progress import LANES
    from c3_five_tradeoff_commands import decode_submissions
    from c3_official_window_workflow import native_certificate_for_storage
    verify(read(root/'freeze.json')['bindings'])
    require(Path(simulator.LIBS[0]._name).resolve()==(RUNTIME/'libvta_fsim.so').resolve(),'GRAPH_RUNTIME_CHANGED')
    capacity=json.loads(tvm.get_global_func('vta.runtime.queue_capacity_status')())
    replay=json.loads(tvm.get_global_func('vta.runtime.replay_status')())
    require(all(capacity[k]==33554432 for k in ('insn_capacity_bytes','uop_capacity_bytes',
            'insn_submit_threshold_bytes')) and capacity['submissions']==0 and
            capacity['replay_policy']=='disabled' and replay['mode']=='disabled' and
            replay['template_count']==0,'GRAPH_RUNTIME_QUEUE_OR_REPLAY_CHANGED')
    contract=read(root/'contract.json');prepared=Path(contract['prepared']);compiled=Path(contract['compiled'])
    qualified=Path(contract['operators']);source=prepared/variant
    out=root/(variant+'_'+mode);out.mkdir(exist_ok=False)
    graph=read(source/'graph.json');mapping=read(source/'namespace.json');events=read(source/'events.json')
    params={str(k):v.numpy() for k,v in relay.load_param_dict((source/'params.bin').read_bytes()).items()}
    functions=read(compiled/variant/'functions.json');specs={}
    units=read(qualified/'objects.json')
    for row in functions:
        index=next(i for i,u in enumerate(units) if u['variant']==variant and u['symbol']==row['symbol'])
        reference=read(qualified/'objects'/('%02d'%index)/'prepare/R/reference_contract.json')
        spec=from_workload(row['workload'],row['abi'],reference['postprocess'])
        require(spec.contract()==reference,'GRAPH_OPERATOR_REFERENCE_CHANGED');specs[row['symbol']]=spec
    reference_lib=tvm.runtime.load_module(str(source/'reference_fsim.so'))
    cpu=CPUCallback(reference_lib,read(source/'CPU_manifest.json'))
    cases,weight_name,observable=make_cases(graph,params,specs,cpu)
    write(out/'reference_protocol.json',{'CPU_function_calls':cpu.calls,'weight_parameter':weight_name,
          'observability':observable,'GraphExecutor_reference_calls':0,
          'reference_library':str(source/'reference_fsim.so'),'trusted_CPU_code':True,
          'independent_integer_convolutions':True})
    lib=tvm.runtime.load_module(str(source/'graph_fsim.so'))
    once=tvm.runtime.load_module(str(prepared/'graph_once_fsim.so'))
    observer_lib=None;proxy=None;module=lib
    if mode=='detailed':
        observer_lib=tvm.runtime.load_module(str(prepared/'graph_observer_fsim.so'))
        text='\n'.join('%d %s 3 1'%(r['node'],r['symbol']) for r in mapping['calls'])
        proxy=tvm.get_global_func('c3.yolo416.observe_module')(lib,text);module=proxy
    executor=graph_executor.create((source/'graph.json').read_text(),module,[tvm.device('ext_dev',0),tvm.cpu(0)])
    executor.load_params((source/'params.bin').read_bytes())
    data=executor.get_input('data');parameters={name:executor.get_input(name) for name in params}
    buffers={'data':data,**parameters};storage={k:identity(v) for k,v in buffers.items()}
    outputs=[executor.get_output(i) for i in range(8)]
    rowptr=graph['node_row_ptr'];sids=graph['attrs']['storage_id'][1]
    null_sids={sids[rowptr[i]] for i,n in enumerate(graph['nodes']) if n['op']=='null'}
    for i,expected_shape in ((0,[1,255,26,26]),(4,[1,255,13,13])):
        nid,nout,version=graph['heads'][i];entry=rowptr[nid]+nout
        require(version==0 and graph['attrs']['shape'][1][entry]==expected_shape and
                graph['attrs']['dltype'][1][entry]=='float32' and sids[entry] not in null_sids,
                'GRAPH_MUTABLE_PREDICTION_HEAD_ALIAS_OR_ABI')
    metadata=tvm.get_global_func('c3.yolo416.graph_metadata')
    tvm.get_global_func('c3.yolo416.graph_reset_counts')()
    clear=tvm.get_global_func('vta.runtime.official_submit_ledger_clear')
    submit=tvm.get_global_func('vta.runtime.official_submit_ledger')
    collection=tvm.get_global_func('vta.runtime.official_collection_status')
    require(json.loads(collection())['mode']==mode,'GRAPH_COLLECTION_MODE_CHANGED')
    require(json.loads(submit())['attempted_submits']==0,'REFERENCE_EXECUTED_UNEXPECTED_SUBMISSION')
    hardware=read(root/'hardware_archive.json')['fifo_depths']
    command_depths={i:hardware[k] for i,k in enumerate(('load_queue','gemm_queue','store_queue'),1)}
    token_depths=dict(zip(LANES,[hardware[k] for k in ('l2g_queue','g2l_queue','g2s_queue','s2g_queue')]))
    planned=[r for r in read(root/'call_ledger.json') if r['variant']==variant and r['mode']==mode]
    require(len(planned)==len(cases)==6,'GRAPH_CALL_BUDGET_CHANGED')
    definitions=None;native_bases=None;rows=[]
    for index,(case,expected) in enumerate(zip(cases,planned),1):
        data.copyfrom(case['data']);parameters[weight_name].copyfrom(case['weight'])
        for i in (0,4):outputs[i].copyfrom(np.full_like(case['heads'][i],np.nan))
        clear()
        if proxy is not None:proxy['c3_yolo416_observer_begin']()
        write(out/('call_%02d_begin.json'%index),expected)
        dump,execution=capture_helper_call(index,
             lambda:tvm.get_global_func('c3.yolo416.graph_diagnostic_once')(executor.module),
             lambda:json.loads(metadata())['completed_calls'],metadata,submit,
             lambda value:write(out/('call_%02d_execution.json'%index),value))
        (out/('native_call_%02d.txt'%index)).write_text(dump)
        helper_gate(execution,index)
        actual_heads=[v.numpy() for v in outputs]
        require(all(np.array_equal(a,b) for a,b in zip(actual_heads,case['heads'])),'GRAPH_RAW_HEAD_OUTPUT_MISMATCH')
        require(np.array_equal(data.numpy(),case['data']),'GRAPH_DATA_INPUT_MODIFIED')
        for name,array in parameters.items():
            wanted=case['weight'] if name==weight_name else params[name]
            require(np.array_equal(array.numpy(),wanted),'GRAPH_READONLY_PARAMETER_MODIFIED:'+name)
        require({k:identity(v) for k,v in buffers.items()}==storage,'GRAPH_LONG_PARAMETER_STORAGE_CHANGED')
        observed=None;snapshot_rows=[]
        if proxy is not None:
            observed=proxy['c3_yolo416_observer_end']()
            require(len(observed)==len(case['convolutions'])==12,'GRAPH_SNAPSHOT_CENSUS')
            for capture,golden in zip(observed,case['convolutions']):
                require(str(capture['symbol'])==golden['symbol'],'GRAPH_SNAPSHOT_FUNCTION_ORDER')
                for before,after,wanted in zip(capture['inputs_before'],capture['inputs_after'],golden['inputs']):
                    require(np.array_equal(before.numpy(),wanted) and np.array_equal(after.numpy(),wanted),
                            'GRAPH_ACTUAL_VTA_INPUT_OR_READONLY_MISMATCH')
                require(len(capture['outputs'])==1 and np.array_equal(capture['outputs'][0].numpy(),golden['output']),
                        'GRAPH_INTERMEDIATE_INTEGER_OUTPUT_MISMATCH')
                snapshot_rows.append({'node':int(str(capture['node'])),'symbol':golden['symbol'],
                    'input_sha256':[array_sha(x) for x in golden['inputs']],
                    'output_sha256':array_sha(golden['output']),'all_elements_compared':True})
        native=certify_native(events,dump,execution['submits'],definitions);definitions=native['cache_definitions']
        require(native['submission_count']==12,'GRAPH_UNEXPECTED_SUBMISSION_PATH')
        binding=storage_bindings(graph,mapping,native,observed)
        if native_bases is None:native_bases=native['tensor_domain_dram_bases']
        require(native_bases==native['tensor_domain_dram_bases'],'GRAPH_NATIVE_STORAGE_CHANGED_ACROSS_CALLS')
        blocks=decode_submissions(dump);require(len(blocks)==12,'GRAPH_DUMP_SUBMISSION_CENSUS')
        progress=[certify_progress(r,command_depths,token_depths) for r in blocks]
        require(all(p['status']=='FINITE_FIFO_PROGRESS_PROVED' for p in progress),'GRAPH_FINITE_FIFO_UNRESOLVED')
        state=json.loads(collection());require(state['mode']==mode and state['debug_flag']==0,'GRAPH_MODE_CHANGED')
        require(mode!='minimal' or state['detailed_clock_reads']==0,'GRAPH_MINIMAL_DETAIL_CLOCK_READS')
        write(out/('native_certificate_%02d.json'%index),native_certificate_for_storage(native))
        write(out/('storage_binding_%02d.json'%index),binding)
        write(out/('progress_%02d.json'%index),{'fragments':progress,
             'cross_fragment_relation':'actual successful blocking Sync; equal producer/consumer counts leave terminal tokens empty',
             'SRAM_semantics':'closed function qualifications; no assumption that ACC contents are cleared at submission boundaries'})
        write(out/('intermediate_comparison_%02d.json'%index),snapshot_rows)
        np.savez_compressed(out/('raw_outputs_%02d.npz'%index),**{'head_%d'%i:a for i,a in enumerate(actual_heads)})
        row={**expected,'status':'PASS','correct':True,'helper':execution,'storage':storage,
             'head_sha256':[array_sha(a) for a in actual_heads],'all_8_raw_heads_elementwise_equal':True,
             'intermediate_integer_comparisons':12 if proxy is not None else 0,
             'native_submit_count':12,'native_dump_sha256':sha(out/('native_call_%02d.txt'%index))}
        write(out/('call_%02d_end.json'%index),row);rows.append(row)
        print(json.dumps({'variant':variant,'mode':mode,'complete_graph_call':index,'status':'PASS'}),flush=True)
    verify(read(root/'freeze.json')['bindings'])
    write(out/'summary.json',{'status':'WHOLE_GRAPH_LOCAL_QUALIFICATION_PASSED','calls':rows,
          'GraphExecutor_calls':6,'FPGA_calls':0,'performance_conclusion':None,
          'reference_CPU_component_is_shared_trusted_code':True,'observer_used':proxy is not None})
    require(once is not None and lib is not None and reference_lib is not None,'GRAPH_LIBRARY_LIFETIME')


def mode_gate(root,variant):
    from c3_five_tradeoff_commands import decode_submissions
    from c3_tps_notification_probe import canonical_payload
    comparisons=[]
    for i in range(1,7):
        modes=[]
        for mode in ('detailed','minimal'):
            folder=root/(variant+'_'+mode)
            blocks=decode_submissions((folder/('native_call_%02d.txt'%i)).read_text())
            modes.append([[(r['stage'],r['bits'],canonical_payload(r)) for r in b] for b in blocks])
        require(modes[0]==modes[1],'GRAPH_MODE_CHANGES_ACTUAL_INSTRUCTION_PATH')
        a=read(root/(variant+'_detailed')/('call_%02d_end.json'%i))
        b=read(root/(variant+'_minimal')/('call_%02d_end.json'%i))
        require(a['head_sha256']==b['head_sha256'],'GRAPH_MODE_CHANGES_RAW_OUTPUTS')
        comparisons.append({'call_index':i,'all_encoded_fields_equal':True,'all_raw_outputs_equal':True})
    write(root/(variant+'_mode_equivalence.json'),{'status':'GRAPH_MODE_INSTRUCTION_AND_OUTPUTS_EQUAL',
          'calls':comparisons,'formal_path_uses_no_snapshot_proxy':True})


def accounting(root):
    records=[];started=completed=0;unknown=[]
    for variant in ('TH','P'):
        for mode in ('detailed','minimal'):
            folder=root/(variant+'_'+mode);s=c=0;uncertain=False
            for p in sorted(folder.glob('call_*_begin.json')):
                row=read(p);i=row['call_index'];ep=folder/('call_%02d_execution.json'%i)
                end=folder/('call_%02d_end.json'%i);e=read(ep) if ep.exists() else {}
                meta=e.get('metadata',{})
                if type(meta.get('started_calls')) is int:s=max(s,meta['started_calls'])
                if type(meta.get('completed_calls')) is int:c=max(c,meta['completed_calls'])
                uncertain |= e.get('helper_returned') is not True
                done=read(end) if end.exists() else {}
                records.append({**row,'helper_returned':e.get('helper_returned'),
                    'certified':done.get('status')=='PASS' and done.get('correct') is True,
                    'execution_evidence':str(ep) if ep.exists() else None})
            started+=s;completed+=c
            if uncertain:unknown.append(variant+'_'+mode)
    return {'registered':len(records),'helper_started_observed_lower_bound':started,
            'helper_completed_observed_lower_bound':completed,
            'all_certificates_passed':sum(r['certified'] for r in records),
            'unresolved_workers':unknown,'calls':records,'FPGA_calls':0,'performance_samples':0}


def run_worker(root,variant,mode):
    import psutil
    env=env_for_worker()
    for k in ('VTA_UOP_BUFFER_BYTES','VTA_INSN_BUFFER_BYTES','VTA_INSN_SUBMIT_THRESHOLD_BYTES'):
        env[k]='33554432'
    args=[str(PYTHON),str(Path(__file__).resolve()),'_worker',str(root),'--variant',variant,'--mode',mode]
    start=time.monotonic();peak=0;limit=None
    with (root/(variant+'_'+mode+'.log')).open('x') as log:
        child=subprocess.Popen(args,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while child.poll() is None:
            try:
                p=psutil.Process(child.pid)
                peak=max(peak,sum(x.memory_info().rss for x in [p,*p.children(recursive=True)]))
            except psutil.Error:pass
            if peak>8*1024**3 or time.monotonic()-start>600:
                limit='MEMORY' if peak>8*1024**3 else 'TIME';os.killpg(child.pid,signal.SIGKILL);break
            time.sleep(.1)
        code=child.wait()
    result={'returncode':code,'resource_limit':limit,'sampled_peak_tree_RSS':peak,
            'resource_control_seconds':time.monotonic()-start,'argv':args}
    write(root/(variant+'_'+mode+'_worker.json'),result)
    return result


def run(root,prepared):
    root.mkdir(exist_ok=False);status='STOPPED_GRAPH_INPUT_IDENTITY'
    try:
        with (root/'protocol_tests.log').open('x') as log:
            test=subprocess.run([str(PYTHON),'-m','unittest','test_c3_yolo416_graph_local','-v'],
                                env=env_for_worker(),stdout=log,stderr=subprocess.STDOUT)
        require(test.returncode==0,'GRAPH_PROTOCOL_TESTS_FAILED')
        seal=read(prepared/'completion_manifest.json')
        verify({str(prepared/p):v for p,v in seal['artifact_sha256'].items()})
        require(read(prepared/'status.json')['status']==
                'WHOLE_TH_P_GRAPHS_AND_HELPER_BUILT_LOCAL_QUALIFICATION_PENDING','GRAPH_PREPARATION_INCOMPLETE')
        base=read(prepared/'contract.json');qualified=Path(base['operator_qualification'])
        bindings=read(prepared/'freeze.json')['bindings'].copy()
        for p in HERE.glob('c3_*.py'):bindings[str(p.resolve())]=sha(p)
        for p in HERE.glob('test_c3_yolo416_graph*.py'):bindings[str(p.resolve())]=sha(p)
        bindings[str(root/'protocol_tests.log')]=sha(root/'protocol_tests.log')
        bindings.update({str(prepared/p):v for p,v in seal['artifact_sha256'].items()})
        verify(bindings)
        write(root/'freeze.json',{'bindings':bindings,'max_GraphExecutor_calls':24,
              'worker_GiB':8,'worker_seconds':600,'max_submissions_per_graph_call':16,
              'max_vector_steps_per_closed_function':5000000,'FPGA_calls':0})
        write(root/'contract.json',{'prepared':str(prepared),'operators':str(qualified),
              'compiled':base['compiled_source'],'variants':['TH','P'],'I_equals_P':True,
              'graph_cases_per_mode':6,'total_graph_calls':24,
              'seeds':list(SEEDS),'reference_CPU_is_trusted_common_code':True,
              'modes':{'detailed':'snapshot proxy and native stream','minimal':'native stream, no snapshot proxy'},
              'stop':'first failure; no retry, patch, added candidate, sample replacement or board timing'})
        shutil.copyfile(qualified/'hardware_archive.json',root/'hardware_archive.json')
        write(root/'call_ledger.json',ledger());status='STOPPED_GRAPH_LOCAL_QUALIFICATION'
        for variant in ('TH','P'):
            good=True
            for mode in ('detailed','minimal'):
                result=run_worker(root,variant,mode)
                if result['returncode'] or result['resource_limit']:
                    write(root/'stop.json',{'variant':variant,'mode':mode,'worker':result});good=False;break
            if not good:break
            mode_gate(root,variant)
        else:status='WHOLE_TH_P_GRAPH_LOCAL_QUALIFIED_ARM_AND_FPGA_QUALIFICATION_PENDING'
    except Exception as e:
        write(root/'failure_controller.json',{'type':type(e).__name__,'message':str(e)});traceback.print_exc()
    write(root/'call_accounting.json',accounting(root))
    write(root/'status.json',{'status':status,'FPGA_calls':0,'performance_conclusion':None})
    print(json.dumps({'status':status,**{k:v for k,v in accounting(root).items() if k!='calls'}}),flush=True)
    return 0 if status.startswith('WHOLE_TH_P_GRAPH_LOCAL_QUALIFIED') else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('run','_worker'));p.add_argument('output',type=Path)
    p.add_argument('--prepared',type=Path);p.add_argument('--variant',choices=('TH','P'))
    p.add_argument('--mode',choices=('detailed','minimal'));a=p.parse_args();root=a.output.resolve()
    if a.action=='run':sys.exit(run(root,a.prepared.resolve()))
    try:qualify_worker(root,a.variant,a.mode)
    except Exception as e:
        write(root/('failure_'+a.variant+'_'+a.mode+'.json'),{'type':type(e).__name__,'message':str(e)})
        traceback.print_exc();sys.exit(1)

