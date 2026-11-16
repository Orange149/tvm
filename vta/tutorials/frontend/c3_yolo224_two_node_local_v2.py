"""Complete fixed graph qualification: B/A, detailed/minimal, six calls each.

Only two graph functions are routed. Preserve all legacy Syncs and serial
settings, certify all 22 submissions, and compare all integer intermediates
and eight outputs to an independent-convolution graph reference.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

from c3_tps_sync_release_local import HERE, HOLD, PYTHON, read, write, sha, verify, require, env_for_worker, capture_helper_call
from c3_yolo224_two_node_prepare import SYMBOLS, ENTRY

SEEDS = (0, 20250901, 20260910)


def make_cases(graph, params, specs, cpu):
    import numpy as np
    from c3_yolo416_graph_reference import evaluate
    from c3_yolo416_graph_local_v2 import recording_specs
    data_node = next(i for i,n in enumerate(graph['nodes']) if n['op']=='null' and n['name']=='data')
    shape = graph['attrs']['shape'][1][graph['node_row_ptr'][data_node]]
    require(shape == [1,3,224,224], 'FIXED_YOLO224_INPUT')
    first = next(n for n in graph['nodes'] if n.get('attrs',{}).get('func_name') in specs)
    parent, output, version = first['inputs'][1]
    require(output == version == 0 and graph['nodes'][parent]['op'] == 'null', 'WEIGHT_PARAMETER_BINDING')
    weight_name = graph['nodes'][parent]['name'];w0 = params[weight_name]
    def golden(seed, weight):
        data = np.random.default_rng(seed).uniform(0,1,size=shape).astype('float32')
        parameters = {**params, weight_name:weight};records = []
        heads,audit = evaluate(graph,parameters,{'data':data},recording_specs(specs,records),cpu)
        require(len(heads)==8 and len(records)==12 and all(np.isfinite(a).all() for a in heads), 'REFERENCE_COVERAGE')
        return {'seed':seed,'data':data,'weight':weight,'heads':heads,'convolutions':records,'reference_audit':audit}
    seeded = [golden(s,w0) for s in SEEDS];base=seeded[0];changed=None;choice=None
    candidates=[np.negative(w0.astype(np.int16)).clip(-128,127).astype('int8'),
                np.full_like(w0,127),np.full_like(w0,-128)]
    for i,weight in enumerate(candidates):
        if np.array_equal(weight,w0):continue
        proposal=golden(0,weight)
        if any(not np.array_equal(a,b) for a,b in zip(proposal['heads'],base['heads'])):
            changed,choice=proposal,i;break
    require(changed is not None,'WEIGHT_ROUNDTRIP_NOT_OBSERVABLE_AT_FINAL_OUTPUT')
    return [*seeded,base,changed,base],weight_name,{'candidate_index':choice,'selection_uses_target_or_timing':False}


def traces(prepared, variant):
    import tvm,vta
    from c3_yolo224_sync_structure import project
    functions=read(prepared/'functions.json');replacement=read(prepared/'replacements.json')[variant]
    result={}
    for symbol,row in functions.items():
        node=next((n for n,s in SYMBOLS.items() if s==symbol),None)
        if node is None:
            fn=tvm.ir.load_json(Path(row['source_TIR']).read_text())
        else:
            parent=tvm.ir.load_json((Path(replacement[str(node)]['qualified_directory'])/'parent.json').read_text())
            fn=list(parent.functions.values())[0]
        events,_,flags=project(fn)
        require(flags==row['runtime_flags'],'RUNTIME_SETTINGS_CHANGED')
        result[symbol]=events
    return result


def make_route(prepared, variant, base, replacement_modules, diagnostic, get):
    functions=read(prepared/'functions.json')
    return get('c3.yolo224.route_module')(base,replacement_modules[34],replacement_modules[46],
        SYMBOLS[34],SYMBOLS[46],ENTRY,'\n'.join(functions),diagnostic)


def worker(root, variant, mode):
    import ctypes
    import numpy as np
    import tvm,vta
    from tvm import relay
    from tvm.contrib import graph_executor
    from vta.testing import simulator
    from c3_yolo416_graph_cpu_reference import CPUCallback
    from c3_official_window_reference import from_workload,array_sha
    from c3_yolo416_graph_local_v2 import helper_gate,identity
    from c3_yolo224_two_node_native_v2 import certify_graph,submit_signature
    from c3_tps_window_progress import LANES
    verify(read(root/'freeze.json')['bindings'])
    prepared=Path(read(root/'contract.json')['prepared'])
    actual=Path(simulator.LIBS[0]._name).resolve()
    require(actual==(prepared/'runtime_sim/libvta_fsim.so').resolve(),'WRONG_GRAPH_RUNTIME')
    require(hasattr(ctypes.CDLL(None),'C3Yolo224SetGraphDump'),'GRAPH_CAPTURE_SYMBOL_MISSING')
    out=root/(variant+'_'+mode);out.mkdir()
    graph=read(prepared/'graph.json');functions=read(prepared/'functions.json')
    params={str(k):v.numpy() for k,v in relay.load_param_dict((prepared/'params.bin').read_bytes()).items()}
    specs={s:from_workload(row['workload'],row['abi'],row['reference']['postprocess']) for s,row in functions.items()}
    for s,spec in specs.items():require(spec.contract()==functions[s]['reference'],'REFERENCE_CONTRACT_CHANGED')
    base=tvm.runtime.load_module(str(prepared/'base_fsim.so'))
    cpu=CPUCallback(base,read(prepared/'CPU_manifest.json'))
    cases,weight_name,observable=make_cases(graph,params,specs,cpu)
    write(out/'reference_protocol.json',{'weight_parameter':weight_name,'observability':observable,
          'CPU_function_calls':len(cpu.calls),'GraphExecutor_reference_calls':0})
    replacements=read(prepared/'replacements.json')[variant]
    modules={n:tvm.runtime.load_module(replacements[str(n)]['fsim_module']) for n in (34,46)}
    route_helper=tvm.runtime.load_module(str(prepared/'route_fsim.so'))
    once=tvm.runtime.load_module(str(prepared/'graph_once_fsim.so'))
    observer=tvm.runtime.load_module(str(HOLD/'20261017_c3_yolo416_graph_prepare_run02/graph_observer_fsim.so'))
    routed=make_route(prepared,variant,base,modules,True,tvm.get_global_func)
    calls=[(i,n['attrs']['func_name']) for i,n in enumerate(graph['nodes']) if n.get('attrs',{}).get('func_name') in functions]
    require(len(calls)==12,'GRAPH_VTA_CALL_COVERAGE')
    proxy=tvm.get_global_func('c3.yolo416.observe_module')(routed,'\n'.join('%d %s 3 1'%x for x in calls))
    executor=graph_executor.create((prepared/'graph.json').read_text(),proxy,[tvm.device('ext_dev',0),tvm.cpu(0)])
    executor.load_params((prepared/'params.bin').read_bytes())
    data=executor.get_input('data');parameters={name:executor.get_input(name) for name in params}
    storage={k:identity(a) for k,a in {'data':data,**parameters}.items()}
    outputs=[executor.get_output(i) for i in range(8)]
    metadata=tvm.get_global_func('c3.yolo416.graph_metadata');tvm.get_global_func('c3.yolo416.graph_reset_counts')()
    clear=tvm.get_global_func('vta.runtime.official_submit_ledger_clear');ledger=tvm.get_global_func('vta.runtime.official_submit_ledger')
    require(json.loads(ledger())['attempted_submits']==0,'REFERENCE_ENQUEUED_VTA')
    h=read(prepared/'hardware_archive.json')['fifo_depths']
    commands={i:h[k] for i,k in enumerate(('load_queue','gemm_queue','store_queue'),1)}
    tokens=dict(zip(LANES,[h[k] for k in ('l2g_queue','g2l_queue','g2s_queue','s2g_queue')]))
    source_traces=traces(prepared,variant);definitions=None;rows=[]
    planned=[r for r in read(root/'call_ledger.json') if r['variant']==variant and r['mode']==mode]
    for index,(case,row) in enumerate(zip(cases,planned),1):
        data.copyfrom(case['data']);parameters[weight_name].copyfrom(case['weight'])
        for i in (0,4):outputs[i].copyfrom(np.full_like(case['heads'][i],np.nan))
        clear();proxy['c3_yolo416_observer_begin']();write(out/('call_%02d_begin.json'%index),row)
        text,execution=capture_helper_call(index,
            lambda:tvm.get_global_func('c3.yolo416.graph_diagnostic_once')(executor.module),
            lambda:json.loads(metadata())['completed_calls'],metadata,ledger,
            lambda value:write(out/('call_%02d_execution.json'%index),value))
        (out/('native_%02d.txt'%index)).write_text(text)
        helper_gate(execution,index);signature=submit_signature(execution['submits'],mode)
        require(len(signature)==22,'ACTUAL_FULL_GRAPH_SUBMISSIONS')
        require(all(np.array_equal(a.numpy(),b) for a,b in zip(outputs,case['heads'])),'GRAPH_FINAL_OUTPUT_MISMATCH')
        require(np.array_equal(data.numpy(),case['data']),'GRAPH_INPUT_CHANGED')
        require(all(np.array_equal(a.numpy(),case['weight'] if name==weight_name else params[name]) for name,a in parameters.items()),
                'GRAPH_READONLY_PARAMETER_CHANGED')
        require(storage=={k:identity(a) for k,a in {'data':data,**parameters}.items()},'GRAPH_PARAMETER_STORAGE_CHANGED')
        observed=proxy['c3_yolo416_observer_end']();require(len(observed)==12,'INTEGER_OBSERVER_COVERAGE')
        for actual_row,golden in zip(observed,case['convolutions']):
            require(str(actual_row['symbol'])==golden['symbol'],'INTEGER_CALL_ORDER_CHANGED')
            for group in ('inputs_before','inputs_after'):
                require(all(np.array_equal(a.numpy(),b) for a,b in zip(actual_row[group],golden['inputs'])), 'INTEGER_INPUT_MISMATCH')
            require(np.array_equal(actual_row['outputs'][0].numpy(),golden['output']),'INTEGER_OUTPUT_MISMATCH')
        certificate,definitions=certify_graph(graph,functions,source_traces,text,execution['submits'],definitions,commands,tokens)
        # Cache definitions and large ancestor bitsets stay in memory only;
        # persist the actual UOP checks and every submission proof separately.
        for call in certificate['calls']:
            for fragment in call['submissions']:
                if fragment['kind']=='material':
                    native=fragment['native'];native.pop('cache_definitions',None)
                    native['ancestors']=[hex(x) for x in native['ancestors']]
        write(out/('certificate_%02d.json'%index),certificate)
        end={**row,'status':'PASS','helper':execution,'all_8_heads_correct':True,
             'all_12_integer_intermediates_correct':True,'head_sha256':[array_sha(a) for a in case['heads']],
             'input_sha256':array_sha(case['data']),'signature':signature,'weight_parameter':weight_name}
        write(out/('call_%02d_end.json'%index),end);rows.append(end)
        print(json.dumps({'variant':variant,'mode':mode,'completed_graph_calls':index,'status':'PASS'}),flush=True)
    require(len(rows)==6,'LOCAL_CALL_BUDGET')
    verify(read(root/'freeze.json')['bindings'])
    write(out/'summary.json',{'status':'SIX_COMPLETE_GRAPH_CALLS_CERTIFIED','calls':rows})
    require(route_helper is not None and once is not None and observer is not None,'PLUGIN_LIFETIME')


def run(root,prepared):
    root.mkdir(exist_ok=False);status='STOPPED_INPUT_QUALIFICATION'
    try:
        require(read(prepared/'status.json')['status']=='YOLO224_TWO_NODE_GRAPH_PAIR_PREPARED_LOCAL_PENDING','GRAPH_NOT_PREPARED')
        contract=read(prepared/'contract.json');require(contract['expected_submissions']==22,'COMPLETE_ENTRY_AND_EXIT_SYNC_CONTRACT')
        seal=read(prepared/'completion_manifest.json')
        bindings={str(prepared/p):v for p,v in seal['artifact_sha256'].items()};verify(bindings)
        bindings.update(read(prepared/'freeze.json')['bindings'])
        for p in HERE.glob('c3_*.py'):bindings[str(p)]=sha(p)
        write(root/'freeze.json',{'bindings':bindings,'max_GraphExecutor_calls':24,'FPGA_calls':0})
        write(root/'contract.json',{'prepared':str(prepared),'max_GraphExecutor_calls':24,'expected_submissions':22,
                                  'order':['B_detailed','B_minimal','A_detailed','A_minimal'],'no_retry':True})
        cases=[('three_seed',s) for s in SEEDS]+[(k,0) for k in ('W0','W1','W0_return')]
        write(root/'call_ledger.json',[{'variant':v,'mode':m,'call_index':i,'kind':k,'seed':s,'GraphExecutor_calls':1}
              for v in ('B','A') for m in ('detailed','minimal') for i,(k,s) in enumerate(cases,1)])
        import psutil
        for variant in ('B','A'):
            for mode in ('detailed','minimal'):
                status='STOPPED_'+variant+'_'+mode
                env=env_for_worker();env['TVM_LIBRARY_PATH']=str(prepared/'runtime_sim');env['C3_OFFICIAL_WINDOW_COLLECTION']=mode
                for key in ('VTA_UOP_BUFFER_BYTES','VTA_INSN_BUFFER_BYTES','VTA_INSN_SUBMIT_THRESHOLD_BYTES'):env[key]='33554432'
                argv=[str(PYTHON),str(Path(__file__).resolve()),'_worker',str(root),'--variant',variant,'--mode',mode]
                start=time.monotonic();peak=0;limit=None
                with (root/(variant+'_'+mode+'.log')).open('x') as log:
                    child=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    while child.poll() is None:
                        try:
                            proc=psutil.Process(child.pid);peak=max(peak,sum(p.memory_info().rss for p in [proc,*proc.children(recursive=True)]))
                        except psutil.Error:pass
                        if peak>8*1024**3 or time.monotonic()-start>600:
                            limit='MEMORY' if peak>8*1024**3 else 'TIME';os.killpg(child.pid,signal.SIGKILL);break
                        time.sleep(.1)
                    code=child.wait()
                write(root/(variant+'_'+mode+'_worker.json'),{'returncode':code,'limit':limit,'peak_RSS':peak,
                      'resource_control_seconds':time.monotonic()-start})
                require(code==0 and limit is None,'FIRST_LOCAL_WORKER_FAILED_NO_RETRY')
        ends=[read(p) for p in root.glob('*_*/call_*_end.json')]
        require(len(ends)==24,'ALL_24_GRAPH_CALLS_REQUIRED')
        # Compare signatures across all seeds, weights, modes and variants
        # only within the same program; B/A legitimately serialize differently.
        for variant in ('B','A'):
            values=[e['signature'] for e in ends if e['variant']==variant]
            require(all(v==values[0] for v in values),'WITHIN_PROGRAM_SUBMISSION_PATH_CHANGED')
        for i in range(1,7):
            values=[read(root/(v+'_'+m)/('call_%02d_end.json'%i)) for v in ('B','A') for m in ('detailed','minimal')]
            require(all(v['head_sha256']==values[0]['head_sha256'] and v['input_sha256']==values[0]['input_sha256'] for v in values),
                    'B_A_CASES_OR_OUTPUTS_DIFFER')
        status='YOLO224_TWO_NODE_GRAPH_PAIR_LOCAL_QUALIFIED'
    except Exception as error:
        write(root/'failure.json',{'type':type(error).__name__,'message':str(error),'no_retry':True});traceback.print_exc()
    write(root/'status.json',{'status':status,'FPGA_calls':0,'performance_conclusion':None})
    write(root/'completion_manifest.json',{'status':status,
          'artifact_sha256':{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}})
    print(status,flush=True);return 0 if status.endswith('LOCAL_QUALIFIED') else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','_worker']);p.add_argument('output',type=Path)
    p.add_argument('--prepared',type=Path);p.add_argument('--variant',choices=['B','A']);p.add_argument('--mode',choices=['detailed','minimal'])
    args=p.parse_args()
    if args.action=='run':sys.exit(run(args.output.resolve(),args.prepared.resolve()))
    worker(args.output.resolve(),args.variant,args.mode)
