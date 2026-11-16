"""Build the whole FSim graph with process-owned plugin code and independent graph helper after operator gates.

No GraphExecutor.run(), fused target call, RPC or FPGA operation occurs here.
Every candidate is built from the exact final modules of the qualified graph.
"""
import argparse
import re
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from c3_tps_sync_release_local import HERE,REPO,read,write,sha,verify,require,env_for_worker


def build(root,variant):
    import tvm,vta
    from c3_official_window_baseline import buffer_abi,normalize_llvm
    from c3_yolo416_graph_binding import bind_graph
    from c3_yolo416_graph_cpu_reference import cpu_instruction_gate
    from c3_yolo416_graph_trace import compose
    from c3_official_window_method import _trace,_limits
    verify(read(root/'freeze.json')['bindings'])
    source=Path(read(root/'contract.json')['compiled_source']);out=root/variant;out.mkdir(exist_ok=False)
    rows=read(source/variant/'all_final.json');functions={};CPU={};VTA={}
    module=tvm.IRModule()
    for symbol,row in rows.items():
        one=tvm.ir.load_json(Path(row['file']).read_text())
        require(len(one.functions)==1,'SOURCE_FUNCTION_NOT_ISOLATED')
        gv,fn=next(iter(one.functions.items()))
        require(str(fn.attrs['global_symbol'])==symbol and symbol not in functions,'SOURCE_SYMBOL_CHANGED')
        module[gv.name_hint]=fn;functions[symbol]=fn
        if row['VTA']:VTA[symbol]=fn
        else:CPU[symbol]={'abi':buffer_abi(fn),'source_final':row['file'],'sha256':sha(row['file']),
                         'instruction_gate':cpu_instruction_gate(fn)}
    require(len(VTA)==11 and len(CPU)>0,'GRAPH_FUNCTION_COVERAGE_CHANGED')
    (out/'codegen_input.json').write_text(tvm.ir.save_json(module))
    with vta.Environment(dict(vta.get_env().cfg_dict,TARGET='sim')):
        target=tvm.target.Target(vta.get_env().target,host=vta.get_env().target_host)
        with tvm.transform.PassContext(opt_level=3,disabled_pass={'AlterOpLayout','tir.CommonSubexprElimTIR'}):
            lib=tvm.build(module,target=target)
            replay=tvm.build(tvm.ir.load_json(tvm.ir.save_json(module)),target=target)
        llvm=lib.get_source('ll')
        require(normalize_llvm(llvm)==normalize_llvm(replay.get_source('ll')),'FULL_GRAPH_CODEGEN_ROUNDTRIP_CHANGED')
        require(re.search(r'^target triple = "x86_64-[^"]+"',llvm,re.M) is not None,'WRONG_FSIM_HOST')
        (out/'module.ll').write_text(llvm)
        require(all(lib.get_function(symbol,True) is not None for symbol in functions),'GRAPH_ENTRY_SYMBOL_MISSING')
        binding=bind_graph(llvm,VTA)
        write(out/'LLVM_initializer_binding.json',binding)
        lib.export_library(str(out/'graph_fsim.so'))
        shutil.copyfile(out/'graph_fsim.so',out/'reference_fsim.so')
        require(sha(out/'graph_fsim.so')==sha(out/'reference_fsim.so') and
                (out/'graph_fsim.so').stat().st_ino!=(out/'reference_fsim.so').stat().st_ino,
                'REFERENCE_LIBRARY_NOT_DISTINCT_BYTE_IDENTICAL_COPY')
    write(out/'CPU_manifest.json',CPU)
    graph=read(source/variant/'graph.json')
    require(len(graph['heads'])==8 and sum(n.get('attrs',{}).get('func_name') in VTA
            for n in graph['nodes'])==12,'GRAPH_CALL_OR_HEAD_COVERAGE_CHANGED')
    require(all(n['attrs']['func_name'] in functions or n['attrs']['func_name'] in ('__copy','__nop')
                for n in graph['nodes'] if n['op']=='tvm_op'),'GRAPH_CALL_SYMBOL_UNBOUND')
    traces={};counts={}
    for row in read(source/variant/'functions.json'):
        parent=tvm.ir.load_json(Path(row['parent']).read_text())
        trace,count=_trace(list(parent.functions.values())[0],_limits(None))
        traces[row['symbol']]={'abi':row['abi'],'events':trace};counts[row['symbol']]=count
    events,mapping=compose(graph,traces)
    write(out/'events.json',events);write(out/'namespace.json',mapping)
    write(out/'per_function_trace_counts.json',counts)
    for name in ('graph.json','params.bin'):shutil.copyfile(source/variant/name,out/name)
    write(out/'result.json',{'status':'WHOLE_GRAPH_FSIM_LIBRARY_BUILT_QUALIFICATION_PENDING',
          'variant':variant,'VTA_functions':len(VTA),'CPU_functions':len(CPU),
          'graph_calls':12,'raw_outputs':8,'target_calls':0,'FPGA_calls':0,
          'CPU_reference_trust_boundary':'same bound compiled CPU functions; independent integer convolutions'})
    verify(read(root/'freeze.json')['bindings'])


def prepare(root,source,qualified):
    root.mkdir(exist_ok=False);status='STOPPED_INPUT_QUALIFICATION'
    try:
        for folder in (source,qualified):
            seal=read(folder/'completion_manifest.json')
            verify({str(folder/p):v for p,v in seal['artifact_sha256'].items()})
        require(read(qualified/'status.json')['status']==
                'ALL_22_FUNCTIONS_LOCAL_QUALIFIED_GRAPH_QUALIFICATION_PENDING','OPERATOR_QUALIFICATION_NOT_COMPLETE')
        require(read(qualified/'freeze.json')['source']==str(source),'QUALIFIED_GRAPH_SOURCE_CHANGED')
        require(read(source/'contract.json')['network']=='yolov3_tiny_416' and
                read(source/'identity_deduplication.json')['I_equals_P'] is True,'GRAPH_SCOPE_OR_DEDUPLICATION_CHANGED')
        bindings=read(qualified/'freeze.json')['bindings'].copy()
        for p in [*HERE.glob('c3_*.py'),HERE/'c3_yolo416_graph_once.cc',HERE/'c3_yolo416_graph_observer.cc',
                  HERE/'c3_yolo416_plugin_lifetime.cc',
                  *HERE.glob('test_c3_yolo416_graph*.py')]:
            bindings[str(p.resolve())]=sha(p)
        for folder in (source,qualified):
            for rel,digest in read(folder/'completion_manifest.json')['artifact_sha256'].items():
                bindings[str(folder/rel)]=digest
            bindings[str(folder/'completion_manifest.json')]=sha(folder/'completion_manifest.json')
        lifetime=HERE/'c3_yolo416_plugin_lifetime.cc'
        write(root/'helper_lifetime.json',{'source':str(lifetime),'sha256':sha(lifetime),
              'policy':'RTLD_NODELETE plugin code; ordinary object destructors preserved',
              'target_IR_runtime_and_call_path_changed':False})
        bindings[str(root/'helper_lifetime.json')]=sha(root/'helper_lifetime.json')
        verify(bindings)
        write(root/'freeze.json',{'bindings':bindings,'max_target_calls':0,'FPGA_calls':0})
        write(root/'contract.json',{'compiled_source':str(source),'operator_qualification':str(qualified),
              'variants':['TH','P'],'I_equals_P':True,'build_only':True,
              'helper':'one zero-argument GraphExecutor.run; no added synchronize or implicit warmup'})
        # Compilation workers remain serial and use the existing resource bounds.
        import os,signal,time,psutil
        from c3_tps_sync_release_local import PYTHON
        for variant in ('TH','P'):
            status='STOPPED_'+variant+'_GRAPH_BUILD'
            args=[str(PYTHON),str(Path(__file__).resolve()),'_worker',str(root),'--variant',variant]
            start=time.monotonic();peak=0;limit=None
            with (root/(variant+'_build.log')).open('x') as log:
                child=subprocess.Popen(args,env=env_for_worker(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                while child.poll() is None:
                    try:
                        p=psutil.Process(child.pid)
                        peak=max(peak,sum(x.memory_info().rss for x in [p,*p.children(recursive=True)]))
                    except psutil.Error:pass
                    if peak>8*1024**3 or time.monotonic()-start>600:
                        limit='MEMORY' if peak>8*1024**3 else 'TIME';os.killpg(child.pid,signal.SIGKILL);break
                    time.sleep(.1)
                code=child.wait()
            write(root/(variant+'_worker.json'),{'returncode':code,'resource_limit':limit,'sampled_peak_tree_RSS':peak,
                  'resource_control_seconds':time.monotonic()-start})
            if code or limit:break
        else:
            from tvm.contrib import cc
            options=['-std=c++17','-O2','-ldl',*['-I'+str(REPO/p) for p in
                ('include','3rdparty/dmlc-core/include','3rdparty/dlpack/include')]]
            cc.create_shared(str(root/'graph_once_fsim.so'),[str(HERE/'c3_yolo416_graph_once.cc'),str(HERE/'c3_yolo416_plugin_lifetime.cc')],options=options)
            cc.create_shared(str(root/'graph_observer_fsim.so'),[str(HERE/'c3_yolo416_graph_observer.cc'),str(HERE/'c3_yolo416_plugin_lifetime.cc')],options=options)
            status='WHOLE_TH_P_GRAPHS_AND_HELPER_BUILT_LOCAL_QUALIFICATION_PENDING'
            write(root/'prepare_complete.json',{'status':status,'target_calls':0,'FPGA_calls':0,
                  'helper_sha256':sha(root/'graph_once_fsim.so'),
                  'observer_sha256':sha(root/'graph_observer_fsim.so')})
    except Exception as e:
        write(root/'failure_controller.json',{'type':type(e).__name__,'message':str(e)});traceback.print_exc()
    write(root/'status.json',{'status':status,'target_calls':0,'FPGA_calls':0,'performance_conclusion':None})
    print(status,flush=True)
    return 0 if status.startswith('WHOLE_TH_P_') else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','_worker'));p.add_argument('output',type=Path)
    p.add_argument('--source',type=Path);p.add_argument('--qualified',type=Path);p.add_argument('--variant',choices=('TH','P'))
    a=p.parse_args();root=a.output.resolve()
    if a.action=='prepare':sys.exit(prepare(root,a.source.resolve(),a.qualified.resolve()))
    try:build(root,a.variant)
    except Exception as e:
        write(root/('failure_'+a.variant+'.json'),{'type':type(e).__name__,'message':str(e)})
        traceback.print_exc();sys.exit(1)
