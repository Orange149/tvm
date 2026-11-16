"""Build-only ARM artifacts after complete operator and whole-graph gates.

No RPC/FPGA/target call. Every isolated operator and full graph is built from
the exact locally qualified final IR; CPU code and graph parameters persist.
"""
import argparse
import os
from pathlib import Path
import shlex
import shutil
import signal
import struct
import subprocess
import sys
import time
import traceback
from c3_tps_sync_release_local import HERE,REPO,PYTHON,read,write,sha,verify,require,env_for_worker
from c3_tps_aggregation_prepare import SDK,compare_initializer_bindings


def elf_gate(path):
    blob=Path(path).read_bytes()
    require(blob[:6]==b'\x7fELF\x02\x01' and struct.unpack_from('<H',blob,18)[0]==183,
            'ARM_ELF_MACHINE_CHANGED')


def compiler_identity():
    tokens=shlex.split(os.environ['CXX']);require(bool(tokens),'EMPTY_CROSS_COMPILER')
    compiler=shutil.which(tokens[0]);require(compiler is not None,'CROSS_COMPILER_NOT_FOUND')
    compiler=str(Path(compiler).resolve())
    return {'CXX':os.environ['CXX'],'compiler':compiler,'sha256':sha(compiler),
            'version':subprocess.check_output([compiler,'--version'],text=True)}


def build(root,index):
    import tvm,vta
    from tvm.contrib import cc
    from c3_official_window_llvm import llvm_alu_contract
    from c3_yolo416_graph_binding import bind_graph,project
    from c3_yolo416_graph_host_reference import raw_pointer_gate,raw_pointer_llvm_gate
    from c3_official_window_baseline import buffer_abi
    from c3_tps_sync_release_prepare import complete_function
    from build_vta_yolov3_tiny_relay_residency_dispatch_pair import cross_compiler
    verify(read(root/'freeze.json')['bindings'])
    require(compiler_identity()==read(root/'compiler_identity.json'),'ACTUAL_CROSS_COMPILER_CHANGED')
    task=read(root/'tasks.json')[index];out=root/task['key'];out.mkdir(parents=True,exist_ok=False)
    module=tvm.ir.load_json(Path(task['final_IR']).read_text());sim=Path(task['fsim_llvm']).read_text()
    cfg=dict(vta.get_env().cfg_dict,TARGET='axu5evb')
    frozen=read(REPO/'3rdparty/vta-hw/config/vta_config.json')
    require(all(cfg.get(k)==v for k,v in frozen.items()),'ARM_VTA_STRUCTURE_CHANGED')
    with vta.Environment(cfg):
        target=tvm.target.Target(vta.get_env().target,host=vta.get_env().target_host)
        with tvm.transform.PassContext(opt_level=3,disabled_pass={'AlterOpLayout','tir.CommonSubexprElimTIR'}):
            lib=tvm.build(module,target=target)
        llvm=lib.get_source('ll');require('target triple = "aarch64-linux-gnu"' in llvm,'ARM_TARGET_CHANGED')
        (out/'module.ll').write_text(llvm)
        comparison={}
        if task['kind']=='operator':
            _,fn=complete_function(module,task['symbol'])
            comparison[task['symbol']]=compare_initializer_bindings(sim,llvm,
                llvm_alu_contract(fn,sim),llvm_alu_contract(fn,llvm))
        else:
            symbols=task['VTA_symbols']
            fns={str(fn.attrs['global_symbol']):fn for fn in module.functions.values()
                 if str(fn.attrs['global_symbol']) in symbols}
            require(set(fns)==set(symbols),'ARM_FULL_GRAPH_SYMBOL_COVERAGE')
            a,b=bind_graph(sim,fns),bind_graph(llvm,fns)
            write(out/'fsim_handle_binding.json',a);write(out/'arm_handle_binding.json',b)
            cpu={str(fn.attrs['global_symbol']):{'abi':buffer_abi(fn),
                 'instruction_gate':raw_pointer_gate(fn)['instruction_gate'],
                 'raw_pointer_gate':raw_pointer_gate(fn),
                 'raw_pointer_llvm_gate':raw_pointer_llvm_gate(llvm,str(fn.attrs['global_symbol']))}
                 for fn in module.functions.values()
                 if str(fn.attrs['global_symbol']) not in symbols}
            require(len(cpu)==21,'ARM_CPU_REFERENCE_COVERAGE_CHANGED')
            write(out/'CPU_reference_manifest.json',cpu)
            for symbol,fn in fns.items():
                first,_=project(sim,symbol);second,_=project(llvm,symbol)
                comparison[symbol]=compare_initializer_bindings(first,second,
                    a['functions'][symbol]['initializer_contract'],b['functions'][symbol]['initializer_contract'])
        write(out/'initializer_comparison.json',comparison)
        invocations=[];original=cc._linux_compile
        def retain(destination,objects,options,compile_cmd,compile_shared=False):
            folder=out/'link_inputs';folder.mkdir(exist_ok=True)
            inputs=[objects] if isinstance(objects,str) else list(objects);saved=[]
            for i,p in enumerate(inputs):
                copy=folder/('%02d_'%i+Path(p).name);shutil.copyfile(p,copy)
                saved.append({'file':str(copy),'sha256':sha(copy)})
            invocations.append({'compiler':compile_cmd,'options':options,'inputs':saved,
                                'compile_shared':compile_shared})
            return original(destination,objects,options,compile_cmd,compile_shared)
        cc._linux_compile=retain
        try:lib.export_library(str(out/'module.so'),fcompile=cross_compiler())
        finally:cc._linux_compile=original
        write(out/'compiler_invocations.json',invocations)
        if task['kind']=='graph':
            shutil.copyfile(out/'module.so',out/'reference_arm.so')
            require(sha(out/'module.so')==sha(out/'reference_arm.so') and
                    (out/'module.so').stat().st_ino!=(out/'reference_arm.so').stat().st_ino,
                    'ARM_REFERENCE_COPY_NOT_INDEPENDENT')
    elf_gate(out/'module.so')
    write(out/'contract.json',{'status':'ARM_INITIALIZERS_MATCH_FSIM_FPGA_PENDING','task':task,
        'module_sha256':sha(out/'module.so'),'arm_llvm_sha256':sha(out/'module.ll'),
        'compiler':compiler_identity(),'SDK_sha256':sha(SDK),'target':str(target),
        'PassContext':{'opt_level':3,'disabled_pass':['AlterOpLayout','tir.CommonSubexprElimTIR']},
        'target_calls':0,'board_contacted':False})
    verify(read(root/'freeze.json')['bindings'])


def bounded_worker(root,index):
    import psutil
    args=[str(PYTHON),str(Path(__file__).resolve()),'_worker',str(root),'--index',str(index)]
    shell='set -e\nunset LD_LIBRARY_PATH\nsource '+shlex.quote(str(SDK))+'\nexec '+shlex.join(args)
    start=time.monotonic();peak=0;limit=None
    with (root/('%02d_build.log'%index)).open('x') as log:
        child=subprocess.Popen(['bash','-c',shell],env=env_for_worker(),stdout=log,
                               stderr=subprocess.STDOUT,start_new_session=True)
        while child.poll() is None:
            try:
                p=psutil.Process(child.pid);peak=max(peak,sum(x.memory_info().rss for x in [p,*p.children(recursive=True)]))
            except psutil.Error:pass
            if peak>8*1024**3 or time.monotonic()-start>600:
                limit='MEMORY' if peak>8*1024**3 else 'TIME';os.killpg(child.pid,signal.SIGKILL);break
            time.sleep(.1)
        code=child.wait()
    write(root/('%02d_worker.json'%index),{'returncode':code,'resource_limit':limit,
        'sampled_peak_tree_RSS':peak,'resource_control_seconds':time.monotonic()-start})
    require(code==0 and limit is None,'ARM_BUILD_WORKER_FAILED')


def prepare(root,operators,graph):
    root.mkdir(exist_ok=False);status='STOPPED_ARM_INPUT_QUALIFICATION'
    try:
        bindings={}
        for folder in (operators,graph):
            seal=read(folder/'completion_manifest.json')
            bindings.update({str(folder/p):d for p,d in seal['artifact_sha256'].items()})
            bindings[str(folder/'completion_manifest.json')]=sha(folder/'completion_manifest.json')
        require(read(operators/'status.json')['status']==
            'ALL_22_FUNCTIONS_LOCAL_QUALIFIED_GRAPH_QUALIFICATION_PENDING','OPERATORS_NOT_QUALIFIED')
        require(read(graph/'status.json')['status']==
            'WHOLE_TH_P_GRAPH_LOCAL_QUALIFIED_ARM_AND_FPGA_QUALIFICATION_PENDING','WHOLE_GRAPH_NOT_QUALIFIED')
        require(read(graph/'contract.json')['operators']==str(operators),'QUALIFICATION_LINEAGE_CHANGED')
        compiled=Path(read(operators/'freeze.json')['source'])
        prepared=Path(read(graph/'contract.json')['prepared'])
        probe='from c3_yolo416_arm_prepare import compiler_identity; import json; print(json.dumps(compiler_identity()))'
        shell='set -e\nunset LD_LIBRARY_PATH\nsource '+shlex.quote(str(SDK))+'\nexec '+shlex.join([str(PYTHON),'-c',probe])
        import json
        compiler=json.loads(subprocess.check_output(['bash','-c',shell],env=env_for_worker(),text=True,timeout=30))
        write(root/'compiler_identity.json',compiler)
        bindings[compiler['compiler']]=compiler['sha256']
        bindings[str(root/'compiler_identity.json')]=sha(root/'compiler_identity.json')
        tasks=[]
        for i,row in enumerate(read(operators/'objects.json')):
            tasks.append({'key':'operators/%02d'%i,'kind':'operator','variant':row['variant'],
                'object_id':row['object_id'],'symbol':row['symbol'],'abi':row['abi'],
                'final_IR':row['final'],'fsim_llvm':str(operators/'objects'/('%02d'%i)/'prepare/R/module.ll')})
        for variant in ('TH','P'):
            tasks.append({'key':'graph/'+variant,'kind':'graph','variant':variant,
                'final_IR':str(prepared/variant/'codegen_input.json'),
                'fsim_llvm':str(prepared/variant/'module.ll'),
                'VTA_symbols':[r['symbol'] for r in read(compiled/variant/'functions.json')]})
        for p in [*HERE.glob('c3_*.py'),HERE/'c3_yolo416_graph_once.cc',HERE/'c3_yolo416_graph_observer.cc',
                  HERE/'c3_yolo416_cpu_reference.cc',HERE/'c3_yolo416_rpc_observer.cc',SDK]:
            bindings[str(p.resolve())]=sha(p)
        for task in tasks:
            for key in ('final_IR','fsim_llvm'):bindings[task[key]]=sha(task[key])
        verify(bindings);write(root/'freeze.json',{'bindings':bindings,'new_target_calls':0,'FPGA_calls':0})
        write(root/'tasks.json',tasks)
        write(root/'source_contract.json',{'operators':str(operators),'whole_graph':str(graph),
              'compiled':str(compiled),'graph_prepared':str(prepared),'I_equals_P':True})
        status='STOPPED_ARM_BUILD'
        for i in range(len(tasks)):bounded_worker(root,i)
        # Helpers are compiled in the same SDK context, never linked into or
        # used to transform the target module.
        script='''import sys\nfrom pathlib import Path\nfrom tvm.contrib import cc\nfrom build_vta_yolov3_tiny_relay_residency_dispatch_pair import cross_compiler\nroot=Path(sys.argv[1]);here=Path(sys.argv[2]);repo=Path(sys.argv[3])\noptions=['-std=c++17','-O2','-ldl']+['-I'+str(repo/p) for p in ('include','3rdparty/dmlc-core/include','3rdparty/dlpack/include')]\nfor name in ('graph_once','graph_observer','cpu_reference','rpc_observer'):\n cross_compiler()(str(root/(name+'_arm.so')),[str(here/('c3_yolo416_'+name+'.cc'))],options=options)\n'''
        (root/'compile_helpers.py').write_text(script)
        helper_source_sha=sha(root/'compile_helpers.py')
        args=[str(PYTHON),str(root/'compile_helpers.py'),str(root),str(HERE),str(REPO)]
        shell='set -e\nunset LD_LIBRARY_PATH\nsource '+shlex.quote(str(SDK))+'\nexec '+shlex.join(args)
        import psutil
        started=time.monotonic();peak=0;limit=None
        with (root/'helpers_build.log').open('x') as log:
            proc=subprocess.Popen(['bash','-c',shell],env=env_for_worker(),stdout=log,
                                  stderr=subprocess.STDOUT,start_new_session=True)
            while proc.poll() is None:
                try:
                    p=psutil.Process(proc.pid);peak=max(peak,sum(x.memory_info().rss for x in [p,*p.children(recursive=True)]))
                except psutil.Error:pass
                if peak>8*1024**3 or time.monotonic()-started>180:
                    limit='MEMORY' if peak>8*1024**3 else 'TIME';os.killpg(proc.pid,signal.SIGKILL);break
                time.sleep(.1)
            code=proc.wait()
        write(root/'helpers_build.json',{'returncode':code,'resource_limit':limit,'compiler':compiler,
              'sampled_peak_tree_RSS':peak,'resource_control_seconds':time.monotonic()-started,
              'script_sha256':helper_source_sha,'arguments':args})
        require(code==0 and limit is None,'ARM_GRAPH_HELPERS_BUILD_FAILED')
        require(sha(root/'compile_helpers.py')==helper_source_sha,'HELPER_BUILD_SCRIPT_CHANGED')
        for name in ('graph_once','graph_observer','cpu_reference','rpc_observer'):elf_gate(root/(name+'_arm.so'))
        verify(bindings);status='ALL_TH_P_ARM_MODULES_AND_GRAPH_HELPERS_BUILT_FPGA_PENDING'
        write(root/'prepare_complete.json',{'status':status,'isolated_functions':22,'graphs':2,
              'target_calls':0,'FPGA_calls':0})
    except Exception as e:
        write(root/'failure_controller.json',{'type':type(e).__name__,'message':str(e)});traceback.print_exc()
    write(root/'status.json',{'status':status,'target_calls':0,'FPGA_calls':0})
    print(status,flush=True)
    return 0 if status.startswith('ALL_TH_P_ARM') else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','_worker'));p.add_argument('output',type=Path)
    p.add_argument('--operators',type=Path);p.add_argument('--graph',type=Path);p.add_argument('--index',type=int)
    a=p.parse_args();root=a.output.resolve()
    if a.action=='prepare':sys.exit(prepare(root,a.operators.resolve(),a.graph.resolve()))
    try:build(root,a.index)
    except Exception as e:
        write(root/('failure_%02d.json'%a.index),{'type':type(e).__name__,'message':str(e)})
        traceback.print_exc();sys.exit(1)
