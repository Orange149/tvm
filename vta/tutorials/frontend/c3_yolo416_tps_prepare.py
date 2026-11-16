"""Build-only full-network TH/P gate for the frozen YOLO416 comparison.

All selection precedes compilation; no operator is executed here. A failed
attempt is closed without retries, tile substitution, fallback or board work.
Historical scripts and the production schedule/runtime are unchanged.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
from c3_tps_sync_release_local import HERE,REPO,HOLD,RUNTIME,HELPER,PYTHON,read,write,sha,verify,require,env_for_worker
from c3_yolo416_tps_comparison_scope import BASE,OFFICIAL,group_calls
from c3_yolo416_tps_model import select

SCOPE=HOLD/'20261017_c3_yolo416_tophub_tps_scope01'
AUTHOR=HOLD/'20261012_c3_banerjee_double_buffer_local_run01/author_sources'
HARDWARE=HOLD/'20261016_c3_tps_sync_release_local_run04/hardware_archive.json'


def canonical(config):
    result=json.loads(json.dumps(config));result.pop('index',None);return result


def choose_all(inventory,cfg):
    capacity={k:1<<cfg['LOG_'+k+'_BUFF_SIZE'] for k in ('INP','WGT','ACC')}
    return [{**r,'TPS_selection':select(r['workload'],capacity)} for r in inventory]


@contextmanager
def schedule_context(namespace,rows,observed):
    import tvm,vta
    from vta.top import vta_conv2d
    from c3_official_window_baseline import plain
    original=vta_conv2d._schedule_conv2d_packed_impl
    def schedule(cfg,outs,residency_mode='original',**kwargs):
        require(residency_mode in ('original',0) and not kwargs,'NONORIGINAL_TPS_ROUTE')
        stages=[];seen=set()
        def visit(t):
            if t.op in seen:return
            seen.add(t.op)
            if str(t.op.tag)=='conv2d_dense':stages.append(t)
            for x in getattr(t.op,'input_tensors',[]):visit(x)
        for t in outs:visit(t)
        require(len(stages)==1,'EXPECTED_ONE_CONVOLUTION')
        data,weight=stages[0].op.input_tensors
        if 'pad' in str(data.op.tag):data=data.op.input_tensors[0]
        shapes=[list(map(int,t.shape)) for t in (data,weight)]
        matches=[r for r in rows if [r['workload'][1][1],r['workload'][2][1]]==shapes]
        require(len(matches)==1,'TPS_SCHEDULE_WORKLOAD_NOT_UNIQUE')
        row=matches[0];actual=plain(cfg.to_json_dict())
        require(not cfg.is_fallback and canonical(actual)==canonical(row['TPS_selection']['complete_config_entity']),
                'ACTUAL_TPS_CONFIG_CHANGED')
        require(namespace['get_env']().dev.vta_axis.same_as(vta.get_env().dev.vta_axis),'AUTHOR_ENVIRONMENT_CHANGED')
        observed.append({'object_id':row['object_id'],'actual_config':actual})
        return namespace['schedule_conv2d_packed'](cfg,outs)
    vta_conv2d._schedule_conv2d_packed_impl=schedule
    try:yield
    finally:vta_conv2d._schedule_conv2d_packed_impl=original


def build_worker(root,variant):
    import numpy as np
    import tvm,vta
    from tvm import autotvm,relay,tir
    from c3_official_window_baseline import (key,select_records,assert_effective_config,plain,
        module_functions,buffer_abi,normalize_llvm,parameter_semantics)
    from c3_banerjee_local import author_namespace
    from c3_banerjee_completion import CloseCallBoundary,SplitPaddedInputChannels,ImprovedDoubleBuffer
    from c3_yolo416_tps_dma import SplitStridedOutputChannels
    from c3_uop_identity_applicability import inspect_function
    from prepare_vta_c3_yolo_unseen_resolution import graph_memory_contract
    verify(read(root/'freeze.json')['bindings'])
    require(Path(tvm._ffi.base._LIB._name).resolve()==(REPO/'build/libtvm.so').resolve(),'COMPILER_DSO_CHANGED')
    rows=read(root/'selection.json');by_workload={key(r['workload']):r for r in rows}
    records,_=select_records();queries=[];scheduled=[];parents={};finals={};all_final={}
    out=root/variant;out.mkdir(exist_ok=False)
    module=tvm.ir.load_json((BASE/'packed_relay.json').read_text())
    params=relay.load_param_dict((BASE/'params_input.bin').read_bytes())

    class Context(autotvm.task.ApplyHistoryBest):
        def query(self,target,workload):
            if workload[0]!='conv2d_packed.vta':return super().query(target,workload)
            require(key(workload) in by_workload,'EXACT_WORKLOAD_MISS')
            row=by_workload[key(workload)]
            if variant=='TH':
                actual=self._query_inside(target,workload)
                config=assert_effective_config(row['TopHub_source'],actual)
            else:
                config=row['TPS_selection']['complete_config_entity']
                actual=autotvm.task.ConfigEntity.from_json_dict(config)
                require(not actual.is_fallback,'TPS_FALLBACK')
            queries.append({'object_id':row['object_id'],'workload':plain(workload),
                            'config':plain(actual.to_json_dict())})
            return actual

    @tvm.transform.module_pass(opt_level=0,name='tir.c3.Yolo416PostFoldCapture')
    def capture(mod,_ctx):
        if not module_functions(mod):return mod
        name='parent_%03d'%len(parents)
        require(len(module_functions(mod))==1,'EXPECTED_ONE_FUSED_VTA_FUNCTION_PER_MODULE')
        _,fn=module_functions(mod)[0];symbol=str(fn.attrs['global_symbol'])
        require(symbol not in parents,'DUPLICATE_VTA_PARENT')
        parents[symbol]=name
        (out/(name+'.json')).write_text(tvm.ir.save_json(mod))
        (out/(name+'.tir')).write_text(mod.script())
        return mod

    @tvm.instrument.pass_instrument
    class Archive:
        def __init__(self):self.order=[]
        def run_before_pass(self,mod,info):
            self.order.append(info.name)
            if info.name=='tir.InjectCopyIntrin' and module_functions(mod):
                folder=out/'pre_dma';folder.mkdir(exist_ok=True)
                _,fn=module_functions(mod)[0];symbol=str(fn.attrs['global_symbol'])
                (folder/(symbol+'.json')).write_text(tvm.ir.save_json(mod))
                (folder/(symbol+'.tir')).write_text(mod.script())
        def run_after_pass(self,mod,info):
            if info.name!='tir.vta.CPUAccessRewrite':return
            for gv,fn in mod.functions.items():
                if not isinstance(fn,tir.PrimFunc) or fn.attrs is None or 'global_symbol' not in fn.attrs:continue
                symbol=str(fn.attrs['global_symbol'])
                require(symbol not in all_final,'DUPLICATE_FINAL_FUNCTION')
                one=tvm.IRModule({gv:fn},attrs=mod.attrs)
                folder=out/'all_final';folder.mkdir(exist_ok=True)
                (folder/(symbol+'.json')).write_text(tvm.ir.save_json(one))
                all_final[symbol]={'file':str(folder/(symbol+'.json')),'VTA':bool(module_functions(one))}
                if symbol in parents:
                    stem=parents[symbol];(out/(stem+'_final.json')).write_text(tvm.ir.save_json(mod))
                    finals[symbol]=stem

    def guarded(inner,thread_only=False):
        @tir.transform.prim_func_pass(opt_level=0,name='tir.c3.Yolo416_'+inner.info.name.rsplit('.',1)[-1])
        def run(fn,_mod,_ctx):
            # CPU functions and functions without virtual contexts are retained.
            from c3_official_window_baseline import is_vta_mac
            if not is_vta_mac(fn):return fn
            contexts=[]
            tir.stmt_functor.post_order_visit(fn.body,lambda n:contexts.append(n)
                if isinstance(n,tir.For) and str(n.loop_var).startswith('cthread') else None)
            if thread_only and not contexts:return fn
            one=inner(tvm.IRModule({'main':fn}))
            return list(one.functions.values())[0]
        return run

    with vta.Environment(dict(vta.get_env().cfg_dict,TARGET='axu5evb')):
        env=vta.get_env();target=tvm.target.Target(env.target,host=env.target_host)
        namespace=author_namespace(AUTHOR,safe_uop_order=False) if variant=='P' else None
        base=vta.build_config(debug_flag=0,opt_level=3,disabled_pass={'AlterOpLayout','tir.CommonSubexprElimTIR'})
        config=dict(base.config);passes=list(config['tir.add_lower_pass'])
        if variant=='P':
            i=next(i for i,(_,p) in enumerate(passes) if p.info.name=='tir.InjectCopyIntrin')
            passes.insert(i,(1,guarded(SplitPaddedInputChannels())))
            if read(root/'contract.json').get('strided_output_dma_adapter',False):
                passes.insert(i+1,(1,guarded(SplitStridedOutputChannels())))
            i=next(i for i,(phase,_) in enumerate(passes) if phase>=2)
            passes[i:i]=[(2,guarded(CloseCallBoundary())),(2,guarded(ImprovedDoubleBuffer(),True))]
        i=next(i for i,(_,p) in enumerate(passes) if p.info.name=='tir.vta.FoldUopLoop')+1
        require(passes[i][1].info.name=='tir.vta.CPUAccessRewrite','PARENT_CAPTURE_GAP')
        passes.insert(i,(3,capture));config['tir.add_lower_pass']=passes
        archive=Archive();relay.backend.te_compiler.get().clear()
        from contextlib import nullcontext
        private=schedule_context(namespace,rows,scheduled) if variant=='P' else nullcontext()
        with private,Context(records):
            with tvm.transform.PassContext(opt_level=3,config=config,instruments=[archive],
                    disabled_pass={'AlterOpLayout','tir.CommonSubexprElimTIR'}):
                graph,lib,lowered_params=relay.build(module,target=target,params=params)
        llvm=lib.get_source('ll')
        (out/'graph.json').write_text(graph);(out/'module.ll').write_text(llvm)
        (out/'params.bin').write_bytes(relay.save_param_dict(lowered_params))
        write(out/'queries.json',queries);write(out/'schedule_hits.json',scheduled)
        write(out/'pass_sequence.json',archive.order);write(out/'all_final.json',all_final)
        require(json.loads(graph)==read(BASE/'B_raw/graph.json'),'GRAPH_STRUCTURE_OR_STORAGE_CHANGED')
        old_params=relay.load_param_dict((BASE/'B_raw/params.bin').read_bytes())
        require(set(old_params)==set(lowered_params) and all(np.array_equal(old_params[k].numpy(),lowered_params[k].numpy())
                and old_params[k].dtype==lowered_params[k].dtype for k in old_params),'GRAPH_PARAMETERS_CHANGED')
        write(out/'parameter_inventory.json',parameter_semantics(lowered_params))
        require(set(parents)==set(finals)=={r['symbol'] for r in rows},'COMPLETE_VTA_FUNCTION_COVERAGE_CHANGED')
        require({q['object_id'] for q in queries}=={r['object_id'] for r in rows},'CONFIG_QUERY_COVERAGE_CHANGED')
        memory=graph_memory_contract(graph)
        require(memory['command_backing_bytes']==2*33554432 and memory['fits'],'GRAPH_192_MIB_ADMISSION_FAILED')
        write(out/'memory_contract.json',memory)
        functions=[]
        for row in rows:
            symbol=row['symbol'];stem=parents[symbol]
            parent=tvm.ir.load_json((out/(stem+'.json')).read_text())
            gv,fn=module_functions(parent)[0]
            require(buffer_abi(fn)==row['abi'],'COMPLETE_FUSION_ABI_CHANGED')
            final=tvm.ir.load_json((out/(stem+'_final.json')).read_text())
            replay=vta.transform.CPUAccessRewrite()(tvm.ir.load_json(tvm.ir.save_json(parent)))
            require(tvm.ir.structural_equal(replay,final),'PARENT_REMAINING_LOWERING_ROUNDTRIP_CHANGED')
            if variant=='TH':
                oldrow=next(r for r in read(BASE/'baseline_identity.json')['callsites'] if r['global_symbol']==symbol)
                oldparent=tvm.ir.load_json(Path(oldrow['parent_module']).read_text())
                oldfinal=tvm.ir.load_json(Path(oldrow['final_module']).read_text())
                require(tvm.ir.structural_equal(parent,oldparent) and tvm.ir.structural_equal(final,oldfinal),
                        'TH_NO_TRANSFORM_TIR_CHANGED')
            census=inspect_function(fn)
            functions.append({'object_id':row['object_id'],'symbol':symbol,'nodes':row['nodes'],
                              'parent':str(out/(stem+'.json')),'final':str(out/(stem+'_final.json')),
                              'abi':row['abi'],'workload':row['workload'],'identity_census':census})
        if variant=='TH':
            require(normalize_llvm(llvm)==normalize_llvm((BASE/'B_raw/module.ll').read_text()),'TH_NO_TRANSFORM_LLVM_CHANGED')
        else:
            require({h['object_id'] for h in scheduled}=={r['object_id'] for r in rows},'TPS_NOT_USED_FOR_EVERY_FUNCTION')
            previous=read(root/'TH/all_final.json')
            require(set(previous)==set(all_final),'CPU_FUNCTION_SET_CHANGED')
            for symbol,row in all_final.items():
                if row['VTA']:continue
                a=tvm.ir.load_json(Path(row['file']).read_text());b=tvm.ir.load_json(Path(previous[symbol]['file']).read_text())
                require(tvm.ir.structural_equal(a,b,map_free_vars=True),'NON_VTA_CPU_FUNCTION_CHANGED:'+symbol)
        write(out/'functions.json',functions)
        write(out/'result.json',{'status':'FULL_GRAPH_COMPILED_LOCAL_QUALIFICATION_PENDING','variant':variant,
              'functions':len(functions),'graph_VTA_calls':sum(len(r['nodes']) for r in rows),
              'raw_outputs':len(json.loads(graph)['heads']),'new_target_calls':0,'board_contacted':False,
              'MAC_redundant_identities':sum(f['identity_census']['MAC_redundant_identities'] for f in functions),
              'identity_analysis_unresolved':[f['object_id'] for f in functions if f['identity_census']['status']!='FINITE_DOMAIN_CONTENT_SCREEN_COMPLETE'],
              'compiled_module_exported':False,'CPU_functions_retained':True,'graph_and_parameters_equal':True})
    verify(read(root/'freeze.json')['bindings'])


def worker(root,variant):
    import psutil
    argv=[str(PYTHON),str(Path(__file__).resolve()),'_worker',str(root),'--variant',variant]
    start=time.monotonic();peak=0;limit=None
    with (root/(variant+'_build.log')).open('x') as log:
        child=subprocess.Popen(argv,env=env_for_worker(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while child.poll() is None:
            try:
                p=psutil.Process(child.pid);peak=max(peak,sum(x.memory_info().rss for x in [p,*p.children(recursive=True)]))
            except psutil.Error:pass
            if peak>8*1024**3 or time.monotonic()-start>600:
                limit='MEMORY' if peak>8*1024**3 else 'TIME';os.killpg(child.pid,signal.SIGKILL);break
            time.sleep(.1)
        code=child.wait()
    write(root/(variant+'_worker.json'),{'returncode':code,'limit':limit,'peak_sampled_tree_rss':peak,
          'resource_control_seconds':time.monotonic()-start,'argv':argv})
    return code==0 and limit is None


def prepare(root):
    root.mkdir(exist_ok=False);status='STOPPED_INPUT_IDENTITY'
    try:
        scope_seal=read(SCOPE/'completion_manifest.json')
        verify({str(SCOPE/p):v for p,v in scope_seal['artifact_sha256'].items()})
        bindings=read(SCOPE/'freeze.json')['bindings'].copy()
        paths=list(HERE.glob('c3_*.py'))+[HERE/'test_c3_yolo416_tps_model.py',HERE/'test_c3_yolo416_tps_dma.py',HELPER,RUNTIME/'libvta_fsim.so',HARDWARE]
        paths+=list(AUTHOR.glob('*'))+list((REPO/'vta/python').rglob('*.py'))
        bindings.update({str(p.resolve()):sha(p) for p in paths if p.is_file()})
        from c3_official_window_baseline import compiler_sources
        bindings.update(compiler_sources());bindings.update(read(HARDWARE)['source_bindings'])
        verify(bindings);write(root/'freeze.json',{'bindings':bindings,'target_calls':0,'FPGA_calls':0})
        shutil.copyfile(HARDWARE,root/'hardware_archive.json')
        rows=choose_all(read(SCOPE/'inventory.json'),read(REPO/'3rdparty/vta-hw/config/vta_config.json'))
        write(root/'selection.json',rows)
        write(root/'contract.json',{'network':'yolov3_tiny_416','variants':['TH','P','I_if_distinct'],
              'source_scope':str(SCOPE),'stage':'full graph compilation and structural gates before any target call',
              'operator_calls':12,'unique_functions':11,'graph_heads':8,'max_target_calls_this_stage':0,
              'configuration_rule':'all supported thread families; analytical score then Tile.identity; no measured latency',
              'strided_output_dma_adapter':True,
              'adapter_basis':'same scalar loop/address sequence; CO outside DMA pragma only for unrepresentable strided STORE',
              'limits':{'worker_GiB':8,'worker_seconds':600},'stop':'no retry, tile substitution or silent fallback'})
        for variant in ('TH','P'):
            status='STOPPED_'+variant+'_BUILD'
            if not worker(root,variant):break
        else:
            p=read(root/'P/result.json')
            status='TH_P_GRAPHS_COMPILED_QUALIFICATION_PENDING'
            write(root/'prepare_complete.json',{'status':status,'P_result_sha256':sha(root/'P/result.json'),
                  'I_deduplicates_P':p['MAC_redundant_identities']==0 and not p['identity_analysis_unresolved']})
    except Exception as error:
        write(root/'failure_controller.json',{'type':type(error).__name__,'message':str(error)});traceback.print_exc()
    write(root/'status.json',{'status':status,'target_calls':0,'FPGA_calls':0,'performance_conclusion':None})
    print(json.dumps({'status':status,'target_calls':0,'FPGA_calls':0}),flush=True)
    return 0 if status=='TH_P_GRAPHS_COMPILED_QUALIFICATION_PENDING' else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','_worker'));p.add_argument('output',type=Path)
    p.add_argument('--variant',choices=('TH','P'));a=p.parse_args();root=a.output.resolve()
    if a.action=='prepare':sys.exit(prepare(root))
    try:build_worker(root,a.variant)
    except Exception as error:
        write(root/('failure_'+a.variant+'.json'),{'type':type(error).__name__,'message':str(error)})
        traceback.print_exc();sys.exit(1)
