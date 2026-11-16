"""One frozen node46 TPS baseline: compile, certify, 14 local FSim calls max.

No board path, tuning, optimization variant or production source modification.
The oc2/ci1 representative is fixed by the preceding applicability audit.
This is the inherited oc2 support family, not the full author-stack optimum.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import signal
import traceback

from c3_tps_sync_release_local import (HERE, REPO, HOLD, RUNTIME, HELPER, PYTHON,
    read, write, sha, verify, require, env_for_worker, call_accounting,
    qualification_worker, compare_modes, KINDS, SEEDS)

APPLICABILITY = HOLD/'20261017_c3_node46_tps_applicability01'
GRAPH = HOLD/'20261008_c3_yolo224_incremental_prepare_run03/G2'
AUTHOR = HOLD/'20261012_c3_banerjee_double_buffer_local_run01/author_sources'
HARDWARE = HOLD/'20261016_c3_tps_sync_release_local_run04/hardware_archive.json'
SYMBOL = 'tvmgen_default_fused_nn_conv2d_add_right_shift_clip_cast'
SHAPES = [[1,24,14,14,1,16],[16,24,3,3,16,16],[16,1,1,1,16],[1,16,14,14,1,16]]
WORKLOAD = ['conv2d_packed.vta',['TENSOR',SHAPES[0],'int8'],['TENSOR',SHAPES[1],'int8'],
            [1,1],[1,1,1,1],[1,1],'NCHW1n16c','int32']


def extract_relay(text):
    anchor='%167 = fn ('
    require(text.count(anchor)==1,'NODE46_RELAY_NOT_UNIQUE')
    block='fn ('+text.split(anchor,1)[1].split('} /* ty=fn',1)[0]+'}'
    for expected in ('Tensor[(16, 24, 3, 3, 16, 16), int8]', 'channels=256',
                     'right_shift(%24, 8', 'a_min=-127f', 'a_max=127f'):
        require(expected in block,'NODE46_RELAY_CONTRACT:'+expected)
    return '#[version = "0.0.5"]\n'+block.replace(', Primitive=1','').replace('fn (','def @main(',1)


def frozen_route():
    return {'candidate_id':'node46-paper-TPS-oc2-ci1-baseline','identity':{
        'workload':WORKLOAD,'public_mode':'original','implementation_mode':0,
        'complete_config_entity':{'code_hash':None,'entity':[
            [n,'sp',[-1,v]] for n,v in zip(('tile_b','tile_h','tile_w','tile_ci','tile_co'),(1,14,14,1,4))
        ]+[['oc_nthread','ot',2],['h_nthread','ot',1]]}}}


def tensor_traffic(events):
    from c3_valid_window import dma_cells
    traffic={}
    for domain,label,width in [(1,'WGT',256),(2,'INP',16),(3,'ADD',64)]:
        visits={};padding=0
        requests=[e for e in events if e['kind']=='load' and e['domain']==domain]
        for e in requests:
            for _,origin in dma_cells(e):
                if origin==('zero',):
                    padding+=1
                    continue
                visits[origin]=visits.get(origin,0)+1
        payload=sum(visits.values())*width
        require(payload==sum(e['x']*e['y']*width for e in requests),'DDR_PAYLOAD_MISMATCH')
        traffic[label]={'requests':len(requests),'origin_count':len(visits),'visits':sum(visits.values()),
            'multiplicities':sorted(set(visits.values())),'bytes_per_vector':width,
            'payload_bytes':payload,'padding_vector_writes':padding}
    return traffic


@contextmanager
def node46_author_schedule(namespace):
    # Isolated worker hook; the old node24-only context is not repurposed.
    from vta.top import vta_conv2d
    original=vta_conv2d._schedule_conv2d_packed_impl
    def schedule(cfg,outs,residency_mode='original',**kwargs):
        require(residency_mode in ('original',0) and not kwargs,'UNEXPECTED_SCHEDULE_ROUTE')
        return namespace['schedule_conv2d_packed'](cfg,outs)
    vta_conv2d._schedule_conv2d_packed_impl=schedule
    try:yield
    finally:vta_conv2d._schedule_conv2d_packed_impl=original


def check_schedule_environment(namespace):
    import vta
    captured=namespace['get_env']()
    active=vta.get_env()
    require(captured.dev.vta_axis.same_as(active.dev.vta_axis),'AUTHOR_SCHEDULE_ENVIRONMENT_MISMATCH')
    return {'same_coproc_axis_identity':True,'captured_target':captured.TARGET,'active_target':active.TARGET}


def compile_worker(root):
    import tvm
    import vta
    from vta.testing import simulator
    from c3_banerjee_local import author_namespace
    from c3_banerjee_completion import CloseCallBoundary,ImprovedDoubleBuffer,SplitPaddedInputChannels
    from c3_tps_sync_release_prepare import complete_function,compiled_module_contract,canonical_config,_hardware
    from c3_official_window_baseline import buffer_abi,normalize_llvm,compiler_sources,plain
    from c3_official_window_method import audit_function
    from c3_official_window_reference import from_workload,postprocess_from_output_terms,semantic_gates
    from c3_valid_window import certify_trace,dma_cells
    from c3_tps_window_method import acc_spacing
    from vta.top.residency_dispatch import ExplicitResidencyDispatch
    verify(read(root/'freeze.json')['bindings'])
    require(Path(tvm._ffi.base._LIB._name).resolve()==(REPO/'build/libtvm.so').resolve(),'WRONG_COMPILER_DSO')
    require(Path(simulator.LIBS[0]._name).resolve()==(RUNTIME/'libvta_fsim.so').resolve(),'WRONG_SIM_DSO')
    out=root/'prepare'/'R';out.mkdir(parents=True)
    relay_text=extract_relay((GRAPH/'relay_000.txt').read_text())
    (out/'source_relay.txt').write_text(relay_text)
    module=tvm.relay.parse(relay_text)
    write(out/'source_relay.json',json.loads(tvm.ir.save_json(module)))
    route=frozen_route();actual_configs=[]

    @tvm.instrument.pass_instrument
    class Archive:
        def __init__(self):self.rows=[];self.order=[]
        def save(self,mod,info,when):
            if when=='before':self.order.append(info.name)
            if any(k in info.name for k in ('LiftAttrScope','InjectVirtualThread','InjectCoProcSync',
                'FoldUopLoop','CPUAccessRewrite','ImprovedDoubleBuffer','CloseCallBoundary','SplitPaddedInputChannels')):
                name='%03d_%s_%s'%(len(self.rows),when,info.name.replace('.','_'))
                (out/(name+'.json')).write_text(tvm.ir.save_json(mod));(out/(name+'.tir')).write_text(mod.script())
                self.rows.append({'name':name,'pass':info.name,'when':when})
        def run_before_pass(self,mod,info):self.save(mod,info,'before')
        def run_after_pass(self,mod,info):self.save(mod,info,'after')
    archive=Archive()
    with vta.Environment(dict(vta.get_env().cfg_dict,TARGET='sim')):
        namespace=author_namespace(AUTHOR,safe_uop_order=False)
        environment_binding=check_schedule_environment(namespace)
        original=namespace['schedule_conv2d_packed']
        def schedule(cfg,outs):
            check_schedule_environment(namespace)
            require(not bool(getattr(cfg,'is_fallback',False)),'FALLBACK')
            actual=plain(cfg.to_json_dict())
            require(canonical_config(actual)==canonical_config(route['identity']['complete_config_entity']),'ACTUAL_CONFIG')
            actual_configs.append(actual)
            return original(cfg,outs)
        namespace['schedule_conv2d_packed']=schedule
        env=vta.get_env();config=dict(vta.build_config().config);passes=list(config['tir.add_lower_pass'])
        at=next(i for i,(_,p) in enumerate(passes) if p.info.name=='tir.InjectCopyIntrin')
        passes.insert(at,(1,SplitPaddedInputChannels()))
        at=next(i for i,(phase,_) in enumerate(passes) if phase>=2)
        passes[at:at]=[(2,CloseCallBoundary()),(2,ImprovedDoubleBuffer())]
        config['tir.add_lower_pass']=passes;disabled={'AlterOpLayout','tir.CommonSubexprElimTIR'}
        target=tvm.target.Target(env.target,host=env.target_host)
        tvm.relay.backend.te_compiler.get().clear()
        with node46_author_schedule(namespace),ExplicitResidencyDispatch([route]) as dispatch:
            with tvm.transform.PassContext(opt_level=3,config=config,instruments=[archive],disabled_pass=disabled):
                graph,unused_lib,params=tvm.relay.build(module,target=target)
        del unused_lib
        require(not params and actual_configs and dispatch.summary()['all_routes_scheduled'],'FUSION_DISPATCH')
        def captured(suffix,when):
            rows=[r for r in archive.rows if r['pass'].endswith(suffix) and r['when']==when]
            require(len(rows)==1,'NONUNIQUE_CAPTURE:'+suffix)
            return tvm.ir.load_json((out/(rows[0]['name']+'.json')).read_text())
        parent=captured('FoldUopLoop','after');direct=captured('CPUAccessRewrite','after')
        require(tvm.ir.structural_equal(parent,captured('CPUAccessRewrite','before')),'PARENT_GAP')
        restored=tvm.ir.load_json(tvm.ir.save_json(parent))
        require(tvm.ir.structural_equal(parent,restored),'PARENT_ROUNDTRIP')
        final=vta.transform.CPUAccessRewrite()(restored)
        require(tvm.ir.structural_equal(final,direct),'FINAL_ROUNDTRIP')
        gv,function=complete_function(parent,SYMBOL);abi=buffer_abi(function)
        require([r['shape'] for r in abi]==SHAPES,'ABI_SHAPE')
        require([r['dtype'] for r in abi]==['int8','int8','int32','int8'],'ABI_DTYPE')
        for name,mod in [('parent',parent),('final',final)]:
            (out/(name+'.json')).write_text(tvm.ir.save_json(mod));(out/(name+'.tir')).write_text(mod.script())
        audit=audit_function(function);write(out/'events.json',audit['events'])
        require(audit['counts']['expanded_uop_steps']<=5000000,'EXPANSION_LIMIT')
        semantic=certify_trace(audit['events'])
        tail=postprocess_from_output_terms(semantic['output_terms'],abi)
        require(tail==[{'op':'add','source':{'kind':'buffer','param':2}},
                       {'op':'shr','source':{'kind':'imm','value':8}},
                       {'op':'min','source':{'kind':'imm','value':127}},
                       {'op':'max','source':{'kind':'imm','value':-127}}],'FUSION_TAIL')
        spec=from_workload(WORKLOAD,abi,tail)
        independent=semantic_gates(spec,semantic['output_terms'])
        hardware=_hardware(root)
        spacing=acc_spacing(audit['events'],hardware['verified_acc_interval'])
        write(out/'reference_contract.json',spec.contract());write(out/'independent_semantics.json',independent)
        write(out/'logical_certificate.json',{'status':'LOGICAL_PASSED_NATIVE_PENDING','counts':audit['counts'],
            'statistics':audit['statistics'],'acc_spacing':spacing,'full_fusion_tail':tail,'roundtrip':True})
        write(out/'tensor_requests.json',tensor_traffic(audit['events']))
        with tvm.transform.PassContext(opt_level=3,disabled_pass=disabled):
            lib=tvm.build(final,target=target);direct_lib=tvm.build(direct,target=target)
        llvm=lib.get_source('ll');require(normalize_llvm(llvm)==normalize_llvm(direct_lib.get_source('ll')),'LLVM_ROUNDTRIP')
        (out/'module.ll').write_text(llvm);lib.export_library(str(out/'fsim_module.so'))
        contract=compiled_module_contract(final,SYMBOL,llvm);require(contract['abi']==abi,'FINAL_ABI')
        write(out/'compiled_cache_contract.json',contract['compiled_cache_contract'])
        write(out/'alu_initializer_contract.json',contract['alu_initializer_contract'])
        write(out/'graph.json',json.loads(graph))
        write(out/'compilation_context.json',{'route':route,'actual_configs':actual_configs,'dispatch':dispatch.summary(),
            'safe_uop_order':False,'pass_order':archive.order,'captures':archive.rows,'target':str(target),
            'host':str(target.host),'environment':env.cfg_dict,'extra_passes':[[int(i),p.info.name] for i,p in passes],
            'author_environment_binding':environment_binding,
            'disabled_passes':sorted(disabled),'compiler_sources':compiler_sources(),
            'not_exact_author_stack':True,'node':46,'new_optimization':False})
    descriptor={'program':'R','node':46,'symbol':SYMBOL,'global_var':gv.name_hint,'abi':abi,'workload':WORKLOAD,
        'parent_module':str(out/'parent.json'),'final_module':str(out/'final.json'),'module_path':str(out/'fsim_module.so'),
        'reference_contract':str(out/'reference_contract.json'),'artifact_bindings':{str(p):sha(p) for p in out.iterdir() if p.is_file()}}
    write(out/'descriptor.json',descriptor)
    write(root/'prepare/result.json',{'status':'NODE46_PREPARED','programs':{'R':descriptor}})
    write(root/'prepare_complete.json',{'status':'NODE46_PREPARED','result_sha256':sha(root/'prepare/result.json')})


def inspect_worker(root):
    """Classify the unchanged, fully qualified baseline; no candidate or calls."""
    import tvm
    from c3_uop_identity_applicability import inspect_function,classify
    from c3_tps_aggregation_capacity import organization
    verify(read(root/'freeze.json')['bindings'])
    parent=tvm.ir.load_json((root/'prepare/R/parent.json').read_text())
    fn=parent[list(parent.functions)[0]]
    census=inspect_function(fn)
    require(census['status']=='FINITE_DOMAIN_CONTENT_SCREEN_COMPLETE','BASELINE_IDENTITY_SUPPORT_UNRESOLVED')
    events=read(root/'prepare/R/events.json')
    classes=classify([e for e in events if e['kind']=='kernel'])
    mac=[r for r in classes if r['mode_reset']==[0,0]]
    require(sum(r['signatures'] for r in mac)==census['MAC_identities'] and
            sum(r['content_classes'] for r in mac)==census['MAC_content_classes'],'INDEPENDENT_CLASS_CENSUS_CHANGED')
    write(root/'identity_census.json',census)
    rows=[]
    for mode in ('detailed','minimal'):
        for index in range(1,8):
            unit=root/f'R_{mode}';text=(unit/f'native_call_{index:02d}.txt').read_text()
            native=read(unit/f'native_certificate_{index:02d}.json')
            org=organization(text,native)
            require(org['host_identity_base_uops_total']==census['identity_base_uops'],'ACTUAL_IDENTITY_CENSUS_CHANGED')
            rows.append({'mode':mode,'call':index,**org})
    fields=('MAC_GEMM','RESET_GEMM','ALU','all_instructions','compute_NOP','UOP_LOAD','UOP_LOAD_bytes',
            'serialized_uop_bytes','UOP_SRAM_wraps','submissions')
    require(all(tuple(x[k] for k in fields)==tuple(rows[0][k] for k in fields) for x in rows),'QUALIFIED_ORGANIZATION_DIFFERS')
    write(root/'organization.json',rows)
    traffic=read(root/'prepare/R/tensor_requests.json')
    require(traffic['WGT']['payload_bytes']==884736 and traffic['WGT']['multiplicities']==[1], 'WEIGHT_ONCE_NOT_CONFIRMED')
    write(root/'inspection.json',{'status':'QUALIFIED_NODE46_BASELINE_CONTENT_AUDITED',
          'MAC_identities':census['MAC_identities'],'MAC_content_classes':census['MAC_content_classes'],
          'MAC_redundant_identities':census['MAC_redundant_identities'],
          'native_capacity_pressure':rows[0]['UOP_SRAM_wraps']>0,
          'same_detector_as_node34':True,'optimized_variant_built':False,
          'new_inspection_target_calls':0,'performance_conclusion':None,
          'next_gate':'independent optimized-variant contract and full qualification, if justified'})
    verify(read(root/'freeze.json')['bindings'])


def worker(root,phase,mode=None):
    os.environ['C3_OFFICIAL_WINDOW_COLLECTION']=mode or 'detailed'
    if phase=='compile':compile_worker(root)
    elif phase=='inspect':inspect_worker(root)
    else:qualification_worker(root,'R',mode)


def run_worker(root,phase,mode=None):
    import psutil
    name=phase+('_'+mode if mode else '')
    argv=[str(PYTHON),str(Path(__file__).resolve()),'_worker',str(root),'--phase',phase]
    if mode:argv+=['--mode',mode]
    start=time.monotonic();peak=0;limit=None
    with (root/(name+'.log')).open('x') as log:
        environment=env_for_worker()
        environment.update(VTA_UOP_BUFFER_BYTES='33554432',VTA_INSN_BUFFER_BYTES='33554432',
                           VTA_INSN_SUBMIT_THRESHOLD_BYTES='33554432')
        p=subprocess.Popen(argv,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while p.poll() is None:
            try:
                tree=[psutil.Process(p.pid)]+psutil.Process(p.pid).children(recursive=True)
                peak=max(peak,sum(x.memory_info().rss for x in tree if x.is_running()))
            except psutil.Error:pass
            if peak>8*1024**3 or time.monotonic()-start>600:
                limit='MEMORY' if peak>8*1024**3 else 'TIME'
                os.killpg(p.pid,signal.SIGKILL);p.wait();break
            time.sleep(.1)
    row={'returncode':p.returncode,'limit':limit,'peak_sampled_tree_rss':peak,
         'resource_control_seconds':time.monotonic()-start,'argv':argv}
    write(root/(name+'_worker.json'),row)
    return p.returncode==0 and limit is None


def start_body(root):
    app=read(APPLICABILITY/'audit.json');verify(app['source_sha256'])
    require(app['prospective']['prospective_oc2_representative']['id']=='b1h14w14ci1co4oc2hth1','SELECTION_DRIFT')
    hardware=read(HARDWARE);verify(hardware['source_bindings'])
    shutil.copyfile(HARDWARE,root/'hardware_archive.json')
    from c3_official_window_baseline import compiler_sources
    compiler=compiler_sources()
    paths=list(HERE.glob('c3_*.py'))+[HELPER,RUNTIME/'libvta_fsim.so',
        GRAPH/'relay_000.txt',APPLICABILITY/'audit.json',HARDWARE]
    paths+=list(AUTHOR.glob('*'))
    paths+=list((REPO/'vta/python').rglob('*.py'))
    paths+=[REPO/'3rdparty/vta-hw/config/vta_config.json',REPO/'3rdparty/vta-hw/config/pkg_config.py',
            HERE/'test_c3_node46_tps_local.py']
    bindings={str(p.resolve()):sha(p) for p in paths if p.is_file()}
    bindings.update(compiler);bindings.update(hardware['source_bindings'])
    write(root/'freeze.json',{'node':46,'max_calls':14,'bindings':bindings})
    write(root/'call_ledger.json',[{'program':'R','mode':mode,'call_index':i,'kind':kind,'seed':seed,
        'target_calls':1,'performance_sample':False} for mode in ('detailed','minimal')
        for i,(kind,seed) in enumerate(zip(KINDS,SEEDS),1)])
    write(root/'contract.json',{'node':46,'route':frozen_route(),'max_target_calls':14,
        'order':['compile','R_detailed','R_minimal','read_only_identity_inspection'],'limits':{'worker_rss_GiB':8,'worker_seconds':600,'expanded_steps':5000000},
        'on_failure':'stop; no retry, substitute tile, extra candidate, or board call',
        'new_optimization':False,'requires_runtime_native_and_FIFO_certificates':True})
    status='STOPPED_COMPILE'
    if run_worker(root,'compile'):
        ready=read(root/'prepare_complete.json')
        require(ready['status']=='NODE46_PREPARED' and ready['result_sha256']==sha(root/'prepare/result.json'),'PREPARE_STATUS')
        require(read(root/'prepare/result.json')['status']=='NODE46_PREPARED','PREPARE_RESULT_STATUS')
        verify(read(root/'freeze.json')['bindings'])
        for mode in ('detailed','minimal'):
            status='STOPPED_'+mode.upper()
            if not run_worker(root,'qualify',mode):break
        else:
            compare_modes(root,'R');status='NODE46_LOCAL_QUALIFIED'
    if status=='NODE46_LOCAL_QUALIFIED':
        status='STOPPED_IDENTITY_INSPECTION'
        if run_worker(root,'inspect'):status='NODE46_LOCAL_QUALIFIED'
    return status


def start(root):
    root.mkdir(exist_ok=False)
    try:
        status=start_body(root)
    except Exception as error:
        status='STOPPED_CONTROLLER_GATE'
        write(root/'failure_controller.json',{'type':type(error).__name__,'message':str(error)})
        traceback.print_exc()
    accounting=call_accounting(root);write(root/'call_accounting.json',accounting)
    write(root/'status.json',{'status':status,'paused':True,'board_contacted':False})
    print(json.dumps({'status':status,'accounting':accounting},indent=2))
    return 0 if status=='NODE46_LOCAL_QUALIFIED' else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','_worker']);p.add_argument('output',type=Path)
    p.add_argument('--phase',choices=['compile','qualify','inspect']);p.add_argument('--mode',choices=['detailed','minimal'])
    a=p.parse_args();root=a.output.resolve()
    if a.action=='run':sys.exit(start(root))
    else:
        try:worker(root,a.phase,a.mode)
        except Exception as e:
            write(root/('failure_'+a.phase+('_'+a.mode if a.mode else '')+'.json'),{'type':type(e).__name__,'message':str(e)})
            traceback.print_exc();sys.exit(1)
