"""One bounded local A/B probe: quotient periodic MAC closure signatures.

The qualified node34 B is adopted unchanged. A changes only the two MAC
closures, preserving DMA, tensor requests, uops, loops and logical tokens.
No board path, runtime modification, tuning or additional variant.
"""
import argparse
from copy import deepcopy
import json
import os
import re
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

from c3_node34_tps_local import (HERE,REPO,HOLD,RUNTIME,HELPER,PYTHON,SYMBOL,
                               read,write,sha,verify,require,env_for_worker)
from c3_tps_sync_release_local import qualification_worker,compare_modes,call_accounting,KINDS,SEEDS

BASELINE=HOLD/'20261017_c3_node34_tps_local_run02'
EVIDENCE=HOLD/'20261017_c3_node34_tps_remaining_readonly01'


def transform(function):
    """Only replace proven k%2 subexpressions inside MAC initializers.

    Original host counters, DMA expressions, scope order and tokens survive.
    A fresh Let immediately outside each initializer projects its closure.
    Unknown uses of k or additional closure variables fail closed.
    """
    import tvm
    from tvm import tir
    changes=[];inverse={}
    def visit(node,host):
        if isinstance(node,tir.AttrStmt) and node.attr_key=='coproc_uop_scope':
            calls=[]
            tir.stmt_functor.post_order_visit(node.body,lambda x:calls.append(x)
                if isinstance(x,tir.Call) and x.op.name=='tir.vta.uop_push' else None)
            require(calls,'EMPTY_INITIALIZER')
            if not all(int(c.args[0])==0 and int(c.args[1])==0 for c in calls):return node
            free=list(tir.analysis.undefined_vars(node.body))
            require(len(free)==1 and str(free[0].dtype)=='int32','MAC_CLOSURE_SUPPORT')
            counter=free[0]
            owners=[f for f in host if f.loop_var.same_as(counter)]
            require(len(owners)==1,'UNBOUND_MAC_COUNTER')
            loop=owners[0]
            require(isinstance(loop.min,tir.IntImm) and isinstance(loop.extent,tir.IntImm) and
                    int(loop.min)==0 and int(loop.extent)==32 and loop.kind==tir.ForKind.SERIAL and
                    loop.thread_binding is None and not loop.annotations,'COUNTER_DOMAIN_NOT_FROZEN')
            bank=tir.Var('c3_uop_bank_'+str(len(changes)),'int32');matched=[]
            def replace(expr):
                if isinstance(expr,tir.FloorMod) and expr.a.same_as(counter) and isinstance(expr.b,tir.IntImm) and int(expr.b)==2:
                    matched.append(True);return bank
            body=tir.stmt_functor.ir_transform(node.body,replace,None,['tir.FloorMod'])
            remaining=list(tir.analysis.undefined_vars(body))
            require(matched and len(remaining)==1 and remaining[0].same_as(bank),'RAW_COUNTER_REMAINS_IN_INITIALIZER')
            scope=tir.AttrStmt(node.node,node.attr_key,node.value,body,node.span)
            expression=tir.floormod(counter,2)
            inverse[bank]=expression
            changes.append({'domain':[0,32],'projection':'k%2','signature_range':[0,1],
                            'matched_expressions':len(matched),'static_MAC_scope':len(changes)})
            return tir.LetStmt(bank,expression,scope)
        if isinstance(node,tir.AttrStmt):
            return tir.AttrStmt(node.node,node.attr_key,node.value,visit(node.body,host),node.span)
        if isinstance(node,tir.SeqStmt):return tir.SeqStmt([visit(x,host) for x in node.seq])
        if isinstance(node,tir.For):
            return tir.For(node.loop_var,node.min,node.extent,node.kind,visit(node.body,host+[node]),
                           node.thread_binding,node.annotations,node.span)
        if isinstance(node,tir.LetStmt):return tir.LetStmt(node.var,node.value,visit(node.body,host),node.span)
        if isinstance(node,tir.IfThenElse):
            raise ValueError('UNFROZEN_HOST_BRANCH')
        return node
    candidate=function.with_body(visit(function.body,[]))
    require(len(changes)==2,'EXPECTED_TWO_MAC_SCOPES')
    def undo(node):
        if isinstance(node,tir.LetStmt) and node.var in inverse:
            return tir.stmt_functor.substitute(node.body,{node.var:inverse[node.var]})
    restored=tir.stmt_functor.ir_transform(candidate.body,None,undo,['tir.LetStmt'])
    require(tvm.ir.structural_equal(restored,function.body),'EXACT_INVERSE_DID_NOT_RESTORE_PARENT')
    return candidate,{'status':'EXACT_INVERSE_AND_CLOSURE_PROJECTION_PASSED','changes':changes,
                      'DMA_control_and_non_MAC_body_restored_exactly':True}


def event_contract(before,after):
    before=json.loads(json.dumps(before));after=json.loads(json.dumps(after))
    require(len(before)==len(after),'LOGICAL_EVENT_CENSUS_CHANGED')
    mapping={};calls=0
    for old,new in zip(before,after):
        a,b=deepcopy(old),deepcopy(new)
        if old['kind']=='kernel':
            for key in ('cache_identity','signature','source_scope_sha256'):
                a.pop(key,None);b.pop(key,None)
            if old['mode_reset']==[0,0]:
                require(new['signature']==[old['signature'][0]%2],'PROJECTION_VALUE_MISMATCH')
                calls+=1
            else:require(old['signature']==new['signature'],'NON_MAC_SIGNATURE_CHANGED')
            key=(new['scope'],tuple(new['signature']))
            require(key not in mapping or mapping[key]==new['uops'],'NEW_SIGNATURE_CONTENT_MUTATION')
            mapping[key]=new['uops']
        require(a==b,'COMPUTATION_TENSOR_REQUEST_OR_TOKEN_CHANGED')
    require(calls==128,'MAC_CALL_DOMAIN_CHANGED')
    return {'status':'EXACT_ORDERED_EVENT_CONTENT_EQUAL_EXCEPT_DECLARED_CACHE_IDENTITY',
            'MAC_calls':calls,'candidate_cache_identities':len(mapping),'ordered_events':len(before)}


def project_lowered_module(lowered):
    """Project after tvm.lower, then use build's target-map entry to skip re-lowering."""
    import tvm
    from c3_tps_sync_release_prepare import complete_function
    gv,fn=complete_function(lowered,SYMBOL)
    projected,proof=transform(fn)
    return tvm.IRModule({gv:projected},attrs=lowered.attrs),proof


def llvm_signature_provenance(llvm):
    """Bounded final-output gate: both MAC closure stores must hold raw_k & 1.

    Fail closed for other SSA forms. The inverse IR proof and event contract
    remain necessary; this gate additionally checks the optimization survived.
    """
    from c3_official_window_llvm import _function_bodies,_calls
    functions=_function_bodies(llvm)
    rows=[];roots=set()
    for function,body in functions.items():
        for name,fields in _calls(body,'VTAPushGEMMOp'):
            if fields[-1]=='i32 0':continue
            require(len(fields)==4 and fields[-1]=='i32 4','FINAL_MAC_SIGNATURE_ABI')
            ptr=fields[2].split()[-1]
            require(ptr.startswith('%'),'FINAL_MAC_SIGNATURE_NOT_LOCAL_STORAGE')
            stores=re.findall(r'\bstore i32 (%[-\w.$]+), ptr '+re.escape(ptr)+r'(?=[,\s])',body)
            require(stores,'FINAL_MAC_SIGNATURE_STORE_NOT_FOUND')
            raw=[]
            for value in stores:
                definition=re.search(r'^\s*'+re.escape(value)+r' = and i32 (%[-\w.$]+), 1\s*$',body,re.M)
                require(definition is not None,'FINAL_MAC_SIGNATURE_IS_NOT_PROJECTED_BANK')
                counter=definition[1];raw.append(counter);roots.add(counter)
                # Actual DMA offset arithmetic must retain the complete counter.
                for scale in (49,9):
                    require(re.search(r'= mul (?:nuw )?(?:nsw )?i32 '+re.escape(counter)+', '+str(scale)+r'\s*$',body,re.M),
                            'FINAL_DMA_RAW_COUNTER_NOT_PRESERVED')
            rows.append({'host_function':function,'handle':fields[0],'initializer':fields[1],
                         'signature_storage':ptr,'stores':stores,'raw_counters':raw,
                         'signature_range':[0,1],'raw_DMA_scales':[49,9]})
    require(len(rows)==2 and len(roots)==1,'FINAL_EXPECTED_TWO_MAC_PROJECTED_SITES')
    return {'status':'FINAL_LLVM_MAC_SIGNATURE_PROVENANCE_PASSED','sites':rows,
            'known_SSA_support_only':True,'runtime_signature_encoding':'one int32, values 0 or 1'}


def prepare_worker(root):
    import tvm,vta
    from c3_tps_sync_release_prepare import complete_function,compiled_module_contract,_hardware
    from c3_official_window_baseline import buffer_abi,normalize_llvm
    from c3_official_window_method import audit_function
    from c3_official_window_reference import from_workload,postprocess_from_output_terms,semantic_gates
    from c3_valid_window import certify_trace
    from c3_tps_window_method import acc_spacing
    from c3_node34_tps_local import tensor_traffic
    from vta.testing import simulator
    verify(read(root/'freeze.json')['bindings'])
    require(Path(tvm._ffi.base._LIB._name).resolve()==(REPO/'build/libtvm.so').resolve(),'COMPILER_DSO_CHANGED')
    require(Path(simulator.LIBS[0]._name).resolve()==(RUNTIME/'libvta_fsim.so').resolve(),'RUNTIME_DSO_CHANGED')
    out=root/'prepare/R';out.mkdir(parents=True)
    before=tvm.ir.load_json((BASELINE/'prepare/R/parent.json').read_text())
    hardware=_hardware(root)
    with vta.Environment(dict(vta.get_env().cfg_dict,TARGET='sim')):
        target=tvm.target.Target(vta.get_env().target,host=vta.get_env().target_host)
        gv,old=complete_function(before,SYMBOL)
        fn,decision=transform(old)
        parent=tvm.IRModule({gv:fn},attrs=before.attrs)
        restored=tvm.ir.load_json(tvm.ir.save_json(parent))
        require(tvm.ir.structural_equal(parent,restored),'CANDIDATE_PARENT_ROUNDTRIP')
        final=vta.transform.CPUAccessRewrite()(parent)
        final_rt=vta.transform.CPUAccessRewrite()(restored)
        require(tvm.ir.structural_equal(final,final_rt),'CANDIDATE_FINAL_ROUNDTRIP')
        for name,mod in [('parent',parent),('final',final)]:
            (out/(name+'.json')).write_text(tvm.ir.save_json(mod));(out/(name+'.tir')).write_text(mod.script())
        write(out/'transform.json',decision)
        audit=audit_function(fn);events=audit['events']
        write(out/'events.json',events)
        delta=event_contract(read(BASELINE/'prepare/R/events.json'),events)
        require(delta['candidate_cache_identities']==14,'UNEXPECTED_IDENTITY_CENSUS')
        write(out/'event_contract.json',delta)
        semantic=certify_trace(events);abi=buffer_abi(fn)
        old_d=read(BASELINE/'prepare/result.json')['programs']['R']
        require(abi==old_d['abi'],'ABI_CHANGED')
        tail=postprocess_from_output_terms(semantic['output_terms'],abi)
        spec=from_workload(old_d['workload'],abi,tail)
        require(spec.contract()==read(BASELINE/'prepare/R/reference_contract.json'),'FUSION_REFERENCE_CHANGED')
        write(out/'reference_contract.json',spec.contract())
        write(out/'independent_semantics.json',semantic_gates(spec,semantic['output_terms']))
        write(out/'logical_certificate.json',{'counts':audit['counts'],'acc_spacing':acc_spacing(events,hardware['verified_acc_interval']),
              'status':'EXACT_EVENT_AND_LOGICAL_PASSED_NATIVE_PENDING'})
        traffic=tensor_traffic(events)
        require(traffic==read(BASELINE/'prepare/R/tensor_requests.json'),'TENSOR_TRAFFIC_CHANGED')
        write(out/'tensor_requests.json',traffic)
        disabled={'AlterOpLayout','tir.CommonSubexprElimTIR'}
        with tvm.transform.PassContext(opt_level=3,disabled_pass=disabled):
            # No-op rebuild binds the new compilation scaffold to sealed B.
            original_final=vta.transform.CPUAccessRewrite()(before)
            old_final=tvm.ir.load_json((BASELINE/'prepare/R/final.json').read_text())
            require(tvm.ir.structural_equal(original_final,old_final),'ADOPTED_B_FINAL_CHANGED')
            blib=tvm.build(original_final,target=target)
            require(normalize_llvm(blib.get_source('ll'))==normalize_llvm((BASELINE/'prepare/R/module.ll').read_text()),
                    'NO_TRANSFORM_BASELINE_LLVM_CHANGED')
            # tvm.build(IRModule) calls lower again and inlines integer Lets.
            # Lower B normally first; insert the projection only after that.
            lowered=tvm.lower(original_final)
            lowered_rt=tvm.lower(final_rt)
            require(tvm.ir.structural_equal(lowered,lowered_rt),'PREPROJECTION_LOWERED_B_CHANGED')
            map_b=tvm.build({target:lowered})
            require(normalize_llvm(map_b.get_source('ll'))==normalize_llvm(blib.get_source('ll')),
                    'TARGET_MAP_BASELINE_LLVM_CHANGED')
            late,late_proof=project_lowered_module(lowered)
            late_rt,_=project_lowered_module(lowered_rt)
            require(tvm.ir.structural_equal(late,late_rt),'LATE_PROJECTION_ROUNDTRIP')
            lib=tvm.build({target:late});rtlib=tvm.build({target:late_rt})
        llvm=lib.get_source('ll')
        require(normalize_llvm(llvm)==normalize_llvm(rtlib.get_source('ll')),'CANDIDATE_LLVM_ROUNDTRIP')
        require(normalize_llvm(llvm)!=normalize_llvm(blib.get_source('ll')),'CANDIDATE_EXECUTION_REPRESENTATION_EQUALS_B')
        signature_gate=llvm_signature_provenance(llvm)
        write(out/'final_signature_provenance.json',signature_gate)
        write(out/'late_projection.json',late_proof)
        (out/'codegen_input.json').write_text(tvm.ir.save_json(late))
        (out/'codegen_input.tir').write_text(late.script())
        (out/'module.ll').write_text(llvm);lib.export_library(str(out/'fsim_module.so'))
        contract=compiled_module_contract(final,SYMBOL,llvm)
        old_cache=read(BASELINE/'prepare/R/compiled_cache_contract.json')
        def relation(c):
            handles={};initializers={}
            return [(handles.setdefault(s['handle'],len(handles)),initializers.setdefault(s['initializer'],len(initializers)),s['signature_bytes']) for s in c['sites']]
        require(relation(contract['compiled_cache_contract'])==relation(old_cache),'STATIC_INITIALIZER_BINDING_RELATION_CHANGED')
        require(contract['abi']==abi,'FINAL_ABI_CHANGED')
        write(out/'compiled_cache_contract.json',contract['compiled_cache_contract'])
        write(out/'alu_initializer_contract.json',contract['alu_initializer_contract'])
        write(out/'compilation_contract.json',{'adopted_B_noop_IR_and_LLVM_equal':True,'candidate_roundtrip_equal':True,
              'target_map_B_equals_normal_B':True,'projection_stage':'after tvm.lower, before target-map build',
              'final_LLVM_signature_provenance_passed':True,'candidate_execution_differs_from_B':True,
              'target':str(target),'host':str(target.host),'opt_level':3,'disabled_pass':sorted(disabled),
              'candidate':'A','internal_worker_label':'R','runtime_unchanged':True})
    desc={'program':'R','variant':'A','global_var':gv.name_hint,'symbol':SYMBOL,'abi':abi,'workload':old_d['workload'],
          'parent_module':str(out/'parent.json'),'final_module':str(out/'final.json'),'module_path':str(out/'fsim_module.so'),
          'reference_contract':str(out/'reference_contract.json'),'artifact_bindings':{str(p):sha(p) for p in out.iterdir() if p.is_file()}}
    write(root/'prepare/result.json',{'status':'IDENTITY_A_PREPARED','programs':{'R':desc}})
    write(root/'prepare_complete.json',{'status':'IDENTITY_A_PREPARED','result_sha256':sha(root/'prepare/result.json')})


def native_pair(before,after,bcert,acert):
    """Keep non-UOP physical access/order; certify each own UOP stream separately."""
    from c3_sync_release_audit import Graph,access_model,requirements,Obligations
    graphs=[Graph(before),Graph(after)]
    accesses=[access_model(before,bcert),access_model(after,acert)]
    mats=[sorted(c['material_map'],key=lambda r:r['tir_event']) for c in (bcert,acert)]
    require([m['tir_event'] for m in mats[0]]==[m['tir_event'] for m in mats[1]],'ACTUAL_MATERIAL_BINDING_CHANGED')
    mapping={a['instruction']:b['instruction'] for a,b in zip(*mats)}
    for a,b in zip(*mats):
        require(a['kind']==b['kind'] and a['description']==b['description'],'ACTUAL_MATERIAL_KIND_CHANGED')
        for field in ('reads','writes','dead_reads'):
            left={k:v for k,v in getattr(accesses[0][a['instruction']],field).items() if k!='UOP'}
            right={k:v for k,v in getattr(accesses[1][b['instruction']],field).items() if k!='UOP'}
            require(left==right,'ACTUAL_TENSOR_PHYSICAL_ACCESS_CHANGED')
    for graph,access in zip(graphs,accesses):Obligations(requirements(access),graph.ancestors,True)
    retained=0;dead_unordered=0
    for r in requirements(accesses[0],include_dead=True):
        if r['domain']=='UOP':continue
        a,b=r['source'],r['target']
        require(a in mapping and b in mapping,'NON_UOP_ACCESS_WITHOUT_MATERIAL')
        if not (graphs[0].ancestors[b]>>a&1):dead_unordered+=1;continue
        require(graphs[1].ancestors[mapping[b]]>>mapping[a]&1,'ORIGINAL_PHYSICAL_ORDER_LOST')
        retained+=1
    controls=[]
    for cert in (bcert,acert):
        controls.append([(r['tir_event'],r['kind'],r['lane'],r['ordinal']) for r in cert['dependency_map']])
    require(controls[0]==controls[1],'ACTUAL_LOGICAL_TOKEN_BINDING_CHANGED')
    return {'status':'NON_UOP_ACCESS_AND_ORIGINAL_ORDER_RETAINED','retained_obligations':retained,
            'original_dead_read_unordered':dead_unordered,'material_events':len(mapping),
            'logical_token_identity_sequence_equal':True,'each_native_UOP_reference_separately_certified':True}


def compare_worker(root):
    from c3_five_tradeoff_commands import decode_submissions
    from c3_tps_device_cost_readonly import decode
    from collections import Counter
    compare_modes(root,'R')
    rows=[]
    for mode in ('detailed','minimal'):
        for index in range(1,8):
            bdir=BASELINE/f'R_{mode}';adir=root/f'R_{mode}'
            bt=(bdir/f'native_call_{index:02d}.txt').read_text();at=(adir/f'native_call_{index:02d}.txt').read_text()
            bc=read(bdir/f'native_certificate_{index:02d}.json');ac=read(adir/f'native_certificate_{index:02d}.json')
            require(sha(bdir/f'native_call_{index:02d}.txt')==bc['dump_sha256'] and
                    sha(adir/f'native_call_{index:02d}.txt')==ac['dump_sha256'],'DUMP_IDENTITY_CHANGED')
            pair=native_pair(decode_submissions(bt)[0],decode_submissions(at)[0],bc,ac)
            write(adir/f'baseline_pair_{index:02d}.json',pair)
            be=read(bdir/f'call_{index:02d}_end.json');ae=read(adir/f'call_{index:02d}_end.json')
            require(be['input_sha256']==ae['input_sha256'] and be['reference_sha256']==ae['reference_sha256'] and
                    be['output_sha256']==ae['output_sha256'],'INPUT_OR_OUTPUT_NOT_PAIRED')
            work=[]
            for text in (bt,at):
                c=Counter()
                for r in decode(text):c[r['kind']]+=r['steps']
                work.append(dict(c))
            require(work[0]==work[1],'ACTUAL_WORK_CHANGED')
            rows.append({'mode':mode,'call_index':index,'same_inputs_and_outputs':True,'work':work[0],
                         'B':{k:v for k,v in bc['uop_values'].items() if k!='checks'},
                         'A':{k:v for k,v in ac['uop_values'].items() if k!='checks'}})
    write(root/'paired_results.json',{'status':'ALL_14_NATIVE_PAIRS_PASSED','calls':rows,'performance_conclusion':None})
    verify(read(root/'freeze.json')['bindings'])


def run_worker(root,phase,mode=None):
    import psutil
    name=phase+('_'+mode if mode else '')
    argv=[str(PYTHON),str(Path(__file__).resolve()),'_worker',str(root),'--phase',phase]
    if mode:argv+=['--mode',mode]
    env=env_for_worker();env.update(VTA_UOP_BUFFER_BYTES='33554432',VTA_INSN_BUFFER_BYTES='33554432',VTA_INSN_SUBMIT_THRESHOLD_BYTES='33554432')
    start=time.monotonic();peak=0;limit=None
    with (root/(name+'.log')).open('x') as log:
        child=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while child.poll() is None:
            try:
                p=psutil.Process(child.pid);peak=max(peak,sum(x.memory_info().rss for x in [p,*p.children(recursive=True)]))
            except psutil.Error:pass
            if peak>8*1024**3 or time.monotonic()-start>600:
                limit='MEMORY' if peak>8*1024**3 else 'TIME';os.killpg(child.pid,signal.SIGKILL);break
            time.sleep(.1)
        code=child.wait()
    write(root/(name+'_worker.json'),{'returncode':code,'limit':limit,'peak_sampled_tree_rss':peak,
         'resource_control_seconds':time.monotonic()-start,'argv':argv})
    return code==0 and limit is None


def run(root):
    root.mkdir(exist_ok=False);status='STOPPED_IDENTITY_ADOPTION'
    try:
        for origin in (BASELINE,EVIDENCE):
            verify({str(origin/p):s for p,s in read(origin/'completion_manifest.json')['artifact_sha256'].items()})
        require(read(BASELINE/'status.json')['status']=='NODE34_LOCAL_QUALIFIED','BASELINE_NOT_QUALIFIED')
        bindings=read(BASELINE/'freeze.json')['bindings'];verify(bindings)
        bindings.update({str(p.resolve()):sha(p) for p in [Path(__file__),HERE/'test_c3_node34_uop_identity.py',
                         BASELINE/'completion_manifest.json',EVIDENCE/'completion_manifest.json']})
        write(root/'freeze.json',{'bindings':bindings,'baseline':str(BASELINE),'variant':'A'})
        shutil.copyfile(BASELINE/'hardware_archive.json',root/'hardware_archive.json')
        write(root/'adoption.json',{'B':str(BASELINE),'B_full_FSim_calls_adopted':14,'B_calls_new':0,
              'helper_worker_label':'R means A in this attempt; it does not replace archived baseline R'})
        write(root/'contract.json',{'variants':['B_adopted','A_periodic_MAC_identity'],'new_FSim_budget':14,
              'target':'local FSim only','order':['prepare','A_detailed','A_minimal','native_pair_compare'],
              'change':'two MAC closures k -> k%2 signature; raw k outside closures retained',
              'fixed':['Tile','threads','tensor requests','MAC terms/order','postprocess','logical tokens','runtime/helper'],
              'limits':{'worker_GiB':8,'worker_seconds':600,'expanded_vector_steps':5000000,'FIFO_states':1000000},
              'on_error':'close attempt without retry or substitute; preserve evidence',
              'performance_samples':0,'FPGA_calls':0})
        write(root/'call_ledger.json',[{'program':'R','mode':mode,'call_index':i,'kind':kind,'seed':seed,
              'target_calls':1,'performance_sample':False} for mode in ('detailed','minimal')
              for i,(kind,seed) in enumerate(zip(KINDS,SEEDS),1)])
        status='STOPPED_PREPARE'
        if run_worker(root,'prepare'):
            ready=read(root/'prepare_complete.json')
            require(ready['status']=='IDENTITY_A_PREPARED' and ready['result_sha256']==sha(root/'prepare/result.json'), 'PREPARE_NOT_COMPLETE')
            require(read(root/'prepare/result.json')['status']=='IDENTITY_A_PREPARED','PREPARE_RESULT_STATUS')
            for mode in ('detailed','minimal'):
                status='STOPPED_'+mode.upper()
                if not run_worker(root,'qualify',mode):break
            else:
                status='STOPPED_NATIVE_COMPARE'
                if run_worker(root,'compare'):status='IDENTITY_A_LOCAL_QUALIFIED'
    except Exception as error:
        write(root/'failure_controller.json',{'type':type(error).__name__,'message':str(error)});traceback.print_exc()
    accounting=call_accounting(root)
    write(root/'call_accounting.json',accounting)
    write(root/'status.json',{'status':status,'paused':True,'FPGA_calls':0,'performance_conclusion':None})
    print(json.dumps({'status':status,'accounting':accounting},indent=2))
    return 0 if status=='IDENTITY_A_LOCAL_QUALIFIED' else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('run','_worker'));p.add_argument('output',type=Path)
    p.add_argument('--phase',choices=('prepare','qualify','compare'));p.add_argument('--mode',choices=('detailed','minimal'))
    a=p.parse_args();root=a.output.resolve()
    if a.action=='run':sys.exit(run(root))
    try:
        if a.phase=='prepare':prepare_worker(root)
        elif a.phase=='compare':compare_worker(root)
        else:qualification_worker(root,'R',a.mode)
    except Exception as error:
        write(root/('failure_'+str(a.phase)+('_'+a.mode if a.mode else '')+'.json'),
              {'type':type(error).__name__,'message':str(error)});traceback.print_exc();sys.exit(1)
