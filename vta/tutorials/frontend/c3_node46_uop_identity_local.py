"""One generic-equivalence-driven node46 A, adopting the qualified B subset.

Local compile and 14 FSim calls only. No production source or board changes.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
from c3_node46_tps_local import (HERE,REPO,HOLD,RUNTIME,HELPER,PYTHON,SYMBOL,
                               read,write,sha,verify,require,env_for_worker)
from c3_tps_sync_release_local import qualification_worker,compare_modes,call_accounting,KINDS,SEEDS
from c3_uop_identity_method import transform,event_contract,llvm_signature_provenance
from c3_node34_uop_identity import native_pair

BASELINE=HOLD/'20261017_c3_node46_tps_local_run01'
EVIDENCE=HOLD/'20261017_c3_node46_tps_identity_readonly01'
PREFLIGHT=HOLD/'20261017_c3_uop_identity_method_preflight02'


def project_lowered_module(lowered):
    import tvm
    from c3_tps_sync_release_prepare import complete_function
    gv,fn=complete_function(lowered,SYMBOL)
    candidate,proof=transform(fn)
    return tvm.IRModule({gv:candidate},attrs=lowered.attrs),proof


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
        delta=event_contract(read(BASELINE/'prepare/R/events.json'),events,decision)
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
            # Actual codegen projects lowered B; the independently audited A parent
            # supplies semantic/native event contracts, not a codegen shortcut.
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
        signature_gate=llvm_signature_provenance(llvm,late_proof)
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


def seal(root,status):
    write(root/'completion_manifest.json',{'status':status,'paused':True,
          'artifact_sha256':{str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()},
          'FPGA_calls':0,'performance_samples':0})


def report(root,status):
    accounting=call_accounting(root)
    write(root/'call_accounting.json',accounting)
    lines=['# 节点46完整μop清单等价类驱动的缓存身份验证','',
           '本轮保持Tile、线程、张量请求、计算顺序、通知及runtime不变，仅按同一静态kernel内的完整initializer等价类归并缓存签名。',
           '',f'状态：`{status}`。FPGA调用与性能样本均为0；完整FSim时长不作为加速证据。','',
           '## 能不能复用','',
           '原版B采纳已认证的14次调用；原attempt的只读检查停止状态保持。优化版A使用新的独立目录和14次调用预算，不补调用、不拼接失败资格。',
           '',f'优化版调用账目：`{json.dumps(accounting,ensure_ascii=False)}`。','']
    if status=='IDENTITY_A_LOCAL_QUALIFIED':
        from c3_tps_aggregation_capacity import organization
        rows=[]
        for mode in ('detailed','minimal'):
            for index in range(1,8):
                unit=root/f'R_{mode}'
                rows.append({'mode':mode,'call':index,**organization(
                    (unit/f'native_call_{index:02d}.txt').read_text(),read(unit/f'native_certificate_{index:02d}.json'))})
        write(root/'organization.json',rows)
        fields=('MAC_GEMM','RESET_GEMM','ALU','compute_NOP','all_instructions','UOP_LOAD','UOP_LOAD_bytes',
                'serialized_uop_bytes','UOP_SRAM_wraps','submissions','host_identity_base_uops_total')
        require(all(tuple(r[k] for k in fields)==tuple(rows[0][k] for k in fields) for r in rows),'QUALIFIED_ORGANIZATION_CHANGED')
        b=read(EVIDENCE/'organization.json')[0];a=rows[0]
        lines+=['A的14次完整融合FSim及全部证书通过，detailed/minimal实际执行编码逐项相同。与B的14组配对输入、参考和输出一致；逐事件计算内容、非UOP物理访问、原必要先后和逻辑token序列保持。',
                '', '## 能复用多少','',
                '两个MAC静态handle分别保留；各自将0..23的原签名映射到完整内容相同的原代表0或1。48个MAC身份降为4个，总kernel身份由58降为14。',
                '', '|指标|B|A|','|---|---:|---:|']
        for field in fields:lines.append(f'|{field}|{b[field]}|{a[field]}|')
        lines+=['','实际载荷由native指令和缓存证书取得，不用等价类条数估算。回绕、装载、DRAM源生命周期及每次引用均重新认证。',
                '', '## 合了值不值','',
                '已形成第二个算子的合格本地B/A对照，可以另行制定FPGA资格与完整调用计时。本轮不判断是否加速。节点46原版没有回绕，且只有一次UOP LOAD；不能沿用节点34减少多次LOAD与回绕的收益解释。',
                '', '## 方法范围与封存','',
                '改写由完整有限域分类自动产生，比较dispatch、mode/reset、全部有序μop参数、硬件循环及因子；保持不同静态handle独立。节点34零调用回归得到与封存优化版相同的完整事件内容；节点24及一个官方程序无冗余时原样保留。',
                '', 'TIR代表映射支持有限Select及周期表示；最终LLVM门禁目前仅支持零起点、2的幂周期映射的已知SSA形式，其他形式登记支持未决，不能宣称已执行覆盖。未知域、分支和双自由整数不静默接纳。',
                '', '当前仍限本地适配作者TPS的oc2/h1支持域。Banerjee底座已有μop优化及相邻相同字节共享；本轮增量是同handle非相邻原签名的完整内容等价归并，不声称作者完整编译链或其他研究均未解决。更宽线程域存在更低解析成本配置，不称本轮Tile为全线程最优。',
                '', '编译在lower后插入投影，通过target-map入口避免再次lower；最终LLVM逐closure存储检查和全部native缓存证书共同约束实际执行。独立预检与本轮prepare均不调用目标。',
                '', '证据：[冻结合同](contract.json)、[编译合同](prepare/R/compilation_contract.json)、[等价类改写](prepare/R/transform.json)、[最终签名](prepare/R/final_signature_provenance.json)、[14组native对照](paired_results.json)、[实际组织](organization.json)、[账目](call_accounting.json)。',
                '', '**本阶段完成并暂停，不执行板测或扩展候选。**']
    else:
        lines+=['未完成资格，停止原因与worker日志封存；不从未完成记录推导正确性或收益。本阶段暂停。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write(root/'status.json',{'status':status,'paused':True,'FPGA_calls':0,'performance_conclusion':None})
    return accounting


def run(root):
    root.mkdir(exist_ok=False);status='STOPPED_IDENTITY_ADOPTION'
    try:
        for origin in (BASELINE,EVIDENCE,PREFLIGHT):
            verify({str(origin/p):s for p,s in read(origin/'completion_manifest.json')['artifact_sha256'].items()})
        require(read(PREFLIGHT/'result.json')['status']=='ZERO_CALL_PREFLIGHT_PASSED','PREFLIGHT_NOT_PASSED')
        require(read(EVIDENCE/'adoption.json')['status']=='NODE46_BASELINE_14_CALL_SUBSET_ADOPTED','QUALIFIED_B_SUBSET_MISSING')
        require(read(EVIDENCE/'inspection.json')['status']=='QUALIFIED_NODE46_BASELINE_CONTENT_AUDITED','B_CONTENT_NOT_AUDITED')
        require(read(BASELINE/'call_accounting.json')['fully_certified']==14,'B_QUALIFICATION_CHANGED')
        bindings=read(EVIDENCE/'freeze.json')['bindings'].copy()
        verify(bindings)
        bindings.update(read(PREFLIGHT/'freeze.json')['bindings'])
        for origin in (BASELINE,EVIDENCE,PREFLIGHT):
            bindings[str(origin/'completion_manifest.json')]=sha(origin/'completion_manifest.json')
        verify(bindings)
        write(root/'freeze.json',{'bindings':bindings,'baseline':str(BASELINE),'variant':'A'})
        shutil.copyfile(BASELINE/'hardware_archive.json',root/'hardware_archive.json')
        write(root/'adoption.json',{'B':str(BASELINE),'qualification_subset':str(EVIDENCE),
              'B_full_FSim_calls_adopted':14,'B_calls_new':0,'source_stop_status_preserved':True,
              'helper_worker_label':'R means A only inside this new attempt'})
        write(root/'contract.json',{'variants':['B_adopted','A_full_content_representative_identity'],
              'new_FSim_budget':14,'target':'local FSim only','order':['prepare','A_detailed','A_minimal','native_pair_compare'],
              'change':'same-handle complete finite initializer equivalence -> original representative signature',
              'fixed':['Tile','threads','tensor requests','MAC terms/order','postprocess','logical tokens','runtime/helper'],
              'limits':{'worker_GiB':8,'worker_seconds':600,'expanded_vector_steps':5000000,'FIFO_states':1000000},
              'on_error':'close without repair/retry/substitute; preserve all evidence','performance_samples':0,'FPGA_calls':0})
        write(root/'call_ledger.json',[{'program':'R','variant':'A','mode':mode,'call_index':i,'kind':kind,'seed':seed,
              'target_calls':1,'performance_sample':False} for mode in ('detailed','minimal')
              for i,(kind,seed) in enumerate(zip(KINDS,SEEDS),1)])
        status='STOPPED_PREPARE'
        if run_worker(root,'prepare'):
            ready=read(root/'prepare_complete.json')
            require(ready['status']=='IDENTITY_A_PREPARED' and ready['result_sha256']==sha(root/'prepare/result.json'),'PREPARE_NOT_COMPLETE')
            require(read(root/'prepare/result.json')['status']=='IDENTITY_A_PREPARED','PREPARE_RESULT_STATUS')
            for mode in ('detailed','minimal'):
                status='STOPPED_'+mode.upper()
                if not run_worker(root,'qualify',mode):break
            else:
                status='STOPPED_NATIVE_COMPARE'
                if run_worker(root,'compare'):status='IDENTITY_A_LOCAL_QUALIFIED'
        verify(read(root/'freeze.json')['bindings'])
    except Exception as error:
        write(root/'failure_controller.json',{'type':type(error).__name__,'message':str(error)});traceback.print_exc()
    accounting=report(root,status)
    seal(root,status)
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

