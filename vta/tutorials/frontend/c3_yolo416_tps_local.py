"""Bounded, serial operator qualification from compiled YOLO416 TH/P graphs.

No board or performance entry point. Every function is prepared before any
target execution. A failed worker closes the whole attempt without retries.
R in the reused worker is only its local baseline label; actual TH/P identity
is always bound in the per-object descriptor and outer ledger.
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
from c3_tps_sync_release_local import (HERE, REPO, HOLD, PYTHON, RUNTIME, HELPER,
    read, write, sha, verify, require, env_for_worker, ledger,
    qualification_worker, compare_modes)


def objects(source):
    return [{**f,'variant':variant} for variant in ('P','TH')
            for f in read(source/variant/'functions.json')]


def prepare_object(root, index):
    import tvm, vta
    from c3_official_window_baseline import buffer_abi, normalize_llvm
    from c3_official_window_method import audit_function
    from c3_official_window_reference import from_workload, semantic_gates
    from c3_valid_window import certify_trace
    from c3_tps_window_method import acc_spacing
    from c3_tps_sync_release_prepare import compiled_module_contract, complete_function
    from c3_yolo416_tps_comparison_scope import OFFICIAL
    verify(read(root/'freeze.json')['bindings'])
    row=read(root/'objects.json')[index];unit=root/'objects'/('%02d'%index)
    unit.mkdir(parents=True,exist_ok=False);out=unit/'prepare'/'R';out.mkdir(parents=True)
    for name in ('freeze.json','hardware_archive.json'):shutil.copyfile(root/name,unit/name)
    write(unit/'call_ledger.json',[{**c,'variant':row['variant'],'object_id':row['object_id']}
          for c in ledger() if c['program']=='R'])
    parent=tvm.ir.load_json(Path(row['parent']).read_text())
    final=tvm.ir.load_json(Path(row['final']).read_text())
    gv,fn=complete_function(parent,row['symbol'])
    abi=buffer_abi(fn);require(abi==row['abi'],'ABI_CHANGED')
    replay=vta.transform.CPUAccessRewrite()(tvm.ir.load_json(tvm.ir.save_json(parent)))
    require(tvm.ir.structural_equal(replay,final),'REMAINING_PASS_ROUNDTRIP_CHANGED')
    audit=audit_function(fn)
    write(out/'audit_counts.json',audit['counts'])
    # The source reference was certified independently of this TPS arrangement.
    reference=read(OFFICIAL/'local_final01'/row['object_id']/'detailed/reference_contract.json')
    spec=from_workload(row['workload'],abi,reference['postprocess'])
    require(spec.contract()==reference,'FROZEN_FUSION_REFERENCE_CHANGED')
    semantic=certify_trace(audit['events'])
    write(out/'independent_semantics.json',semantic_gates(spec,semantic['output_terms']))
    hardware=read(root/'hardware_archive.json')
    write(out/'logical_certificate.json',{'status':'LOGICAL_PASSED_NATIVE_PENDING',
          'counts':audit['counts'],'acc_spacing':acc_spacing(audit['events'],hardware['verified_acc_interval'])})
    write(out/'reference_contract.json',reference)
    # Neither this gate nor qualification changes the archived graph functions.
    with vta.Environment(dict(vta.get_env().cfg_dict,TARGET='sim')):
        target=tvm.target.Target(vta.get_env().target,host=vta.get_env().target_host)
        with tvm.transform.PassContext(opt_level=3,disabled_pass={'AlterOpLayout','tir.CommonSubexprElimTIR'}):
            lib=tvm.build(final,target=target)
            repeated=tvm.build(replay,target=target)
        llvm=lib.get_source('ll')
        require(normalize_llvm(llvm)==normalize_llvm(repeated.get_source('ll')),'FSIM_CODEGEN_ROUNDTRIP_CHANGED')
        lib.export_library(str(out/'fsim_module.so'))
    (out/'module.ll').write_text(llvm)
    compiled=compiled_module_contract(final,row['symbol'],llvm)
    require(compiled['abi']==abi,'COMPILED_ABI_CHANGED')
    write(out/'compiled_contract.json',compiled)
    descriptor={'program':'R','variant':row['variant'],'object_id':row['object_id'],
        'nodes':row['nodes'],'symbol':row['symbol'],'global_var':gv.name_hint,'abi':abi,
        'workload':row['workload'],'parent_module':row['parent'],'final_module':row['final'],
        'module_path':str(out/'fsim_module.so'),'reference_contract':str(out/'reference_contract.json'),
        'artifact_bindings':{str(p):sha(p) for p in out.iterdir() if p.is_file()}}
    descriptor['artifact_bindings'].update({row['parent']:sha(row['parent']),row['final']:sha(row['final'])})
    write(unit/'prepare/result.json',{'status':'GRAPH_FUNCTION_PREPARED','programs':{'R':descriptor}})
    write(unit/'prepare_complete.json',{'status':'GRAPH_FUNCTION_PREPARED','result_sha256':sha(unit/'prepare/result.json')})
    verify(read(root/'freeze.json')['bindings'])


def worker(root, action, index, mode=None):
    import psutil
    args=[str(PYTHON),str(Path(__file__).resolve()),action,str(root),'--index',str(index)]
    if mode:args+=['--mode',mode]
    label='%02d_%s%s'%(index,action,'_'+mode if mode else '')
    env=env_for_worker()
    for key in ('VTA_UOP_BUFFER_BYTES','VTA_INSN_BUFFER_BYTES','VTA_INSN_SUBMIT_THRESHOLD_BYTES'):
        env[key]='33554432'
    start=time.monotonic();peak=0;limit=None
    with (root/(label+'.log')).open('x') as log:
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
    record={'returncode':code,'resource_limit':limit,'sampled_peak_tree_RSS':peak,
            'resource_control_seconds':time.monotonic()-start,'argv':args}
    write(root/(label+'_worker.json'),record)
    return record


def accounting(root):
    rows=[];started_lower=0;completed_lower=0;unresolved=[]
    for unit in sorted((root/'objects').glob('*')) if (root/'objects').exists() else []:
        for mode in ('detailed','minimal'):
            folder=unit/('R_'+mode)
            completed=0;started=0;uncertain=False
            for begin in sorted(folder.glob('call_*_begin.json')):
                row=read(begin);index=row['call_index']
                execution=folder/('call_%02d_execution.json'%index)
                end=folder/('call_%02d_end.json'%index)
                observed=read(execution) if execution.exists() else None
                if observed is not None:
                    count=observed.get('counts')
                    if type(count) is int:completed=max(completed,count);started=max(started,count)
                    meta=observed.get('metadata',{})
                    if type(meta.get('started_calls')) is int:started=max(started,meta['started_calls'])
                uncertain |= observed is None or observed.get('helper_returned') is not True
                rows.append({**row,'unit':unit.name,'attempt_registered':True,
                    'execution_evidence':str(execution) if execution.exists() else None,
                    'all_certificates_passed':end.exists(),
                    'execution':observed})
            started_lower+=started;completed_lower+=completed
            if uncertain:unresolved.append(unit.name+'_'+mode)
    return {'registered':len(rows),'all_certificates_passed':sum(r['all_certificates_passed'] for r in rows),
            'helper_started_observed_lower_bound':started_lower,'helper_completed_observed_lower_bound':completed_lower,
            'execution_count_unresolved_workers':unresolved,
            'calls':rows,'begin_is_not_execution_completion':True,'FPGA_calls':0,'performance_samples':0}


def run(root,source):
    root.mkdir(exist_ok=False);status='STOPPED_INPUT_IDENTITY'
    try:
        with (root/'protocol_tests.log').open('x') as log:
            test=subprocess.run([str(PYTHON),'-m','unittest','test_c3_yolo416_tps_local','-v'],
                                env=env_for_worker(),stdout=log,stderr=subprocess.STDOUT)
        require(test.returncode==0,'PROTOCOL_PREFLIGHT_FAILED')
        seal=read(source/'completion_manifest.json')
        require(seal['status']=='SEALED_TH_P_GRAPHS_COMPILED_QUALIFICATION_PENDING','UNQUALIFIED_SOURCE_STAGE')
        verify({str(source/p):v for p,v in seal['artifact_sha256'].items()})
        bindings=read(source/'freeze.json')['bindings'].copy()
        for p in HERE.glob('c3_*.py'):bindings[str(p.resolve())]=sha(p)
        for p in (HERE/'test_c3_yolo416_tps_local.py',root/'protocol_tests.log'):
            bindings[str(p.resolve())]=sha(p)
        bindings.update({str(source/p):v for p,v in seal['artifact_sha256'].items()})
        verify(bindings)
        write(root/'freeze.json',{'bindings':bindings,'max_target_calls':308,'max_worker_bytes':8*1024**3,
              'max_worker_seconds':600,'max_vector_steps':5000000,'source':str(source)})
        shutil.copyfile(source/'hardware_archive.json',root/'hardware_archive.json')
        items=objects(source);require(len(items)==22,'OBJECT_COVERAGE_CHANGED')
        write(root/'objects.json',items)
        write(root/'call_ledger.json',[{**c,'variant':row['variant'],'object_id':row['object_id'],'unit':i}
             for i,row in enumerate(items) for c in ledger() if c['program']=='R'])
        status='STOPPED_LOCAL_PREPARATION'
        # Complete all static gates before permitting any target call.
        for i,row in enumerate(items):
            result=worker(root,'_prepare',i)
            if result['returncode'] or result['resource_limit']:
                write(root/'stop.json',{'stage':'prepare','index':i,'object_id':row['object_id'],
                      'variant':row['variant'],'worker':result});break
        else:
            status='STOPPED_LOCAL_QUALIFICATION'
            for i,row in enumerate(items):
                good=True
                for mode in ('detailed','minimal'):
                    result=worker(root,'_qualify',i,mode)
                    if result['returncode'] or result['resource_limit']:
                        write(root/'stop.json',{'stage':'qualify','index':i,'object_id':row['object_id'],
                              'variant':row['variant'],'mode':mode,'worker':result})
                        good=False;break
                if not good:break
                compare_modes(root/'objects'/('%02d'%i),'R')
            else:status='ALL_22_FUNCTIONS_LOCAL_QUALIFIED_GRAPH_QUALIFICATION_PENDING'
    except Exception as e:
        write(root/'failure_controller.json',{'type':type(e).__name__,'message':str(e)});traceback.print_exc()
    write(root/'call_accounting.json',accounting(root))
    write(root/'status.json',{'status':status,'FPGA_calls':0,'performance_conclusion':None})
    print(json.dumps({'status':status,**{k:v for k,v in accounting(root).items() if k!='calls'}}),flush=True)
    return 0 if status.startswith('ALL_22_') else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('run','_prepare','_qualify'))
    p.add_argument('output',type=Path);p.add_argument('--source',type=Path);p.add_argument('--index',type=int)
    p.add_argument('--mode',choices=('detailed','minimal'));a=p.parse_args();root=a.output.resolve()
    if a.action=='run':sys.exit(run(root,a.source.resolve()))
    try:
        if a.action=='_prepare':prepare_object(root,a.index)
        else:qualification_worker(root/'objects'/('%02d'%a.index),'R',a.mode)
    except Exception as e:
        write(root/('failure_%02d_%s_%s.json'%(a.index,a.action,a.mode)),
              {'type':type(e).__name__,'message':str(e)})
        traceback.print_exc();sys.exit(1)
