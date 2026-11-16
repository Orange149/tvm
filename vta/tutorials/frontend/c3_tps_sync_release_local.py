#!/usr/bin/env python3
"""Bounded TPS order-restoration and wait-release qualification, local FSim only.

Public phases: prepare, qualify, report. Four fresh qualification workers,
seven complete calls each at most. Any failed worker closes the attempt.
Historical sources and the common runtime/helper are never changed.
"""
import argparse
from copy import deepcopy
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
HOLD = HERE/'report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout'
AUDIT = HOLD/'20261016_c3_sync_release_audit_run01'
QUALIFIED = HOLD/'20261014_c3_tps_window_complete_run03'
OFFICIAL = HOLD/'20261014_c3_official_valid_window_run01'
RUNTIME = OFFICIAL/'runtime_local_final'
HELPER = OFFICIAL/'helper/local_final02/c3_official_window_runtime_helper.so'
PYTHON = Path('/home/orange/miniconda3/envs/vta-resnet/bin/python')
KINDS = ('three_seed','three_seed','three_seed','weight_ABA_W0',
         'weight_ABA_W1','weight_ABA_W0_return','directed')
SEEDS = (0,20250901,20260910,0,0,0,20260910)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify(bindings):
    for path, digest in bindings.items():
        require(sha(path) == digest, 'FROZEN_IDENTITY_CHANGED:'+path)


def ledger():
    return [{'program':program,'mode':mode,'call_index':i,'kind':kind,'seed':seed,
             'target_calls':1,'performance_sample':False}
            for program in ('R','S') for mode in ('detailed','minimal')
            for i,(kind,seed) in enumerate(zip(KINDS,SEEDS),1)]


def validate_calls(planned, actual):
    require(len(planned) == len(actual), 'MISSING_OR_EXTRA_CALL')
    for p,a in zip(planned,actual):
        require(all(a.get(k) == v for k,v in p.items()), 'CALL_IDENTITY_OR_ORDER_CHANGED')
        require(a.get('status') == 'PASS' and a.get('correct') is True, 'CALL_NOT_PASSED')
    return True


def initialize_helper(reset, counts, metadata, persist=None):
    """Validate the frozen helper's reset state without invoking a target."""
    reset()
    state={'metadata':json.loads(metadata()),'counts':int(counts())}
    if persist is not None:
        persist(state)
    require(state['metadata']=={} and state['counts']==0,'HELPER_NOT_FRESH')
    return state


def capture_helper_call(index, invoke, counts, metadata, submissions, persist):
    """Persist live completion counts separately from last-success metadata.

    A throwing invocation leaves metadata empty or stale in the frozen helper.
    Observation errors must neither erase this distinction nor mask its exception.
    """
    execution={'call_index':index,'helper_returned':False}
    try:
        result=invoke()
        execution['helper_returned']=True
    except Exception as error:
        execution['invocation_error']={'type':type(error).__name__,'message':str(error)}
        raise
    finally:
        for name,reader in (('counts',lambda:int(counts())),
                            ('metadata',lambda:json.loads(metadata())),
                            ('submits',lambda:json.loads(submissions()))):
            try:
                execution[name]=reader()
            except Exception as error:
                execution.setdefault('observation_errors',{})[name]={
                    'type':type(error).__name__,'message':str(error)}
        persist(execution)
    return result,execution


def validate_helper_completion(execution, index):
    """Empty/stale/partial metadata is never accepted after a successful call."""
    require(execution.get('helper_returned') is True and execution.get('call_index')==index,
            'HELPER_DID_NOT_RETURN_FOR_CALL')
    require(not execution.get('observation_errors'),'HELPER_OBSERVATION_FAILED')
    meta=execution.get('metadata')
    required={'schema','started_calls','completed_calls','packed_calls_per_invocation',
              'adds_synchronize','parameter_count','last_kind'}
    require(isinstance(meta,dict) and required.issubset(meta),'HELPER_METADATA_INCOMPLETE')
    require(type(execution.get('counts')) is int and type(meta['started_calls']) is int and
            type(meta['completed_calls']) is int and
            execution['counts']==meta['started_calls']==meta['completed_calls']==index,
            'HIDDEN_OR_MISSING_TARGET_CALL')
    require(meta['schema']=='c3_official_window_once_helper_v1' and
            meta['last_kind']=='diagnostic' and meta['packed_calls_per_invocation']==1 and
            meta['adds_synchronize'] is False and meta['parameter_count']==4,
            'HELPER_INVOCATION_CONTRACT_CHANGED')
    return True


def env_for_worker():
    return dict(os.environ, PYTHONPATH=str(HERE)+':'+str(REPO/'python')+':'+str(REPO/'vta/python'),
                OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                TVM_LIBRARY_PATH=str(RUNTIME), VTA_REPLAY_POLICY='disabled')


def run_worker(root, name, arguments):
    import psutil
    started, peak, limit = time.monotonic(), 0, None
    argv = [str(PYTHON), str(Path(__file__).resolve()), *arguments, '--output-dir', str(root)]
    with (root/(name+'.log')).open('x') as log:
        child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                 env=env_for_worker(), start_new_session=True)
        while child.poll() is None:
            try:
                parent = psutil.Process(child.pid)
                peak = max(peak, sum(p.memory_info().rss for p in [parent,*parent.children(recursive=True)]))
            except (psutil.NoSuchProcess,psutil.AccessDenied):
                pass
            if peak > 8*1024**3:
                limit = 'ANALYSIS_MEMORY_LIMIT'
            if time.monotonic()-started > 600:
                limit = 'ANALYSIS_TIME_LIMIT'
            if limit:
                os.killpg(child.pid, signal.SIGKILL)
                break
            time.sleep(.1)
        code = child.wait()
    row = {'name':name,'argv':argv,'returncode':code,'analysis_limit':limit,
           'elapsed_seconds_for_resource_control_only':time.monotonic()-started,
           'peak_sampled_process_tree_rss':peak,'memory_sampling_seconds':.1}
    write(root/(name+'_worker.json'),row)
    print(json.dumps(row),flush=True)
    return row


def call_accounting(root):
    """Use helper counters as evidence; a begin marker is only an intention."""
    units=[]
    for directory in sorted(root.glob('[RS]_*')):
        if not directory.is_dir():
            continue
        begins=sorted(directory.glob('call_*_begin.json'))
        if not begins:
            continue
        executions={p.name.replace('_execution.json','_begin.json'):read(p)
                    for p in directory.glob('call_*_execution.json')}
        unit={'unit':directory.name,'registered_attempts':len(begins),
              'fully_certified':len(list(directory.glob('call_*_end.json')))}
        last=executions.get(begins[-1].name,{})
        observations={'started_calls':[0],'completed_calls':[0]}
        issues=[]
        for name,execution in executions.items():
            meta=execution.get('metadata',{})
            values={field:meta.get(field) for field in observations} if isinstance(meta,dict) else {}
            values['live_completed']=execution.get('counts')
            for field,value in values.items():
                if value is None:
                    continue
                if type(value) is not int or value<0:
                    issues.append(name+':INVALID_COUNTER:'+field)
                    continue
                destination='completed_calls' if field=='live_completed' else field
                observations[destination].append(value)
                if value>len(begins):
                    issues.append(name+':COUNTER_EXCEEDS_REGISTERED_ATTEMPTS:'+field)
        # Completion proves a start even when metadata has not been refreshed.
        observations['started_calls']+=observations['completed_calls']
        latest_completed=last.get('counts')
        if type(latest_completed) is int and latest_completed<max(observations['completed_calls']):
            issues.append('LIVE_COMPLETED_COUNTER_REGRESSED')
        for field in ('started_calls','completed_calls'):
            lower=max(observations[field])
            upper=None if issues else len(begins)
            if field=='completed_calls' and not issues and type(latest_completed) is int:
                upper=latest_completed
            # Last-success metadata only supplies a lower bound. It cannot
            # establish that a subsequent failing invocation never started.
            unit[field]={'exact':lower if upper==lower else None,
                         'lower_bound':lower,'upper_bound':upper}
        unit['evidence_issues']=issues
        units.append(unit)
    result={'registered_attempts':sum(u['registered_attempts'] for u in units),
            'fully_certified':sum(u['fully_certified'] for u in units),'units':units}
    for field in ('started_calls','completed_calls'):
        values=[u[field] for u in units]
        result[field]={'exact':sum(v['exact'] for v in values) if all(v['exact'] is not None for v in values) else None,
                       'lower_bound':sum(v['lower_bound'] for v in values),
                       'upper_bound':sum(v['upper_bound'] for v in values) if all(v['upper_bound'] is not None for v in values) else None}
    return result


def close_attempt(root, status, reason=None):
    require(not (root/'stage_result.json').exists(), 'ATTEMPT_ALREADY_CLOSED')
    entries = sorted(root.glob('[RS]_*/call_*_begin.json'))
    ends = sorted(root.glob('[RS]_*/call_*_end.json'))
    rows = [read(p) for p in ends]
    accounting=call_accounting(root)
    write(root/'stage_result.json', {'status':status,'reason':reason,'paused':True,
        'recorded_call_entries':len(entries),'completed_certified_calls':len(ends),
        'call_accounting':accounting,
        'new_FSim_calls_upper_bound':accounting['started_calls']['upper_bound'],'FPGA_calls':0,
        'performance_conclusion':None,'automatic_retry':False,
        'calls':rows,'complete_qualification':status=='R_S_LOCAL_QUALIFIED'})


def tool_preflight(root):
    require(not root.exists(),'OUTPUT_DIRECTORY_ALREADY_EXISTS')
    preflight=root.with_name(root.name+'_preflight')
    preflight.mkdir(parents=True,exist_ok=False)
    paths=sorted(set(HERE.glob('c3_*.py')) | set(HERE.glob('test_c3_tps_sync_release*.py')))
    sources={str(p):sha(p) for p in paths}
    test_names=['test_c3_tps_sync_release_method','test_c3_tps_sync_release_prepare','test_c3_tps_sync_release_local']
    argv=[str(PYTHON),'-m','unittest','-v',*test_names]
    with (preflight/'tests.log').open('x') as log:
        try:
            result=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,env=env_for_worker(),timeout=180)
            code=result.returncode
        except subprocess.TimeoutExpired:
            code=None
    output=(preflight/'tests.log').read_text()
    match=re.search(r'^Ran (\d+) tests? in ',output,re.MULTILINE)
    count=int(match.group(1)) if match else None
    unchanged=all(sha(p)==digest for p,digest in sources.items())
    passed=code==0 and count==38 and unchanged
    write(preflight/'result.json',{'status':'TOOL_PREFLIGHT_PASSED' if passed else 'STOP_TOOL_PREFLIGHT_FAILED',
          'returncode':code,'test_count':count,'expected_tests':38,'argv':argv,'source_bindings':sources,
          'sources_unchanged':unchanged,'target_calls':0,'research_attempt_started':False})
    require(passed,'TOOL_PREFLIGHT_FAILED_NO_RESEARCH_ATTEMPT:'+str(preflight))
    return preflight


def prepared_contract(root):
    require(not (root/'stage_result.json').exists(),'ATTEMPT_ALREADY_CLOSED')
    result=read(root/'prepare/result.json')
    require(result['status']=='COMPILE_AND_LOGICAL_PREPARATION_PASSED_QUALIFICATION_PENDING',
            'PREPARE_RESULT_NOT_SUCCESSFUL')
    require(result['target_calls']==result['new_fsim_calls']==0,'PREPARATION_EXECUTED_TARGET')
    verify(result['input_bindings'])
    verify(result['programs']['R']['artifact_bindings'])
    verify(result['programs']['O_rebuilt']['artifact_bindings'])
    return result


def prepare(root):
    preflight=tool_preflight(root)
    root.mkdir(parents=True,exist_ok=False)
    (root/'tool_sources').mkdir()
    prior = read(AUDIT/'freeze.json')['bindings']
    verify(prior)
    bindings = dict(prior)
    sealed = read(HOLD/'20261015_c3_tps_aggregation_run02/freeze.json')['source_and_artifact_bindings']
    for path,digest in sealed.items():
        if '/TPS_B_' in path:
            require(sha(path)==digest,'SEALED_O_ARTIFACT_CHANGED:'+path)
            bindings[path]=digest
    old = HOLD/'20261013_c3_banerjee_baseline_completion_run02'
    manifest = read(old/'manifest.json')['files']
    for rel,digest in manifest.items():
        if rel.startswith('local_final06/TPS-oc-reuse-ci4/'):
            path = old/rel
            require(sha(path)==digest,'SEALED_O_BUILD_CHANGED:'+rel)
            bindings[str(path)]=digest
    from c3_tps_aggregation_prepare import common_runtime_bindings
    runtime_contract = read(OFFICIAL/'runtime_contract.json')
    bindings.update(common_runtime_bindings(runtime_contract))
    paths = list(HERE.glob('c3_*.py')) + list(HERE.glob('test_c3_tps_sync_release*.py'))
    paths += [HERE/'run_c3_banerjee_completion.py',HERE/'run_vta_c3_node24_partial_local.py',
              HERE/'run_vta_c3_node24_partial_board.py',REPO/'build/libtvm.so',REPO/'build/libtvm_runtime.so',
              AUDIT/'completion_manifest.json',old/'manifest.json',
              HOLD/'20261013_c3_banerjee_baseline_completion_run01/audit.json']
    paths += list((REPO/'vta/python/vta').rglob('*.py'))
    paths += list((REPO/'src/tir/transforms').glob('*.cc'))
    for p in paths:
        bindings[str(p.resolve())]=sha(p)
    author = HOLD/'20261012_c3_banerjee_double_buffer_local_run01/author_sources'
    for name in ('vta_conv2d.py','runtime.cc','coproc_sync.cc','inject_virtual_thread.cc'):
        p=author/name; bindings[str(p)]=sha(p)
    for p in sorted(HERE.glob('*c3_tps_sync_release*.py')):
        shutil.copy2(p,root/'tool_sources'/p.name)
    shutil.copyfile(QUALIFIED/'hardware_archive.json',root/'hardware_archive.json')
    bindings[str(QUALIFIED/'hardware_archive.json')]=sha(QUALIFIED/'hardware_archive.json')
    bindings[str(RUNTIME/'libvta_fsim.so')]=sha(RUNTIME/'libvta_fsim.so')
    bindings[str(HELPER)]=sha(HELPER)
    verify(bindings)
    verify(read(preflight/'result.json')['source_bindings'])
    shutil.copyfile(preflight/'tests.log',root/'tests.log')
    shutil.copyfile(preflight/'result.json',root/'tests.json')
    bindings[str(preflight/'tests.log')]=sha(preflight/'tests.log')
    bindings[str(preflight/'result.json')]=sha(preflight/'result.json')
    write(root/'freeze.json',{'status':'LOCAL_ORDER_AND_RELEASE_ATTEMPT_FROZEN',
        'bindings':bindings,'source_scope':'O source adoption, R original order, S two MAC wait consumers',
        'programs_to_execute':['R','S'],'modes':['detailed','minimal'],'max_calls':28,
        'max_expanded_vector_steps':5_000_000,'worker_seconds':600,'process_tree_bytes':8*1024**3,
        'single_submission_support':True,'FPGA_calls':0,'performance_samples':0})
    write(root/'call_ledger.json',ledger())
    row=run_worker(root,'prepare',['_prepare_worker'])
    if row['returncode']:
        failure=read(root/'prepare_worker/failure.json') if (root/'prepare_worker/failure.json').exists() else {}
        category=failure.get('category','ANALYSIS_RESOURCE_LIMIT' if row['analysis_limit'] else 'UNCLASSIFIED_PREPARATION_FAILURE')
        close_attempt(root,'STOP_PREPARATION_FAILED',{'worker':row,'category':category,
            'failure_phase':failure.get('failure_phase'),'error':failure.get('error')});return
    verify(bindings)
    prepared_contract(root)
    write(root/'prepare_complete.json',{'status':'O_REBUILT_AND_R_PREPARED','target_calls':0,
        'prepare_result_sha256':sha(root/'prepare/result.json')})


def descriptor(root, program):
    if program=='R':
        return read(root/'prepare/result.json')['programs']['R']
    return read(root/'S_program/descriptor.json')


def prepare_s(root):
    # Called only inside an isolated worker, after both complete R sessions.
    import tvm
    import vta
    from c3_tps_sync_release_method import transform,verify_event_delta
    from c3_official_window_method import audit_function
    from c3_official_window_baseline import buffer_abi
    from c3_tps_sync_release_prepare import compiled_module_contract
    from c3_tps_aggregation_prepare import compare_initializer_bindings
    from c3_five_tradeoff_commands import decode_submissions
    from c3_sync_release_audit import Graph,Obligations,access_model,requirements
    from c3_tps_window_progress import certify_progress,LANES
    d=descriptor(root,'R');verify(d['artifact_bindings'])
    parent=tvm.ir.load_json(Path(d['parent_module']).read_text())
    gv=parent.get_global_var(d['global_var']); function=parent[gv]
    proposed,decisions=transform(function)
    delta=verify_event_delta(audit_function(function)['events'],audit_function(proposed)['events'])
    require(buffer_abi(function)==buffer_abi(proposed),'S_ABI_CHANGED')
    module=tvm.IRModule(parent.functions,parent.type_definitions,parent.attrs)
    module.update_func(gv,proposed)
    final=vta.transform.CPUAccessRewrite()(module)
    out=root/'S_program';out.mkdir()
    # Joint pre-execution graph gate. This is explicitly a hypothesis on R's
    # emitted stream; S's actual emitted stream is separately certified later.
    source=root/'R_detailed'
    records=decode_submissions((source/'native_call_01.txt').read_text())[0]
    native=read(source/'native_certificate_01.json')
    original_graph=Graph(records);accesses=access_model(records,native)
    expected=deepcopy(records)
    moves=read(root/'R_opportunities.json')['MAC_candidates']
    for move in moves:
        old,new=move['consumer'],move['new_consumer']
        require(expected[old]['bits'][1]==1 and expected[new]['bits'][1]==0,'PREFLIGHT_POP_BITS_CHANGED')
        bits=list(expected[old]['bits']);bits[1]=0;expected[old]['bits']=tuple(bits)
        bits=list(expected[new]['bits']);bits[1]=1;expected[new]['bits']=tuple(bits)
    combined=Graph(expected)
    live=Obligations(requirements(accesses),original_graph.ancestors,True)
    physical=Obligations(requirements(accesses,include_dead=True),original_graph.ancestors,False)
    require(live.missing(combined.ancestors) is None,'PREFLIGHT_JOINT_LIVE_CONFLICT')
    require(physical.missing(combined.ancestors) is None,'PREFLIGHT_JOINT_PHYSICAL_CONFLICT')
    h=read(root/'hardware_archive.json')['fifo_depths']
    cd={i:h[k] for i,k in enumerate(('load_queue','gemm_queue','store_queue'),1)}
    td=dict(zip(LANES,[h[k] for k in ('l2g_queue','g2l_queue','g2s_queue','s2g_queue')]))
    progress=certify_progress(expected,cd,td)
    require(progress['status']=='FINITE_FIFO_PROGRESS_PROVED','PREFLIGHT_FIFO_UNRESOLVED')
    write(out/'preflight_joint.json',{'status':'JOINT_HYPOTHESIS_PASSED_ACTUAL_ENCODING_PENDING',
        'moves':moves,'progress':progress,'original_dead_read_unordered':physical.original_unordered,
        'target_calls':0,'not_a_certificate_of_the_generated_instruction_stream':True})
    (out/'parent.json').write_text(tvm.ir.save_json(module));(out/'parent.tir').write_text(module.script())
    (out/'final.json').write_text(tvm.ir.save_json(final))
    with vta.Environment(dict(vta.get_env().cfg_dict,TARGET='sim')):
        target=tvm.target.Target(vta.get_env().target,host=vta.get_env().target_host)
        with tvm.transform.PassContext(opt_level=3,disabled_pass={'AlterOpLayout','tir.CommonSubexprElimTIR'}):
            lib=tvm.build(final,target=target)
        lib.export_library(str(out/'fsim_module.so'))
    llvm=lib.get_source('ll');(out/'module.ll').write_text(llvm)
    new_contract=compiled_module_contract(final,d['symbol'],llvm)
    require(new_contract['abi']==buffer_abi(proposed),'S_COMPILED_ABI_CHANGED')
    old_final=tvm.ir.load_json(Path(d['final_module']).read_text())
    old_llvm=Path(d['module_path']).with_name('module.ll').read_text()
    old_contract=compiled_module_contract(old_final,d['symbol'],old_llvm)
    binding=compare_initializer_bindings(old_llvm,llvm,old_contract['alu_initializer_contract'],
                                         new_contract['alu_initializer_contract'])
    write(out/'initializer_binding.json',binding)
    write(out/'cache_contract.json',new_contract['compiled_cache_contract'])
    write(out/'logical_delta.json',{'decisions':decisions,'event_delta':delta})
    result={**d,'program':'S','parent_module':str(out/'parent.json'),'final_module':str(out/'final.json'),
            'module_path':str(out/'fsim_module.so'),'source_R_descriptor':d}
    result['artifact_bindings']={**d['artifact_bindings'],**{str(p):sha(p) for p in out.iterdir() if p.is_file()}}
    write(out/'descriptor.json',result)


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
    from c3_tps_window_progress import certify_progress,LANES
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


def compare_modes(root, program):
    from c3_five_tradeoff_commands import decode_submissions
    from c3_tps_notification_probe import canonical_payload
    rows=[]
    for index in range(1,8):
        traces=[decode_submissions((root/f'{program}_{mode}'/f'native_call_{index:02d}.txt').read_text())[0]
                for mode in ('detailed','minimal')]
        require([[(r['stage'],r['bits'],canonical_payload(r)) for r in trace] for trace in traces][0] ==
                [[(r['stage'],r['bits'],canonical_payload(r)) for r in trace] for trace in traces][1],
                'MODE_ACTUAL_ENCODING_CHANGED')
        rows.append({'call_index':index,'all_encoded_fields_equal':True})
    write(root/f'{program}_mode_equivalence.json',{'status':'ACTUAL_ENCODING_EQUAL','calls':rows})


def qualify(root):
    require((root/'prepare_complete.json').exists() and not (root/'stage_result.json').exists(),'ATTEMPT_NOT_READY_OR_CLOSED')
    require(not list(root.glob('[RS]_detailed_worker.json')),'QUALIFICATION_ALREADY_STARTED_NO_RETRY')
    verify(read(root/'freeze.json')['bindings'])
    require(read(root/'prepare_complete.json')['status']=='O_REBUILT_AND_R_PREPARED','PREPARE_COMPLETION_NOT_SUCCESSFUL')
    prepared_contract(root)
    require(sha(root/'prepare/result.json')==read(root/'prepare_complete.json')['prepare_result_sha256'],
            'PREPARED_DESCRIPTOR_IDENTITY_CHANGED')
    for program in ('R','S'):
        if program=='S':
            audit=read(root/'R_detailed/physical_certificate_01.json')
            candidates=[r for r in audit['moves'] if not r['reset_producer'] and r['category']=='POSSIBLE_EARLIER_RELEASE']
            for mode in ('detailed','minimal'):
                for index in range(1,8):
                    other=read(root/f'R_{mode}'/f'physical_certificate_{index:02d}.json')
                    require([r for r in other['moves'] if not r['reset_producer'] and
                             r['category']=='POSSIBLE_EARLIER_RELEASE']==candidates,'R_WARM_OPPORTUNITY_MAPPING_CHANGED')
            write(root/'R_opportunities.json',{'MAC_candidates':candidates,'count':len(candidates),
                'not_a_performance_measurement':True})
            if not candidates:
                close_attempt(root,'R_QUALIFIED_NO_REMAINING_SITE');return
            require(len(candidates)==2,'R_MAC_SITE_MAPPING_UNRESOLVED')
            row=run_worker(root,'S_prepare',['_s_prepare_worker'])
            if row['returncode']:
                close_attempt(root,'STOP_S_PREPARATION_FAILED',row);return
        for mode in ('detailed','minimal'):
            row=run_worker(root,f'{program}_{mode}',['_qualify_worker','--program',program,'--mode',mode])
            if row['returncode']:
                close_attempt(root,f'STOP_{program}_QUALIFICATION_FAILED',row);return
        compare_modes(root,program)
    verify(read(root/'freeze.json')['bindings'])
    close_attempt(root,'R_S_LOCAL_QUALIFIED')


def report(root):
    require((root/'stage_result.json').exists(),'ATTEMPT_NOT_CLOSED')
    require(not (root/'completion_manifest.json').exists(),'ATTEMPT_ALREADY_SEALED')
    stage=read(root/'stage_result.json')
    accounting=stage['call_accounting']
    def observed(field):
        value=accounting[field]
        return str(value['exact']) if value['exact'] is not None else '未知（下界%s，上界%s）'%(value['lower_bound'],value['upper_bound'])
    current={p:sha(p) for p in read(root/'freeze.json')['bindings']}
    unchanged=current==read(root/'freeze.json')['bindings']
    failures=[{'path':str(p.relative_to(root)),**read(p)} for p in sorted(root.rglob('failure.json'))]
    lines=['# TPS作者循环恢复与同步后移本地预实验','',
        '状态：`'+stage['status']+'`。本阶段已暂停。', '',
        '本轮只有本地编译、静态检查和完整融合FSim，FPGA调用为0；没有性能结论。', '',
        '完整目标调用登记%d次，完成全部证书%d次。源身份保持：%s。' %
        (stage['recorded_call_entries'],stage['completed_certified_calls'],unchanged), '',
        'helper开始：%s；helper完成：%s。登记记录不是实际执行证据。' %
        (observed('started_calls'),observed('completed_calls')), '',
        '## 能不能撤除等待边界？','']
    if stage['status']=='R_S_LOCAL_QUALIFIED':
        lines+=['R和S两模式共28次完整融合调用均通过。S仅对第二个归约批次后移两处MAC通知消费，首批RESET等待保持；两处改动已经联合认证。','',
                '## 能提前到哪里？','',
                '每处允许四次INP装载在原等待前执行，等待保留在随后WGT覆盖之前。实际位置、地址及必要保护见逐调用joint_certificate；不推广为任意同步激进化。']
    elif stage['status']=='R_QUALIFIED_NO_REMAINING_SITE':
        lines+=['恢复作者循环顺序的R已完成14次资格，但原MAC后移候选已消失。未生成或执行S；作为复现差异封存。']
    else:
        lines+=['本轮在门禁失败或支持未决处停止，未退回旧TPS寻找收益。失败详情见下表和对应worker日志。','',
                '| 位置 | 原因 |','|---|---|']
        for f in failures:lines.append('| `%s` | %s |'%(f['path'],str(f.get('error','')).replace('|','/')))
        if not failures:lines.append('| 控制器/资源门禁 | `%s` |'%str(stage.get('reason')))
        if any(f.get('category')=='TOOL_ERROR' for f in failures):
            lines+=['','失败类别：工具链路错误；不能据此判定R或S方法不合法。']
    lines+=['','## 值不值得进入性能验证？','',
            '本阶段结果只决定本地对照是否成立。R相对O的循环表示恢复归为复现修正；作者完整dthread栈与本地适配仍有差异。即使S通过，也不能由FSim或命令数推导板端加速。','',
            '现有指令聚合结果保持封存。本阶段结束后暂停，FPGA资格与完整调用计时须另行设计。','',
            '## 证据入口','',
            '- `freeze.json`：共同源码、runtime/helper、硬件及历史来源。',
            '- `prepare/result.json`：O重建、R配置、往返、乘积及循环证据（若准备完成）。',
            '- `call_ledger.json`：最多28次调用的冻结预算。',
            '- `R_*`、`S_*`：逐调用输出、实际存储、native、物理及FIFO证书。',
            '- `stage_result.json`：停止状态及已完成调用；`tests.log`：工具测试。','']
    (root/'REPORT.md').write_text('\n'.join(lines))
    files={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
    write(root/'completion_manifest.json',{'schema':'c3_tps_sync_release_local_completion_v1',
        'status':stage['status'],'paused':True,'source_identities_unchanged':unchanged,
        'FPGA_calls':0,'performance_conclusion':None,'artifact_sha256':files})
    verify({str(root/p):digest for p,digest in files.items()})
    print(json.dumps({'status':stage['status'],'report':str(root/'REPORT.md'),'sealed_files':len(files)}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','qualify','report','_prepare_worker','_s_prepare_worker','_qualify_worker'))
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--program',choices=('R','S'))
    parser.add_argument('--mode',choices=('detailed','minimal'))
    args=parser.parse_args();root=args.output_dir.resolve()
    try:
        if args.phase=='prepare':prepare(root)
        elif args.phase=='qualify':qualify(root)
        elif args.phase=='report':report(root)
        elif args.phase=='_prepare_worker':
            verify(read(root/'freeze.json')['bindings'])
            from c3_tps_sync_release_prepare import prepare_worker
            prepare_worker(root)
        elif args.phase=='_s_prepare_worker':
            os.environ['TVM_LIBRARY_PATH']=str(RUNTIME)
            verify(read(root/'freeze.json')['bindings']);prepare_s(root)
        else:qualification_worker(root,args.program,args.mode)
        if args.phase in ('prepare','qualify') and (root/'stage_result.json').exists():
            if read(root/'stage_result.json')['status'].startswith('STOP_'):
                raise SystemExit(1)
    except Exception as error:
        if args.phase.startswith('_'):
            destination=root/(f'{args.program}_{args.mode}' if args.phase=='_qualify_worker' else args.phase[1:])
            destination.mkdir(exist_ok=True)
            if not (destination/'failure.json').exists():
                failure_phase=args.phase
                if args.phase=='_prepare_worker':
                    failure_phase='R_PREPARATION' if (root/'prepare/R').exists() else 'O_REBUILD_OR_ADOPTION'
                category='TOOL_ERROR' if isinstance(error,(AttributeError,TypeError,KeyError,NameError)) else 'CERTIFICATION_OR_SUPPORT_UNRESOLVED'
                write(destination/'failure.json',{'error':str(error),'traceback':traceback.format_exc(),
                    'exception_type':type(error).__name__,'category':category,'failure_phase':failure_phase,
                    'FPGA_calls':0,'automatic_retry':False})
        elif root.exists() and (root/'freeze.json').exists() and not (root/'stage_result.json').exists():
            close_attempt(root,'STOP_CONTROLLER_GATE',{'error':str(error),'traceback':traceback.format_exc()})
        raise


if __name__=='__main__':main()
