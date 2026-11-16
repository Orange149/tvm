#!/usr/bin/env python3
"""Freeze and run a serial, bounded read-only audit of existing native traces."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import psutil
from c3_sync_release_audit import HERE, HOLD, inventory, read, require, sha, write


def verify(bindings):
    for path,digest in bindings.items():
        require(sha(path)==digest,'FROZEN_IDENTITY_CHANGED:'+path)


def run_worker(row_path, result_path, log_path):
    started, peak, limit = time.monotonic(),0,None
    argv=[sys.executable,str(HERE/'c3_sync_release_audit.py'),'--row',str(row_path),'--output',str(result_path)]
    with log_path.open('x') as log:
        worker=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while worker.poll() is None:
            try:
                parent=psutil.Process(worker.pid)
                current=sum(p.memory_info().rss for p in [parent,*parent.children(recursive=True)])
                peak=max(peak,current)
            except (psutil.NoSuchProcess,psutil.AccessDenied):
                pass
            if peak>8*1024**3:limit='ANALYSIS_MEMORY_LIMIT'
            if time.monotonic()-started>600:limit='ANALYSIS_TIME_LIMIT'
            if limit:
                os.killpg(worker.pid,signal.SIGKILL)
                break
            time.sleep(.1)
        code=worker.wait()
    return {'returncode':code,'analysis_limit':limit,'peak_sampled_process_tree_rss':peak,
            'memory_sampling_seconds':.1,'elapsed_seconds':time.monotonic()-started,'argv':argv}


def summarize(results):
    groups={}
    candidates=[]
    for result in results:
        family='TPS_adapted' if result['family']=='TPS_adapted' else (
            'ResNet18_official' if result['object_id'].startswith('resnet') else 'YOLO416_official')
        group=groups.setdefault(family,{'objects':0,'g2l_tokens':0,'census':Counter(),
            'reset_cut_categories':Counter(),'operand_cut_categories':Counter(),'move_categories':Counter(),
            'move_candidate_objects':set(),'move_candidates':0,'move_candidates_finite_batch':0})
        group['objects']+=1
        group['g2l_tokens']+=result['g2l_tokens']
        group['census'].update(result['census'])
        group['reset_cut_categories'].update(result['reset_cut_categories'])
        group['operand_cut_categories'].update(r['category'] for r in result['cuts']
                                               if r['consumer_description'] in ('LOAD INP','LOAD WGT'))
        group['move_categories'].update(result['move_categories'])
        for row in result['moves']:
            if row['category']=='POSSIBLE_EARLIER_RELEASE':
                group['move_candidates']+=1
                group['move_candidate_objects'].add(result['object_id'])
                group['move_candidates_finite_batch']+=row['finite_fifo']=='SUFFICIENT_FULL_BATCH_FITS'
                candidates.append({'object_id':result['object_id'],'family':family,**row})
    for group in groups.values():
        group['move_candidate_objects']=sorted(group['move_candidate_objects'])
        for key in ('census','reset_cut_categories','operand_cut_categories','move_categories'):
            group[key]=dict(group[key])
    return groups,candidates


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    (out/'objects').mkdir()
    (out/'tool_sources').mkdir()
    sources=[HERE/name for name in ('c3_sync_release_audit.py','run_c3_sync_release_audit.py',
        'test_c3_sync_release_audit.py','c3_five_tradeoff_commands.py','c3_thread_sync_audit.py')]
    env=dict(os.environ,PYTHONPATH=str(HERE))
    tests=subprocess.run([sys.executable,'-m','unittest','-v','test_c3_sync_release_audit'],
                         capture_output=True,text=True,env=env,timeout=60)
    (out/'tests.log').write_text(tests.stdout+tests.stderr)
    require(tests.returncode==0,'AUDIT_TOOL_TESTS_FAILED')
    rows=inventory()
    write(out/'inventory.json',rows)
    bindings={str(p):sha(p) for p in sources}
    for source in sources:shutil.copy2(source,out/'tool_sources'/source.name)
    for row in rows:
        bindings.update(row['expected_source_sha256'])
        bindings[row['source_seal']]=row['source_seal_sha256']
    hardware=HOLD/'20261014_c3_tps_window_complete_run03/hardware_archive.json'
    archived=read(hardware)
    verify(archived['source_bindings'])
    bindings.update(archived['source_bindings'])
    bindings[str(hardware)]=sha(hardware)
    frozen_sources=[HERE/'c3_banerjee_local.py',HERE/'run_c3_banerjee_completion.py',
        HERE/'c3_banerjee_completion.py',HERE.parents[2]/'src/tir/transforms/coproc_sync.cc',
        HERE.parents[2]/'vta/python/vta/transform.py',HERE.parents[2]/'vta/runtime/runtime.cc',
        HOLD/'20261012_c3_banerjee_double_buffer_local_run01/author_sources/vta_conv2d.py']
    bindings.update({str(p):sha(p) for p in frozen_sources})
    verify(bindings)
    write(out/'freeze.json',{'status':'READONLY_AUDIT_FROZEN','bindings':bindings,
        'objects':27,'official_objects':26,'TPS_objects':1,'sequential':True,
        'max_process_tree_bytes':8*1024**3,'worker_seconds':600,'new_FSim_calls':0,'FPGA_calls':0,
        'new_executable_programs':0,'scope':'independent graph hypotheses on original sealed instructions',
        'classification':'live required ordering; dead operand physical conflicts unresolved; independent OUT',
        'not_an_optimization_or_performance_attempt':True})
    results,workers=[],[]
    for row in rows:
        name=row['object_id']
        row_path=out/'objects'/(name+'_input.json')
        result_path=out/'objects'/(name+'.json')
        write(row_path,row)
        worker=run_worker(row_path,result_path,out/'objects'/(name+'.log'))
        worker['object_id']=name
        write(out/'objects'/(name+'_worker.json'),worker)
        workers.append(worker)
        if worker['returncode']==0 and result_path.exists():
            result=read(result_path)
            results.append(result)
            print(json.dumps({k:result[k] for k in ('object_id','status','census','reset_cut_categories','move_categories')}),flush=True)
        else:
            print(json.dumps({'object_id':name,'status':'ANALYSIS_UNRESOLVED','worker':worker}),flush=True)
    verify(bindings)
    groups,candidates=summarize(results)
    write(out/'candidate_inventory.json',candidates)
    write(out/'summary.json',{'status':'READONLY_AUDIT_COMPLETE' if len(results)==27 else 'READONLY_AUDIT_WITH_UNRESOLVED_OBJECTS',
        'objects':27,'analysis_complete':len(results),'analysis_unresolved':27-len(results),
        'groups':groups,'workers':workers,'new_FSim_calls':0,'FPGA_calls':0,'new_executable_programs':0,
        'all_selected_source_identities_unchanged':True,'performance_conclusion':None,
        'independent_hypotheses_not_combined':True,'paused_after_stage':True})
    print(json.dumps({'status':'READONLY_AUDIT_FINISHED','completed':len(results),'groups':groups}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args()
    run(args.output_dir.resolve())
