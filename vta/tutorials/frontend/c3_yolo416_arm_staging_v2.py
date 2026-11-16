"""Wait for complete local gates, then build cases/ARM and freeze board pool.

Never contacts a board, never retries a target/build, never changes candidates.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from c3_tps_sync_release_local import HERE,PYTHON,read,write,sha,require,env_for_worker
from c3_yolo416_aux_closeout import seal


def bounded(root,label,args,seconds):
    import psutil
    env=env_for_worker()
    for k in ('VTA_UOP_BUFFER_BYTES','VTA_INSN_BUFFER_BYTES','VTA_INSN_SUBMIT_THRESHOLD_BYTES'):env[k]='33554432'
    start=time.monotonic();peak=0;limit=None
    with (root/(label+'.log')).open('x') as log:
        child=subprocess.Popen(args,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while child.poll() is None:
            try:
                proc=psutil.Process(child.pid);peak=max(peak,sum(p.memory_info().rss for p in [proc,*proc.children(recursive=True)]))
            except psutil.Error:pass
            if peak>8*1024**3 or time.monotonic()-start>seconds:
                limit='MEMORY' if peak>8*1024**3 else 'TIME';os.killpg(child.pid,signal.SIGKILL);break
            time.sleep(.1)
        code=child.wait()
    result={'returncode':code,'resource_limit':limit,'sampled_peak_tree_RSS':peak,
            'resource_control_seconds':time.monotonic()-start,'arguments':args}
    write(root/(label+'_worker.json'),result);return result


def stage(root,pipeline,operators,graph,cases,arm,board_root):
    root.mkdir(exist_ok=False)
    try:
        require(not any(p.exists() for p in (cases,arm,board_root)),'STAGING_DIRECTORY_EXISTS')
        write(root/'contract.json',{'local_pipeline':str(pipeline),'operators':str(operators),'graph':str(graph),
              'cases':str(cases),'ARM':str(arm),'board_prepared':str(board_root),
              'board_contacted':False,'FPGA_calls':0,'VTA_or_GraphExecutor_calls':0,'retry':False})
        start=time.monotonic()
        while True:
            require(time.monotonic()-start<21600,'LOCAL_COMPLETION_NOT_OBSERVED')
            if (pipeline/'failure.json').exists():
                try:failure=read(pipeline/'failure.json')
                except json.JSONDecodeError:time.sleep(1);continue
                raise RuntimeError('LOCAL_PIPELINE_STOPPED:'+str(failure))
            if (pipeline/'result.json').exists():
                try:result=read(pipeline/'result.json')
                except json.JSONDecodeError:time.sleep(1);continue
                require(result['status']=='OPERATORS_AND_WHOLE_GRAPHS_LOCAL_QUALIFIED','LOCAL_PIPELINE_NOT_QUALIFIED')
                break
            time.sleep(5)
        record=bounded(root,'model_cases',[str(PYTHON),str(HERE/'c3_yolo416_operator_cases.py'),str(cases),
                      '--qualified',str(graph)],600)
        if not cases.exists():cases.mkdir()
        if not (cases/'status.json').exists():write(cases/'status.json',{'status':'STOPPED_CASE_OUTER_WORKER','target_calls':0})
        write(cases/'outer_worker.json',record)
        ok=seal(cases,'cases');require(record['returncode']==0 and record['resource_limit'] is None and ok,
                                     'MODEL_CASE_EXPORT_STOPPED_NO_RETRY')
        record=bounded(root,'ARM_build',[str(PYTHON),str(HERE/'c3_yolo416_arm_prepare_v2.py'),'prepare',str(arm),
                      '--operators',str(operators),'--graph',str(graph)],1800)
        if not arm.exists():arm.mkdir()
        if not (arm/'status.json').exists():write(arm/'status.json',{'status':'STOPPED_ARM_OUTER_WORKER','target_calls':0})
        write(arm/'outer_worker.json',record)
        ok=seal(arm,'arm');require(record['returncode']==0 and record['resource_limit'] is None and ok,
                                  'ARM_BUILD_STOPPED_NO_RETRY')
        # Final code preflight is performed with no external connection.
        modules=['test_c3_yolo416_comparison_protocol','test_c3_yolo416_board_transport',
                 'test_c3_yolo416_board_gates','test_c3_yolo416_graph_host_reference',
                 'test_c3_yolo416_rpc_observer','test_c3_yolo416_board_session','test_c3_yolo416_plugin_lifetime']
        record=bounded(root,'board_tools',[str(PYTHON),'-m','unittest',*modules,'-v'],180)
        require(record['returncode']==0 and record['resource_limit'] is None,'BOARD_TOOLS_PREFLIGHT_FAILED')
        record=bounded(root,'board_prepare',[str(PYTHON),str(HERE/'c3_yolo416_board_prepare.py'),str(board_root),
                      '--arm',str(arm),'--cases',str(cases)],600)
        require(record['returncode']==0 and record['resource_limit'] is None and
                (board_root/'freeze.json').exists() and not (board_root/'prepare_failure.json').exists(),
                'BOARD_PREPARATION_NOT_COMPLETE')
        write(root/'result.json',{'status':'ARM_CASES_AND_BOARD_POOL_READY_NO_BOARD_CONTACT',
              'board_freeze_sha256':sha(board_root/'freeze.json'),'FPGA_calls':0})
        print('ARM_CASES_AND_BOARD_POOL_READY_NO_BOARD_CONTACT',flush=True);return 0
    except Exception as e:
        write(root/'failure.json',{'type':type(e).__name__,'message':str(e),'FPGA_calls':0,'retry':False})
        traceback.print_exc();return 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    for name in ('pipeline','operators','graph','cases','arm','board'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();sys.exit(stage(*(x.resolve() for x in
        (a.output,a.pipeline,a.operators,a.graph,a.cases,a.arm,a.board))))
