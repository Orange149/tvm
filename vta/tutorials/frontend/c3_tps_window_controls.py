#!/usr/bin/env python3
"""Requalify only sealed Q1/Q8 on the V4 common runtime; no new Q candidate."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from c3_tps_window_local import HERE,REPO,HOLD,PYTHON,sha,write


def run(output,complete):
    import tvm
    import vta
    from c3_official_window_workflow import _tree_rss
    if json.loads((complete/'stage_result.json').read_text())['status']!='COMPLETE_LOCAL_GATES_PASSED':
        raise ValueError('MAIN_OBJECTS_NOT_LOCALLY_QUALIFIED')
    output.mkdir(exist_ok=False)
    identity=json.loads((complete/'identity.json').read_text())
    bindings=dict(identity['source_bindings'])
    for path,digest in bindings.items():
        if sha(path)!=digest:raise ValueError('MAIN_QUALIFICATION_SOURCE_CHANGED:'+path)
    old=HOLD/'20261014_c3_valid_window_local_run01/local_final02'
    inventory={}
    for label in ('Q1','Q8'):
        source=old/label
        parent_path=source/'post_aggregate_pre_cpu.tir.json'
        final_path=source/'node24_final.tir.json'
        parent=tvm.ir.load_json(parent_path.read_text());final=tvm.ir.load_json(final_path.read_text())
        symbol=str(parent.attrs['global_symbol'])
        (output/f'{label}_parent.json').write_text(tvm.ir.save_json(tvm.IRModule({symbol:parent})))
        (output/f'{label}_final.json').write_text(tvm.ir.save_json(tvm.IRModule({symbol:final})))
        for path in (parent_path,final_path,output/f'{label}_parent.json',output/f'{label}_final.json'):
            bindings[str(path)]=sha(path)
        inventory[label]={'parent_module':str(output/f'{label}_parent.json'),
                          'final_module':str(output/f'{label}_final.json'),'global_var':symbol,
                          'bank_loop_source_verified':False,'sealed_Q_program':True,
                          'source_parent_sha256':sha(parent_path),'source_final_sha256':sha(final_path)}
    bindings[str(Path(__file__).resolve())]=sha(__file__)
    write(output/'objects.json',inventory)
    write(output/'identity.json',{'scope':'sealed Q1/Q8 controls only; variant B means leave Q representation unchanged',
                                 'source_bindings':bindings})
    (output/'hardware_archive.json').write_bytes((complete/'hardware_archive.json').read_bytes())
    rows=[]
    for label in ('Q1','Q8'):
        for mode in ('detailed','minimal'):
            name=f'{label}_{mode}'
            command=[str(PYTHON),str(Path(__file__).resolve()),'--worker',label,
                     '--mode',mode,'--output',str(output/name)]
            env=dict(os.environ,PYTHONPATH=':'.join(str(REPO/p) for p in ('python','vta/python','vta/tutorials/frontend')),
                     OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
            start,peak,limit=time.monotonic(),0,None
            with (output/f'{name}.log').open('x') as stream:
                child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                while child.poll() is None:
                    peak=max(peak,_tree_rss(child.pid))
                    if peak>8*1024**3:limit='AUDIT_MEMORY_LIMIT'
                    elif time.monotonic()-start>600:limit='AUDIT_TIME_LIMIT'
                    if limit:
                        os.killpg(child.pid,signal.SIGKILL);break
                    time.sleep(.2)
                code=child.wait()
            row={'program':label,'mode':mode,'returncode':code,
                 'status':limit or ('PASS' if code==0 else 'LOCAL_GATE_FAILED'),
                 'wall_seconds':time.monotonic()-start,'process_tree_peak_rss':peak}
            rows.append(row);write(output/f'{name}_worker.json',row);print(json.dumps(row),flush=True)
            if code!=0:
                write(output/'stage_result.json',{'status':'STOP_LOCAL_CONTROL_GATE_FAILED','workers':rows})
                return
    for path,digest in bindings.items():
        if sha(path)!=digest:raise ValueError('FROZEN_CONTROL_SOURCE_CHANGED:'+path)
    write(output/'stage_result.json',{'status':'SEALED_CONTROLS_REQUALIFIED_ON_COMMON_RUNTIME',
        'workers':rows,'local_complete_calls':28,'new_Q_candidate_generated':False,'board_contacted':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--complete')
    parser.add_argument('--worker',choices=('Q1','Q8'))
    parser.add_argument('--mode',choices=('detailed','minimal'),default='detailed')
    args=parser.parse_args()
    if args.worker:
        from c3_tps_window_complete import worker
        worker(Path(args.output).resolve(),'B',args.worker,args.mode)
    else:run(Path(args.output).resolve(),Path(args.complete).resolve())
