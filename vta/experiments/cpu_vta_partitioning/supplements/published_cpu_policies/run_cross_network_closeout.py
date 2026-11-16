#!/usr/bin/env python3
"""Finite frozen validation flow. Stops on any error; never reboots the board."""
import argparse
from pathlib import Path
import os
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))

if __name__=='__main__':
    from cpu_vta_partitioning.common import now,read,sha256,write
    from workload_model.common import OUTPUT,ROOT
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute-board',action='store_true');a=p.parse_args()
    if not a.execute_board:raise ValueError('--execute-board is required')
    if read(ROOT/'runs/cross_network_pool_intrinsics_01/state.json')['status']!='COMPLETE':
        raise ValueError('finish all frozen isolated signatures before closeout')
    steps=[('project','run_pool_temporal.py','project'),
           ('intrinsic_features','run_cross_network_final.py','features'),
           ('freeze_temporal_rank','run_pool_temporal.py','rank'),
           ('formal_short','run_common_top20_guarded.py','--execute-board'),
           ('analyze_common_short','run_common_top20.py','analyze'),
           ('freeze_all_long_winners','run_cross_network_final.py','short'),
           ('formal_long_original','run_common_top20.py','long','--execute-board'),
           ('analyze_common_long','run_common_top20.py','analyze'),
           ('formal_long_extra','run_cross_network_final.py','long','--execute-board'),
           ('analyze_all_long','run_cross_network_final.py','long-results'),
           ('report','run_cross_network_final.py','report')]
    logdir=OUTPUT/'closeout_logs';logdir.mkdir(parents=True,exist_ok=True)
    attempt=now().replace(':','-');environment={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1'}
    for number,(label,script,*arguments) in enumerate(steps,1):
        path=logdir/(attempt+'_'+str(number).zfill(2)+'_'+label+'.log')
        state=dict(status='RUNNING',step=label,number=number,total=len(steps),log=str(path),
                   updated_at=now(),reboot_allowed=False,runner_sha256=sha256(Path(__file__)))
        write(OUTPUT/'CLOSEOUT_STATE.json',state,replace=True)
        print('闭环步骤',number,'/',len(steps),label,flush=True)
        with path.open('w') as stream:
            code=subprocess.call([sys.executable,'-u',str(ROOT/script),*arguments],cwd=ROOT,
                                 env=environment,stdout=stream,stderr=subprocess.STDOUT)
        if code:
            write(OUTPUT/'CLOSEOUT_STATE.json',dict(**{k:v for k,v in state.items() if k!='status'},status='STOPPED_ON_ERROR',exit_code=code),replace=True)
            print(path.read_text()[-5000:],flush=True);sys.exit(code)
    write(OUTPUT/'CLOSEOUT_STATE.json',dict(status='PIPELINE_COMPLETE_PENDING_EVIDENCE_AUDIT',
          updated_at=now(),report=str(ROOT.parents[4]/'paper/my_paper/analysis/20261110_cross_network_model/FINAL_REPORT_zh.md'),
          goal_complete=False),replace=True)
