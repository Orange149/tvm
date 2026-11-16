#!/usr/bin/env python3
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
if __name__=='__main__':
    import finite_service_holdout_board as w
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','qemu','validate','board','status'))
    p.add_argument('--execute-board',action='store_true');a=p.parse_args()
    if a.command=='board':result=w.board(execute_board=a.execute_board)
    elif a.command=='validate':result={n:w.validate(w.POOL/n/'capsule') for n in w.selection()['selected_configurations']}
    elif a.command=='status':
        from cpu_vta_partitioning.common import read
        result=read(w.RUN/'state.json') if (w.RUN/'state.json').exists() else dict(status='NOT_STARTED')
    else:result=getattr(w,a.command)()
    print(json.dumps(result,ensure_ascii=False))
