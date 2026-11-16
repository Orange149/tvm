#!/usr/bin/env python3
"""One YOLO pipeline: matched scheduler/stage/worker timeline diagnostic."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
if __name__=='__main__':
    from yolo_scheduler_trace.common import OUTPUT,RUN,CONFIG
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','validate','qemu','board','status','analyze'))
    p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--run-dir',type=Path,default=RUN)
    p.add_argument('--local-config',type=Path,default=CONFIG);p.add_argument('--execute-board',action='store_true');a=p.parse_args()
    if a.command in ('prepare','validate','qemu'):
        from yolo_scheduler_trace import prepare
        result=getattr(prepare,a.command)(a.output)
    elif a.command=='board':
        from yolo_scheduler_trace.campaign import run
        result=run(a.output,a.run_dir,a.local_config,execute_board=a.execute_board)
    elif a.command=='analyze':
        from yolo_scheduler_trace.analysis import analyze
        result=analyze(a.output,a.run_dir)
    else:
        from cpu_vta_partitioning.common import read
        result=read(a.run_dir/'state.json')
    print(json.dumps(result,ensure_ascii=False))
