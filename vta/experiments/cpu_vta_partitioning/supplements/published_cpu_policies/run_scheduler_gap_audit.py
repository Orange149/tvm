#!/usr/bin/env python3
"""Bounded thread sensitivity and scheduler-gap audit; old studies are read-only."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))

def main():
    from scheduler_gap_audit.common import OUTPUT,RUN,CONFIG
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('curves','audit','self-test','prepare','validate','qemu','board','status','analyze'))
    p.add_argument('action',nargs='?',choices=('run','resume'))
    p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--run-dir',type=Path,default=RUN)
    p.add_argument('--local-config',type=Path,default=CONFIG);p.add_argument('--execute-board',action='store_true')
    a=p.parse_args()
    if a.command in ('audit','self-test'):
        from scheduler_gap_audit.capability import audit,self_test
        result=(audit if a.command=='audit' else self_test)(a.output,a.local_config,execute_board=a.execute_board)
    elif a.command=='curves':
        from scheduler_gap_audit.curves import curves
        result=curves(a.output)
    elif a.command in ('prepare','validate','qemu'):
        from scheduler_gap_audit import prepare
        result=getattr(prepare,a.command)(a.output)
    elif a.command=='board':
        if not a.action:p.error('board requires run or resume')
        from scheduler_gap_audit.campaign import run
        result=run(a.output,a.run_dir,a.local_config,execute_board=a.execute_board)
    elif a.command=='analyze':
        from scheduler_gap_audit.analysis import analyze
        result=analyze(a.output,a.run_dir)
    else:
        from cpu_vta_partitioning.common import read
        result=read(a.run_dir/'state.json')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
