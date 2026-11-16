#!/usr/bin/env python3
"""Matched fixed-group/open YOLO policies; no profile or scoring changes."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))


def main(argv=None):
    from yolo_cpu_policy_check.common import OUTPUT,RUN,LOCAL_CONFIG
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('audit','rank','prepare','validate','board','status','analyze'))
    p.add_argument('action',nargs='?',choices=('run','resume'))
    p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--run-dir',type=Path,default=RUN)
    p.add_argument('--local-config',type=Path,default=LOCAL_CONFIG);p.add_argument('--execute-board',action='store_true')
    a=p.parse_args(argv)
    if a.command in ('audit','rank'):
        from yolo_cpu_policy_check import ranking
        result=getattr(ranking,a.command)(a.output)
    elif a.command in ('prepare','validate'):
        from yolo_cpu_policy_check import bundle
        result=getattr(bundle,a.command)(a.output)
    elif a.command=='board':
        if not a.execute_board or not a.action:p.error('board run/resume requires --execute-board')
        from yolo_cpu_policy_check.campaign import run
        result=run(a.output,a.run_dir,a.local_config,execute_board=True)
    elif a.command=='status':
        from yolo_cpu_policy_check.campaign import status
        result=status(a.output,a.run_dir)
    else:
        from yolo_cpu_policy_check.analysis import analyze
        result=analyze(a.output,a.run_dir)
    print(json.dumps(result,ensure_ascii=False));return result


if __name__=='__main__':main()
