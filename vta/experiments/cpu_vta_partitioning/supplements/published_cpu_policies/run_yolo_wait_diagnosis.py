#!/usr/bin/env python3
"""Frozen YOLO waiting diagnosis and one conditional blocking-wait experiment."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))


def main(argv=None):
    from wait_diagnosis.common import OUTPUT,RUN,LOCAL_CONFIG
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('audit','prepare','validate','board','status','analyze'))
    p.add_argument('action',nargs='?',choices=('run','resume'))
    p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--run-dir',type=Path,default=RUN)
    p.add_argument('--local-config',type=Path,default=LOCAL_CONFIG);p.add_argument('--execute-board',action='store_true')
    a=p.parse_args(argv)
    if a.command in ('audit','prepare','validate'):
        from wait_diagnosis import prepare
        result=getattr(prepare,a.command)(a.output)
    elif a.command=='board':
        if not a.action or not a.execute_board:p.error('board run/resume requires --execute-board')
        from wait_diagnosis.campaign import run
        result=run(a.output,a.run_dir,a.local_config,execute_board=True)
    elif a.command=='status':
        from wait_diagnosis.campaign import status
        result=status(a.output,a.run_dir)
    else:
        from wait_diagnosis.analysis import analyze
        result=analyze(a.output,a.run_dir)
    print(json.dumps(result,ensure_ascii=False));return result


if __name__=='__main__':main()
