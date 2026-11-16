#!/usr/bin/env python3
"""Independent three-model YOLO validation; explicit gate for all board work."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))

def main(argv=None):
    from yolo_validation.common import DEFAULT_OUTPUT,DEFAULT_RUN,LOCAL_CONFIG,METHODS
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('audit','prepare','validate','rank','rank-one','board','status','analyze'))
    p.add_argument('action',nargs='?',choices=('run','resume'))
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    p.add_argument('--local-config',type=Path,default=LOCAL_CONFIG)
    p.add_argument('--model',choices=METHODS)
    p.add_argument('--execute-board',action='store_true')
    a=p.parse_args(argv)
    if a.command=='audit':
        from yolo_validation.evidence import audit
        result=audit(a.output)
    elif a.command=='prepare':
        from yolo_validation.evidence import prepare
        result=prepare(a.output)
    elif a.command=='validate':
        from yolo_validation.validation import validate
        result=validate(a.output)
    elif a.command in ('rank','rank-one'):
        from yolo_validation.ranking import rank,rank_one
        if a.command=='rank-one':
            if not a.model:p.error('--model is required')
            result=rank_one(a.output,a.model)
        else:result=rank(a.output)
    elif a.command=='board':
        if not a.execute_board:p.error('board operations require --execute-board')
        if not a.action:p.error('board run or board resume is required')
        from yolo_validation.campaign import run
        result=run(a.output,a.run_dir,a.local_config,execute_board=True)
    elif a.command=='status':
        from yolo_validation.campaign import status
        result=status(a.output,a.run_dir)
    else:
        from yolo_validation.analysis import analyze
        result=analyze(a.output,a.run_dir)
    print(json.dumps(result,ensure_ascii=False));return result

if __name__=='__main__':main()
