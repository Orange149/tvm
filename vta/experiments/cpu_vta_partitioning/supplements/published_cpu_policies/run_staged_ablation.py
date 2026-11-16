#!/usr/bin/env python3
"""Independent audit, full-domain ranking and frozen board-campaign entry."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

def main(argv=None):
    args=list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ('audit','rank','rank-one','prepare'):
        from staged_ablation.common import DEFAULT_OUTPUT,MODELS
        p=argparse.ArgumentParser();p.add_argument('command',choices=('audit','rank','rank-one','prepare'))
        p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
        p.add_argument('--model',choices=[r[0] for r in MODELS]);p.add_argument('--workers',type=int,choices=(1,2),default=2)
        a=p.parse_args(args)
        if a.command=='audit':
            from staged_ablation.audit import audit
            result=audit(a.output)
        elif a.command=='rank':
            from staged_ablation.search import run_all
            result=run_all(a.output,a.workers)
        elif a.command=='rank-one':
            from staged_ablation.search import run_one
            if not a.model:p.error('--model is required')
            result=run_one(a.output,a.model)
        else:
            from staged_ablation.prepare import prepare
            result={'status':'PREPARED_OFFLINE','bundle':str(prepare(a.output))}
        print(json.dumps(result,ensure_ascii=False));return result
    if not args or args[0] in ('--help','-h'):
        print('audit/rank/prepare --output PATH | validate --bundle PATH | board run/resume --execute-board | status/analyze --run-dir PATH');return
    if '--bundle' in args:bundle=Path(args[args.index('--bundle')+1]).resolve()
    elif '--run-dir' in args:
        run=Path(args[args.index('--run-dir')+1]).resolve();bundle=Path(json.loads((run/'state.json').read_text())['bundle'])
        args[args.index('--run-dir')+1]=str(run)
    else:raise ValueError('a frozen --bundle or --run-dir is required')
    if '--bundle' in args:args[args.index('--bundle')+1]=str(bundle)
    controller=bundle/'sources/staged_ablation_controller'
    raise SystemExit(subprocess.run([sys.executable,str(controller/'run_staged_ablation.py'),*args],cwd=controller,
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')).returncode)

if __name__=='__main__':main()
