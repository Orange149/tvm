#!/usr/bin/env python3
"""Independent eight-model search and immutable board campaign entry point."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

def main(argv=None):
    args=list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ('rank','rank-one','prepare'):
        from model_ablation.common import DEFAULT_OUTPUT,MODELS
        p=argparse.ArgumentParser();p.add_argument('command',choices=('rank','rank-one','prepare'))
        p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
        p.add_argument('--model',choices=[r[0] for r in MODELS]);p.add_argument('--workers',type=int,choices=(1,2),default=2)
        parsed=p.parse_args(args)
        if parsed.command=='rank':
            from model_ablation.search import run_all
            result=run_all(parsed.output,parsed.workers)
        elif parsed.command=='rank-one':
            from model_ablation.search import run_one
            if not parsed.model:p.error('--model is required')
            result=run_one(parsed.output,parsed.model)
        else:
            from model_ablation.prepare import prepare
            result={'status':'PREPARED_OFFLINE','bundle':str(prepare(parsed.output))}
        print(json.dumps(result,ensure_ascii=False));return result
    if not args or args[0] in ('--help','-h'):
        print('rank/prepare --output PATH | validate --bundle PATH | board run --bundle PATH --execute-board | board resume/status/analyze --run-dir PATH');return
    if '--bundle' in args:bundle=Path(args[args.index('--bundle')+1]).resolve()
    elif '--run-dir' in args:
        run=Path(args[args.index('--run-dir')+1]).resolve();bundle=Path(json.loads((run/'state.json').read_text())['bundle'])
        args[args.index('--run-dir')+1]=str(run)
    else:raise ValueError('a frozen --bundle or --run-dir is required')
    if '--bundle' in args:args[args.index('--bundle')+1]=str(bundle)
    controller=bundle/'sources/ablation_controller'
    raise SystemExit(subprocess.run([sys.executable,str(controller/'run_model_ablation.py'),*args],cwd=controller,
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')).returncode)

if __name__=='__main__':main()
