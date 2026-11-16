#!/usr/bin/env python3
"""Prepare a capsule, then use only that capsule's immutable board controller."""
import json
from pathlib import Path
import subprocess
import sys

def main(argv=None):
    args=list(sys.argv[1:] if argv is None else argv)
    if not args or args[0]=='--help':
        print('prepare --search-root PATH | validate/rank --bundle PATH | board run --bundle PATH --execute-board | board resume/status/analyze --run-dir PATH')
        return
    if args[0]=='prepare':
        import argparse
        p=argparse.ArgumentParser();p.add_argument('--search-root',type=Path,required=True)
        options=p.parse_args(args[1:])
        from resource_envelope.prepare_campaign import prepare
        print(json.dumps({'status':'PREPARED_OFFLINE','bundle':str(prepare(options.search_root))}));return
    if '--bundle' in args:bundle=Path(args[args.index('--bundle')+1]).resolve()
    elif '--run-dir' in args:
        run=Path(args[args.index('--run-dir')+1]).resolve();bundle=Path(json.loads((run/'state.json').read_text())['bundle'])
        args[args.index('--run-dir')+1]=str(run)
    else:raise ValueError('a frozen --bundle or --run-dir is required')
    if '--bundle' in args:args[args.index('--bundle')+1]=str(bundle)
    # Source-tree edits cannot silently change a campaign in progress.
    controller=bundle/'sources/envelope_controller'
    raise SystemExit(subprocess.run([sys.executable,str(controller/'run_envelope_top20.py'),*args],cwd=controller).returncode)

if __name__=='__main__':main()
