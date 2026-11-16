import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from .common import ROOT, DEFAULT_LOCAL_CONFIG, DEFAULT_RUN_ID

def main(argv=None):
    parser=argparse.ArgumentParser(description='受控并发组件与跨切图验证')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare'); p.add_argument('--base-bundle',type=Path); p.add_argument('--full-bundle',type=Path)
    p.add_argument('--reuse-executor-bundle',type=Path)
    p=sub.add_parser('validate'); p.add_argument('--bundle',required=True,type=Path)
    for name in ('status','analyze','rank'):
        p=sub.add_parser(name); p.add_argument('--run-dir',required=True,type=Path)
    board=sub.add_parser('board').add_subparsers(dest='action',required=True)
    p=board.add_parser('run'); p.add_argument('--bundle',required=True,type=Path); p.add_argument('--run-id',default=DEFAULT_RUN_ID)
    p.add_argument('--local-config',default=DEFAULT_LOCAL_CONFIG,type=Path); p.add_argument('--execute-board',action='store_true')
    p=board.add_parser('resume'); p.add_argument('--run-dir',required=True,type=Path); p.add_argument('--execute-board',action='store_true')
    args=parser.parse_args(argv)
    # A validator-only repair is replayed by its hash-bound archived adapter;
    # do not apply the current controller identity to the old execution capsule.
    if args.command in ('status','analyze') or (args.command=='board' and args.action=='resume'):
        patch=args.run_dir.resolve()/'validation_repair/compat.py'
        if patch.is_file():
            action='resume' if args.command=='board' else args.command
            command=[sys.executable,str(patch),action,'--run-dir',str(args.run_dir.resolve())]
            if args.command=='board' and args.execute_board: command.append('--execute-board')
            subprocess.run(command,check=True)
            return None
    if args.command=='prepare':
        from .prepare import prepare
        result={'bundle':str(prepare(args.base_bundle,args.full_bundle,args.reuse_executor_bundle)),'status':'PREPARED_OFFLINE'}
    elif args.command=='validate':
        from .prepare import validate_bundle
        result={'bundle_id':validate_bundle(args.bundle)['bundle_id'],'status':'VALIDATED_OFFLINE'}
    elif args.command=='status':
        from .campaign import status
        result=status(args.run_dir)
    elif args.command=='analyze':
        from .analysis import analyze
        result=analyze(args.run_dir)
    elif args.command=='rank':
        from .campaign import validate_state, freeze_predictions
        result=freeze_predictions(args.run_dir,validate_state(args.run_dir))
    else:
        if not args.execute_board: parser.error('board access requires --execute-board')
        from .campaign import create,validate_state,run_current,status
        from .runtime import SSH
        if args.action=='run':
            if not re.fullmatch('[A-Za-z0-9_-]+',args.run_id): parser.error('invalid run ID')
            root=ROOT/'runs'/args.run_id; connection=json.loads(args.local_config.read_text())['board']
            state=create(args.bundle,root,connection)
        else:
            root=args.run_dir.resolve(); state=validate_state(root); connection=state['connection']
        run_current(root,SSH(connection,execute_board=True,bundle=state['bundle']))
        result=status(root)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
    return result
