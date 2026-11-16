import argparse
import json
from pathlib import Path
import re
from .common import ROOT, DEFAULT_RUN_ID, DEFAULT_LOCAL_CONFIG

def main(argv=None):
    parser=argparse.ArgumentParser(description='完整DP候选CPU重排与Top-20验证')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--base-bundle',type=Path);p.add_argument('--frame-bundle',type=Path);p.add_argument('--static-source',type=Path);p.add_argument('--source-run',type=Path)
    p=sub.add_parser('validate');p.add_argument('--bundle',type=Path,required=True)
    p=sub.add_parser('rank');p.add_argument('--stage',choices=['static','frame-stream'],required=True);p.add_argument('--run-dir',type=Path,required=True)
    for command in ('status','analyze'):
        p=sub.add_parser(command);p.add_argument('--run-dir',type=Path,required=True)
    board=sub.add_parser('board').add_subparsers(dest='action',required=True)
    p=board.add_parser('run');p.add_argument('--bundle',type=Path,required=True);p.add_argument('--run-id',default=DEFAULT_RUN_ID)
    p.add_argument('--local-config',type=Path,default=DEFAULT_LOCAL_CONFIG);p.add_argument('--execute-board',action='store_true')
    p=board.add_parser('resume');p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--execute-board',action='store_true')
    args=parser.parse_args(argv)
    if args.command=='prepare':
        from .prepare import prepare
        result={'status':'PREPARED_OFFLINE','bundle':str(prepare(args.base_bundle,args.frame_bundle,args.static_source,args.source_run))}
    elif args.command=='validate':
        from .prepare import validate_bundle
        result={'status':'VALIDATED_OFFLINE','bundle_id':validate_bundle(args.bundle)['bundle_id']}
    elif args.command=='status':
        from .campaign import status
        result=status(args.run_dir)
    elif args.command=='rank':
        from .campaign import static_rank,freeze_predictions
        result=(static_rank if args.stage=='static' else freeze_predictions)(args.run_dir)
    elif args.command=='analyze':
        from .analysis import analyze
        result=analyze(args.run_dir)
    else:
        if not args.execute_board:parser.error('board access requires --execute-board')
        from .campaign import create,validate_state,run_current,status
        from .runtime import SSH
        if args.action=='run':
            if not re.fullmatch('[A-Za-z0-9_-]+',args.run_id):parser.error('invalid run ID')
            root=ROOT/'runs'/args.run_id;connection=json.loads(args.local_config.read_text())['board']
            state=create(args.bundle,root,connection)
        else:
            root=args.run_dir.resolve();state=validate_state(root);connection=state['connection']
        run_current(root,SSH(connection,execute_board=True,bundle=state['bundle']))
        result=status(root)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False));return result
