"""Local preparation and explicitly authorized, one-boot diagnosis."""
import argparse
import json
from pathlib import Path
import re

from .common import ROOT, DEFAULT_LOCAL_CONFIG, DEFAULT_RUN_ID


def main(argv=None):
    parser = argparse.ArgumentParser(description='CPU–VTA静态预测误差的小规模诊断')
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--base-bundle', type=Path)
    prepare.add_argument('--source-run', type=Path)
    validate = commands.add_parser('validate')
    validate.add_argument('--bundle', type=Path, required=True)
    for name in ('status', 'analyze'):
        sub = commands.add_parser(name)
        sub.add_argument('--run-dir', type=Path, required=True)
    boards = commands.add_parser('board').add_subparsers(dest='action', required=True)
    run = boards.add_parser('run')
    run.add_argument('--bundle', type=Path, required=True)
    run.add_argument('--run-id', default=DEFAULT_RUN_ID)
    run.add_argument('--local-config', type=Path, default=DEFAULT_LOCAL_CONFIG)
    resume = boards.add_parser('resume')
    resume.add_argument('--run-dir', type=Path, required=True)
    for sub in (run, resume):
        sub.add_argument('--execute-board', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        from .prepare import prepare
        result = {'status': 'PREPARED_OFFLINE', 'bundle': str(prepare(args.base_bundle, args.source_run))}
    elif args.command == 'validate':
        from .prepare import validate_bundle
        manifest = validate_bundle(args.bundle)
        result = {'status': 'VALIDATED_OFFLINE', 'bundle_id': manifest['bundle_id']}
    elif args.command == 'status':
        from .campaign import status
        result = status(args.run_dir)
    elif args.command == 'analyze':
        from .campaign import validate_state, validated_records
        state = validate_state(args.run_dir)
        validated_records(args.run_dir, state)
        from .analysis import analyze
        result = analyze(args.run_dir)
    else:
        if not args.execute_board:
            parser.error('board access requires --execute-board')
        from .campaign import create, validate_state, run_current, status
        from .runtime import SSH
        if args.action == 'run':
            if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id):
                parser.error('invalid run identity')
            connection = json.loads(args.local_config.read_text())['board']
            root = ROOT / 'runs' / args.run_id
            state = create(args.bundle, root, connection)
        else:
            root = args.run_dir.resolve()
            state = validate_state(root)
            connection = state['connection']
        run_current(root, SSH(connection, execute_board=True, bundle=state['bundle']))
        result = {**status(root), 'run_dir': str(root)}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return result
