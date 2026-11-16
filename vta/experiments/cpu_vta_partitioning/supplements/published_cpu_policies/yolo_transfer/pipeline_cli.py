"""Also copied as the standalone entry inside each frozen pipeline capsule."""
import argparse
import json
from pathlib import Path
import sys


def main():
    source_root = Path(__file__).resolve().parent
    if (source_root / 'base_controller').is_dir():
        sys.path.insert(0, str(source_root / 'base_controller'))
        sys.path.insert(0, str(source_root))
    from yolo_transfer import pipeline
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('validate'); p.add_argument('--bundle', type=Path, required=True)
    p = sub.add_parser('board'); p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True); p.add_argument('--local-config', type=Path, required=True)
    p.add_argument('--execute-board', action='store_true')
    p = sub.add_parser('analyze'); p.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'validate':
        manifest = pipeline.validate(args.bundle)
        result = {'bundle_id': manifest['bundle_id'], 'validated': True}
    elif args.command == 'board':
        result = pipeline.run(args.bundle, args.run_dir, args.local_config, execute_board=args.execute_board)
    else:
        from yolo_transfer.analysis import analyze
        result = analyze(args.run_dir)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
