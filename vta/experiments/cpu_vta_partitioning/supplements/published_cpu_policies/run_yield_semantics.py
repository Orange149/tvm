from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from yield_semantics.audit import OUTPUT, board, check, prepare
from cpu_vta_partitioning.common import read

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runtime Yield identity and zero-DNN bounded test')
    parser.add_argument('command', choices=['prepare', 'board', 'validate'])
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--execute-board', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.output)
    elif args.command == 'board':
        result = board(args.output, execute_board=args.execute_board)
    else:
        result = check(read(args.output / 'raw.json'))
        result = dict(passed=result['passed'], summary=result['summary'], records=result['records'])
    print(json.dumps(result, ensure_ascii=False, indent=2))
