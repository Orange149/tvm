#!/usr/bin/env python3
"""Existing-trace alignment; counter preflight is a separately gated command."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))

if __name__=='__main__':
    from placement_alignment.common import OUTPUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('analyze','prepare-counter','board-self-test'))
    parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--execute-board',action='store_true')
    args=parser.parse_args()
    if args.command=='analyze':
        from placement_alignment.analysis import analyze
        result=analyze(args.output)
    elif args.command=='prepare-counter':
        from placement_alignment.counter import prepare
        result=prepare(args.output)
    else:
        from placement_alignment.counter import board_self_test
        result=board_self_test(args.output,execute_board=args.execute_board)
    print(json.dumps(result,ensure_ascii=False))
