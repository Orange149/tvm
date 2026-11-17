#!/usr/bin/env python3
"""ResNet50 five-policy ablation; frozen costs, exact DP, one new board cohort."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / 'src'))


def main(argv=None):
    from resnet50_ablation.common import OUTPUT, RUN, LOCAL_CONFIG, MODELS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('audit','validate','rank','prepare','board','status','analyze'))
    parser.add_argument('action', nargs='?', choices=('run','resume'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--run-dir', type=Path, default=RUN)
    parser.add_argument('--local-config', type=Path, default=LOCAL_CONFIG)
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--verify-search', action='store_true')
    parser.add_argument('--execute-board', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'validate' and args.verify_search:
        from resnet50_ablation.ranking import acceptance
        value = acceptance(args.output)
        value = {k:v for k,v in value.items() if k != 'scoring_sources'}
    elif args.command == 'rank':
        from resnet50_ablation.ranking import rank_all, run_one
        value = run_one(args.output,args.model,args.resume) if args.model else rank_all(args.output,args.resume)
    elif args.command == 'board':
        if not args.execute_board or not args.action:
            parser.error('board run/resume requires --execute-board')
        from resnet50_ablation.campaign import run
        value = run(args.output,args.run_dir,args.local_config,execute_board=True)
    elif args.command == 'analyze':
        from resnet50_ablation.analysis import analyze
        value = analyze(args.output,args.run_dir)
    else:
        from resnet50_ablation import workflow
        value = getattr(workflow,args.command)(args.output,args.run_dir)
    print(json.dumps(value,ensure_ascii=False,indent=2),flush=True)
    if value.get('status') == 'SEARCH_BUDGET_EXCEEDED':
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
