#!/usr/bin/env python3
"""Offline fixed-group branch evaluation. No board command is provided."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))


def main(argv=None):
    from policy_branch_eval.common import OUTPUT,REPORT,offline_guard,require_ranked
    from cpu_vta_partitioning.common import write
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('rank','analyze','validate','run'))
    p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--report-dir',type=Path,default=REPORT)
    a=p.parse_args(argv)
    with offline_guard(a.output) as ledger:
        result={}
        if a.command in ('rank','run'):
            from policy_branch_eval.ranking import rank
            result['ranking']=rank(a.output)
        if a.command in ('analyze','run'):
            from policy_branch_eval.analysis import analyze
            result['analysis']=analyze(a.output,a.report_dir)
        if a.command=='validate':
            require_ranked(a.output)
            from policy_branch_eval.ranking import rank
            result['ranking_replayed']=rank(a.output)
    write(a.output/'OFFLINE_GUARD.json',dict(command=a.command,**ledger,new_board_sessions=0,
        new_DNN_inferences=0),replace=True)
    print(json.dumps(result,ensure_ascii=False))
    return result


if __name__=='__main__':
    main()
