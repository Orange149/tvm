#!/usr/bin/env python3
"""Cross-network model study; audit is local and does not execute the board."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))


def main(argv=None):
    from workload_model.common import OUTPUT,REPORT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('audit','status'))
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--report-dir',type=Path,default=REPORT)
    args=parser.parse_args(argv)
    if args.command=='audit':
        from workload_model.audit import audit
        result=audit(args.output,args.report_dir)
    else:
        from cpu_vta_partitioning.common import read
        result=read(args.output/'PROGRESS.json')
    print(json.dumps(result,ensure_ascii=False))
    return result


if __name__=='__main__':main()
