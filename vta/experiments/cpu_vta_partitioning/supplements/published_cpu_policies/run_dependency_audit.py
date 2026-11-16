#!/usr/bin/env python3
"""Offline-only dependency audit; no board commands or executable launches."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))


def main(argv=None):
    from dependency_audit.common import OUTPUT, REPORT, offline_guard
    from cpu_vta_partitioning.common import write
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('audit','extract','replay','analyze','validate','run'))
    p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--report-dir',type=Path,default=REPORT)
    a=p.parse_args(argv)
    with offline_guard() as guard:
        if a.command=='analyze':
            from dependency_audit.analysis import analyze
            result=analyze(a.output)
        else:
            from dependency_audit import workflow
            result=getattr(workflow,a.command)(a.output)
    write(a.output/'OFFLINE_GUARD.json',dict(command=a.command,**guard,
        board_access=False,new_native_sessions=0,new_DNN_inferences=0),replace=True)
    if a.command in ('run','analyze'):
        from dependency_audit.delivery import publish
        result['delivery']=publish(a.output,a.report_dir)
    print(json.dumps(result,ensure_ascii=False))
    return result


if __name__=='__main__':
    main()
