#!/usr/bin/env python3
"""Offline-only execution-semantics audit; no board commands or FPS fitting."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))

def main():
    from model_semantics_audit.common import OUTPUT,offline_guard
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('audit','project','replay','validate','analyze','run'))
    p.add_argument('--output',type=Path,default=OUTPUT);a=p.parse_args()
    with offline_guard() as ledger:
        from model_semantics_audit.audit import audit
        from model_semantics_audit.project import project
        from model_semantics_audit.replay import replay,validate
        from model_semantics_audit.analysis import analyze
        funcs=dict(audit=audit,project=project,replay=replay,validate=validate,analyze=analyze)
        if a.command=='run':result={name:funcs[name](a.output) for name in ('audit','project','replay','analyze','validate')}
        else:result=funcs[a.command](a.output)
    from cpu_vta_partitioning.common import write
    write(a.output/('OFFLINE_GUARD_'+a.command+'.json'),ledger,replace=True)
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
