#!/usr/bin/env python3
"""Offline-only logical-thread placement hypotheses, no board commands."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))

if __name__=='__main__':
    from thread_history_audit.common import OUTPUT
    from model_semantics_audit.common import offline_guard
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','replay','validate','analyze','run'))
    p.add_argument('--output',type=Path,default=OUTPUT);a=p.parse_args()
    with offline_guard() as ledger:
        from thread_history_audit.workflow import prepare,replay,validate,analyze
        commands=dict(prepare=prepare,replay=replay,validate=validate,analyze=analyze)
        result={k:commands[k](a.output) for k in ('prepare','replay','analyze','validate')} if a.command=='run' else commands[a.command](a.output)
    from cpu_vta_partitioning.common import write
    write(a.output/('OFFLINE_GUARD_'+a.command+'.json'),ledger,replace=True)
    print(json.dumps(result,ensure_ascii=False))
