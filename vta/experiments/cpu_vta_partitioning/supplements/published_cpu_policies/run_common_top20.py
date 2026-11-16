#!/usr/bin/env python3
"""Final common static lists; all pipeline measurements belong to a fresh batch."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))

if __name__=='__main__':
    from workload_model.common import OUTPUT,ROOT
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('rank','prepare','validate','qemu','short','long','analyze','status'))
    p.add_argument('--output',type=Path,default=OUTPUT/'common_top20')
    p.add_argument('--run-dir',type=Path,default=ROOT/'runs/cross_network_common_top20_01')
    p.add_argument('--execute-board',action='store_true')
    a=p.parse_args()
    if a.command=='rank':
        from common_top20.ranking import rank
        result=rank(a.output)
    elif a.command=='prepare':
        from common_top20.prepare import prepare
        result=prepare(a.output)
    elif a.command=='validate':
        from common_top20.prepare import validate
        result={n:validate(a.output/n/'capsule')['bundle_id'] for n in ('resnet18','yolov3_tiny416')}
    elif a.command=='qemu':
        from common_top20.campaign import qemu
        result=qemu(a.output)
    elif a.command in ('short','long'):
        from common_top20.campaign import run
        result=run(a.output,a.run_dir,phase=a.command,execute_board=a.execute_board)
    elif a.command=='analyze':
        from common_top20.analyze import analyze
        result=analyze(a.output,a.run_dir)
    else:
        from cpu_vta_partitioning.common import read
        result=read(a.run_dir/'state.json')
    print(json.dumps(result,ensure_ascii=False))
