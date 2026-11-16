#!/usr/bin/env python3
"""Candidate-only isolated phase collection and pre-performance temporal Top20."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))

if __name__=='__main__':
    from workload_model.common import OUTPUT,ROOT
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','validate','qemu','run','resume','project','rank','status'))
    p.add_argument('--output',type=Path,default=OUTPUT/'pool_temporal')
    p.add_argument('--run-dir',type=Path,default=ROOT/'runs/cross_network_pool_intrinsics_01')
    p.add_argument('--execute-board',action='store_true');a=p.parse_args()
    if a.command=='prepare':
        from pool_temporal.prepare import prepare
        result=prepare(a.output)
    elif a.command=='validate':
        from pool_temporal.prepare import validate
        result={n:validate(a.output/n/'capsule')['bundle_id'] for n in ('resnet18','yolov3_tiny416')}
    elif a.command=='qemu':
        from pool_temporal.campaign import qemu
        result=qemu(a.output)
    elif a.command in ('run','resume'):
        from pool_temporal.campaign import run
        result=run(a.output,a.run_dir,execute_board=a.execute_board)
    elif a.command=='project':
        from pool_temporal.project import project
        result=project(a.output,a.run_dir)
    elif a.command=='rank':
        from pool_temporal.ranking import rank
        result=rank(a.output)
    else:
        from cpu_vta_partitioning.common import read
        result=read(a.run_dir/'state.json')
    print(json.dumps(result,ensure_ascii=False))
