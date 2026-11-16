#!/usr/bin/env python3
"""Observe isolated VTA host/device phase boundaries without changing waiting."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))

if __name__=='__main__':
    from workload_model.common import OUTPUT,ROOT
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('build','prepare','run','resume','status'))
    p.add_argument('--output',type=Path,default=OUTPUT/'device_phase_02')
    p.add_argument('--run-dir',type=Path,default=ROOT/'runs/cross_network_device_phase_02')
    p.add_argument('--execute-board',action='store_true')
    a=p.parse_args()
    if a.command=='build':
        from device_phase.build import build
        result=build(a.output/'build')
    elif a.command=='prepare':
        from device_phase.collect import prepare
        result=prepare(a.output)
    elif a.command in ('run','resume'):
        from device_phase.collect import run
        result=run(a.output,a.run_dir,execute_board=a.execute_board)
    else:
        from cpu_vta_partitioning.common import read
        result=read(a.run_dir/'state.json')
    print(json.dumps(result,ensure_ascii=False))
