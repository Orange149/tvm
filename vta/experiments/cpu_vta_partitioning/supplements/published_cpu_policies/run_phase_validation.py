#!/usr/bin/env python3
"""Six preregistered resource holdouts, with same-cost static controls."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))

if __name__=='__main__':
    from workload_model.common import OUTPUT,ROOT
    from cpu_vta_partitioning.common import read
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','validate','qemu','components','project','rank','pipelines','analyze','status'))
    p.add_argument('--output',type=Path,default=OUTPUT/'phase_validation')
    p.add_argument('--run-dir',type=Path,default=ROOT/'runs/cross_network_phase_validation_01')
    p.add_argument('--local-config',type=Path,default=OUTPUT/'board.local.json')
    p.add_argument('--execute-board',action='store_true')
    a=p.parse_args()
    if a.command=='prepare':
        from phase_validation.prepare import prepare
        result=prepare(a.output)
    elif a.command=='validate':
        from phase_validation.prepare import validate
        result={n:validate(a.output/n/'capsule') for n in ('resnet18','yolov3_tiny416')}
    elif a.command=='qemu':
        from phase_validation.campaign import qemu
        result=qemu(a.output)
    elif a.command in ('components','pipelines'):
        from phase_validation.campaign import board
        result=board(a.output,a.run_dir,a.local_config,scope=a.command,execute_board=a.execute_board)
    elif a.command=='project':
        from phase_validation.project import project
        result=project(a.output,a.run_dir)
    elif a.command=='rank':
        from phase_validation.predict import predict
        result=predict(a.output)
    elif a.command=='analyze':
        from phase_validation.analyze import analyze
        result=analyze(a.output,a.run_dir)
    else:result=read(a.run_dir/'state.json')
    print(json.dumps(result,ensure_ascii=False))
