#!/usr/bin/env python3
"""Project isolated stage/worker clocks into explicit dependency model inputs."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))

if __name__=='__main__':
    from workload_model.common import OUTPUT,ROOT
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('extract','predict','analyze'))
    p.add_argument('--output',type=Path,default=OUTPUT/'phase_profile')
    p.add_argument('--run-dir',type=Path,default=ROOT/'runs/cross_network_phase_profile_01')
    a=p.parse_args()
    if a.command=='extract':
        from phase_model.extract import extract
        result=extract(a.output,a.run_dir)
    elif a.command=='predict':
        from phase_model.predict import predict
        result=predict(a.output)
    else:
        from phase_model.analyze import analyze
        result=analyze(a.output,a.run_dir)
    print(json.dumps(result,ensure_ascii=False))
