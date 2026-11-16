#!/usr/bin/env python3
"""Read-only extraction; frozen device collection code is unchanged."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))

if __name__=='__main__':
    from workload_model.common import OUTPUT,ROOT
    from device_phase.extract import extract
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=OUTPUT/'device_phase_02')
    p.add_argument('--run-dir',type=Path,default=ROOT/'runs/cross_network_device_phase_02')
    a=p.parse_args()
    print(json.dumps(extract(a.output,a.run_dir),ensure_ascii=False))
