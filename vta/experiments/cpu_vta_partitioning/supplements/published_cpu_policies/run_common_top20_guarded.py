#!/usr/bin/env python3
"""Formal timing gate: freeze candidate-only temporal rankings first."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))

if __name__=='__main__':
    from workload_model.common import OUTPUT,ROOT
    from cpu_vta_partitioning.common import read,sha256
    from common_top20.campaign import run
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute-board',action='store_true')
    a=p.parse_args()
    if not a.execute_board:raise ValueError('--execute-board is required')
    temporal=OUTPUT/'pool_temporal';seal=read(temporal/'RANK_FREEZE.json')
    if seal['common_rank_freeze_sha256']!=sha256(OUTPUT/'common_top20/RANK_FREEZE.json'):
        raise ValueError('temporal and common candidate pools differ')
    if seal['pipeline_FPS_read'] is not False:raise ValueError('temporal ranking used pipeline performance')
    for path,expected in seal['files'].items():
        if sha256(temporal/path)!=expected:raise ValueError('temporal rank freeze changed')
    print(json.dumps(run(OUTPUT/'common_top20',ROOT/'runs/cross_network_common_top20_01',
                         phase='short',execute_board=True),ensure_ascii=False))
