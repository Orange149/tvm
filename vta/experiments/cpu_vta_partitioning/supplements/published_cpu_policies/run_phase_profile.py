#!/usr/bin/env python3
"""Build the isolated optional same-clock profiling executor locally."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))


if __name__=='__main__':
    from workload_model.common import OUTPUT
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('build','prepare','qemu','run','resume','status'))
    p.add_argument('--output',type=Path,default=OUTPUT/'phase_profile')
    p.add_argument('--run-dir',type=Path,default=Path(__file__).resolve().parent/'runs/cross_network_phase_profile_01')
    p.add_argument('--local-config',type=Path,default=OUTPUT/'board.local.json')
    p.add_argument('--scope',choices=('components','pipelines'),default='components')
    p.add_argument('--execute-board',action='store_true')
    a=p.parse_args()
    if a.command=='build':
        from phase_profile.build import build
        result=build(a.output/'build')
    elif a.command=='prepare':
        from phase_profile.prepare import prepare
        result=prepare(a.output)
    elif a.command=='qemu':
        from phase_profile.qemu import check
        result=check(a.output)
    elif a.command in ('run','resume'):
        from phase_profile.campaign import run
        from cpu_vta_partitioning.common import read
        if not (a.output/'QEMU_QUALIFICATION.json').exists():raise ValueError('ARM output qualification is required before board access')
        proof=read(a.output/'QEMU_QUALIFICATION.json')
        if (not proof.get('passed') or proof.get('capsule_ids')!={network:read(a.output/network/'capsule/manifest.json')['bundle_id']
                for network in ('resnet18','yolov3_tiny416')}):
            raise ValueError('QEMU qualification does not bind the current capsules')
        result=run(a.output,a.run_dir,a.local_config,scope=a.scope,execute_board=a.execute_board)
    else:
        from cpu_vta_partitioning.common import read
        result=read(a.run_dir/'state.json')
    print(json.dumps(result,ensure_ascii=False))
