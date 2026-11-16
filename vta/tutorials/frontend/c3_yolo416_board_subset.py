"""Freeze comparison strata before any performance is exposed.

Only selects graph/operator kind from a fully qualified frozen pool. It never
selects a layer by timing, changes a program, or imports measurements.
"""
import argparse
from pathlib import Path
import shutil
from c3_tps_sync_release_local import read,write,sha,verify,require
from c3_yolo416_comparison_protocol import manifest
from c3_yolo416_board_transport import transport_contract


def prepare(root, source, kind):
    require(not (source/'board').exists(), 'PARENT_PERFORMANCE_ALREADY_EXPOSED')
    frozen=read(source/'freeze.json');verify(frozen['bindings'])
    require(frozen['status']=='TH_P_OPERATORS_AND_GRAPH_READY_FPGA_PENDING', 'PARENT_POOL_NOT_READY')
    require(not (source/'prepare_failure.json').exists(), 'PARENT_PREPARATION_FAILED')
    original=read(source/'call_manifest.json')
    pool=[p for p in original['pool'] if p['kind']==kind]
    require(len(pool)==(1 if kind=='graph' else 11), 'STRATUM_COVERAGE_CHANGED')
    root.mkdir(exist_ok=False)
    shutil.copyfile(source/'artifacts.json',root/'artifacts.json')
    ledger=manifest(root.name,pool);write(root/'call_manifest.json',ledger)
    write(root/'transport_contract.json',transport_contract(ledger))
    write(root/'stratum_contract.json',{'source':str(source),'source_freeze_sha256':sha(source/'freeze.json'),
        'selection':'all members of frozen kind='+kind,'performance_exposed_before_selection':False,
        'programs_changed':False,'source_measurements_adopted':0,'kind':kind})
    bindings=dict(frozen['bindings']);bindings[str(source/'freeze.json')]=sha(source/'freeze.json')
    bindings[str(Path(__file__).resolve())]=sha(__file__)
    for p in root.iterdir():
        if p.is_file():bindings[str(p.resolve())]=sha(p)
    verify(bindings)
    write(root/'freeze.json',{**frozen,'bindings':bindings,'target_calls':ledger['total_target_calls'],
        'sessions':len(ledger['sessions']),'formal_samples':ledger['formal_samples'],
        'comparison_stratum':kind,'source_pool':str(source)})
    return ledger


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--kind',choices=('graph','operator'),required=True);a=p.parse_args()
    value=prepare(a.output.resolve(),a.source.resolve(),a.kind)
    print({'calls':value['total_target_calls'],'sessions':len(value['sessions']),'formal':value['formal_samples']})
