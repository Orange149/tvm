#!/usr/bin/env python3
"""Real build/board adapter for budget-executor development on an exposed identity."""
import argparse
import json
from pathlib import Path
import time

from analyze_vta_fused_program_service_proxy_norefit import verify_artifacts_compatible
from build_vta_resnet50_fused_program_pool import build_stock, fused_features
from build_vta_resnet50_generic_residency_tir_audit import build_one
from build_vta_resnet50_relay_residency_dispatch_pair import make_relay_program
from run_vta_resnet50_fused_program_pareto_board import (
    CleanStartBoard, assert_board_state, run_one, semantic_params_hash, sha256, write_json,
)
import vta


def read(path):
    return json.loads(Path(path).read_text())


def finish(output):
    write_json(output/'artifact_hashes.json',{'artifacts':{
        str(p.relative_to(output)):sha256(p) for p in output.rglob('*')
        if p.is_file() and p.name!='artifact_hashes.json'}})


def build(args):
    target=Path(args.target_contract)
    local=Path(args.local_qualification)
    verify_artifacts_compatible(target)
    verify_artifacts_compatible(local)
    candidates=[json.loads(s) for s in (local/'candidates_v2.jsonl').read_text().splitlines() if s.strip()]
    qualified={json.loads(s)['candidate_id'] for s in (local/'fsim_results.jsonl').read_text().splitlines()
               if s.strip() and json.loads(s).get('status')=='passed'}
    row=next(r for r in candidates if r['candidate_id']==args.candidate_id)
    if args.candidate_id not in qualified:
        raise ValueError('candidate is not locally qualified')
    output=Path(args.bundle)
    output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    env=vta.get_env()
    relay_program,params=make_relay_program(env,pretrained=True)
    stock=build_stock(relay_program,params,output,env)
    graph=read(output/'stock_reference/graph.json')
    workload=read(target/'contract.json')['workload']
    stock_features=fused_features(stock,graph,workload)
    built=build_one(row,relay_program,params,output,env)
    if read(output/row['public_mode']/'graph.json')!=graph:
        raise ValueError('graph structure changed')
    if semantic_params_hash(output/row['public_mode']/'params.bin')!=semantic_params_hash(output/'stock_reference/params.bin'):
        raise ValueError('parameter semantics changed')
    program=dict(candidate_id=row['candidate_id'],family_id=row['family_id'],public_mode=row['public_mode'],
                 relative_dir=row['public_mode'],**fused_features(built,graph,workload))
    write_json(output/'bundle.json',dict(status='development_single_candidate_built',
        candidate_id=row['candidate_id'],program=program,stock_fused_features=stock_features,
        target_manifest_sha256=sha256(target/'artifact_hashes.json'),
        local_manifest_sha256=sha256(local/'artifact_hashes.json'),
        adapter_sha256=sha256(__file__),stock_build_seconds=stock['build_seconds'],
        candidate_build_seconds=built['build_seconds'],body_seconds=time.monotonic()-start,
        target_labels_previously_exposed=True,board_contacted=False))
    finish(output)


def measure(args):
    bundle=Path(args.bundle)
    verify_artifacts_compatible(bundle)
    spec=read(bundle/'bundle.json')
    if spec['candidate_id']!=args.candidate_id or spec['status']!='development_single_candidate_built':
        raise ValueError('bundle identity/status mismatch')
    output=Path(args.measure_output)
    output.mkdir(parents=True,exist_ok=False)
    board=CleanStartBoard(args.host,args.ssh_known_hosts)
    initial=board.state()
    assert_board_state(initial)
    result=run_one(board,args,output,bundle,spec['program'],spec['stock_fused_features'],0)
    after=board.state()
    assert_board_state(after)
    if initial['BOOT']!=after['BOOT']:
        raise RuntimeError('boot changed')
    write_json(output/'result.json',result)
    write_json(output/'receipt.json',dict(candidate_id=args.candidate_id,
        status='qualified' if result['status']=='passed' else 'rejected',
        correctness_records=len(result.get('correctness',[])),timing_records=len(result.get('timings',[])),
        result_sha256=sha256(output/'result.json'),bundle_manifest_sha256=sha256(bundle/'artifact_hashes.json'),
        boot_id=after['BOOT'],board_after=after,target_labels_previously_exposed=True,
        experiment_role='budget_executor_integration_not_search_holdout'))
    finish(output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['build','measure'])
    parser.add_argument('--candidate-id',required=True)
    parser.add_argument('--bundle',required=True)
    parser.add_argument('--target-contract')
    parser.add_argument('--local-qualification')
    parser.add_argument('--measure-output')
    parser.add_argument('--host',default='192.168.1.247')
    parser.add_argument('--port',type=int,default=9090)
    parser.add_argument('--session-timeout',type=int,default=600)
    parser.add_argument('--ssh-known-hosts')
    parser.add_argument('--default-runtime',default='/var/volatile/vta_c3_ram/runtime')
    args=parser.parse_args()
    if args.phase=='build':build(args)
    else:measure(args)


if __name__=='__main__':
    main()
