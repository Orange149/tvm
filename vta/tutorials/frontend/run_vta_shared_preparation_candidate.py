#!/usr/bin/env python3
"""Prepare a ResNet50 model/stock once, then build one candidate per process.

Development adapter: existing qualification and target labels are exposed.
Shared host compiler artifacts are not device tensor/slot reuse.
"""
import argparse
from pathlib import Path
import json
import shutil
import time

import tvm
from tvm import relay
import vta
from run_vta_single_candidate_process import (
    read, finish, measure, verify_artifacts_compatible, make_relay_program,
    build_stock, build_one, fused_features, semantic_params_hash, sha256, write_json,
)


def prepare(args):
    target=Path(args.target_contract)
    verify_artifacts_compatible(target)
    output=Path(args.shared)
    output.mkdir(parents=True,exist_ok=False)
    env=vta.get_env()
    started=time.monotonic()
    program,params=make_relay_program(env,pretrained=True)
    model_seconds=time.monotonic()-started
    (output/'relay.json').write_text(tvm.ir.save_json(program))
    (output/'params.bin').write_bytes(relay.save_param_dict(params))
    restored=tvm.ir.load_json((output/'relay.json').read_text())
    if not tvm.ir.structural_equal(program,restored,map_free_vars=True):
        raise ValueError('Relay serialization changed structure')
    restored_params=relay.load_param_dict((output/'params.bin').read_bytes())
    if set(params)!=set(restored_params):
        raise ValueError('parameter keys changed')
    for key in params:
        left,right=params[key].numpy(),restored_params[key].numpy()
        if left.dtype!=right.dtype or left.shape!=right.shape or left.tobytes()!=right.tobytes():
            raise ValueError('parameter serialization changed '+key)
    stock=build_stock(program,params,output,env)
    graph=read(output/'stock_reference/graph.json')
    workload=read(target/'contract.json')['workload']
    write_json(output/'shared.json',dict(
        status='development_shared_model_and_stock_ready',
        target_manifest_sha256=sha256(target/'artifact_hashes.json'),
        workload=workload,stock_fused_features=fused_features(stock,graph,workload),
        model_prepare_seconds=model_seconds,stock_build_seconds=stock['build_seconds'],
        body_seconds=time.monotonic()-started,adapter_sha256=sha256(__file__),
        serialization_verified=True,target_labels_previously_exposed=True,
        board_contacted=False))
    finish(output)


def build(args):
    shared=Path(args.shared)
    local=Path(args.local_qualification)
    target=Path(args.target_contract)
    for directory in (shared,local,target):
        verify_artifacts_compatible(directory)
    spec=read(shared/'shared.json')
    if spec['status']!='development_shared_model_and_stock_ready' or spec['target_manifest_sha256']!=sha256(target/'artifact_hashes.json'):
        raise ValueError('shared preparation target mismatch')
    candidates=[json.loads(line) for line in (local/'candidates_v2.jsonl').read_text().splitlines() if line.strip()]
    row=next(r for r in candidates if r['candidate_id']==args.candidate_id)
    qualified=[json.loads(line) for line in (local/'fsim_results.jsonl').read_text().splitlines() if line.strip()]
    if not any(r['candidate_id']==args.candidate_id and r.get('status')=='passed' for r in qualified):
        raise ValueError('candidate not qualified')
    output=Path(args.bundle)
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    # Copying the immutable stock is paid here; no stock compilation occurs.
    shutil.copytree(shared/'stock_reference',output/'stock_reference')
    program=tvm.ir.load_json((shared/'relay.json').read_text())
    params=relay.load_param_dict((shared/'params.bin').read_bytes())
    restored_seconds=time.monotonic()-started
    built=build_one(row,program,params,output,vta.get_env())
    graph=read(output/'stock_reference/graph.json')
    if read(output/row['public_mode']/'graph.json')!=graph:
        raise ValueError('graph structure changed')
    if semantic_params_hash(output/row['public_mode']/'params.bin')!=semantic_params_hash(output/'stock_reference/params.bin'):
        raise ValueError('candidate/stock parameter semantics changed')
    features=fused_features(built,graph,spec['workload'])
    write_json(output/'bundle.json',dict(
        status='development_single_candidate_built',candidate_id=args.candidate_id,
        program=dict(candidate_id=args.candidate_id,family_id=row['family_id'],public_mode=row['public_mode'],
                     relative_dir=row['public_mode'],**features),
        stock_fused_features=spec['stock_fused_features'],
        shared_manifest_sha256=sha256(shared/'artifact_hashes.json'),
        target_manifest_sha256=spec['target_manifest_sha256'],local_manifest_sha256=sha256(local/'artifact_hashes.json'),
        shared_restore_seconds=restored_seconds,candidate_build_seconds=built['build_seconds'],
        stock_rebuilt=False,model_reprepared=False,body_seconds=time.monotonic()-started,
        adapter_sha256=sha256(__file__),board_contacted=False,target_labels_previously_exposed=True))
    finish(output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','build','measure'])
    parser.add_argument('--shared')
    parser.add_argument('--target-contract')
    parser.add_argument('--local-qualification')
    parser.add_argument('--candidate-id')
    parser.add_argument('--bundle')
    parser.add_argument('--measure-output')
    parser.add_argument('--host',default='192.168.1.247')
    parser.add_argument('--port',type=int,default=9090)
    parser.add_argument('--session-timeout',type=int,default=600)
    parser.add_argument('--ssh-known-hosts')
    parser.add_argument('--default-runtime',default='/var/volatile/vta_c3_ram/runtime')
    args=parser.parse_args()
    {'prepare':prepare,'build':build,'measure':measure}[args.phase](args)


if __name__=='__main__':
    main()
