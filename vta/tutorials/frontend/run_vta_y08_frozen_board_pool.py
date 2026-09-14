#!/usr/bin/env python3
"""Collect the preregistered Y08 FPGA-correct full pool without online selection."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

PROCESS_START = time.monotonic()

from tvm import rpc
import vta

import vta.top.vta_conv2d_residency  # noqa: F401
from prepare_vta_p7r120_yolo_rpc_contract import build_and_export
from run_vta_p7r120_yolo_rpc_board import load_ephemeral, runtime_inventory
from run_vta_p7r132_y00_search_confirmation import (
    CleanStartBoard, EXPECTED_DEFAULT_RUNTIME_HASHES, allocate_reusable_buffers,
    profiled_reused_call, timed_reused_call, timing_orders, summarize_timing,
)

SEEDS = (0, 20250901, 20260910)
ROUNDS = 7


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def lines(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def write(path, value): Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def verify(directory, required):
    ledger = read(Path(directory) / 'artifact_hashes.json')['artifacts']
    if not set(required).issubset(ledger): raise ValueError('manifest omits consumed input')
    for name, expected in ledger.items():
        path = (Path(directory) / name).resolve()
        if not path.is_relative_to(Path(directory).resolve()) or sha(path) != expected:
            raise ValueError('artifact mismatch: ' + name)
    return sha(Path(directory) / 'artifact_hashes.json')


def finalize(output):
    write(output/'artifact_hashes.json', {'artifacts': {p.name: sha(p) for p in output.iterdir()
          if p.is_file() and p.name != 'artifact_hashes.json'}, 'source_sha256': sha(__file__)})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qualification',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--orders',type=Path,required=True)
    p.add_argument('--health-contract',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--host',default='192.168.1.247');p.add_argument('--port',type=int,default=9090)
    p.add_argument('--known-hosts',type=Path,required=True)
    p.add_argument('--runtime',default='/var/volatile/vta_c3_ram/runtime')
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    qhash=verify(a.qualification,['summary.json','candidates_v2.jsonl','static_results.jsonl','fsim_results.jsonl'])
    phash=verify(a.protocol,['protocol.json']); ohash=verify(a.orders,['orders.json'])
    protocol=read(a.protocol/'protocol.json'); frozen=read(a.orders/'orders.json')
    if frozen.get('board_contacted') or frozen.get('performance_labels_used'):
        raise ValueError('search order was not frozen before board/performance observation')
    summary=read(a.qualification/'summary.json')
    if summary['board_contacted'] or summary['performance_labels_used'] or summary['fsim_passed']!=6:
        raise ValueError('qualification is not the frozen six-candidate label-free pool')
    cs={c['candidate_id']:c for c in lines(a.qualification/'candidates_v2.jsonl')}
    static={r['candidate_id']:r for r in lines(a.qualification/'static_results.jsonl')}
    fsim={r['candidate_id']:r for r in lines(a.qualification/'fsim_results.jsonl')}
    order=frozen['orders']['random_lazy_build']
    if len(order)!=6 or set(order)!=set(p['candidate_id'] for p in frozen['programs']):
        raise ValueError('frozen six-candidate order mismatch')
    for cid in order:
        if static[cid]['status']!='ok' or fsim[cid]['status']!='passed' or len(fsim[cid]['seeds'])!=3:
            raise ValueError('unqualified board identity')
    candidates=[cs[cid] for cid in order]
    health=read(a.health_contract/'board_collection_contract.json')['health_canary']
    a.output.mkdir(parents=True)
    contract={'schema':'c3_y08_frozen_board_pool_v1','status':'frozen_before_current_board_observation',
              'bindings':{'qualification_ledger':qhash,'protocol_ledger':phash,'orders_ledger':ohash,
                          'candidate_order_source':'P7R357 random_lazy_build'},
              'candidate_order':order,'candidate_ids':order,'health_candidate_id':health['candidate_id'],
              'seeds':list(SEEDS),'rounds':ROUNDS,'failure_policy':'health fail stops; candidate first error rejects; timed error stops',
              'buffer_policy':'health set then one output-data-weight Y08 set reused for all shape-compatible candidates',
              'search_policy_execution':'forbidden; full pool labels only; P7R355 policies evaluated later',
              'persistent_board_write':False,'protocol':protocol}
    write(a.output/'contract.json',contract);write(a.output/'pre_observation_hashes.json',{'contract.json':sha(a.output/'contract.json')})
    correctness=[]; timing=[]; health_rows=[]
    try:
        env=vta.get_env()
        with tempfile.TemporaryDirectory(prefix='c3_y08_board_') as td:
            td=Path(td); start=time.monotonic()
            cert={'health':build_and_export(health,env,td)}
            cert['health']['cross_compile_wall_ms']=(time.monotonic()-start)*1000
            for c in candidates:
                start=time.monotonic()
                cert[c['candidate_id']]=build_and_export(c,env,td,static[c['candidate_id']]['tir_sha256'])
                cert[c['candidate_id']]['cross_compile_wall_ms']=(time.monotonic()-start)*1000
            write(a.output/'cross_compile.json',cert)
            board=CleanStartBoard(a.host,a.known_hosts); before=board.state()
            if before.get('FPGA')!='operating' or before.get('UDMABUF')!='201326592' or before.get('STORAGE_ERRORS'):
                raise RuntimeError('board preflight failed: '+repr(before))
            if board.runtime_hashes(a.runtime)!=EXPECTED_DEFAULT_RUNTIME_HASHES: raise RuntimeError('runtime hash mismatch')
            old=board.rpc_state()
            if old.get('CWD')!=a.runtime: raise RuntimeError('unexpected RPC')
            board.stop_rpc(old); reload_info=board.reload_frozen_bitstream()
            fresh=board.start_rpc(a.runtime,'y08_frozen_pool.log')
            write(a.output/'clean_start.json',{'before':before,'old_rpc':old,'reload':reload_info,'fresh_rpc':fresh})
            remote=rpc.connect(a.host,a.port,session_timeout=300); inventory,functions=runtime_inventory(remote)
            write(a.output/'rpc_inventory.json',inventory); device=remote.ext_dev(0)
            hm=load_ephemeral(remote,td/(health['candidate_id']+'.so'))
            mods={cid:load_ephemeral(remote,td/(cid+'.so')) for cid in order}
            hb=allocate_reusable_buffers(device,health['identity']['workload'])
            tb=allocate_reusable_buffers(device,candidates[0]['identity']['workload'])
            for seed in SEEDS:
                row=profiled_reused_call(hm['main'],hb,health['identity']['workload'],seed,functions)
                health_rows.append(row)
                if not row.get('correct'): break
            write(a.output/'health.json',{'status':'passed' if len(health_rows)==3 and all(r['correct'] for r in health_rows) else 'failed','rows':health_rows})
            if len(health_rows)!=3 or not all(r['correct'] for r in health_rows): raise RuntimeError('health gate failed')
            with (a.output/'correctness.jsonl').open('x') as f:
                for pos,c in enumerate(candidates):
                    rows=[]
                    for seed in SEEDS:
                        start=time.perf_counter(); row=profiled_reused_call(mods[c['candidate_id']]['main'],tb,c['identity']['workload'],seed,functions)
                        row['host_wall_ms']=(time.perf_counter()-start)*1000;rows.append(row)
                        if not row.get('correct'): break
                    item={'candidate_id':c['candidate_id'],'family_id':c['family_id'],'public_mode':c['public_mode'],
                          'position':pos,'status':'passed' if len(rows)==3 and all(r['correct'] for r in rows) else 'failed','seeds':rows}
                    correctness.append(item);f.write(json.dumps(item,sort_keys=True)+'\n');f.flush()
                    print('correctness {}/6 {}'.format(pos+1,item['status']),flush=True)
            correct=[r['candidate_id'] for r in correctness if r['status']=='passed']
            if not correct: raise RuntimeError('no FPGA-correct candidates')
            round_orders=timing_orders(correct,ROUNDS);write(a.output/'timing_orders.json',round_orders)
            with (a.output/'timing.jsonl').open('x') as f, (a.output/'health_brackets.jsonl').open('x') as hf:
                for ri,ro in enumerate(round_orders):
                    for bracket in ('before',):
                        x=timed_reused_call(hm,device,hb,health['identity']['workload'],0,functions);x.update(round=ri,bracket=bracket);hf.write(json.dumps(x)+'\n');hf.flush()
                    for pos,cid in enumerate(ro):
                        c=cs[cid];x=timed_reused_call(mods[cid],device,tb,c['identity']['workload'],0,functions)
                        x.update(candidate_id=cid,round=ri,position=pos);timing.append(x);f.write(json.dumps(x)+'\n');f.flush()
                    x=timed_reused_call(hm,device,hb,health['identity']['workload'],0,functions);x.update(round=ri,bracket='after');hf.write(json.dumps(x)+'\n');hf.flush()
                    print('timing {}/{}'.format(ri+1,ROUNDS),flush=True)
            stats=summarize_timing(timing,correct);write(a.output/'timing_summary.json',stats)
            after=board.state();write(a.output/'post_state.json',after)
            write(a.output/'summary.json',{'status':'completed_full_pool','boot_id':before['BOOT'],'candidate_count':6,
                  'fpga_correct':len(correct),'correctness_invocations':sum(len(r['seeds']) for r in correctness),
                  'timing_samples':len(timing),'oracle_candidate_id':stats['pool_oracle_candidate_id'],
                  'oracle_latency_ms':stats['pool_oracle_latency_ms'],'all_timed_correct':all(r['correct'] for r in timing),
                  'process_wall_ms':(time.monotonic()-PROCESS_START)*1000,
                  'board_contacted':True,'performance_labels_collected':True})
    except Exception as exc:
        write(a.output/'failure.json',{'type':type(exc).__name__,'message':str(exc),'health_rows':len(health_rows),
              'correctness_rows':len(correctness),'timing_rows':len(timing)})
        finalize(a.output);raise
    finalize(a.output)


if __name__=='__main__': main()
