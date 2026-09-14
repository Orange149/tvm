#!/usr/bin/env python3
"""Freeze protocol-defined orders using only local candidate qualification."""
import argparse
import hashlib
import json
from pathlib import Path
import random


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(directory, required):
    artifacts=json.loads((directory/'artifact_hashes.json').read_text())['artifacts']
    if not set(required).issubset(artifacts):
        raise ValueError('manifest omits consumed artifact')
    for name,h in artifacts.items():
        if not (directory/name).resolve().is_relative_to(directory.resolve()):
            raise ValueError('manifest path escapes directory')
        if digest(directory/name)!=h:raise ValueError('artifact mismatch: '+name)


def index_unique(rows):
    result={}
    for row in rows:
        cid=row['candidate_id']
        if cid in result:raise ValueError('duplicate candidate identity: '+cid)
        result[cid]=row
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qualification',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    verify(a.qualification, ['summary.json','static_results.jsonl',
                            'candidates_v2.jsonl','fsim_results.jsonl'])
    verify(a.protocol, ['protocol.json'])
    protocol=json.loads((a.protocol/'protocol.json').read_text())
    if protocol.get('sort_ties')!='candidate_id':
        raise ValueError('unsupported tie rule')
    if protocol.get('pareto_axes')!=['operator_LOAD_plus_STORE_bytes',
                                   'operator_LOAD_plus_STORE_calls',
                                   'weight_barrier_indicator']:
        raise ValueError('unsupported Pareto axes')
    summary=json.loads((a.qualification/'summary.json').read_text())
    if summary['board_contacted'] or summary['performance_labels_used'] or summary['status']!='completed_local_no_board':
        raise ValueError('local qualification is not label-isolated')
    def rows(name):return [json.loads(s) for s in (a.qualification/name).read_text().splitlines() if s.strip()]
    static=index_unique(rows('static_results.jsonl'))
    candidates=index_unique(rows('candidates_v2.jsonl'))
    fsim=index_unique(rows('fsim_results.jsonl'))
    if set(static)!=set(candidates) or set(fsim)!=set(candidates):
        raise ValueError('qualification identity sets differ')
    if any(c.get('performance_label') is not None or
           c.get('board_status', 'not_dispatched')!='not_dispatched'
           for c in candidates.values()):
        raise ValueError('candidate contains board exposure')
    passed=[cid for cid,r in fsim.items() if r['status']=='passed']
    programs=[]
    for cid in passed:
        r=static[cid];c=candidates[cid]
        if r.get('status')!='ok':raise ValueError('FSim pass lacks static success')
        d=r['transfer_signature']['descriptor_aggregates']
        if any(type(d[k]) is not int or d[k]<0
               for k in ('expanded_bytes','expanded_calls')):
            raise ValueError('DMA metrics must be nonnegative integers')
        programs.append(dict(candidate_id=cid,family_id=c['family_id'],mode=c['public_mode'],
                             bytes=int(d['expanded_bytes']),calls=int(d['expanded_calls']),
                             barrier=int(c['public_mode']=='weight_resident_barrier')))
    def dominates(x,y):return all(x[k]<=y[k] for k in ('bytes','calls','barrier')) and any(x[k]<y[k] for k in ('bytes','calls','barrier'))
    front=sorted(r['candidate_id'] for r in programs if not any(dominates(x,r) for x in programs))
    shuffled=sorted(passed);random.Random(protocol['random_seed']).shuffle(shuffled)
    orders={'random_lazy_build':shuffled,
            'operator_bytes_lazy_build':[r['candidate_id'] for r in sorted(programs,key=lambda r:(r['bytes'],r['candidate_id']))],
            'operator_calls_lazy_build':[r['candidate_id'] for r in sorted(programs,key=lambda r:(r['calls'],r['candidate_id']))],
            'operator_pareto_hash_lazy_build':front,
            'operator_pareto_hash_prebuild_ablation':front}
    if set(orders)!=set(protocol['policies']):raise ValueError('policy drift')
    result=dict(status='orders_frozen_before_candidate_fpga_or_latency',programs=programs,orders=orders,
                qualification_manifest_sha256=digest(a.qualification/'artifact_hashes.json'),
                protocol_manifest_sha256=digest(a.protocol/'artifact_hashes.json'),
                source_sha256=digest(Path(__file__)),candidate_count=len(programs),front_count=len(front),
                board_contacted=False,performance_labels_used=False,
                scope='local order freeze; no FPGA correctness, speed or oracle retention claim')
    a.output.mkdir(parents=True)
    dest=a.output/'orders.json';dest.write_text(json.dumps(result,indent=2)+'\n')
    (a.output/'artifact_hashes.json').write_text(json.dumps({'artifacts':{'orders.json':digest(dest)}},indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
