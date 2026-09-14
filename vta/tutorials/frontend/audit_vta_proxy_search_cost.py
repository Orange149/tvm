#!/usr/bin/env python3
"""Post-hoc R50E search-cost ablation; sums measured phases, not process wall."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified(directory):
    manifest = directory / 'artifact_hashes.json'
    entries = json.loads(manifest.read_text())['artifacts']
    for name, expected in entries.items():
        if digest(directory / name) != expected:
            raise ValueError('artifact mismatch: ' + str(directory / name))
    return digest(manifest)


def replay(order, rows, common_seconds, stock_seconds, prepaid=()):
    """Prepaid builds remain charged even when the first measured point wins."""
    if len(order) != len(set(order)) or len(prepaid) != len(set(prepaid)):
        raise ValueError('duplicate identity')
    spent = common_seconds + stock_seconds
    built = set(prepaid)
    spent += sum(rows[c]['build_seconds'] for c in built)
    best = math.inf
    trace = []
    dma = 0
    for i, cid in enumerate(order, 1):
        row = rows[cid]
        if cid not in built:
            spent += row['build_seconds']
            built.add(cid)
        spent += row['board_call_seconds']
        dma += row['logical_dma_bytes']
        if row['status'] == 'passed':
            best = min(best, row['ratio'])
        trace.append(dict(candidate_id=cid, dispatches=i, candidate_builds=len(built),
                          measured_phase_seconds=spent, logical_dma_bytes=dma,
                          best_ratio=best if math.isfinite(best) else None))
    return trace


def target(trace, oracle, epsilon):
    return next((r for r in trace if r['best_ratio'] is not None and
                 r['best_ratio'] <= oracle * (1 + epsilon)), None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    dirs = {n: next(args.root.glob(f'20260913_p7r{n}_*')) for n in (342,345,346,347,348)}
    manifests = {str(n): verified(p) for n,p in dirs.items()}
    analysis = json.loads((dirs[348]/'analysis.json').read_text())
    pool = json.loads((dirs[346]/'pool.json').read_text())
    board = [json.loads((dirs[n]/'summary.json').read_text()) for n in (345,347)]
    if board[0]['pool_artifact_manifest_sha256'] != manifests['342'] or board[1]['pool_artifact_manifest_sha256'] != manifests['346']:
        raise ValueError('board/pool binding mismatch')
    programs = {p['candidate_id']: p for p in pool['programs']}
    rows = {}
    for result in board:
        for row in result['candidate_results']:
            cid = row['candidate_id']
            if cid in rows:
                raise ValueError('duplicate result')
            calls = row['correctness'] + row['timings']
            rows[cid] = dict(status=row['status'], ratio=row.get('median_paired_latency_ratio'),
                build_seconds=programs[cid]['build_seconds'],
                board_call_seconds=sum(c['host_wall_ms'] for c in calls)/1000,
                logical_dma_bytes=sum(c['runtime_profile_complete']['load_buffer_2d_bytes']+
                    c['runtime_profile_complete']['store_buffer_2d_bytes'] for c in calls))
    if set(rows) != set(programs):
        raise ValueError('incomplete oracle')
    common = analysis['common_operator_qualification_cost']
    common_seconds = common['static_diagnostic_wall_seconds'] + common['fsim_diagnostic_wall_seconds']
    stock_seconds = pool['stock_reference']['build_seconds']
    features = analysis['candidate_table']
    front = analysis['operator_proxy_front_candidate_ids']
    ids = sorted(rows)
    oracle = min(r['ratio'] for r in rows.values() if r['status']=='passed')
    bytes_order = [r['candidate_id'] for r in sorted(features,key=lambda r:(r['operator_dma_bytes'],r['candidate_id']))]
    calls_order = [r['candidate_id'] for r in sorted(features,key=lambda r:(r['operator_dma_calls'],r['candidate_id']))]
    policies = {
        'frozen_front_prebuild': (front,front),
        'reversed_front_prebuild_order_ablation': (list(reversed(front)),front),
        'front_lazy_build_posthoc': (front,()),
        'bytes_lazy_build': (bytes_order,()),
        'calls_lazy_build': (calls_order,()),
        'full_prebuild_hash_order': (ids,ids),
    }
    traces = {name:replay(order,rows,common_seconds,stock_seconds,prepaid)
              for name,(order,prepaid) in policies.items()}
    random_traces = []
    for seed in range(1000):
        order=ids.copy(); random.Random(seed).shuffle(order)
        random_traces.append(replay(order,rows,common_seconds,stock_seconds))
    epsilons=(0.,0.02,0.05)
    targets={name:{str(e):target(trace,oracle,e) for e in epsilons}
             for name,trace in traces.items()}
    random_summary={}
    for epsilon in epsilons:
        hits=[target(trace,oracle,epsilon) for trace in random_traces]
        if any(hit is None for hit in hits):
            raise ValueError('full permutation missed oracle')
        random_summary[str(epsilon)]={
            k: {'min':min(h[k] for h in hits),'median':statistics.median(h[k] for h in hits),
                'max':max(h[k] for h in hits)}
            for k in ('dispatches','candidate_builds','measured_phase_seconds','logical_dma_bytes')}
    budgets=(100,150,200,300,500)
    success={str(b):{
        **{name: bool((hit:=target(trace,oracle,.02)) and hit['measured_phase_seconds']<=b)
           for name,trace in traces.items()},
        'random_lazy_build_fraction':sum(target(t,oracle,.02)['measured_phase_seconds']<=b for t in random_traces)/len(random_traces)
    } for b in budgets}
    output=dict(schema='c3_proxy_search_cost_posthoc_v1',workload='R50E',
        status='posthoc_equal_target_and_measured_phase_budget_audit',
        source_manifests=manifests,source_sha256=digest(Path(__file__)),
        common_qualification_seconds=common_seconds,stock_build_seconds=stock_seconds,
        oracle_ratio=oracle,policy_traces=traces,targets=targets,
        random_seeds=1000,random_targets=random_summary,success_at_oracle_plus_2pct_by_seconds=success,
        boundaries=[
            'Post-hoc replay of exposed R50E labels; not another prospective experiment.',
            '1000 shuffled orders are simulations, not independent workload or board samples.',
            'Every policy pays the same full operator qualification; no claim about avoiding its cost.',
            'Measured-phase sum omits uninstrumented graph/parameter setup, hashing, selection, RPC clean-start/upload/allocation and orchestration; not complete process wall.',
            'Pool oracle is used only by evaluation. Reaching oracle is not an implementable stopping certificate.',
            'All search candidates are slower than stock; deployments should retain the protected stock incumbent.',
            'Random/bytes/calls are controls, not reproductions of AutoTVM-XGB, Rieber or ML2Tuner.',
            'Build costs from separate collection waves are reused without a new interleaved construction experiment.'
        ])
    args.output.mkdir(parents=True)
    (args.output/'analysis.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    (args.output/'artifact_hashes.json').write_text(json.dumps({'artifacts':{'analysis.json':digest(args.output/'analysis.json')}},indent=2)+'\n')
    print(json.dumps(dict(targets=targets,random_targets=random_summary,success=success),indent=2))


if __name__=='__main__':
    main()
