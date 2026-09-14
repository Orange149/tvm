"""Retrospective validity-ranking composition, NOT an ML2Tuner reproduction."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics


def features(c):
    w = c['identity']['workload']; k = c['knobs']
    vals = [k[n] for n in ('tile_b','tile_h','tile_w','tile_ci','tile_co','oc_nthread','h_nthread')]
    vals += w[1][1] + w[2][1]
    return [math.log2(1 + v) for v in vals] + [int(c['implementation_mode'] == m) for m in (0,1,4)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-audits', type=Path, required=True)
    p.add_argument('--target-candidates', type=Path, required=True)
    p.add_argument('--target-baseline', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from vta_residency_capacity_bound import residency_bounds
    train = {}; hashes = {}
    def read(path):
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    for audit_path in sorted(a.training_audits.glob('*/audit.json')):
        audit = json.loads(audit_path.read_text())
        sources = audit['input_hashes']
        cp = next(Path(s) for s in sources if s.endswith('candidates_v2.jsonl'))
        sp = next(Path(s) for s in sources if s.endswith('static_results.jsonl'))
        for path in (cp,sp):
            if hashlib.sha256(path.read_bytes()).hexdigest() != sources[str(path)]:
                raise ValueError('training source changed')
        rows = {r['candidate_id']:r for r in read(sp)}
        for c in read(cp): train[c['candidate_id']] = (c, rows[c['candidate_id']]['status']=='ok')
    target = read(a.target_candidates)
    target_workloads = {json.dumps(c['identity']['workload'], sort_keys=True) for c in target}
    train = {cid: pair for cid,pair in train.items()
             if json.dumps(pair[0]['identity']['workload'],sort_keys=True) not in target_workloads}
    if set(train) & {c['candidate_id'] for c in target}: raise ValueError('identity overlap')
    model = RandomForestClassifier(n_estimators=128,max_depth=6,min_samples_leaf=2,
                                   class_weight='balanced',random_state=20260913,n_jobs=1)
    model.fit(np.array([features(c) for c,_ in train.values()]), [y for _,y in train.values()])
    probs = model.predict_proba(np.array([features(c) for c in target]))[:,list(model.classes_).index(True)]
    ids = [c['candidate_id'] for c in target]
    ranked = [cid for _,cid in sorted(zip(-probs,ids))]
    rejected = {c['candidate_id'] for c in target if residency_bounds(c)['decision']=='reject_capacity'}
    # Targets are exposed retrospective evaluation labels, never training features.
    labels = {r['candidate_id']:r['status']=='ok' for r in read(a.target_baseline)}
    if set(labels)!=set(ids): raise ValueError('target label identity mismatch')
    def evaluate(order, gate):
        valid = 0; compiles = 0; gross = 0; first4 = None
        for cid in order:
            gross += 1
            if gate and cid in rejected: continue
            compiles += 1; valid += int(labels[cid])
            if valid == 4 and first4 is None: first4 = {'compiler_calls':compiles,'gross_candidates':gross}
        return {'first4':first4, 'total_compiler_calls':compiles, 'valid_preserved':valid}
    random_results = {False:[], True:[]}
    for seed in range(100):
        order=sorted(ids);random.Random(seed).shuffle(order)
        for gate in random_results: random_results[gate].append(evaluate(order,gate))
    summary = {'scope':'retrospective same-target composition; RF validity surrogate, not ML2Tuner Model P/V/A reproduction; no latency search',
               'training_count':len(train), 'target_count':len(target),
               'target_workload_excluded_from_training':True,
               'validity_only':evaluate(ranked,False),'validity_plus_capacity':evaluate(ranked,True),
               'random':{str(gate):{'median_compiler_calls_first4':statistics.median(
                   r['first4']['compiler_calls'] for r in rows),
                   'median_gross_first4':statistics.median(r['first4']['gross_candidates'] for r in rows)}
                   for gate,rows in random_results.items()},
               'rejected_valid':sum(labels[cid] for cid in rejected),
               'input_hashes':hashes,'board_contacted':False,
               'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    a.output.mkdir(parents=True)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (a.output/'ranked_ids.json').write_text(json.dumps(ranked,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='input_hashes'},indent=2))


if __name__ == '__main__': main()
