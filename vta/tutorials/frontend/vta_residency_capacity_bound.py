"""Development-only residency-aware weight bound; never a correctness certificate."""
import argparse
import hashlib
import json
from pathlib import Path


def weight_bound(candidate):
    """Return unknown outside the audited packed int8, single-thread schedule scope."""
    unknown = {'decision': 'unknown', 'reason': 'outside audited formula scope'}
    try:
        identity = candidate['identity']
        w = identity['workload']
        hw = identity['hardware_fingerprint']
        k = candidate['knobs']
        mode = candidate['public_mode']
        if mode not in ('original', 'input_stationary', 'weight_resident_barrier'):
            return unknown
        if any(k[n] != 1 for n in ('oc_nthread', 'h_nthread', 'tile_b')):
            return unknown
        if w[0] != 'conv2d_packed.vta' or w[2][2] != 'int8':
            return unknown
        co, ci, kh, kw, bo, bi = w[2][1]
        if (bo, bi) != (hw['block_out'], hw['block_in']):
            return unknown
        values = [co, ci, kh, kw, bo, bi, k['tile_ci'], k['tile_co'],
                  hw['wgt_buff_size']]
        if any(type(x) is not int or x <= 0 for x in values):
            return unknown
        if ci % k['tile_ci'] or co % k['tile_co']:
            return unknown
        ci_region = ci if mode == 'weight_resident_barrier' else k['tile_ci']
        co_region = co if mode == 'input_stationary' else k['tile_co']
        required = ci_region * co_region * kh * kw * bo * bi
        return {'decision': 'reject_capacity' if required > hw['wgt_buff_size']
                else 'not_rejected', 'weight_region_bytes': required,
                'weight_capacity_bytes': hw['wgt_buff_size'],
                'scope': 'schedule-specific necessary weight bound only; not allocation proof'}
    except (KeyError, TypeError, ValueError, IndexError):
        return unknown


def residency_bounds(candidate):
    """Extend the development bound to ACC/input for unit-stride packed int8."""
    result = weight_bound(candidate)
    if result['decision'] == 'unknown':
        return result
    try:
        identity = candidate['identity']; w = identity['workload']
        hw = identity['hardware_fingerprint']; k = candidate['knobs']
        if w[3] != [1, 1] or w[5] != [1, 1] or w[1][2] != 'int8' or w[-1] != 'int32':
            return {'decision': 'unknown', 'reason': 'spatial formula scope'}
        n, _, ih, iw, batch, bi = w[1][1]
        co, _, kh, kw, bo, _ = w[2][1]
        pt, pl, pb, pr = w[4]
        oh, ow = ih + pt + pb - kh + 1, iw + pl + pr - kw + 1
        th, tw = k['tile_h'], k['tile_w']
        if n != 1 or batch != 1 or th <= 0 or tw <= 0 or oh % th or ow % tw:
            return {'decision': 'unknown', 'reason': 'partial/batched spatial tile'}
        mode = candidate['public_mode']
        width_group = 2 if mode == 'weight_resident_barrier' and (ow // tw) % 2 == 0 else 1
        region_w = tw * width_group
        region_co = co if mode == 'input_stationary' else k['tile_co']
        required = {'weight': result['weight_region_bytes'],
                    'accumulator': region_co * bo * th * region_w * 4,
                    'input': k['tile_ci'] * bi * (th + kh - 1) * (region_w + kw - 1)}
        capacities = {'weight': hw['wgt_buff_size'], 'accumulator': hw['acc_buff_size'],
                      'input': hw['inp_buff_size']}
        if any(type(v) is not int or v <= 0 for v in capacities.values()):
            return {'decision': 'unknown', 'reason': 'invalid capacities'}
        exceeded = [m for m in required if required[m] > capacities[m]]
        return {'decision': 'reject_capacity' if exceeded else 'not_rejected',
                'required_bytes': required, 'capacity_bytes': capacities,
                'exceeded_memories': exceeded,
                'scope': 'development schedule-specific bounds, not correctness certificate'}
    except (KeyError, TypeError, ValueError, IndexError):
        return {'decision': 'unknown', 'reason': 'incomplete spatial metadata'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualification', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--all-memories', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = [args.qualification / n for n in
             ('candidates_v2.jsonl', 'static_results.jsonl')]
    candidates, static = [[json.loads(s) for s in p.read_text().splitlines() if s.strip()]
                          for p in paths]
    indexed = {r['candidate_id']: r for r in static}
    if len(indexed) != len(static) or len({c['candidate_id'] for c in candidates}) != len(candidates):
        raise ValueError('duplicate identities')
    if set(indexed) != {c['candidate_id'] for c in candidates}:
        raise ValueError('identity sets differ')
    rows = []
    for c in candidates:
        r = indexed[c['candidate_id']]
        rows.append({'candidate_id': c['candidate_id'], 'family_id': c['family_id'],
                     'mode': c['public_mode'], **(residency_bounds(c) if args.all_memories else weight_bound(c)),
                     'observed_static_status': r['status'],
                     'observed_failure': (r.get('failure') or {}).get('subcategory')})
    rejected = [r for r in rows if r['decision'] == 'reject_capacity']
    summary = {'scope': 'retrospective development audit, not prospective search savings',
               'rule': 'all_memories' if args.all_memories else 'weight_only',
               'count': len(rows), 'rejected': len(rejected),
               'rejected_static_ok': sum(r['observed_static_status'] == 'ok' for r in rejected),
               'rejected_observed_capacity_failure': sum(
                   r['observed_failure'] == 'allocation_capacity' for r in rejected),
               'unknown': sum(r['decision'] == 'unknown' for r in rows),
               'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'input_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
               'rows': rows, 'board_contacted': False}
    args.output.mkdir(parents=True)
    (args.output / 'audit.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
