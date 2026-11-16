"""Scope-aware admission audit; gate waits never count as executed service."""
import copy
import math
import statistics as st

from analyze_cpu_stage_overlap_validation import analyze_gate_rows, overlap

IDENTITY = ('boot_id', 'runner_sha256', 'runtime_bindings', 'program_identity',
            'numeric_input_sha256', 'threads', 'affinity', 'spin_count',
            'queue_depth', 'cpu_frequency_khz')


def analyze(rows, mode, scope, discard=7):
    if scope not in ('cpu', 'all'):
        raise ValueError('Unknown gate scope')
    normalized = copy.deepcopy(rows)
    gated = (0, 1, 2) if scope == 'all' else (0, 2)
    for raw, row in zip(rows, normalized):
        if mode != 'off' and raw.get('stage_gate_scope') != scope:
            raise ValueError('Scope identity drift')
        for i in range(3):
            points = [raw.get(f'stage{i}_cpu_gate_{f}_ms')
                      for f in ('request', 'acquire', 'release', 'wait')]
            if mode != 'off' and i in gated:
                if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in points):
                    raise ValueError('Missing admission timing')
                req, acq, rel, wait = points
                if not (req <= acq <= raw[f'stage{i}_start_ms'] <= raw[f'stage{i}_end_ms'] <= rel):
                    raise ValueError('Admission order failure')
                if abs(acq-req-wait) > 1e-5:
                    raise ValueError('Admission wait mismatch')
            elif any(x is not None for x in points):
                raise ValueError('Unexpected admission timing')
        # Reuse the frozen CPU-window auditor, while independently validating
        # VTA admission above and all-stage leases below.
        for field in ('request', 'acquire', 'release', 'wait'):
            row[f'stage1_cpu_gate_{field}_ms'] = None
    result = analyze_gate_rows(normalized, mode, discard)
    if mode == 'exclusive':
        leases = sorted((r[f'stage{i}_cpu_gate_acquire_ms'], r[f'stage{i}_cpu_gate_release_ms'])
                        for r in rows for i in gated)
        if any(a[1] > b[0] for a, b in zip(leases, leases[1:])):
            raise ValueError('Exclusive admission leases overlap')
    lo, hi = rows[discard]['completion_ms'], rows[-1]['completion_ms']
    intervals = [sorted((max(r[f'stage{i}_start_ms'], lo), min(r[f'stage{i}_end_ms'], hi))
                        for r in rows if r[f'stage{i}_end_ms'] > lo and r[f'stage{i}_start_ms'] < hi)
                 for i in range(3)]
    result['pairwise_stage_overlap_ms'] = {f'{i}:{j}': overlap(intervals[i], intervals[j])
                                           for i, j in ((0, 1), (0, 2), (1, 2))}
    result['scope'] = scope
    result['overlap_window'] = 'all submitted intervals clipped to stable completion horizon'
    for i in gated:
        result['stages'][i]['gate_wait_mean_ms'] = st.mean(r[f'stage{i}_cpu_gate_wait_ms'] for r in rows[discard:]) if mode != 'off' else None
    return result


def effect(cpu_only, all_stages):
    if any(cpu_only.get(k) is None or cpu_only[k] != all_stages.get(k) for k in IDENTITY):
        raise ValueError('Paired identity drift')
    b, c = cpu_only['gate_metrics'], all_stages['gate_metrics']
    if (b['gate_mode'], b['scope'], c['gate_mode'], c['scope']) != ('exclusive', 'cpu', 'exclusive', 'all'):
        raise ValueError('Wrong paired interventions')
    br, cr = [s['run_mean_ms'] for s in b['stages']], [s['run_mean_ms'] for s in c['stages']]
    trace = b['pairwise_stage_overlap_ms']['0:1'] + b['pairwise_stage_overlap_ms']['1:2'] > 0 and all(v == 0 for v in c['pairwise_stage_overlap_ms'].values())
    return {'front_run_reduction_fraction': 1-cr[0]/br[0],
            'cpu_run_sum_reduction_fraction': 1-(cr[0]+cr[2])/(br[0]+br[2]),
            'run_deltas_ms': [x-y for x, y in zip(br, cr)],
            'ii_delta_ms_cpu_minus_all': b['ii_ms']-c['ii_ms'], 'trace_passed': trace}


def repeated(pairs):
    groups, identity = {}, None
    for pair in pairs:
        effect(pair['cpu_only'], pair['all_stages'])
        current = {k:pair['cpu_only'][k] for k in IDENTITY if k != 'boot_id'}
        if identity is None:
            identity = current
        if current != identity or pair['boot_id'] != pair['cpu_only']['boot_id']:
            raise ValueError('Cross-boot identity drift')
        block = groups.setdefault(pair['boot_id'], {})
        if pair['block'] in block:
            raise ValueError('Duplicate boot/block')
        block[pair['block']] = pair
    if len(groups) != 3 or any(set(g) != {0, 1} for g in groups.values()):
        raise ValueError('Exactly three boots and two balanced pairs required')
    rows = []
    for boot, blocks in groups.items():
        e = [effect(p['cpu_only'], p['all_stages']) for p in blocks.values()]
        rows.append({'boot_id':boot, 'front_delta_ms':st.mean(x['run_deltas_ms'][0] for x in e),
                     'front_reduction_fraction':st.mean(x['front_run_reduction_fraction'] for x in e),
                     'trace_passed':all(x['trace_passed'] for x in e)})
    mean = st.mean(r['front_delta_ms'] for r in rows)
    half = 4.3026527299*st.stdev(r['front_delta_ms'] for r in rows)/math.sqrt(3)
    return {'boot_rows':rows, 'front_ci95_ms':[mean-half, mean+half], 'mean_front_reduction_ms':mean,
            'df':2, 'primary_gate_passed':all(r['front_delta_ms'] > 0 and r['trace_passed'] for r in rows)
            and st.median(r['front_reduction_fraction'] for r in rows) >= .05 and mean-half > 0}
