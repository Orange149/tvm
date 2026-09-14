"""Freeze additional same-geometry candidates, excluding indexed historical v2 pools."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--catalog-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    from run_vta_p7r119_yolo_barrier_pilot import create_task, candidate_identity, config_knobs, MODES
    import vta
    base = json.loads(a.reference.read_text().splitlines()[0])
    seen = set(); sources = {}
    for path in sorted(a.catalog_root.rglob('candidates_v2.jsonl')):
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for line in path.read_text().splitlines():
            if line.strip():
                seen.add(json.loads(line)['candidate_id'])
    template = dict(base, implementation_mode=0)
    task = create_task(template, vta.get_env())
    families = []
    for index in range(len(task.config_space)):
        entity = task.config_space.get(index).to_json_dict()
        knobs = config_knobs(entity)
        if any(knobs[k] != 1 for k in ('tile_b', 'h_nthread', 'oc_nthread')):
            continue
        b = {'identity': base['identity'], 'complete_config_entity': entity,
             'debug': {'config_index': index}}
        rows = []
        for mode, number in MODES.items():
            c = candidate_identity(b, mode, number)
            rows.append({**c, 'workload_id': base['workload_id'],
                         'family_id': 'capacity_check_%d' % index, 'knobs': knobs,
                         'public_mode': mode, 'implementation_mode': number,
                         'performance_label': None, 'board_status': 'not_dispatched'})
        if any(r['candidate_id'] in seen for r in rows):
            continue
        rank = hashlib.sha256(('capacity-validation-v1:%d' % index).encode()).hexdigest()
        families.append((rank, rows))
    selected = [c for _, rows in sorted(families)[:16] for c in rows]
    if len(selected) != 48:
        raise ValueError('insufficient unreserved candidate families')
    a.output.mkdir(parents=True)
    dest = a.output/'candidates_v2.jsonl'
    dest.write_text(''.join(json.dumps(c, sort_keys=True)+'\n' for c in selected))
    rule = Path(__file__).with_name('vta_residency_capacity_bound.py')
    protocol = {'scope': 'additional same-geometry local validation; not new workload or FPGA holdout; exclusion limited to indexed v2 pools',
                'rule_sha256': hashlib.sha256(rule.read_bytes()).hexdigest(),
                'candidate_sha256': hashlib.sha256(dest.read_bytes()).hexdigest(),
                'catalog_sources': sources, 'families_available': len(families),
                'selection': 'first 16 families by fixed hash of ConfigEntity index; no bound or lowering outcome used',
                'baseline': 'lower all 48, including every analytically rejected point',
                'no_replacement': True, 'board_contacted': False}
    (a.output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    print(json.dumps({k:v for k,v in protocol.items() if k!='catalog_sources'}, indent=2))


if __name__ == '__main__':
    main()
