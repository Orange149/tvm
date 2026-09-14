"""Measured, development-pool lowering comparison; no FPGA or performance labels."""
import time
ENTRY = time.monotonic()
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--worker', choices=['baseline', 'tile_bound', 'prescreen'])
    p.add_argument('--with-tile-bound', action='store_true')
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    if a.worker:
        from vta_residency_capacity_bound import residency_bounds
        from run_vta_p7r119_yolo_barrier_pilot import instantiate
        import tvm
        import vta
        candidates = [json.loads(s) for s in a.candidates.read_text().splitlines() if s.strip()]
        rows = []
        with (a.output / 'rows.jsonl').open('x') as stream:
            for c in candidates:
                start = time.monotonic()
                decision = residency_bounds(c) if a.worker == 'prescreen' else None
                if a.worker == 'tile_bound':
                    # Ablation: same units/capacities, ordinary tile lifetime only.
                    # The real candidate passed to instantiate remains unchanged.
                    decision = residency_bounds({**c, 'public_mode': 'original'})
                row = {'candidate_id': c['candidate_id'], 'decision': decision}
                if decision and decision['decision'] == 'reject_capacity':
                    row['status'] = 'prescreen_reject'
                else:
                    try:
                        s, tensors = instantiate(c, vta.get_env())
                        with vta.build_config(disabled_pass={'tir.CommonSubexprElimTIR'}):
                            module = tvm.lower(s, tensors, name='main')
                        row.update(status='ok', tir_sha256=hashlib.sha256(
                            tvm.ir.save_json(module).encode()).hexdigest())
                    except Exception as exc:
                        row.update(status='failed', error=str(exc))
                row['seconds'] = time.monotonic() - start
                rows.append(row)
                stream.write(json.dumps(row) + '\n'); stream.flush()
        summary = {'policy': a.worker, 'entry_to_completion_seconds': time.monotonic()-ENTRY,
                   'count': len(rows), 'lower_attempts': sum(r['status']!='prescreen_reject' for r in rows),
                   'ok': sum(r['status']=='ok' for r in rows)}
    else:
        # Commit protocol before observing timings; independent interpreter per arm.
        protocol = {'order': ['baseline', 'prescreen', 'prescreen', 'baseline'],
                    'candidate_sha256': hashlib.sha256(a.candidates.read_bytes()).hexdigest(),
                    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    'bound_source_sha256': hashlib.sha256(Path(__file__).with_name(
                        'vta_residency_capacity_bound.py').read_bytes()).hexdigest(),
                    'scope': 'known development pool, measured lowering cost; excludes FSim, DMA extraction, FPGA and search',
                    'failure_policy': 'no replacement; compare all successful TIR identities'}
        if a.with_tile_bound:
            protocol['order'] = ['baseline', 'tile_bound', 'prescreen',
                                 'prescreen', 'tile_bound', 'baseline']
            protocol['tile_bound_scope'] = 'ordinary tile footprint ablation; not stock AutoTVM or Rieber reproduction'
        (a.output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
        runs = []
        for i, policy in enumerate(protocol['order']):
            dest = a.output / ('run%d_%s' % (i, policy))
            start = time.monotonic()
            with (a.output / ('run%d.log' % i)).open('x') as log:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), '--candidates',
                                str(a.candidates.resolve()), '--output', str(dest.resolve()),
                                '--worker', policy], stdout=log, stderr=subprocess.STDOUT, check=True)
            runs.append({'policy': policy, 'process_wall_seconds': time.monotonic()-start,
                         **json.loads((dest/'summary.json').read_text())})
            print(json.dumps(runs[-1]), flush=True)
        successes = []
        for i, policy in enumerate(protocol['order']):
            rows = [json.loads(s) for s in (a.output/('run%d_%s' % (i,policy))/'rows.jsonl').read_text().splitlines()]
            successes.append({r['candidate_id']:r['tir_sha256'] for r in rows if r['status']=='ok'})
        if not all(s==successes[0] for s in successes):
            raise RuntimeError('successful TIR identities differ across arms')
        summary = {'scope': protocol['scope'], 'runs': runs, 'all_successful_tir_sets_equal': True,
                   'board_contacted': False}
    (a.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()
