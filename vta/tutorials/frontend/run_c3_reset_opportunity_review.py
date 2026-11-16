"""Serial bounded read-only review of the six archived G2 functions."""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from collections import Counter
from c3_reset_opportunity_review import CENSUS
from c3_yolo224_sync_structure import HOLD, HERE, read, write, require, sha


def main():
    out = HOLD / '20261016_c3_reset_opportunity_review_readonly01'
    out.mkdir(exist_ok=False)
    manifest = read(CENSUS / 'completion_manifest.json')
    for name, digest in manifest['artifact_sha256'].items():
        require(sha(CENSUS / name) == digest, 'CENSUS_CHANGED:' + name)
    sources = [HERE / n for n in ['c3_reset_opportunity_review.py', 'test_c3_reset_opportunity_review.py',
                'run_c3_reset_opportunity_review.py', 'c3_yolo224_sync_structure.py', 'c3_valid_window.py',
                'c3_official_window_method.py']]
    write(out / 'freeze.json', {'sources': {str(p): sha(p) for p in sources},
          'census_manifest': sha(CENSUS / 'completion_manifest.json'), 'target_calls': 0,
          'board_operations': 0, 'worker_seconds': 600, 'worker_tree_bytes': 8 * 1024**3,
          'max_expanded_vector_steps': 5000000, 'scope': 'G2 RESET candidates; separate and joint TIR edge hypotheses'})
    with (out / 'tests.log').open('x') as log:
        p = subprocess.run([sys.executable, '-m', 'unittest', '-v', 'test_c3_reset_opportunity_review'],
                           stdout=log, stderr=subprocess.STDOUT, timeout=60)
    require(p.returncode == 0, 'TEST_FAILURE')
    from c3_official_window_workflow import _tree_rss
    rows = [r for r in read(CENSUS / 'summary.json')['G2']['entries'] if
            r['RESET_categories'].get('ADDRESS_DISJOINT_PREFIX_NEEDS_FULL_CERTIFICATE', 0)]
    unique = {r['object_id']: r for r in rows}
    write(out / 'inventory.json', {'graph_calls': rows, 'functions': list(unique)})
    (out / 'objects').mkdir()
    for key, row in unique.items():
        base = out / 'objects' / key
        write(base.with_suffix('.input.json'), row)
        start = time.monotonic(); peak = 0; limit = None
        with base.with_suffix('.log').open('x') as log:
            proc = subprocess.Popen([sys.executable, str(HERE / 'c3_reset_opportunity_review.py'),
                 '--row', str(base.with_suffix('.input.json')), '--output', str(base.with_suffix('.json'))],
                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while proc.poll() is None:
                peak = max(peak, _tree_rss(proc.pid))
                if peak > 8 * 1024**3 or time.monotonic() - start > 600:
                    limit = 'RESOURCE_LIMIT'; os.killpg(proc.pid, signal.SIGKILL); break
                time.sleep(.2)
            code = proc.wait()
        write(base.with_suffix('.worker.json'), {'code': code, 'limit': limit, 'peak_bytes': peak, 'seconds': time.monotonic()-start})
        require(code == 0, 'WORKER_FAILED:' + key)
        d = read(base.with_suffix('.json'))
        print(key, d['candidate_count'], d['structural_classes'], d['phases'], flush=True)
    classes, phases = Counter(), Counter()
    summary = []
    for row in rows:
        d = read(out / 'objects' / (row['object_id'] + '.json'))
        classes.update(d['structural_classes']); phases.update(d['phases'])
        summary.append({'node': row['node'], 'object': row['object_id'], **{k: d[k] for k in
                       ['candidate_count', 'structural_classes', 'phases', 'released_MAC_count', 'direct_physical_conflict_count']}})
    write(out / 'summary.json', {'graph_weighted_candidates': sum(classes.values()),
          'classes': dict(classes), 'phases': dict(phases), 'nodes': summary,
          'target_calls': 0, 'performance_conclusion': False})
    for p, digest in read(out / 'freeze.json')['sources'].items(): require(sha(p) == digest, 'SOURCE_CHANGED')
    write(out / 'completion_manifest.json', {'status': 'STATIC_RESET_REVIEW_COMPLETE',
          'artifacts': {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}})
    print(out, flush=True)


if __name__ == '__main__': main()
