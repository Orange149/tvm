"""Synthetic, board-free regression checks for frozen search order generation."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name('freeze_vta_budget_search_orders.py')
POLICIES = ['random_lazy_build', 'operator_bytes_lazy_build',
            'operator_calls_lazy_build', 'operator_pareto_hash_lazy_build',
            'operator_pareto_hash_prebuild_ablation']
AXES = ['operator_LOAD_plus_STORE_bytes', 'operator_LOAD_plus_STORE_calls',
        'weight_barrier_indicator']


class SearchOrderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.q = self.root / 'qualification'
        self.p = self.root / 'protocol'
        self.q.mkdir()
        self.p.mkdir()
        self.summary = dict(board_contacted=False, performance_labels_used=False,
                            status='completed_local_no_board')
        self.write(self.q, 'summary.json', self.summary)
        self.protocol = dict(random_seed=17, policies=POLICIES,
                             sort_ties='candidate_id', pareto_axes=AXES)
        self.write(self.p, 'protocol.json', self.protocol)
        candidates = [dict(candidate_id=c, family_id=c, public_mode='original')
                      for c in 'abcd']
        static = [dict(candidate_id=c, status='ok', transfer_signature=dict(
            descriptor_aggregates=dict(expanded_bytes=b, expanded_calls=n)))
                  for c, b, n in [('a', 10, 4), ('b', 10, 2),
                                  ('c', 20, 1), ('d', 1, 1)]]
        self.lines('candidates_v2.jsonl', candidates)
        self.lines('static_results.jsonl', static)
        self.lines('fsim_results.jsonl', [dict(candidate_id=c, status=(
            'passed' if c != 'd' else 'failed')) for c in 'abcd'])
        self.seal(self.q)
        self.seal(self.p)

    def write(self, directory, name, value):
        (directory / name).write_text(json.dumps(value))

    def lines(self, name, rows):
        (self.q / name).write_text('\n'.join(json.dumps(r) for r in rows))

    def seal(self, directory):
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in directory.iterdir() if p.name != 'artifact_hashes.json'}
        self.write(directory, 'artifact_hashes.json', {'artifacts': hashes})

    def run_freezer(self, name='output'):
        return subprocess.run([sys.executable, str(SCRIPT), '--qualification',
                               str(self.q), '--protocol', str(self.p),
                               '--output', str(self.root / name)],
                              capture_output=True, text=True)

    def test_orders_ties_front_and_failed_exclusion(self):
        result = self.run_freezer()
        self.assertEqual(result.returncode, 0, result.stderr)
        orders = json.loads(result.stdout)['orders']
        self.assertEqual(orders['operator_bytes_lazy_build'], ['a', 'b', 'c'])
        self.assertEqual(orders['operator_calls_lazy_build'], ['c', 'b', 'a'])
        self.assertEqual(orders['operator_pareto_hash_lazy_build'], ['b', 'c'])
        self.assertEqual(orders['operator_pareto_hash_prebuild_ablation'], ['b', 'c'])
        self.assertNotIn('d', orders['random_lazy_build'])
        again = self.run_freezer('other')
        self.assertEqual(json.loads(again.stdout)['orders'], orders)

    def test_rejects_exposed_labels(self):
        for field in ['board_contacted', 'performance_labels_used']:
            with self.subTest(field=field):
                self.write(self.q, 'summary.json', {**self.summary, field: True})
                self.seal(self.q)
                result = self.run_freezer()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('not label-isolated', result.stderr)
                self.assertFalse((self.root / 'output').exists())

    def test_rejects_tampered_artifact(self):
        (self.q / 'static_results.jsonl').write_text('')
        result = self.run_freezer()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('artifact mismatch', result.stderr)

    def test_rejects_policy_drift(self):
        self.write(self.p, 'protocol.json', {**self.protocol, 'policies': ['unknown']})
        self.seal(self.p)
        result = self.run_freezer()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('policy drift', result.stderr)

    def test_rejects_unimplemented_protocol_semantics(self):
        for field, value in [('sort_ties', 'latency'), ('pareto_axes', ['bytes'])]:
            with self.subTest(field=field):
                self.write(self.p, 'protocol.json', {**self.protocol, field: value})
                self.seal(self.p)
                self.assertNotEqual(self.run_freezer().returncode, 0)

    def test_rejects_omitted_consumed_artifact(self):
        manifest = json.loads((self.q / 'artifact_hashes.json').read_text())
        del manifest['artifacts']['static_results.jsonl']
        self.write(self.q, 'artifact_hashes.json', manifest)
        self.assertIn('omits consumed artifact', self.run_freezer().stderr)

    def test_rejects_duplicate_identity(self):
        path = self.q / 'fsim_results.jsonl'
        path.write_text(path.read_text() + '\n' + json.dumps(
            dict(candidate_id='a', status='passed')))
        self.seal(self.q)
        self.assertIn('duplicate candidate identity', self.run_freezer().stderr)

    def test_rejects_identity_set_mismatch(self):
        self.lines('fsim_results.jsonl', [dict(candidate_id='a', status='passed')])
        self.seal(self.q)
        self.assertIn('identity sets differ', self.run_freezer().stderr)

    def test_preserves_existing_output(self):
        self.assertEqual(self.run_freezer().returncode, 0)
        before = (self.root / 'output' / 'orders.json').read_bytes()
        self.assertNotEqual(self.run_freezer().returncode, 0)
        self.assertEqual((self.root / 'output' / 'orders.json').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
