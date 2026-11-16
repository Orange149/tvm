from copy import deepcopy
from pathlib import Path
import unittest

from c3_five_tradeoff_commands import decode_submissions
from c3_tps_notification_probe import (propose, certify_rewrite, finite_batch_progress,
    relations, material, token_counts)

ROOT = Path(__file__).parent / 'report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout/20261014_c3_tps_window_complete_run03'
COMMAND = {1: 512, 2: 512, 3: 512}
TOKEN = {k: 1024 for k in ((1, 2), (2, 1), (2, 3), (3, 2))}


def fixture(v):
    return decode_submissions((ROOT / f'TPS_{v}_detailed/native_call_02.txt').read_text())[0]


class ProbeTests(unittest.TestCase):
    def test_original_baseline_unchanged(self):
        before = fixture('B')
        after, edits = propose(before)
        self.assertEqual(before, after)
        self.assertEqual(edits, [])
        certify_rewrite(before, after, COMMAND, TOKEN)

    def test_six_nops_removed_but_all_dependencies_and_payloads_retained(self):
        for v, count in (('N', 83), ('G', 77)):
            before = fixture(v)
            after, edits = propose(before)
            self.assertEqual(len(edits), 6)
            self.assertEqual(len(after), count)
            self.assertEqual(sum(e['rule'].startswith('two_ready') for e in edits), 4)
            cert = certify_rewrite(before, after, COMMAND, TOKEN)
            self.assertEqual(cert['progress']['token_pairs'], 32)

    def test_collapsing_two_ready_tokens_rejected(self):
        before = fixture('G')
        after, _ = propose(before)
        item = next(r for r in after if r['description'] == 'GEMM' and r['bits'][0])
        item['bits'] = (0, *item['bits'][1:])
        with self.assertRaisesRegex(ValueError, 'UNEXPLAINED_ENCODING_CHANGE'):
            certify_rewrite(before, after, COMMAND, TOKEN)

    def test_payload_change_rejected(self):
        before = fixture('G')
        after, _ = propose(before)
        item = next(r for r in after if r['description'] == 'LOAD UOP')
        item['detail'] = item['detail'].replace('x: size=28', 'x: size=27')
        with self.assertRaisesRegex(ValueError, 'UNEXPLAINED_ENCODING_CHANGE'):
            certify_rewrite(before, after, COMMAND, TOKEN)

    def test_deferred_middle_release_deadlocks(self):
        rows = fixture('G')
        # Move the first release NOP past BOTH readiness waits to the next GEMM.
        # Moving it only to the first wait is not the same proposal and need
        # not deadlock; this test deliberately checks the full-wait cycle.
        source = next(r for r in rows if r['description'] == 'NOP-COMPUTE-STAGE'
                      and r['bits'] == (0, 0, 1, 0))
        target = next(r for r in rows[source['index']+1:] if r['description'] == 'GEMM')
        rows.remove(source)
        rows.insert(rows.index(target) + 1, source)
        for i, row in enumerate(rows): row['index'] = i
        with self.assertRaisesRegex(ValueError, 'circular wait'):
            finite_batch_progress(rows, COMMAND, TOKEN)

    def test_acyclic_graph_alone_does_not_prove_small_fifo_progress(self):
        with self.assertRaisesRegex(ValueError, 'FIFO_BOUND_UNPROVED'):
            finite_batch_progress(fixture('B'), {1: 1, 2: 1, 3: 1}, TOKEN)

    def test_token_capacity_not_assumed_infinite(self):
        with self.assertRaisesRegex(ValueError, 'TOKEN_WRITE_BOUND_UNPROVED'):
            finite_batch_progress(fixture('B'), COMMAND, {k: 1 for k in TOKEN})

    def test_unknown_pattern_left_untouched(self):
        rows = fixture('G')
        row = next(r for r in rows if r['description'] == 'NOP-COMPUTE-STAGE'
                   and r['bits'] == (1, 0, 0, 0))
        row['bits'] = (1, 1, 0, 0)
        _, edits = propose(rows)
        self.assertNotIn(row['index'], [e['removed_original_instruction'] for e in edits])

    def test_notification_migration_adds_four_entry_and_two_exit_constraints(self):
        b, n = fixture('B'), fixture('N')
        self.assertEqual(len(material(b)), len(material(n)))
        added = relations(n) - relations(b)
        self.assertFalse(relations(b) - relations(n))
        waits = [(i, j) for i, j in added if material(n)[i]['description'] == 'LOAD WGT']
        releases = [(i, j) for i, j in added if material(n)[i]['description'] == 'GEMM']
        self.assertEqual(len(waits), 4)
        self.assertEqual(len(releases), 10)  # two groups, each four INP + one WGT
        self.assertEqual(len({i for i, _ in releases}), 2)
        for i, j in added:
            self.assertNotIn((j, i), relations(b))  # previously unordered


if __name__ == '__main__': unittest.main()
