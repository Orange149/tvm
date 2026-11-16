"""Validate comparison scope and legacy-slot interpretation without targets."""
from collections import Counter
import unittest
from c3_tps_broader_baseline_compare import pool_entry, SLOTS
from c3_yolo416_comparison_protocol import manifest, validate


class ComparisonTests(unittest.TestCase):
    def pool(self):
        return [pool_entry(n, {v: {'actual_variant': v, 'execution_identity': {'node': n, 'v': v}}
                               for v in ('A', 'C')}) for n in (34, 46)]

    def test_exact_240_calls_and_actual_variant_identities(self):
        ledger = manifest('preflight', self.pool())
        self.assertEqual(ledger['total_target_calls'], 240)
        self.assertEqual(len(ledger['sessions']), 40)
        self.assertEqual(ledger['formal_samples'], 72)
        for row in ledger['calls']:
            self.assertEqual(row['program_identity']['actual_variant'], SLOTS[row['variant']])
        self.assertEqual(Counter((r['object_id'], r['variant'], r['round'], r['seed'])
                                 for r in ledger['calls'] if r['timed']).most_common(1)[0][1], 1)

    def test_wrong_identity_and_extra_variant_rejected(self):
        with self.assertRaisesRegex(ValueError, 'VARIANT_IDENTITY_MISMATCH'):
            pool_entry(34, {'A': {'actual_variant': 'C'}, 'C': {'actual_variant': 'A'}})
        with self.assertRaisesRegex(ValueError, 'EXACT_A_C_PAIR_REQUIRED'):
            pool_entry(34, {'A': {}, 'B': {}, 'C': {}})

    def test_partial_or_replaced_call_refused(self):
        ledger = manifest('preflight', self.pool())
        rows = [{**r, 'status': 'PASS', 'target_calls': 1, 'elapsed_ms': 1.0} for r in ledger['calls']]
        self.assertTrue(validate(ledger, rows))
        with self.assertRaisesRegex(ValueError, 'INCOMPLETE_ATTEMPT'):
            validate(ledger, rows[:-1])
        rows[0] = {**rows[1]}
        with self.assertRaisesRegex(ValueError, 'CALL_IDENTITY_OR_ORDER_CHANGED'):
            validate(ledger, rows)


if __name__ == '__main__':
    unittest.main()
