"""Fixed V4 call ledger regressions; no compiler or target invocation."""
import unittest
from c3_tps_window_protocol import manifest,analyse,BASIS


class CallManifestTest(unittest.TestCase):
    def test_full_pool_budget_and_seed_once_per_round(self):
        m=manifest(list(BASIS),'unit_fixture')
        self.assertEqual(len(m['calls']),480)
        self.assertEqual(sum(c['formal_sample'] for c in m['calls']),144)
        self.assertTrue(all(s['calls']==6 for s in m['sessions']))
        for p in BASIS:
            formal=[c for c in m['calls'] if c['program']==p and c['formal_sample']]
            self.assertEqual(len(formal),18)
            self.assertEqual(len({(c['round'],c['seed']) for c in formal}),18)

    def test_incomplete_attempt_has_no_performance_result(self):
        m=manifest(['TPS_B','TPS_N','TPS_G'],'unit_fixture')
        with self.assertRaisesRegex(ValueError,'INCOMPLETE'):
            analyse(m,[])

    def test_fixed_bng_statistics_and_nonformal_exclusion(self):
        m=manifest(['TPS_B','TPS_N','TPS_G'],'unit_fixture')
        rows=[dict(c,status='PASS',target_calls=1,latency_ms=
                   {'TPS_B':10,'TPS_N':12,'TPS_G':9}[c['program']] if c['formal_sample'] else 10000)
              for c in m['calls']]
        result=analyse(m,rows)
        self.assertEqual(result['programs']['TPS_G']['T_ms'],9)
        self.assertAlmostEqual(result['comparisons']['TPS_G;TPS_B']['Gain_percent'],10)
        self.assertEqual(result['comparisons']['TPS_G;TPS_B']['wins'],6)


if __name__=="__main__":unittest.main()
