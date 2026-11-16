"""Zero-target gates for the two fixed stronger TPS baselines."""
import unittest
from c3_tps_broader_baseline_local import selection, route, sources, EXPECTED


class BroaderBaselineTests(unittest.TestCase):
    def test_exact_archived_all_thread_selections(self):
        for node in (34, 46):
            chosen = selection(node)
            self.assertEqual(chosen['selected']['id'], EXPECTED[node])
            self.assertFalse(chosen['performance_read'])
            self.assertEqual(chosen['selected']['tile']['oc_nthread'], 1)
            self.assertEqual(chosen['selected']['tile']['h_nthread'], 1)

    def test_full_config_not_only_index_or_threads(self):
        for node, co in ((34, 32), (46, 8)):
            chosen = selection(node)
            r = route(node, chosen)
            self.assertEqual(r['identity']['workload'], sources(node).WORKLOAD)
            entity = {name: values for name, kind, values in r['identity']['complete_config_entity']['entity']}
            self.assertEqual(entity['tile_co'], [-1, co])
            self.assertEqual(entity['tile_ci'], [-1, 1])
            self.assertEqual(len(entity), 7)

    def test_real_distinct_fusion_extraction(self):
        for node, shift in ((34, 'right_shift(%57, 6'), (46, 'right_shift(%24, 8')):
            source = sources(node)
            text = source.extract_relay((source.GRAPH/'relay_000.txt').read_text())
            self.assertIn(shift, text)
            self.assertIn('a_min=-127f', text)
            self.assertIn('a_max=127f', text)

    def test_unregistered_node_rejected(self):
        with self.assertRaisesRegex(ValueError, 'FIXED_TWO_NODE_SCOPE'):
            sources(24)


if __name__ == '__main__':
    unittest.main()
