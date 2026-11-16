"""No target calls: baseline identity, real coverage and bounded TPS screening."""
from copy import deepcopy
import unittest
import c3_yolo416_tps_comparison_scope as p


class ScopeTests(unittest.TestCase):
    def setUp(self):
        self.calls=p.read(p.BASE/'baseline_identity.json')['callsites']
        self.graph=p.read(p.BASE/'B_raw/graph.json')
        self.capacity={'INP':32768,'WGT':262144,'ACC':131072}

    def test_real_coverage_and_shared_callsite(self):
        groups=p.group_calls(self.calls,self.graph)
        self.assertEqual(len(groups),11)
        shared=next(g for g in groups if len(g)==2)
        self.assertEqual([r['earliest_graph_node'] for r in shared],[29,57])
        self.assertEqual(len(self.graph['heads']),8)

    def test_config_difference_rejected(self):
        calls=deepcopy(self.calls);calls[0]['effective_config_entity']['entity'][0][2][-1]=99
        with self.assertRaisesRegex(ValueError,'EFFECTIVE_CONFIG_CHANGED'):p.group_calls(calls,self.graph)

    def test_workload_difference_rejected(self):
        calls=deepcopy(self.calls);calls[0]['workload'][1][1][2]=224
        with self.assertRaisesRegex(ValueError,'EXACT_WORKLOAD_CHANGED'):p.group_calls(calls,self.graph)

    def test_missing_call_rejected(self):
        with self.assertRaisesRegex(ValueError,'EXPECTED_12'):p.group_calls(self.calls[:-1],self.graph)

    def test_pointwise_not_silently_sent_to_3x3_formula(self):
        row=next(r for r in self.calls if r['workload'][2][1][2:4]==[1,1])
        result=p.screen(row,self.capacity)
        self.assertFalse(result['existing_3x3_cost_applied'])
        self.assertIn('PENDING',result['status'])

    def test_real_3x3_screen_is_analytical_not_latency_selection(self):
        result=p.screen(self.calls[0],self.capacity)
        self.assertFalse(result['compiled_TPS']);self.assertFalse(result['selection_read_performance'])
        self.assertTrue(result['all_supported_threads_first']['necessary_conditions_passed'])
        self.assertGreater(result['necessary_domain'],0)


if __name__=='__main__':unittest.main()
