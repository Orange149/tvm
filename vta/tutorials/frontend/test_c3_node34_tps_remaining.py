"""Exact initializer-class regressions; no build or target execution."""
from copy import deepcopy
import unittest
from c3_node34_tps_remaining import classify_kernels


def kernel(scope,signature,src=0):
    return {'kind':'kernel','mode_reset':[0,0],'scope':scope,'signature':[signature],
            'uops':[{'args':[0,0,0,src,0,0,0,0],'loops':[[2,1,1,0]]}]}


class InitializerClassTests(unittest.TestCase):
    def test_nonadjacent_periodic_content_exactly_classified(self):
        rows,mapping=classify_kernels([kernel(2,i,i%2) for i in range(4)])
        self.assertEqual(rows[0]['signature_count'],4)
        self.assertEqual(rows[0]['complete_initializer_classes'],2)
        self.assertTrue(rows[0]['parity_equals_full_content_on_observed_domain'])
        self.assertEqual(mapping[2,(0,)],mapping[2,(2,)])
        self.assertNotEqual(mapping[2,(0,)],mapping[2,(1,)])

    def test_equal_addresses_with_different_loops_not_equivalent(self):
        events=[kernel(2,0),kernel(2,1),kernel(2,2)]
        events[-1]['uops'][0]['loops'][0][0]=3
        rows,mapping=classify_kernels(events)
        self.assertEqual(rows[0]['complete_initializer_classes'],2)
        self.assertFalse(rows[0]['parity_equals_full_content_on_observed_domain'])
        self.assertNotEqual(mapping[2,(0,)],mapping[2,(2,)])

    def test_separate_handles_and_alu_not_aliased(self):
        alu=kernel(4,0);alu['mode_reset']=[1,0]
        rows,mapping=classify_kernels([kernel(2,0),kernel(3,0),alu])
        self.assertEqual(len(rows),2)
        self.assertNotEqual(mapping[2,(0,)],mapping[3,(0,)])
        self.assertNotIn((4,(0,)),mapping)

    def test_same_signature_different_content_is_error(self):
        with self.assertRaisesRegex(ValueError,'SIGNATURE_CONTENT_MUTATION'):
            classify_kernels([kernel(2,0),kernel(2,0,1)])


if __name__=='__main__':unittest.main()
