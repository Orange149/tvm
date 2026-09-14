import unittest
from vta_residency_capacity_bound import weight_bound, residency_bounds


def candidate(mode, ci=1, co=4):
    return {'public_mode': mode,
            'knobs': dict(tile_ci=ci, tile_co=co, tile_b=1, oc_nthread=1, h_nthread=1),
            'identity': {'workload': ['conv2d_packed.vta', None,
                                      ['TENSOR', [64, 32, 3, 3, 16, 16], 'int8']],
                         'hardware_fingerprint': dict(block_in=16, block_out=16,
                                                       wgt_buff_size=262144)}}


class BoundTest(unittest.TestCase):
    def test_acc_lifetime_and_width_group(self):
        c = candidate('input_stationary')
        c['knobs'].update(tile_h=13, tile_w=13)
        c['identity']['workload'] = ['conv2d_packed.vta',
            ['TENSOR', [1, 32, 26, 26, 1, 16], 'int8'],
            ['TENSOR', [64, 32, 3, 3, 16, 16], 'int8'],
            [1, 1], [1, 1, 1, 1], [1, 1], 'NCHW1n16c', 'int32']
        c['identity']['hardware_fingerprint'].update(acc_buff_size=131072, inp_buff_size=32768)
        r = residency_bounds(c)
        self.assertEqual(r['required_bytes']['accumulator'], 64*16*13*13*4)
        self.assertIn('accumulator', r['exceeded_memories'])
        c['public_mode'] = 'weight_resident_barrier'
        r = residency_bounds(c)
        self.assertEqual(r['required_bytes']['accumulator'], 4*16*13*26*4)
        self.assertEqual(r['required_bytes']['input'], 16*15*28)
        c['identity']['workload'][3] = [2, 2]
        self.assertEqual(residency_bounds(c)['decision'], 'unknown')

    def test_lifetime_changes_capacity(self):
        self.assertEqual(weight_bound(candidate('original'))['weight_region_bytes'], 9216)
        self.assertEqual(weight_bound(candidate('input_stationary'))['weight_region_bytes'], 147456)
        r = weight_bound(candidate('weight_resident_barrier'))
        self.assertEqual(r['weight_region_bytes'], 294912)
        self.assertEqual(r['decision'], 'reject_capacity')

    def test_capacity_is_not_validity(self):
        self.assertEqual(weight_bound(candidate('weight_resident_barrier', 16, 1))['decision'],
                         'not_rejected')

    def test_unknown_does_not_prune(self):
        for c in [{}, candidate('hybrid'), candidate('original', ci=3)]:
            self.assertEqual(weight_bound(c)['decision'], 'unknown')
        c = candidate('input_stationary')
        c['knobs']['oc_nthread'] = 2
        self.assertEqual(weight_bound(c)['decision'], 'unknown')


if __name__ == '__main__':
    unittest.main()
