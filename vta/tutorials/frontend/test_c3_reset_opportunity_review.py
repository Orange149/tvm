import unittest
from c3_reset_opportunity_review import ancestors, moved_graph, analyze
from c3_yolo224_sync_structure import classify_events


def fixture(inp=10):
    def kernel(reset):
        return {'kind': 'kernel', 'mode_reset': [0, reset],
                'uops': [{'args': [0, reset, 0, 0, 0, 0, 0, 0], 'loops': []}]}
    def token(kind, lane): return {'kind': kind, 'pair': lane}
    def load(domain, addr):
        return {'kind': 'load', 'domain': domain, 'sram': addr, 'x': 1, 'y': 1,
                'pad': [0, 0, 0, 0], 'tensor': 0 if domain == 2 else 1}
    store = {'kind': 'store', 'sram': 0, 'x': 1, 'y': 1, 'tensor': 3}
    return [kernel(0), store.copy(), token('push', [3, 2]), store.copy(),
            token('push', [3, 2]), token('pop', [3, 2]), kernel(1), token('push', [2, 1]),
            token('pop', [3, 2]), kernel(1), token('push', [2, 1]),
            token('pop', [2, 1]), load(2, 0), load(1, 0),
            token('pop', [2, 1]), load(2, inp), load(1, 1)]


class Tests(unittest.TestCase):
    def test_retained_path_is_not_released(self):
        p = [[], [(0, 'fifo')], [(0, 'token'), (1, 'fifo')], [(2, 'fifo')]]
        q = moved_graph(p, [{'push': 0, 'pop': 2, 'following_weight': 3}])
        self.assertIn(0, ancestors(q, 2))
        self.assertIn((0, 'token'), p[2])

    def test_store_tail_is_released_but_mac_is_not(self):
        e = fixture()
        candidates = [r for r in classify_events(e)['waits'] if r['classification'] == 'ADDRESS_DISJOINT_PREFIX_NEEDS_FULL_CERTIFICATE']
        r = analyze(e, candidates)[0]
        self.assertEqual(r['all_reset_moves']['kinds'], {'STORE': 1, 'RESET': 1})
        self.assertEqual(r['all_reset_moves']['new_INP_physical_conflicts'], [])
        self.assertTrue(r['single_and_joint_material_loss_equal'])

    def test_new_physical_reset_collision_is_detected(self):
        e = fixture()
        candidates = [r for r in classify_events(e)['waits'] if r['classification'] == 'ADDRESS_DISJOINT_PREFIX_NEEDS_FULL_CERTIFICATE']
        e[9]['uops'][0]['args'][3] = 10
        r = analyze(e, candidates)[0]
        self.assertEqual(r['all_reset_moves']['new_INP_physical_conflicts'][0]['INP_address'], 10)


if __name__ == '__main__': unittest.main()
