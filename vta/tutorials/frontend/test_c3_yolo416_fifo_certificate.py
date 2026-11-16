import random
import unittest
from c3_tps_window_progress import LANES, certify_progress
from c3_yolo416_fifo_certificate import certify_action_graph


def records(rows):
    return [{'index': i, 'stage': s, 'bits': b} for i, (s, b) in enumerate(rows)]


class ActionGraphTests(unittest.TestCase):
    def check(self, rows, capacity=1):
        rs = records(rows); commands = {s: capacity for s in (1, 2, 3)}
        tokens = {l: capacity for l in LANES}
        old = certify_progress(rs, commands, tokens, max_states=100000)
        new = certify_action_graph(rs, commands, tokens)
        self.assertNotEqual(old['status'], 'ANALYSIS_STATE_LIMIT')
        self.assertEqual(old['status'] == 'FINITE_FIFO_PROGRESS_PROVED',
                         new['status'] == 'FINITE_FIFO_PROGRESS_PROVED')
        return new

    def test_finite_fetch_capacity_changes_result(self):
        rows = [(2, (1, 0, 0, 0)), (2, (0, 0, 0, 0)),
                (2, (0, 0, 0, 0)), (1, (0, 0, 0, 1))]
        self.assertEqual(self.check(rows)['status'], 'CYCLIC_ACTION_GRAPH_UNRESOLVED')
        self.assertEqual(self.check(rows, 3)['status'], 'FINITE_FIFO_PROGRESS_PROVED')

    def test_token_capacity_and_ready_release_cycle(self):
        self.check([(1, (0, 0, 0, 1))]*3+[(2, (1, 0, 0, 0))]*3)
        self.check([(1, (0, 1, 0, 1)), (2, (1, 0, 1, 0))], 2)

    def test_two_pops_and_pushes_keep_serial_order(self):
        self.check([(1, (0, 0, 0, 1)), (3, (0, 0, 1, 0)),
                    (2, (1, 1, 1, 1)), (1, (0, 1, 0, 0)), (3, (1, 0, 0, 0))])

    def test_deterministic_small_graphs_match_exhaustive_enumerator(self):
        rng = random.Random(20261017)
        for case in range(120):
            stages = [1, 2, 3]+[rng.randint(1, 3) for _ in range(3)]
            bits = [[0]*4 for _ in stages]
            for src, dst in LANES:
                ps = [i for i, s in enumerate(stages) if s == src]
                cs = [i for i, s in enumerate(stages) if s == dst]
                count = rng.randint(0, min(len(ps), len(cs)))
                for i in rng.sample(ps, count): bits[i][2 if dst == src-1 else 3] = 1
                for i in rng.sample(cs, count): bits[i][0 if src == dst-1 else 1] = 1
            with self.subTest(case=case): self.check(list(zip(stages, bits)), 1+case % 2)

    def test_malformed_and_unbalanced_reject(self):
        commands = {s: 1 for s in (1, 2, 3)}; tokens = {l: 1 for l in LANES}
        for rs in ([{'index': 4, 'stage': 1, 'bits': (0, 0, 0, 0)}],
                   records([(1, (0, 0, 0, 1))]), records([(1, (1, 0, 0, 0))])):
            with self.assertRaises(ValueError): certify_action_graph(rs, commands, tokens)
        with self.assertRaises(ValueError): certify_action_graph([], {1: 0}, tokens)


if __name__ == '__main__': unittest.main()
