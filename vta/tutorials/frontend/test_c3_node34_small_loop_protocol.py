"""Pure protocol checks for node34 stop gates; no board access."""

import unittest

from run_vta_c3_node34_board_operator import ORDERS, VERSIONS, evaluate as evaluate_operator
from run_vta_c3_node34_board_graph import evaluate as evaluate_graph


class Node34ProtocolTest(unittest.TestCase):
    def make_operator_rows(self, baseline, bridge):
        latency = {"G2": baseline, "O1": baseline + 1,
                   "I1": baseline - 1, "W1": baseline + 2, "BRIDGE": bridge}
        return [{"phase": "timing", "round": r, "variant": name,
                 "latency_ms": latency[name]}
                for r in range(6) for name in VERSIONS for _ in range(3)]

    def test_interleaving_has_each_version_once_per_round(self):
        self.assertEqual(len(ORDERS), 6)
        for order in ORDERS:
            self.assertEqual(set(order), set(VERSIONS))
            self.assertEqual(len(order), len(VERSIONS))

    def test_operator_requires_beating_input_and_original(self):
        result = evaluate_operator(self.make_operator_rows(10.0, 9.5))
        self.assertGreater(result["decision"]["mode4_median_improvement_pct"], 2)
        self.assertFalse(result["decision"]["beats_all_original_input_controls"])
        self.assertFalse(result["decision"]["graph_eligible"])

    def test_operator_positive_gate(self):
        result = evaluate_operator(self.make_operator_rows(10.0, 8.5))
        self.assertTrue(result["decision"]["graph_eligible"])

    def test_missing_operator_sample_fails(self):
        rows = self.make_operator_rows(10.0, 8.5)
        with self.assertRaises(ValueError):
            evaluate_operator(rows[:-1])

    def make_graph_rows(self, g2, w1, bridge):
        latency = {"G2": g2, "W1": w1, "BRIDGE": bridge}
        return [{"phase": "timing", "block": b, "variant": name,
                 "latency_ms": latency[name]}
                for b in range(2) for name in latency for _ in range(5)]

    def test_graph_second_boot_gate(self):
        result = evaluate_graph(self.make_graph_rows(430, 415, 420))
        self.assertTrue(result["decision"]["request_second_linux_boot"])
        self.assertAlmostEqual(result["single_graph_equivalent_fps"]["G2"], 1000 / 430)

    def test_graph_subpercent_gain_is_not_eligible(self):
        result = evaluate_graph(self.make_graph_rows(430, 429, 428))
        self.assertFalse(result["decision"]["request_second_linux_boot"])


if __name__ == "__main__":
    unittest.main()
