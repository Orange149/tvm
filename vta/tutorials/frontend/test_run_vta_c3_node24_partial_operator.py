"""Pure protocol checks for the frozen node24 operator comparison."""

import unittest

from run_vta_c3_node24_partial_operator import ORDERS, VERSIONS, analyze


class Node24OperatorProtocolTest(unittest.TestCase):
    def test_positions_and_versions(self):
        self.assertEqual(len(ORDERS), 6)
        for order in ORDERS:
            self.assertEqual(len(order), len(VERSIONS))
            self.assertEqual(set(order), set(VERSIONS))
        primary = tuple(name for name in VERSIONS if name != "input_alt")
        for position in range(6):
            self.assertEqual({tuple(name for name in order if name != "input_alt")[position]
                              for order in ORDERS}, set(primary))

    def test_eligible_winner_and_missing_samples(self):
        latencies = {
            "G2": 12.0, "input_fixed": 10.0, "input_alt": 9.0,
            "a12": 10.3, "f12": 8.5, "a13": 10.1, "f13": 8.0,
        }
        rows = [
            {"phase": "timing", "round": round_no, "variant": name,
             "latency_ms": value}
            for round_no in range(6) for name, value in latencies.items()
            for _ in range(3)
        ]
        result = analyze(rows)
        self.assertEqual(result["frozen_graph_winner"], "f13")
        self.assertTrue(result["decisions"]["f12"]["graph_eligible"])
        self.assertTrue(result["decisions"]["f13"]["graph_eligible"])
        with self.assertRaisesRegex(ValueError, "missing three-seed"):
            analyze(rows[:-1])

    def test_request_saving_does_not_imply_eligibility(self):
        latencies = {
            "G2": 9.0, "input_fixed": 10.0, "input_alt": 8.0,
            "a12": 10.0, "f12": 8.5, "a13": 10.0, "f13": 8.8,
        }
        rows = [
            {"phase": "timing", "round": round_no, "variant": name,
             "latency_ms": value}
            for round_no in range(6) for name, value in latencies.items()
            for _ in range(3)
        ]
        result = analyze(rows)
        self.assertIsNone(result["frozen_graph_winner"])


if __name__ == "__main__":
    unittest.main()
