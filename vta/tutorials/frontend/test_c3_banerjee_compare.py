import unittest
from unittest.mock import patch
import c3_banerjee_compare_board as c
import c3_residency_tradeoff as p
import c3_residency_tradeoff_board as base


class ComparisonTests(unittest.TestCase):
    def test_balance(self):
        self.assertTrue(c.balance()["positions_once"])
        self.assertEqual(len(c.PROGRAMS), 6)

    def test_counts(self):
        self.assertEqual(c.expectations("Ordinary-DB")["mac"], 896)
        self.assertEqual(c.expectations("Banerjee-DB-Adapted")["wgt_payload_bytes"], 2016 * 1024)
        self.assertEqual(c.expectations(c.NEW)["wgt_loads"], 40)
        self.assertEqual(c.expectations("R13-G")["wgt_loads"], 34)
        with self.assertRaises(ValueError):
            c.expectations("F13-B")

    def test_no_global_mutation(self):
        self.assertNotIn("Ordinary-DB", p.PROGRAMS)
        self.assertIsNot(c.Runner.__init__, base.Runner.__init__)
        self.assertEqual(p.SCHEMA, "c3_node24_residency_instruction_tradeoff_v1_1")

    def test_invalid_signature(self):
        signature = c.expectations("Ordinary-DB")
        profile = dict(push_gemm_op_calls=1008, driver_run_calls=1)
        self.assertEqual(c.verify_profile("Ordinary-DB", profile, signature)["mac"], 896)
        profile["push_gemm_op_calls"] = 1009
        with self.assertRaises(ValueError):
            c.verify_profile("Ordinary-DB", profile, signature)

    def test_requests_not_zero_or_ignored(self):
        with self.assertRaises(ValueError):
            base.validate_request_profile({"load_buffer_2d_wgt_bytes": 0},
                                          {"load_buffer_2d_wgt_bytes": 2064384})

    def test_seed_positions(self):
        self.assertEqual([p.seed_order(r)[0] for r in range(6)], list(p.SEEDS) * 2)


if __name__ == "__main__":
    unittest.main()
