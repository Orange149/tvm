#!/usr/bin/env python3

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("prepare_vta_p7r132_y00_search_confirmation.py")
SPEC = importlib.util.spec_from_file_location("p7r132", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class P7R132Test(unittest.TestCase):
    def test_default_real_pool_is_valid_and_has_no_oracle(self):
        directory = MODULE.DEFAULT_LOCAL
        pool, eligible = MODULE.build_pool(
            MODULE.read_jsonl(directory / "candidates_v2.jsonl"),
            MODULE.read_jsonl(directory / "static_results.jsonl"),
            MODULE.read_jsonl(directory / "fsim_results.jsonl"),
        )
        self.assertEqual(len(eligible), 11)
        MODULE.validate_pool(pool, require_oracle=False)
        self.assertTrue(all("oracle" not in row for row in pool["workloads"]["Y00"]["candidates"]))

    def test_lexicographic_order_is_complete_and_unique(self):
        directory = MODULE.DEFAULT_LOCAL
        pool, eligible = MODULE.build_pool(
            MODULE.read_jsonl(directory / "candidates_v2.jsonl"),
            MODULE.read_jsonl(directory / "static_results.jsonl"),
            MODULE.read_jsonl(directory / "fsim_results.jsonl"),
        )
        order = MODULE.lexicographic_order(pool)
        self.assertEqual(len(order), len(set(order)))
        self.assertEqual(set(order), set(eligible))


if __name__ == "__main__":
    unittest.main()
