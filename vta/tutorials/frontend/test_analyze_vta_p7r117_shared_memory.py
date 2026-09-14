#!/usr/bin/env python3

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("analyze_vta_p7r117_shared_memory.py")
SPEC = importlib.util.spec_from_file_location("p7r117_shared_memory", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SharedMemoryAnalysisTest(unittest.TestCase):
    def test_page_alignment(self):
        self.assertEqual(MODULE.aligned(1), 4096)
        self.assertEqual(MODULE.aligned(4096), 4096)
        self.assertEqual(MODULE.aligned(4097), 8192)

    def test_distribution_even_count(self):
        self.assertEqual(
            MODULE.distribution([1, 7, 3, 5]),
            {"count": 4, "min": 1.0, "median": 4.0, "max": 7.0},
        )

    def test_empty_range_formats_as_missing(self):
        empty = MODULE.distribution([])
        self.assertEqual(MODULE.format_range(empty), "--")


if __name__ == "__main__":
    unittest.main()
