import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from run_queue_capacity import batch_signature, queue_records, suggested_capacity


class QueueCapacityTests(unittest.TestCase):
    def test_production_policy(self):
        source = Path(__file__).resolve().parents[2] / "apps/native_deploy/test_queue_capacity.cc"
        with tempfile.TemporaryDirectory(prefix="vta_queue_test_") as directory:
            binary = Path(directory) / "queue_test"
            subprocess.run(["g++", "-std=c++11", "-Wall", "-Wextra", "-Werror", str(source),
                            "-o", str(binary)], check=True)
            result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
            self.assertIn("19 queue capacity policy checks passed", result.stdout)

    def test_margin_alignment_and_cap(self):
        self.assertEqual(suggested_capacity(0), 4096)
        self.assertEqual(suggested_capacity(2792), 5632)
        self.assertEqual(suggested_capacity(5312), 10752)
        self.assertEqual(suggested_capacity(1 << 25), 1 << 25)

    def test_diagnostic_parser_does_not_count_other_logging(self):
        self.assertEqual(queue_records(b'other\n[VTA_QUEUE] {"submit":1}\n'), [{"submit": 1}])
        with self.assertRaises(json.JSONDecodeError):
            queue_records(b'[VTA_QUEUE] bad\n')

    def test_batch_multiplicity_not_global_island_order(self):
        a = dict(insn_bytes=16, uop_bytes=4, load_bytes=256, store_bytes=256, reason="explicit_sync")
        b = dict(a, insn_bytes=32)
        self.assertEqual(batch_signature([a, b, a]), batch_signature([b, a, a]))
        self.assertNotEqual(batch_signature([a, b, a]), batch_signature([b, a]))
        self.assertNotEqual(batch_signature([a]), batch_signature([dict(a, load_bytes=512)]))


if __name__ == "__main__":
    unittest.main()
