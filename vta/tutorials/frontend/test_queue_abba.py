import math
import unittest

from analyze_queue_abba import interval
from run_queue_capacity_abba import metrics


class QueueABBATests(unittest.TestCase):
    def test_warmup_discard_and_intervals(self):
        rows = [dict(frame_id=i, completion_ms=i*10., total_latency_ms=10000. if i < 2 else 50.) for i in range(302)]
        result = metrics(rows)
        self.assertEqual(result["ii_ms"], 10)
        self.assertEqual(result["fps"], 100)
        self.assertEqual(result["submit_to_complete_p95_ms"], 50)
        self.assertEqual(result["scored_intervals"], 299)

    def test_bad_count_or_order_rejected(self):
        with self.assertRaises(AssertionError):
            metrics([])
        rows = [dict(frame_id=i, completion_ms=0., total_latency_ms=1.) for i in range(302)]
        with self.assertRaises(AssertionError):
            metrics(rows)

    def test_no_claim_from_one_or_two_boots(self):
        self.assertIsNone(interval([0.])["noninferiority_pass"])
        self.assertIsNone(interval([0., 0.])["upper95_ii_change_pct"])

    def test_margin_and_uncertainty(self):
        self.assertTrue(interval([math.log(1.01)] * 3)["noninferiority_pass"])
        self.assertFalse(interval([math.log(1.03)] * 3)["noninferiority_pass"])
        self.assertFalse(interval([-.1, 0., .1])["noninferiority_pass"])


if __name__ == "__main__":
    unittest.main()
