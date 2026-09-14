"""Unit tests for the C3 historical shortlist replay."""

from __future__ import annotations

import unittest

from plan_vta_safe_tuning import order_candidates, pareto_fronts, replay_prefix, run_replay


def _candidate(candidate_id, outcome, position, total_bytes=None, calls=None, incumbent=False):
    features = None
    if outcome == "pass":
        features = {
            "total_dma_bytes": total_bytes,
            "load_bytes": total_bytes - 10,
            "store_bytes": 10,
            "load_calls": calls,
            "store_calls": 1,
            "small_load_calls": calls // 2,
            "strided_load_calls": calls // 3,
            "padded_load_calls": calls // 4,
            "input_reload_ratio": float(calls),
            "weight_reload_ratio": float(total_bytes),
        }
    return {
        "candidate_id": candidate_id,
        "pool_position": position,
        "is_incumbent": incumbent,
        "outcome": outcome,
        "latency_ms": float(total_bytes) if outcome == "pass" else None,
        "request_features": features,
    }


class SafeTuningReplayTest(unittest.TestCase):
    def setUp(self):
        self.workload = {
            "workload_id": "W00",
            "workload": ["conv2d_packed.vta"],
            "incumbent_index": 0,
            "candidates": [
                _candidate("incumbent", "pass", 0, 100, 4, incumbent=True),
                _candidate("compile", "compile", 1),
                _candidate("wrong", "wrong_answer", 2),
                _candidate("low-bytes", "pass", 3, 80, 8),
            ],
        }

    def test_b4_keeps_all_candidates_and_incumbent(self):
        ordered = order_candidates(self.workload, "B4")
        self.assertEqual(len(ordered), 4)
        self.assertTrue(any(item["is_incumbent"] for item in ordered))
        self.assertEqual([item["outcome"] for item in ordered], ["pass", "pass", "wrong_answer", "compile"])

    def test_b5_orders_valid_by_total_bytes(self):
        ordered = order_candidates(self.workload, "B5")
        self.assertEqual(ordered[0]["candidate_id"], "low-bytes")

    def test_b6_pareto_does_not_delete_candidates(self):
        ordered = order_candidates(self.workload, "B6")
        self.assertEqual(len(ordered), 4)
        self.assertEqual({item["candidate_id"] for item in ordered}, {"incumbent", "compile", "wrong", "low-bytes"})

    def test_pareto_fronts_respect_multiple_features(self):
        valid = [item for item in self.workload["candidates"] if item["outcome"] == "pass"]
        fronts = pareto_fronts(valid, ("total_dma_bytes", "load_calls"))
        self.assertEqual(fronts["incumbent"], 0)
        self.assertEqual(fronts["low-bytes"], 0)

    def test_budget_counts_failed_positions(self):
        metrics = replay_prefix(self.workload, self.workload["candidates"], 4)
        self.assertEqual(metrics["attempted_positions"], 4)
        self.assertEqual(metrics["outcomes"]["pass"], 2)
        self.assertEqual(metrics["outcomes"]["compile"], 1)
        self.assertEqual(metrics["outcomes"]["wrong_answer"], 1)

    def test_random_replay_keeps_incumbent_and_uses_all_permutations(self):
        pool = {"workloads": [self.workload]}
        results = run_replay(pool, (4,), random_permutations=1000, random_seed=17)
        random_result = next(result for result in results if result["method"] == "B2")
        self.assertEqual(random_result["metrics"]["permutations"], 1000)
        self.assertEqual(random_result["metrics"]["incumbent_seen_rate"], 1.0)

    def test_post_dispatch_rankings_are_never_marked_evaluable(self):
        pool = {"workloads": [self.workload]}
        results = run_replay(pool, (4,), random_permutations=1000, random_seed=17)
        status = {result["method"]: result["evaluation_status"] for result in results}
        self.assertEqual(status["B2"], "evaluable_historical_random_baseline")
        self.assertEqual(status["B4"], "diagnostic_oracle_upper_bound_leaky")
        self.assertEqual(status["B5"], "not_evaluable_missing_prefeatures")
        self.assertEqual(status["B6"], "not_evaluable_missing_prefeatures")
        for result in results:
            if result["method"] != "B2":
                self.assertNotIn("metrics", result)
                self.assertIn("diagnostic_metrics_leaky", result)


if __name__ == "__main__":
    unittest.main()
