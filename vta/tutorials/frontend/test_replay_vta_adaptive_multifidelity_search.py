#!/usr/bin/env python3

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import replay_vta_adaptive_multifidelity_search as adaptive
from test_replay_vta_multifidelity_search import pool


class AdaptiveMultiFidelityReplayTest(unittest.TestCase):
    def grouped_pool(self):
        value = pool()
        for row in value["workloads"]["test"]["candidates"]:
            row["family_id"] = "same-tile-family"
        return value

    def test_sparse_path_is_selected_from_observed_lowering_survival(self):
        value = self.grouped_pool()
        rows = value["workloads"]["test"]["candidates"]
        for row in rows[:2]:
            row["oracle"]["phases"]["lower"]["status"] = "invalid"
            row["oracle"]["phases"]["lower"]["reveal"] = {
                "failure_category": "synthetic"
            }
            for phase in ("fsim", "compile", "fpga", "measure"):
                row["oracle"]["phases"][phase] = {
                    "status": "not_run", "wall_ms": None,
                    "reveal": None, "resource_cost": None,
                }
            row["oracle"]["latency_ms"] = None
        run = adaptive.replay(value, "test", 0)
        self.assertEqual(run["first_family_lower_passes"], 1)
        self.assertEqual(run["adaptive_path"], "sparse_family_wave")
        self.assertEqual([action["phase"] for action in run["actions"][:3]],
                         ["lower", "lower", "lower"])

    def test_dense_path_is_selected_without_future_latency_visibility(self):
        value = self.grouped_pool()
        first = adaptive.replay(value, "test", 0)
        poisoned = copy.deepcopy(value)
        for row in poisoned["workloads"]["test"]["candidates"]:
            if row["oracle"]["latency_ms"] is not None:
                row["oracle"]["latency_ms"] *= 1000
        second = adaptive.replay(poisoned, "test", 0)
        self.assertEqual(first["first_family_lower_passes"], 3)
        self.assertEqual(first["adaptive_path"], "dense_fixed_4_to_2")
        self.assertEqual(
            [(row["candidate_id"], row["phase"]) for row in first["actions"][:3]],
            [(row["candidate_id"], row["phase"]) for row in second["actions"][:3]],
        )


if __name__ == "__main__":
    unittest.main()
