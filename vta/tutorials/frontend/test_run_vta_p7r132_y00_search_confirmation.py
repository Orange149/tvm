#!/usr/bin/env python3

import copy
import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("run_vta_p7r132_y00_search_confirmation.py")
SPEC = importlib.util.spec_from_file_location("p7r133", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class P7R133Test(unittest.TestCase):
    def test_timing_orders_cover_every_candidate_every_round(self):
        ids = ["a", "b", "c"]
        orders = MODULE.timing_orders(ids, rounds=7)
        self.assertEqual(len(orders), 7)
        self.assertTrue(all(set(order) == set(ids) for order in orders))
        self.assertTrue(all(len(order) == len(ids) for order in orders))

    def test_completed_pool_rejects_failed_and_labels_passed(self):
        contract = MODULE.read_json(MODULE.DEFAULT_CONTRACT / "prospective_pool.json")
        candidates = contract["workloads"]["Y00"]["candidates"]
        correctness = []
        stats = {}
        for index, candidate in enumerate(candidates):
            passed = index == 0
            correctness.append({
                "candidate_id": candidate["candidate_id"],
                "status": "passed" if passed else "failed",
                "seeds": [{"correct": passed}],
            })
            if passed:
                stats[candidate["candidate_id"]] = {"median_ms": 1.0}
        timing = {"candidate_stats": stats}
        complete = MODULE.completed_pool(copy.deepcopy(contract), correctness, timing)
        outcomes = [row["oracle"] for row in complete["workloads"]["Y00"]["candidates"]]
        self.assertEqual(sum(item["latency_ms"] is not None for item in outcomes), 1)
        self.assertEqual(sum(item["phases"]["fpga"]["status"] == "invalid" for item in outcomes), 10)

    def test_logical_resource_cost_and_five_phase_pool(self):
        prospective = MODULE.read_json(MODULE.DEFAULT_CONTRACT / "prospective_pool.json")
        candidates = prospective["workloads"]["Y00"]["candidates"]
        profile = {
            "load_buffer_2d_bytes": 100,
            "store_buffer_2d_bytes": 20,
            "load_buffer_2d_calls": 4,
            "store_buffer_2d_calls": 1,
            "driver_run_calls": 1,
        }
        correctness = []
        stats = {}
        timing_rows = []
        qualification = {}
        compile_walls = {}
        for index, candidate in enumerate(candidates):
            candidate_id = candidate["candidate_id"]
            passed = index == 0
            correctness.append({
                "candidate_id": candidate_id,
                "status": "passed" if passed else "failed",
                "seeds": [{
                    "correct": passed,
                    "host_wall_ms": 7.0,
                    "runtime_profile_complete": profile,
                }],
            })
            qualification[candidate_id] = {"lower_wall_ms": 1.0, "fsim_wall_ms": 2.0}
            compile_walls[candidate_id] = 3.0
            if passed:
                stats[candidate_id] = {"median_ms": 1.0}
                timing_rows.append({
                    "candidate_id": candidate_id,
                    "host_wall_ms": 8.0,
                    "runtime_profile_complete": profile,
                })
        complete = MODULE.completed_pool(
            copy.deepcopy(prospective), correctness, {"candidate_stats": stats},
            qualification, compile_walls, timing_rows,
        )
        first = complete["workloads"]["Y00"]["candidates"][0]["oracle"]
        self.assertEqual(set(first["phases"]), {"lower", "fsim", "compile", "fpga", "measure"})
        self.assertEqual(first["phases"]["fpga"]["resource_cost"]["logical_vta_load_bytes"], 100)
        self.assertEqual(first["phases"]["measure"]["wall_ms"], 8.0)

    def test_bound_prequalification_costs_cover_frozen_pool(self):
        contract = MODULE.read_json(MODULE.DEFAULT_CONTRACT / "board_collection_contract.json")
        costs, summary = MODULE.qualification_costs(
            MODULE.DEFAULT_CONTRACT,
            contract["candidate_order_for_unbiased_full_label_collection"],
        )
        self.assertEqual(len(costs), 11)
        self.assertEqual(summary["static_identity_count"], 24)
        self.assertEqual(summary["static_pass_count"], 18)
        self.assertEqual(summary["fsim_pass_count"], 11)
        self.assertGreater(summary["static_known_wall_ms"], 0)
        self.assertGreater(summary["fsim_known_wall_ms"], 0)

    def test_profile_cost_counts_logical_dma_and_kernel_invocations(self):
        cost = MODULE.profile_resource_cost({
            "load_buffer_2d_bytes": 1000,
            "store_buffer_2d_bytes": 200,
            "load_buffer_2d_calls": 7,
            "store_buffer_2d_calls": 3,
            "driver_run_calls": 2,
        })
        self.assertEqual(cost["logical_vta_load_bytes"], 1000)
        self.assertEqual(cost["logical_vta_store_bytes"], 200)
        self.assertEqual(cost["logical_vta_dma_calls"], 10)
        self.assertEqual(cost["fpga_kernel_invocations"], 2)

    def test_representative_timing_cost_uses_median_observation(self):
        rows = []
        for wall, load in ((9.0, 300), (7.0, 100), (8.0, 200)):
            rows.append({
                "host_wall_ms": wall,
                "runtime_profile_complete": {
                    "load_buffer_2d_bytes": load,
                    "store_buffer_2d_bytes": 20,
                    "load_buffer_2d_calls": 4,
                    "store_buffer_2d_calls": 1,
                    "driver_run_calls": 2,
                },
            })
        cost = MODULE.representative_timing_cost(rows)
        self.assertEqual(cost["wall_ms"], 8.0)
        self.assertEqual(cost["resource_cost"]["logical_vta_load_bytes"], 200)
        self.assertEqual(cost["resource_cost"]["fpga_kernel_invocations"], 2)


if __name__ == "__main__":
    unittest.main()
