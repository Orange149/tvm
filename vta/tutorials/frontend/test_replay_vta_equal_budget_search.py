#!/usr/bin/env python3

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("replay_vta_equal_budget_search.py")
SPEC = importlib.util.spec_from_file_location("equal_budget", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def resource(value):
    return {name: float(value) for name in MODULE.RESOURCE_KEYS}


def outcome(failure=None, latency=1.0, delta=0.0, costs=(1.0, 2.0, 3.0, 4.0, 5.0)):
    phases = {}
    stopped = False
    for phase, cost in zip(MODULE.PHASES, costs):
        if stopped:
            status = "not_run"
            wall = None
        elif failure == phase:
            status = "invalid"
            wall = cost
            stopped = True
        else:
            status = "ok"
            wall = cost
        phases[phase] = {
            "status": status,
            "wall_ms": wall,
            "resource_cost": None if status == "not_run" else resource(cost),
        }
    return {"phases": phases, "latency_ms": None if failure else latency,
            "delta_t_ms": None if failure else delta}


def candidate(cid, workload, rule, valid=0.5, delta_feature=0.0, result=None):
    return {
        "candidate_id": cid,
        "workload_id": workload,
        "residence_mode": "input_stationary",
        "same_tile_control_id": None,
        "predispatch": {
            "feature_provenance": "frozen_before_target_labels",
            "rule_score": rule,
            "knobs": {"tile": rule},
            "validity": {
                "valid": valid,
                "static_dma_total_bytes": float(rule),
                "static_dma_total_calls": float(rule),
                "static_padded_dma_calls": 0.0,
                "static_submissions": 1.0,
            },
            "delta_t": {"bytes": delta_feature},
        },
        "oracle": result,
    }


def pool():
    return {
        "schema": MODULE.POOL_SCHEMA,
        "pool_id": "synthetic",
        "frozen_before_target_labels": True,
        "delta_feature_sets": {
            "bytes_calls": ["bytes"],
            "request": ["bytes"],
            "command": ["bytes"],
        },
        "workloads": {
            "train": {
                "sealed_reference": {
                    "reference_kind": "tophub", "reference_id": "train-top",
                    "exact_identity": True, "latency_ms": 1.0,
                },
                "candidates": [
                    candidate("t-good", "train", 1, 1, -1, outcome(latency=.9, delta=-.1)),
                    candidate("t-bad", "train", 2, 0, 1, outcome("compile")),
                ],
            },
            "test": {
                "sealed_reference": {
                    "reference_kind": "tophub", "reference_id": "test-top",
                    "exact_identity": True, "latency_ms": 1.0,
                },
                "candidates": [
                    candidate("a", "test", 3, 1, -2, outcome(latency=.99, delta=-.1)),
                    candidate("b", "test", 1, 0, 2, outcome("lower")),
                    candidate("c", "test", 2, 1, -1, outcome(latency=1.2, delta=.1)),
                ],
            },
        },
    }


class EqualBudgetSearchTest(unittest.TestCase):
    def test_sealed_reference_in_candidate_list_rejected(self):
        value = pool()
        value["workloads"]["test"]["candidates"][0]["candidate_id"] = "test-top"
        with self.assertRaisesRegex(ValueError, "sealed reference"):
            MODULE.validate_pool(value)

    def test_tophub_requires_exact_identity(self):
        value = pool()
        value["workloads"]["test"]["sealed_reference"]["exact_identity"] = False
        with self.assertRaisesRegex(ValueError, "exact sealed identity"):
            MODULE.validate_pool(value)

    def test_oracle_mutation_cannot_change_first_selection(self):
        value = pool()
        candidates = value["workloads"]["test"]["candidates"]
        training = MODULE.held_out_training_records([value], "test")
        first, _ = MODULE.choose_next(
            "v_plus_delta_t", candidates, [], training, 7, value["delta_feature_sets"]
        )
        poisoned = copy.deepcopy(candidates)
        for item in poisoned:
            item["oracle"] = outcome(latency=1000.0 if item["candidate_id"] == first["candidate_id"] else .001)
        second, _ = MODULE.choose_next(
            "v_plus_delta_t", poisoned, [], training, 7, value["delta_feature_sets"]
        )
        self.assertEqual(first["candidate_id"], second["candidate_id"])

    def test_invalid_consumes_gross_budget_and_phase_costs_partition(self):
        value = pool()
        run = MODULE.run_offline_workload(value, "test", "rules_only", 0)
        self.assertEqual(run["dispatches"][0]["candidate_id"], "b")
        metrics = MODULE.prefix_metrics(run, value["workloads"]["test"], 1)
        self.assertEqual(metrics["gross_dispatches"], 1)
        self.assertEqual(metrics["outcome_counts"]["lower_invalid"], 1)
        self.assertEqual(metrics["wall_clock_known_total_ms"], 1.0)
        self.assertTrue(metrics["resource_cost_complete"])
        self.assertEqual(metrics["resource_cost_known_totals"]["logical_vta_load_bytes"], 1.0)
        self.assertIsNone(metrics["regret_ms"])

    def test_sealed_reference_only_affects_post_run_threshold(self):
        value = pool()
        run = MODULE.run_offline_workload(value, "test", "rules_only", 0)
        original_order = [item["candidate_id"] for item in run["dispatches"]]
        value["workloads"]["test"]["sealed_reference"]["latency_ms"] = .01
        rerun = MODULE.run_offline_workload(value, "test", "rules_only", 0)
        self.assertEqual(original_order, [item["candidate_id"] for item in rerun["dispatches"]])
        self.assertIsNone(MODULE.trials_to_threshold(rerun, value["workloads"]["test"], 5))

    def test_all_policies_and_summary_have_required_metrics(self):
        value = pool()
        MODULE.validate_pool(value)
        runs, summary = MODULE.run_offline(value, (1, 2), 2)
        self.assertEqual(len(runs), 2 * 2 * len(MODULE.POLICIES))
        self.assertEqual(set(summary), set(MODULE.POLICIES))
        for policy in MODULE.POLICIES:
            budget = summary[policy]["budgets"]["2"]
            self.assertIn("regret_percent_defined", budget)
            self.assertIn("pool_success_at_2_percent_rate", budget)
            self.assertIn("success_at_2_percent_rate", budget)
            self.assertIn("fpga_known_wall_ms", budget)
            self.assertIn("fsim_known_wall_ms", budget)
            self.assertEqual(budget["resource_cost_complete_rate"], 1.0)
            self.assertIn("trials_to_5_percent", summary[policy])
            self.assertIn("trials_to_pool_5_percent", summary[policy])

    def test_prospective_emits_only_next_identity(self):
        value = pool()
        target = copy.deepcopy(value)
        for candidate_value in target["workloads"]["test"]["candidates"]:
            candidate_value.pop("oracle")
        target["workloads"] = {"test": target["workloads"]["test"]}
        target["workloads"]["test"]["sealed_reference"]["latency_ms"] = None
        result = MODULE.prospective_dispatch(target, "test", "validity_v", 0, [], [value])
        self.assertEqual(result["status"], "dispatch_ready")
        self.assertFalse(result["oracle_fields_exposed_to_selector"])
        self.assertFalse(result["sealed_reference_used_by_selector"])

    def test_full_pool_oracle_is_derived_after_order(self):
        value = pool()
        value["workloads"]["test"]["sealed_reference"] = {
            "reference_kind": "full_pool_oracle", "reference_id": None,
            "exact_identity": False, "latency_ms": None,
        }
        MODULE.validate_pool(value)
        run = MODULE.run_offline_workload(value, "test", "rules_only", 0)
        self.assertEqual(MODULE.trials_to_threshold(run, value["workloads"]["test"], 2), 3)

    def test_delta_feature_ablation_aliases_are_frozen(self):
        self.assertEqual(MODULE.DELTA_POLICY_FEATURE_SET["v_plus_delta_t"], "bytes_calls")
        self.assertEqual(
            set(MODULE.DELTA_POLICY_FEATURE_SET.values()), {"bytes_calls", "request", "command"}
        )

    def test_paper_rule_selects_family_winners_without_oracle(self):
        value = pool()
        candidates = value["workloads"]["test"]["candidates"]
        for item in candidates:
            item["same_tile_control_id"] = "family"
            item["predispatch"]["validity"].update({
                "static_dma_total_bytes": 10.0 + item["predispatch"]["rule_score"],
                "static_dma_total_calls": item["predispatch"]["rule_score"],
                "static_padded_dma_calls": 0.0,
                "static_submissions": 1.0,
            })
        chosen, diagnostics = MODULE.choose_next(
            "paper_minimum_access", candidates, [], [], 0, value["delta_feature_sets"]
        )
        self.assertEqual(chosen["candidate_id"], "b")
        self.assertEqual(diagnostics["selection"], "same_tile_minimum_access_winner")

    def test_shared_memory_rule_uses_absolute_bytes_then_calls(self):
        value = pool()
        candidates = value["workloads"]["test"]["candidates"]
        for index, item in enumerate(candidates):
            item["predispatch"]["validity"].update({
                "static_dma_total_bytes": 5.0 if index < 2 else 6.0,
                "static_dma_total_calls": 2.0 if index == 0 else 1.0,
                "static_padded_dma_calls": 0.0,
                "static_submissions": 1.0,
            })
        chosen, _ = MODULE.choose_next(
            "shared_memory_lexicographic", candidates, [], [], 0, value["delta_feature_sets"]
        )
        self.assertEqual(chosen["candidate_id"], "b")

    def test_service_proxy_can_prefer_fewer_requests_over_fewer_bytes(self):
        value = pool()
        candidates = value["workloads"]["test"]["candidates"]
        metrics = (
            (1.0, 3.0, 1.0),
            (1000.0, 1.0, 1.0),
            (2000.0, 2.0, 2.0),
        )
        for item, (byte_count, calls, submissions) in zip(candidates, metrics):
            item["predispatch"]["validity"].update({
                "static_dma_total_bytes": byte_count,
                "static_dma_total_calls": calls,
                "static_padded_dma_calls": 0.0,
                "static_submissions": submissions,
            })
        chosen, diagnostics = MODULE.choose_next(
            "shared_memory_service_proxy", candidates, [], [], 0,
            value["delta_feature_sets"]
        )
        self.assertEqual(chosen["candidate_id"], "b")
        self.assertEqual(diagnostics["request_equivalent_bytes"], 64 * 1024)
        self.assertEqual(diagnostics["extra_submission_equivalent_bytes"], 128 * 1024)

    def test_output_is_immutable(self):
        value = pool()
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "existing"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                MODULE.write_offline(output, value, [], {}, {}, "cmd")


if __name__ == "__main__":
    unittest.main()
