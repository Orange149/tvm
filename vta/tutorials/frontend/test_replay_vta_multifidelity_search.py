#!/usr/bin/env python3

import copy
import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location(
    "multifidelity", ROOT / "replay_vta_multifidelity_search.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def resources(value):
    return {key: float(value) for key in MODULE.RESOURCE_KEYS}


def phase(status="ok", wall=1.0, reveal=None, resource=0.0):
    return {"status": status, "wall_ms": wall, "reveal": reveal,
            "resource_cost": resources(resource)}


def candidate(cid, workload, dma_bytes, dma_calls, latency=1.0, failure=None):
    phases = {}
    stopped = False
    for name in MODULE.PHASES:
        if stopped:
            phases[name] = {"status": "not_run", "wall_ms": None,
                            "reveal": None, "resource_cost": None}
            continue
        if name == failure:
            phases[name] = phase("invalid", reveal={"failure_category": "synthetic"})
            stopped = True
            continue
        reveal = None
        if name == "lower":
            reveal = {"features": {
                "load_dma_bytes": float(dma_bytes), "store_dma_bytes": 0.0,
                "load_dma_calls": float(dma_calls), "store_dma_calls": 0.0,
            }}
        if name == "fsim":
            reveal = {"features": {"submissions": 1.0}}
        phases[name] = phase(reveal=reveal, resource=1.0 if name in ("fpga", "measure") else 0.0)
    return {
        "candidate_id": cid, "workload_id": workload, "family_id": cid,
        "residence_mode": "original",
        "prelower": {"feature_provenance": "frozen_before_target_labels",
                     "features": {"knob_tile": float(dma_calls)},
                     "geometry_signature_sha256": "geometry-" + workload,
                     "hardware_fingerprint_sha256": "hardware",
                     "schedule_version": "schedule"},
        "oracle": {"phases": phases, "latency_ms": None if failure else float(latency)},
    }


def pool():
    train = [candidate("train-ok", "train", 10, 1),
             candidate("train-bad", "train", 20, 2, failure="lower")]
    test = [candidate("a", "test", 100, 1, latency=1.0),
            candidate("b", "test", 50, 4, latency=2.0),
            candidate("c", "test", 10, 1, failure="fsim")]
    return {
        "schema": MODULE.pool_schema.SCHEMA,
        "frozen_before_target_labels": True,
        "workloads": {
            "train": {"pool_oracle_latency_ms": 1.0, "candidates": train},
            "test": {"pool_oracle_latency_ms": 1.0, "candidates": test},
        },
    }


class MultiFidelityReplayTest(unittest.TestCase):
    def test_future_latency_cannot_change_first_action(self):
        value = pool()
        first = MODULE.replay(value, "test", "validity_frontier_service", 3, 2)
        poisoned = copy.deepcopy(value)
        for item in poisoned["workloads"]["test"]["candidates"]:
            if item["oracle"]["latency_ms"] is not None:
                item["oracle"]["latency_ms"] *= 1000
        second = MODULE.replay(poisoned, "test", "validity_frontier_service", 3, 2)
        self.assertEqual(first["actions"][0]["candidate_id"],
                         second["actions"][0]["candidate_id"])
        self.assertEqual(first["actions"][0]["phase"], "lower")

    def test_phase_failure_terminates_candidate(self):
        value = pool()
        run = MODULE.replay(value, "test", "exhaustive_then_service", 0, 2)
        failed = [item for item in run["actions"] if item["candidate_id"] == "c"]
        self.assertEqual([item["phase"] for item in failed], ["lower", "fsim"])
        self.assertEqual(failed[-1]["outcome"]["status"], "invalid")

    def test_exhaustive_pays_all_lowering_before_fsim(self):
        run = MODULE.replay(pool(), "test", "exhaustive_then_service", 0, 2)
        self.assertEqual([item["phase"] for item in run["actions"][:3]],
                         ["lower", "lower", "lower"])

    def test_target_training_is_observation_gated(self):
        value = pool()
        states = {item["candidate_id"]: MODULE.state_for(item)
                  for item in value["workloads"]["test"]["candidates"]}
        before = MODULE.training_examples(value, "test", "lower", states)
        self.assertEqual(len(before), 2)
        states["a"]["revealed"]["_outcome_lower"] = "ok"
        after = MODULE.training_examples(value, "test", "lower", states)
        self.assertEqual(len(after), 3)

    def test_renamed_target_geometry_is_excluded_from_training(self):
        value = pool()
        alias = copy.deepcopy(value["workloads"]["test"])
        for item in alias["candidates"]:
            item["workload_id"] = "alias"
        value["workloads"]["alias"] = alias
        examples = MODULE.training_examples(value, "test", "lower", {})
        self.assertEqual(len(examples), 2)

    def test_four_to_two_wave_compiles_two_before_first_fpga(self):
        run = MODULE.replay(
            pool(), "test", "hardware_diverse_frontier_service", 0,
            frontier_width=2, promotion_width=2,
        )
        first_fpga = next(index for index, item in enumerate(run["actions"])
                          if item["phase"] == "fpga")
        self.assertEqual(sum(item["phase"] == "compile"
                             for item in run["actions"][:first_fpga]), 2)

    def test_service_proxy_can_reject_tiny_many_request_candidate(self):
        value = pool()
        states = {item["candidate_id"]: MODULE.state_for(item)
                  for item in value["workloads"]["test"]["candidates"][:2]}
        for item in value["workloads"]["test"]["candidates"][:2]:
            state = states[item["candidate_id"]]
            state["revealed"]["lower"] = item["oracle"]["phases"]["lower"]["reveal"]
        # b has fewer bytes but four requests, so request service dominates.
        self.assertLess(MODULE.service_score(states["a"]), MODULE.service_score(states["b"]))

    def test_resource_and_phase_costs_are_counted_once(self):
        value = pool()
        run = MODULE.replay(value, "test", "exhaustive_then_service", 0, 2)
        metric = MODULE.prefix_to_target(run, 1.0, 2.0)
        prefix = run["actions"][:metric["actions_executed_until_stop"]]
        expected_wall = sum(item["outcome"]["wall_ms"] for item in prefix)
        self.assertEqual(metric["known_wall_ms"], expected_wall)
        expected_fpga = sum(item["outcome"]["resource_cost"]["fpga_kernel_invocations"]
                            for item in prefix)
        self.assertEqual(metric["resource_cost"]["fpga_kernel_invocations"], expected_fpga)


if __name__ == "__main__":
    unittest.main()
