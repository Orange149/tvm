"""Tests for the frozen three-boot C1 shared-memory Top-K analysis."""

import unittest

from analyze_vta_shared_memory_topk_board import analyze


class BoardAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.copy = "candidate::ordinary_copy"
        self.shared = "candidate::pool_anchor_k2_shared"
        self.contract = {
            "union_configuration_ids": [self.copy, self.shared],
            "strategies": {
                "S0_m1_copy": [self.copy],
                "S1_m1_payload": [self.copy],
                "S2_m1_certified_shared": [self.shared],
            },
        }

    def boots(self, shared_ii=9.0):
        result = []
        for index in range(3):
            result.append({
                "schema": "vta_shared_memory_topk_board_boot_v1",
                "selection_contract_sha256": "hash",
                "boot_id": "boot{}".format(index),
                "rows": [
                    {
                        "configuration_id": self.copy,
                        "correct": True,
                        "ii_ms": 10.0,
                        "framework_materialization_bytes": 200,
                        "additional_slot_bytes": 0,
                        "vta_internal_dma_bytes": 1200,
                        "vta_dma_calls": 10,
                    },
                    {
                        "configuration_id": self.shared,
                        "correct": True,
                        "ii_ms": shared_ii,
                        "framework_materialization_bytes": 0,
                        "additional_slot_bytes": 256,
                        "vta_internal_dma_bytes": 1200,
                        "vta_dma_calls": 10,
                    },
                ],
            })
        return result

    def test_positive_selection_and_mechanism_gate(self):
        result = analyze(self.contract, self.boots(), "hash")
        self.assertTrue(result["gates"]["shared_mechanism_invariants_passed"])
        self.assertTrue(result["gates"]["selection_efficiency_claim_allowed"])

    def test_negative_result_is_retained(self):
        result = analyze(self.contract, self.boots(shared_ii=11.0), "hash")
        self.assertFalse(result["gates"]["selection_efficiency_claim_allowed"])
        self.assertIn("negative_ranking", result["conclusion"])

    def test_dma_change_blocks_claim_even_when_shared_is_faster(self):
        boots = self.boots()
        for boot in boots:
            boot["rows"][1]["vta_internal_dma_bytes"] = 1199
        result = analyze(self.contract, boots, "hash")
        self.assertFalse(result["gates"]["shared_mechanism_invariants_passed"])
        self.assertFalse(result["gates"]["selection_efficiency_claim_allowed"])

    def test_duplicate_boot_rejected(self):
        boots = self.boots()
        boots[2]["boot_id"] = boots[1]["boot_id"]
        with self.assertRaisesRegex(ValueError, "distinct"):
            analyze(self.contract, boots, "hash")


if __name__ == "__main__":
    unittest.main()
