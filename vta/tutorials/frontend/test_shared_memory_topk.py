"""Pure regression tests for shared-memory-aware Top-K selection."""

import unittest

from shared_memory_topk import (
    POLICY_EXTERNAL_K2,
    copy_realization,
    make_anchor_certificate,
    order_within_m1_band,
    policy_feature_row,
    shared_realization,
    topology_diverse,
    validate_certificate,
)


def topology(candidate="cpu-00-02_t4__vta-03-17_t1__cpu-18-20_t1", score=10.0):
    return {
        "candidate_id": candidate,
        "topology_id": "__".join(part.rsplit("_t", 1)[0] for part in candidate.split("__")),
        "path": [],
        "m1_score_ms": score,
        "edge_ids": ["a", "b"],
        "vta_islands": 1,
        "boundary_payload_bytes": 100,
        "theoretical_k2_slot_bytes": 512,
        "vta_logical_dma": {
            "load_buffer_2d_calls": 7,
            "store_buffer_2d_calls": 3,
            "load_buffer_2d_bytes": 1000,
            "store_buffer_2d_bytes": 200,
            "load_buffer_2d_inp_bytes": 600,
            "load_buffer_2d_wgt_bytes": 300,
            "load_buffer_2d_acc_bytes": 100,
            "store_buffer_2d_out_bytes": 200,
            "load_buffer_2d_inp_calls": 3,
            "load_buffer_2d_wgt_calls": 2,
            "load_buffer_2d_acc_calls": 2,
            "store_buffer_2d_out_calls": 3,
            "load_buffer_2d_small_calls": 2,
            "load_buffer_2d_strided_calls": 3,
            "store_buffer_2d_small_calls": 0,
            "store_buffer_2d_strided_calls": 1,
        },
        "vta_internal_cpu_helper_logical_access": {"read_bytes": 8, "write_bytes": 4},
    }


def certificate(row):
    return {
        "candidate_id": row["candidate_id"],
        "topology_id": row["topology_id"],
        "edge_ids": row["edge_ids"],
        "policy_id": POLICY_EXTERNAL_K2,
        "logical_boundary_bytes": 100,
        "additional_slot_bytes": 512,
        "functional_gate_passed": True,
        "correctness_boot_count": 3,
        "binding_status": "exact_candidate_three_boot_functional",
        "performance_claim_allowed": False,
        "boundary_api_service_ms": 0.04,
        "throughput_improvement_ms": -0.5,
        "throughput_improvement_ci95_ms": {"lower": -2.0, "upper": 1.0},
    }


class SharedMemoryPolicyTests(unittest.TestCase):
    def test_copy_and_shared_keep_logical_bytes_separate(self):
        row = topology()
        copy = copy_realization(row)
        shared = shared_realization(row, certificate(row))
        self.assertEqual(copy["logical_boundary_bytes"], shared["logical_boundary_bytes"])
        self.assertEqual(copy["framework_materialization_bytes"], 200)
        self.assertEqual(shared["framework_materialization_bytes"], 0)
        self.assertEqual(shared["additional_slot_bytes"], 512)
        self.assertIsNone(shared["physical_ddr_bytes"])

    def test_unknown_policy_fails_closed_to_copy(self):
        result = shared_realization(topology(), None)
        self.assertTrue(result["fallback_to_copy"])
        self.assertEqual(result["realized_policy"], "ordinary_copy")
        self.assertEqual(result["framework_materialization_bytes"], 200)

    def test_capacity_overflow_fails_closed(self):
        row = topology()
        result = shared_realization(row, certificate(row), capacity_bytes=511)
        self.assertTrue(result["fallback_to_copy"])
        self.assertEqual(result["binding_status"], "shared_slot_capacity_exceeded")

    def test_anchor_certificate_uses_measured_smaller_slot(self):
        row = topology()
        result = {
            "predicted_saved_bytes": 256,
            "modes": {
                "external": {"slot_actual": 512, "byte_exact": True},
                "anchor": {"slot_actual": 256, "byte_exact": True, "steady_growth": 0},
            },
        }
        anchor = make_anchor_certificate(row, certificate(row), result, "test")
        self.assertEqual(anchor["additional_slot_bytes"], 256)
        self.assertEqual(anchor["pool_anchor_saved_bytes"], 256)

    def test_cross_zero_latency_interval_is_not_used(self):
        row = topology()
        result = shared_realization(row, certificate(row))
        self.assertEqual(result["performance_status"], "latency_abstention")
        self.assertIsNone(result["throughput_delta_ms"])

    def test_structured_dma_features_preserve_memory_class_and_fragmentation(self):
        row = topology()
        result = policy_feature_row(row, copy_realization(row))
        self.assertTrue(result["detailed_dma_complete"])
        self.assertEqual(result["vta_load_inp_bytes"], 600)
        self.assertEqual(result["vta_load_wgt_bytes"], 300)
        self.assertEqual(result["vta_load_acc_bytes"], 100)
        self.assertEqual(result["vta_store_out_bytes"], 200)
        self.assertEqual(result["vta_fragmented_dma_calls"], 6)
        self.assertAlmostEqual(result["vta_average_load_request_bytes"], 1000 / 7)
        self.assertAlmostEqual(
            result["vta_load_request_density_per_mib"], 7 * (1 << 20) / 1000
        )

    def test_nonzero_calls_without_bytes_are_rejected(self):
        row = topology()
        row["vta_logical_dma"]["load_buffer_2d_bytes"] = 0
        with self.assertRaisesRegex(ValueError, "nonzero DMA calls"):
            policy_feature_row(row, copy_realization(row))

    def test_certificate_mismatch_rejected(self):
        row = topology()
        bad = certificate(row)
        bad["edge_ids"] = ["a"]
        with self.assertRaisesRegex(ValueError, "edge-set mismatch"):
            validate_certificate(bad, row)

    def test_topology_dedup_keeps_best_thread_realization(self):
        first = topology(score=11.0)
        second = topology(candidate="cpu-00-02_t4__vta-03-17_t1__cpu-18-20_t4", score=10.0)
        rows = topology_diverse([first, second])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["candidate_id"], second["candidate_id"])

    def test_m1_winner_remains_first_and_band_can_reorder(self):
        a = policy_feature_row(topology("cpu-00-00_t1__vta-01-01_t1__cpu-02-20_t1", 10.0), copy_realization(topology("cpu-00-00_t1__vta-01-01_t1__cpu-02-20_t1", 10.0)))
        b_top = topology("cpu-00-01_t1__vta-02-02_t1__cpu-03-20_t1", 10.1)
        b_top["boundary_payload_bytes"] = 400
        b = policy_feature_row(b_top, copy_realization(b_top))
        c_top = topology("cpu-00-02_t1__vta-03-03_t1__cpu-04-20_t1", 10.15)
        c_top["boundary_payload_bytes"] = 50
        c = policy_feature_row(c_top, copy_realization(c_top))
        ordered = order_within_m1_band([a, b, c], 0.02, ("framework_materialization_bytes",))
        self.assertEqual(ordered[0]["candidate_id"], a["candidate_id"])
        self.assertEqual(ordered[1]["candidate_id"], c["candidate_id"])


if __name__ == "__main__":
    unittest.main()
