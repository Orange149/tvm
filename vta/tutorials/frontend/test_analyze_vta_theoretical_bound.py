"""Tests for the offline VTA theoretical-bound model."""

import math
import unittest

from analyze_vta_theoretical_bound import analyze_workload, convolution_geometry, derive_hardware


class TheoreticalBoundTest(unittest.TestCase):
    def setUp(self):
        self.workload = [
            "conv2d_packed.vta",
            ["TENSOR", [1, 4, 56, 56, 1, 16], "int8"],
            ["TENSOR", [8, 4, 3, 3, 16, 16], "int8"],
            [2, 2],
            [1, 1, 1, 1],
            [1, 1],
            "NCHW1n16c",
            "int32",
        ]
        self.row = {
            "workload": self.workload,
            "config_index": 681,
            "unique_tensor_bytes": {
                "input_bytes": 200704,
                "weight_bytes": 73728,
                "output_bytes": 100352,
            },
            "static_dma": {
                "totals": {"load_buffer_2d_bytes": 717824, "store_buffer_2d_bytes": 100352}
            },
        }
        self.hardware = derive_hardware({"LOG_BLOCK": 4}, 100, 128)

    def test_hardware_peak_and_ideal_axi_bandwidth(self):
        self.assertEqual(self.hardware["block_in"], 16)
        self.assertEqual(self.hardware["block_out"], 16)
        self.assertEqual(self.hardware["peak_macs_per_second"], 25_600_000_000)
        self.assertEqual(
            self.hardware["optimistic_unidirectional_axi_bytes_per_second"], 1_600_000_000
        )

    def test_packed_convolution_geometry_and_macs(self):
        geometry = convolution_geometry(self.workload)
        self.assertEqual(geometry["output_height"], 28)
        self.assertEqual(geometry["output_width"], 28)
        self.assertEqual(geometry["input_channels"], 64)
        self.assertEqual(geometry["output_channels"], 128)
        self.assertEqual(geometry["macs"], 28 * 28 * 64 * 128 * 3 * 3)

    def test_overlap_and_no_overlap_formulas(self):
        result = analyze_workload(self.row, self.hardware, "W01")
        expected_compute = (28 * 28 * 64 * 128 * 3 * 3) / 25_600_000_000 * 1000
        expected_read = 717824 / 1_600_000_000 * 1000
        expected_write = 100352 / 1_600_000_000 * 1000
        expected_shared_dma = 818176 / 1_600_000_000 * 1000
        self.assertTrue(math.isclose(result["compute_lower_bound_ms"], expected_compute))
        self.assertTrue(
            math.isclose(result["current_static_full_duplex_read_lower_bound_ms"], expected_read)
        )
        self.assertTrue(
            math.isclose(result["current_static_full_duplex_write_lower_bound_ms"], expected_write)
        )
        self.assertEqual(
            result["current_static_perfect_overlap_lower_bound_ms"],
            max(expected_compute, expected_read, expected_write),
        )
        self.assertTrue(
            math.isclose(
                result["current_static_no_overlap_reference_ms"],
                expected_compute + expected_read + expected_write,
            )
        )
        self.assertTrue(
            math.isclose(
                result["current_static_shared_serial_dma_scenario_ms"], expected_shared_dma
            )
        )

    def test_current_static_traffic_cannot_be_below_unique_bytes(self):
        result = analyze_workload(self.row, self.hardware, "W01")
        self.assertGreaterEqual(
            result["current_static_total_dma_bytes"], result["irreducible_unique_total_bytes"]
        )
        self.assertGreaterEqual(result["current_static_traffic_amplification"], 1.0)


if __name__ == "__main__":
    unittest.main()
