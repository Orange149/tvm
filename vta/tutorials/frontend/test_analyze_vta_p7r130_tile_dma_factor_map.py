#!/usr/bin/env python3

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("analyze_vta_p7r130_tile_dma_factor_map.py")
SPEC = importlib.util.spec_from_file_location("p7r130", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(cid, family, mode, tile_w, load_bytes, calls, latency=None):
    value = {
        "workload_id": "Y",
        "family_id": family,
        "candidate_id": cid,
        "mode": mode,
        "static_status": "ok",
        "fsim_status": "passed",
        "fpga_status": "passed",
        "latency_ms": latency,
        "load_dma_bytes": load_bytes,
        "load_dma_calls": calls,
    }
    value.update({name: 1 for name in MODULE.KNOBS})
    value["tile_w"] = tile_w
    for name in MODULE.DMA_FIELDS:
        value.setdefault(name, 1)
    return value


class FactorMapTest(unittest.TestCase):
    def test_residence_pair_is_same_family_control(self):
        rows = [row("a", "F0", "original", 1, 100, 10, 10),
                row("b", "F0", "input_stationary", 1, 50, 5, 8)]
        pair = MODULE.residence_pairs(rows)[0]
        self.assertEqual(pair["load_dma_bytes_delta_fraction"], -0.5)
        self.assertAlmostEqual(pair["latency_delta_fraction"], -0.2)

    def test_one_knob_pair_rejects_two_knob_change(self):
        left = row("a", "F0", "original", 1, 100, 10)
        right = row("b", "F1", "original", 2, 50, 5)
        self.assertEqual(len(MODULE.one_knob_pairs([left, right])), 1)
        right["tile_h"] = 2
        self.assertEqual(MODULE.one_knob_pairs([left, right]), [])

    def test_alignment(self):
        self.assertEqual(MODULE.aligned_4k(1), 4096)
        self.assertEqual(MODULE.aligned_4k(4097), 8192)


if __name__ == "__main__":
    unittest.main()
